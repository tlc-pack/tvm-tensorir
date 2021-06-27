/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file lower_match_buffer.cc
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../ir/functor_common.h"

namespace tvm {
namespace tir {
class MatchBufferLower : public StmtExprMutator {
 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    for (const MatchBufferRegion& match_buffer : op->match_buffers) {
      CheckAndUpdateVarMap(match_buffer);
    }

    Stmt stmt = StmtExprMutator ::VisitStmt_(op);
    op = stmt.as<BlockNode>();
    ICHECK(op != nullptr);
    Array<BufferRegion> reads = MutateArray(
        op->reads, std::bind(&MatchBufferLower::VisitBufferRegion, this, std::placeholders::_1));
    Array<BufferRegion> writes = MutateArray(
        op->writes, std::bind(&MatchBufferLower::VisitBufferRegion, this, std::placeholders::_1));

    if (reads.same_as(op->reads) && writes.same_as(op->writes) && op->match_buffers.empty()) {
      return stmt;
    } else {
      auto n = CopyOnWrite(op);
      n->match_buffers = {};
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      return Stmt(n);
    }
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var v = GetRef<Var>(op);
    auto it = var_map_.find(v);
    if (it != var_map_.end()) {
      return it->second;
    } else {
      return std::move(v);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();
    ICHECK(op != nullptr);

    auto it = match_buffers_.find(op->buffer);
    if (it == match_buffers_.end()) {
      return stmt;
    } else {
      const Buffer& buffer = it->first;
      const BufferRegion& source = it->second;

      auto n = CopyOnWrite(op);
      n->indices = ConvertIndices(op->indices, MatchBufferRegion(buffer, source));
      n->buffer = source->buffer;
      return Stmt(n);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();
    ICHECK(op != nullptr);

    auto it = match_buffers_.find(op->buffer);
    if (it == match_buffers_.end()) {
      return expr;
    } else {
      const Buffer& buffer = it->first;
      const BufferRegion& source = it->second;
      Array<PrimExpr> indices = ConvertIndices(op->indices, MatchBufferRegion(buffer, source));
      return BufferLoad(source->buffer, indices);
    }
  }

  BufferRegion VisitBufferRegion(const BufferRegion& buffer_region) const {
    const Buffer& buffer = buffer_region->buffer;
    auto it = match_buffers_.find(buffer);
    if (it == match_buffers_.end()) {
      return buffer_region;
    } else {
      const BufferRegion& source = it->second;

      Region region;
      region.reserve(source->region.size());
      size_t target_cur_pos = 0;
      for (const Range& source_range : source->region) {
        if (is_one(source_range->extent)) {
          // This dimesion is ignored by the target buffer
          region.push_back(source_range);
        } else {
          // Check the current pos is less than the indices size.
          ICHECK_LT(target_cur_pos, buffer_region->region.size());
          const Range& target_range = buffer_region->region[target_cur_pos];
          region.push_back(
              Range::FromMinExtent(source_range->min + target_range->min, target_range->extent));
        }
      }
      // Check the shape is exactly matched.
      ICHECK_EQ(target_cur_pos, buffer_region->region.size());
      return BufferRegion(buffer, region);
    }
  }

 private:
  void CheckAndUpdateVarMap(const MatchBufferRegion& match_buffer) {
    // Step.1. Check
    const Buffer& buffer = match_buffer->buffer;
    const BufferRegion& source = match_buffer->source;
    const Buffer& source_buffer = source->buffer;

    // Step.1.1. Check scope & dtype
    ICHECK_EQ(buffer->scope, source_buffer->scope)
        << "MatchBuffer " << buffer << " scope mismatch:" << buffer->scope << "vs."
        << source_buffer->scope;
    ICHECK_EQ(buffer->dtype, source_buffer->dtype)
        << "MatchBuffer " << buffer << " data type mismatch:" << buffer->dtype << "vs."
        << source_buffer->dtype;

    // Step.1.2. Check data alignment
    if (source_buffer->data_alignment % buffer->data_alignment != 0) {
      LOG(WARNING) << "Trying to bind buffer to another one with lower alignment requirement "
                   << " required_alignment=" << buffer->data_alignment
                   << ", provided_alignment=" << source_buffer->data_alignment;
    }
    if (is_zero(buffer->elem_offset)) {
      ICHECK(is_zero(source_buffer->elem_offset))
          << "Trying to bind a Buffer with offset into one without offset "
          << " required elem_offset=" << buffer->elem_offset
          << ", provided elem_offset=" << source_buffer->elem_offset;
    }

    // Step.1.3. Check offset_factor
    // TODO(Siyuan): check offset_factor

    // Step.1.4. Check dimension
    // Note that matching from high-dimensional buffer to low-dimensional buffer is allowed.
    // e.g. A(4, 4) = B[i: i + 4, j, k : k + 4]
    ICHECK_EQ(source_buffer->shape.size(), source->region.size());
    ICHECK(CheckMatchDimension(match_buffer))
        << "The dimension mismatched in match buffer: " << match_buffer->buffer;

    // Step.2. Update
    match_buffers_[buffer] = source;
    // Step.2.1. Update buffer data
    Bind(buffer->data, source_buffer->data, buffer->name + ".data");

    // Step.2.2. Update element offset
    // Note we create Load via vload and try to reuse index calculate.
    {
      Array<PrimExpr> indices;
      indices.reserve(source->region.size());
      for (const Range& range : source->region) {
        indices.push_back(range->min);
      }

      Load load = Downcast<Load>(source_buffer.vload(indices, source_buffer->dtype));
      Bind(buffer->elem_offset, load->index, buffer->name + ".elem_offset");
    }

    // Step 2.3. Check and update strides
    // Check if target buffer strides are defined
    if (!buffer->strides.empty()) {
      ICHECK_EQ(buffer->strides.size(), buffer->shape.size());
      PrimExpr stride = make_const(DataType::Int(32), 1);
      size_t dim = buffer->strides.size();
      for (size_t i = source_buffer->shape.size(); i > 0; --i) {
        const PrimExpr& shape = source_buffer->shape[i - 1];
        const Range& range = source->region[i - 1];
        if (!is_one(range->extent)) {
          Bind(buffer->strides[dim - 1], stride,
               buffer->name + ".strides_" + std::to_string(dim - 1));
          ICHECK(dim-- > 0);
        }
        stride *= shape;
      }
    }

    // Step 2.4. Check and update shape
    {
      size_t dim = buffer->shape.size();
      for (size_t i = source_buffer->shape.size(); i > 0; --i) {
        const Range& range = source->region[i - 1];
        if (!is_one(range->extent)) {
          Bind(buffer->shape[dim - 1], range->extent,
               buffer->name + ".shape_" + std::to_string(dim - 1));
          ICHECK(dim-- > 0);
        }
      }
    }
  }

  bool CheckMatchDimension(const MatchBufferRegion& match_buffer) {
    const Buffer& target = match_buffer->buffer;
    const BufferRegion& source = match_buffer->source;
    size_t num_unit_dim = 0;
    for (const Range& range : source->region) {
      if (is_one(range->extent)) ++num_unit_dim;
    }
    // Check the shape is exactly matched.
    return num_unit_dim + target->shape.size() == source->region.size();
  }

  Array<PrimExpr> ConvertIndices(const Array<PrimExpr> indices,
                                 const MatchBufferRegion& match_buffer) {
    const Buffer& target = match_buffer->buffer;
    const BufferRegion& source = match_buffer->source;
    ICHECK_EQ(indices.size(), target->shape.size());

    Array<PrimExpr> result;
    result.reserve(source->region.size());
    size_t target_cur_pos = 0;
    for (const Range& range : source->region) {
      if (is_one(range->extent)) {
        // This dimesion is ignored by the target buffer
        result.push_back(range->min);
      } else {
        // Check the current pos is less than the indices size.
        ICHECK_LT(target_cur_pos, indices.size());
        const PrimExpr& index = indices[target_cur_pos];
        result.push_back(range->min + index);
        ++target_cur_pos;
      }
    }
    // Check the shape is exactly matched.
    ICHECK_EQ(target_cur_pos, indices.size());
    return result;
  }

  void Bind(const PrimExpr& arg, const PrimExpr& value, const std::string& arg_name = "argument") {
    CHECK_EQ(arg.dtype(), value.dtype())
        << "The data type mismatched: " << arg->dtype << " vs. " << value->dtype;
    if (arg->IsInstance<VarNode>()) {
      Var v = Downcast<Var>(arg);
      auto it = var_map_.find(v);
      if (it == var_map_.end()) {
        var_map_[v] = value;
        analyzer_.Bind(v, value);
      } else {
        AssertBinding(it->second == value, arg_name);
      }
    } else {
      AssertBinding(arg == value, arg_name);
    }
  }

  void AssertBinding(const PrimExpr& cond, const std::string& arg_name = "argument") {
    CHECK(analyzer_.CanProve(cond))
        << "The buffer match constraint for " << arg_name << " unmet: " << cond;
  }

 private:
  /*! \brief Var mapping for buffer signature (data, strides, element_offset, etc.) */
  std::unordered_map<Buffer, BufferRegion, ObjectHash, ObjectEqual> match_buffers_;
  /*! \brief Var mapping for buffer signature (data, strides, element_offset, etc.) */
  std::unordered_map<Var, PrimExpr, ObjectHash, ObjectEqual> var_map_;
  /*! \brief The analyzer */
  arith::Analyzer analyzer_;
};

PrimFunc LowerMatchBuffer(PrimFunc func) {
  auto fptr = func.CopyOnWrite();
  fptr->body = MatchBufferLower()(std::move(fptr->body));
  return func;
}

namespace transform {

Pass LowerMatchBuffer() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerMatchBuffer(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerMatchBuffer", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerMatchBuffer").set_body_typed(LowerMatchBuffer);

}  // namespace transform

}  // namespace tir
}  // namespace tvm