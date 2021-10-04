/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
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

#include <tvm/arith/iter_affine_map.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../runtime/thread_storage_scope.h"
#include "../schedule/utils.h"
#include "./ir_utils.h"
/*!
 * \brief Automatically generate thread binding for auto copy blocks
 * \file lower_auto_copy.cc
 */

namespace tvm {
namespace tir {

class AutoCopyMutator : public StmtExprMutator {
 public:
  Stmt RewritePaddingBody(Stmt stmt) { return RewriteBufferAccess(stmt, padded_buffer_map_); }

 private:
  /**
   * \brief fuse consecutive loops
   * \param stmt the outer-most loop
   * \return the fused loop
   */
  Stmt FuseNestLoops(Stmt stmt) {
    if (!stmt->IsInstance<ForNode>()) {
      return stmt;
    }
    std::vector<const ForNode*> loops;
    Stmt body = stmt;
    while (const ForNode* loop = body.as<ForNode>()) {
      loops.push_back(loop);
      body = loop->body;
    }
    Var fused_var = loops[0]->loop_var.copy_with_suffix("_fused");
    Array<PrimExpr> substitute_value;
    substitute_value.resize(loops.size());
    PrimExpr tot = fused_var;
    for (int i = static_cast<int>(loops.size()) - 1; i >= 0; i--) {
      substitute_value.Set(i, floormod(tot, loops[i]->extent));
      tot = floordiv(tot, loops[i]->extent);
    }
    auto f_substitute = [&](const Var& v) -> Optional<PrimExpr> {
      for (int i = 0; i < static_cast<int>(loops.size()); i++) {
        if (v.same_as(loops[i]->loop_var)) {
          return substitute_value[i];
        }
      }
      return NullOpt;
    };
    PrimExpr fused_extent = 1;
    for (int i = 0; i < static_cast<int>(loops.size()); i++) {
      fused_extent *= loops[i]->extent;
    }
    Stmt new_stmt = Substitute(body, f_substitute);
    new_stmt = For(fused_var, 0, fused_extent, ForKind::kSerial, new_stmt);
    return new_stmt;
  }
  /**
   * \brief a combination of split, bind, vectorize,
   *        a helper function to perform coalesced load/store
   * \param body the stmt to do transformation
   * \param vector_bytes the annotated vectorization bytes
   * \return the stmt after transformation
   */
  Stmt SplitBindVectorize(Stmt body, int vector_bytes) {
    const ForNode* loop = body.as<ForNode>();
    int tot_threads = threadIdx_x_ * threadIdx_y_ * threadIdx_z_;
    int vector_len = vector_bytes * 8 / data_bits_;
    if (!loop || !is_zero(indexmod(loop->extent, (vector_len * tot_threads)))) {
      return body;
    }
    PrimExpr outer_loop_extent = indexdiv(loop->extent, tot_threads * vector_len);
    Array<PrimExpr> factors{outer_loop_extent};
    std::vector<std::string> thread_axis;
    int new_loop_num = 2;
    if (threadIdx_z_ != 1) {
      factors.push_back(threadIdx_z_);
      thread_axis.push_back("threadIdx.z");
      new_loop_num++;
    }
    if (threadIdx_y_ != 1) {
      factors.push_back(threadIdx_y_);
      thread_axis.push_back("threadIdx.y");
      new_loop_num++;
    }
    if (threadIdx_x_ != 1) {
      factors.push_back(threadIdx_x_);
      thread_axis.push_back("threadIdx.x");
      new_loop_num++;
    }
    factors.push_back(vector_len);
    std::vector<Var> new_loop_vars;
    new_loop_vars.reserve(new_loop_num);
    for (int i = 0; i < new_loop_num; i++) {
      new_loop_vars.push_back(loop->loop_var.copy_with_suffix("_" + std::to_string(i)));
    }

    PrimExpr substitute_value = 0;
    for (int i = 0; i < new_loop_num; i++) {
      substitute_value *= factors[i];
      substitute_value += new_loop_vars[i];
    }
    body = Substitute(loop->body, [&](const Var& v) -> Optional<PrimExpr> {
      if (v.same_as(loop->loop_var)) {
        return substitute_value;
      } else {
        return NullOpt;
      }
    });

    For new_loop = For(new_loop_vars[new_loop_num - 1], 0, vector_len, ForKind::kVectorized, body);

    for (int i = new_loop_num - 2; i >= 1; i--) {
      new_loop =
          For(new_loop_vars[i], 0, factors[i], ForKind::kThreadBinding, new_loop,
              IterVar(Range(nullptr), Var(thread_axis[i - 1]), kThreadIndex, thread_axis[i - 1]));
    }

    new_loop = For(new_loop_vars[0], 0, outer_loop_extent, ForKind::kSerial, new_loop);
    return std::move(new_loop);
  }
  

  /**
   * \brief get the index mapping of a specific block
   * \param block the specific block
   * \return the mapping
   */
  std::pair<Array<PrimExpr>, Array<Var>> GetMapping(Block block) {
    Stmt body = block->body;
    Array<Var> loop_vars;
    while (const ForNode* loop = body.as<ForNode>()) {
      loop_vars.push_back(loop->loop_var);
      body = loop->body;
    }
    const BufferStoreNode* buf_store = TVM_TYPE_AS(buf_store, body, BufferStoreNode);
    const Array<Range>& write_region = block->writes[0]->region;
    const Array<PrimExpr>& write_index = buf_store->indices;
    ICHECK(write_region.size() == write_index.size() &&
           block->writes[0]->buffer.same_as(buf_store->buffer));
    Array<PrimExpr> result;
    arith::Analyzer analyzer;
    for (int i = 0; i < static_cast<int>(write_region.size()); i++) {
      PrimExpr pattern = analyzer.Simplify(write_index[i] - write_region[i]->min);
      if (!is_zero(pattern)) {
        result.push_back(pattern);
      }
    }
    // todo(jinhongyii): is there some cases where the size is different?
    ICHECK(result.size() == loop_vars.size());
    return std::make_pair(result, loop_vars);
  }

  /**
   * \brief relax the threads whose rank is higher than both the storage rank of target buffer and
   * source buffer
   * @param block the specific block
   * @param mapping the index mapping
   * @return the block after relaxation
   */
  Block RelaxThreads(Block block, const std::pair<Array<PrimExpr>, Array<Var>>& mapping) {
    Stmt body = block.as<BlockNode>()->body;
    Map<Var, Range> var_range;
    Array<Var> loop_vars;
    Array<PrimExpr> mapping_pattern;
    std::tie(mapping_pattern, loop_vars) = mapping;
    while (const ForNode* loop = body.as<ForNode>()) {
      var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
      body = loop->body;
    }
    for (const ForNode* loop : outer_loops_) {
      if (loop->kind == ForKind::kThreadBinding) {
        const String& thread_tag = loop->thread_binding.value()->thread_tag;
        if (CanRelaxStorageUndereThread(src_scope_, runtime::ThreadScope::Create(thread_tag)) &&
            CanRelaxStorageUndereThread(tgt_scope_, runtime::ThreadScope::Create(thread_tag))) {
          var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
        }
      }
    }
    int n = loop_vars.size();
    const Buffer& read_buffer = block->reads[0]->buffer;
    const Buffer& write_buffer = block->writes[0]->buffer;
    const Array<Range>& read_region = block->reads[0]->region;
    const Array<Range>& write_region = block->writes[0]->region;
    Map<Var, arith::IntSet> var_dom = AsIntSet(var_range);
    Array<arith::IntSet> relaxed_read_region = arith::EvalSet(read_region, var_dom);
    Array<arith::IntSet> relaxed_write_region = arith::EvalSet(write_region, var_dom);
    Array<PrimExpr> read_index;
    Array<PrimExpr> write_index;
    Array<Var> relaxed_loop_vars;
    Map<Var, PrimExpr> extent;
    arith::Analyzer analyzer;
    Array<PrimExpr> original_expr;
    BufferStore original_store = runtime::Downcast<BufferStore>(body);
    BufferLoad original_load = Downcast<BufferLoad>(original_store->value);
    Array<PrimExpr> new_generated_dims;
    Array<Var> old_dims;
    for (int i = 0, j = 0; i < static_cast<int>(relaxed_read_region.size()); i++) {
      PrimExpr ext =
          analyzer.Simplify(relaxed_read_region[i].max() - relaxed_read_region[i].min() + 1);
      if (is_one(ext)) {
        read_index.push_back(relaxed_read_region[i].min());
      } else {
        Var var = loop_vars[0].copy_with_suffix("_relaxed" + std::to_string(j++));
        relaxed_loop_vars.push_back(var);
        if (is_one(read_region[i]->extent)) {
          original_expr.push_back(original_load->indices[i]);
          new_generated_dims.push_back(var);
        } else {
          old_dims.push_back(var);
        }
        read_index.push_back(relaxed_read_region[i].min() + var);
        extent.Set(var, ext);
      }
    }
    Array<arith::IterSumExpr> iter_map =
        arith::DetectIterMap(original_expr, var_range, Bool(true), false, &analyzer);
    Map<Var, PrimExpr> inverse_mapping = arith::InverseAffineIterMap(iter_map, new_generated_dims);
    Map<Var, PrimExpr> substitute_mapping;
    for (const auto& pair : inverse_mapping) {
      substitute_mapping.Set(pair.first, pair.second);
    }
    for (int i = 0; i < n; i++) {
      substitute_mapping.Set(loop_vars[i], old_dims[i]);
    }
    for (int i = 0, j = 0; i < static_cast<int>(write_region.size()); i++) {
      PrimExpr new_bound = Substitute(write_region[i]->min, substitute_mapping);
      arith::IntSet relaxed_bound = arith::EvalSet(new_bound, var_dom);
      if (is_one(write_region[i]->extent)) {
        write_index.push_back(relaxed_bound.min());
      } else {
        write_index.push_back(new_bound + Substitute(mapping_pattern[j++], substitute_mapping));
      }
    }
    BufferLoad new_buf_load = BufferLoad(read_buffer, read_index);
    BufferStore new_buf_store = BufferStore(write_buffer, new_buf_load, write_index);
    Stmt ret = new_buf_store;
    for (int i = static_cast<int>(relaxed_loop_vars.size()) - 1; i >= 0; i--) {
      PrimExpr loop_extent = extent[relaxed_loop_vars[i]];
      ret = For(relaxed_loop_vars[i], 0, loop_extent, ForKind::kSerial, ret);
    }
    ObjectPtr<BlockNode> new_block = runtime::make_object<BlockNode>(*block.get());
    Array<Range> new_read_region;
    Array<Range> new_write_region;
    Range none;
    for (const arith::IntSet& int_set : relaxed_read_region) {
      new_read_region.push_back(int_set.CoverRange(none));
    }
    for (const arith::IntSet& int_set : relaxed_write_region) {
      new_write_region.push_back(int_set.CoverRange(none));
    }
    new_block->body = ret;
    new_block->reads = {BufferRegion(read_buffer, new_read_region)};
    new_block->writes = {BufferRegion(write_buffer, new_write_region)};
    return Block(new_block);
  }

  /**
   * \brief transform from A[f(i,j)] = B[i,j] to A[i,j] = B[f^{-1}(i,j)]
   * @param block the specific block
   * @param mapping the index mapping
   * @return the result stmt
   */
  Stmt InverseMappingTransform(Block block, const std::pair<Array<PrimExpr>, Array<Var>>& mapping) {
    Stmt body = block->body;
    Map<Var, Range> var_range;
    Array<PrimExpr> loop_vars;
    Array<PrimExpr> mapping_pattern = mapping.first;
    while (const ForNode* loop = body.as<ForNode>()) {
      var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
      loop_vars.push_back(loop->loop_var);
      body = loop->body;
    }
    arith::Analyzer analyzer;
    Array<arith::IterSumExpr> iter_map =
        arith::DetectIterMap(mapping_pattern, var_range, Bool(true), true, &analyzer);
    CHECK_EQ(iter_map.size(), loop_vars.size());
    Map<Var, PrimExpr> inverse_mapping = arith::InverseAffineIterMap(iter_map, loop_vars);
    Array<PrimExpr> write_index;
    Array<PrimExpr> read_index;
    Array<Var> new_loop_vars;
    int n = loop_vars.size();
    Map<Var, PrimExpr> substitute_map;
    for (int i = 0, j = 0; i < static_cast<int>(block->writes[0]->region.size()); i++) {
      if (is_zero(block->writes[0]->region[i]->extent)) {
        write_index.push_back(block->writes[0]->region[i]->min);
      } else {
        Var var = runtime::Downcast<Var>(loop_vars[j]).copy_with_suffix("_inverse");
        new_loop_vars.push_back(var);
        substitute_map.Set(runtime::Downcast<Var>(loop_vars[j++]), var);
        write_index.push_back(block->writes[0]->region[i]->min + var);
      }
    }
    for (int i = 0, j = 0; i < static_cast<int>(block->reads[0]->region.size()); i++) {
      if (is_zero(block->reads[0]->region[i]->extent)) {
        read_index.push_back(block->reads[0]->region[i]->min);
      } else {
        read_index.push_back(
            block->reads[0]->region[i]->min +
            Substitute(inverse_mapping[Downcast<Var>(loop_vars[j++])], substitute_map));
      }
    }
    BufferLoad new_buf_load = BufferLoad(block->reads[0]->buffer, read_index);
    BufferStore new_buf_store = BufferStore(block->writes[0]->buffer, new_buf_load, write_index);
    Stmt ret = new_buf_store;
    for (int i = static_cast<int>(new_loop_vars.size()) - 1; i >= 0; i--) {
      PrimExpr extent = block->writes[0]->region[i]->extent;
      ret = For(new_loop_vars[i], 0, extent, ForKind::kSerial, ret);
    }
    return ret;
  }

  /**
   * \brief rewrite wmma load
   * @param block the specific block
   * @param match_buffers the match_buffer to be appended to the new block
   * @return the result stmt
   */
  Stmt RewriteWmmaLoad(Block block, Array<MatchBufferRegion>* match_buffers) {
    Stmt body = block->body;
    std::vector<int> index;
    arith::Analyzer analyzer;
    Map<Var, PrimExpr> substitute_map;
    int i = 0;
    while (const ForNode* loop = body.as<ForNode>()) {
      if (!analyzer.CanProveEqual(loop->min, 0)) {
        return block->body;
      }
      if (analyzer.CanProveEqual(loop->extent, 1)) {
        continue;
      } else if (analyzer.CanProveEqual(loop->extent, 16)) {
        index.push_back(i);
      } else {
        return block->body;
      }
      substitute_map.Set(loop->loop_var, 0);
      i++;
      body = loop->body;
    }
    if (index.size() != 2 || index[1] != i - 1 || index[0] != i - 2) {
      return block->body;
    }
    Buffer src_buffer = block->reads[0]->buffer;
    Buffer tgt_buffer = block->writes[0]->buffer;
    padding_constraint[src_buffer.get()] = 3;
    TensorIntrin wmma_load;
    if (tgt_buffer.scope() == "wmma.matrix_a") {
      wmma_load = tir::TensorIntrin::Get("wmma_load_a");
    } else {
      wmma_load = tir::TensorIntrin::Get("wmma_load_b");
    }

    auto param = wmma_load->implementation->params[0];
    Buffer new_src_buffer = wmma_load->implementation->buffer_map.at(param);
    match_buffers->push_back(MatchBufferRegion(new_src_buffer, block->reads[0]));
    param = wmma_load->implementation->params[1];
    Buffer new_tgt_buffer = wmma_load->implementation->buffer_map.at(param);
    match_buffers->push_back(MatchBufferRegion(new_tgt_buffer, block->writes[0]));

    PrimExpr frag_index = floordiv(new_tgt_buffer->elem_offset, 256) +
                          floordiv(floormod(new_tgt_buffer->elem_offset, 256), 16);

    auto new_src_pointer = Call(
        runtime::DataType::Handle(), builtin::tvm_access_ptr(),
        {TypeAnnotation(new_src_buffer->dtype), new_src_buffer->data, new_src_buffer->elem_offset,
         new_src_buffer->strides[new_src_buffer->strides.size() - 2] * 16, 1});

    return Evaluate(Call(
        runtime::DataType::Handle(), builtin::tvm_load_matrix_sync(),
        {new_tgt_buffer->data, 16, 16, 16, frag_index, new_src_pointer,
         new_src_buffer->strides[new_src_buffer->strides.size() - 2], StringImm("row_major")}));
  }

  /**
   * \brief rewrite wmma store
   * @param block the specific block
   * @param match_buffers the match_buffer to be appended to the new block
   * @return the result stmt
   */
  Stmt RewriteWmmaStore(Block block, Array<MatchBufferRegion>* match_buffers) {
    Stmt body = block->body;
    std::vector<int> index;
    arith::Analyzer analyzer;
    Map<Var, PrimExpr> substitute_map;
    int i = 0;
    while (const ForNode* loop = body.as<ForNode>()) {
      if (!analyzer.CanProveEqual(loop->min, 0)) {
        return block->body;
      }
      if (analyzer.CanProveEqual(loop->extent, 1)) {
        continue;
      } else if (analyzer.CanProveEqual(loop->extent, 16)) {
        index.push_back(i);
      } else {
        return block->body;
      }
      substitute_map.Set(loop->loop_var, 0);
      i++;
      body = loop->body;
    }
    if (index.size() != 2 || index[1] != i - 1 || index[0] != i - 2) {
      return block->body;
    }
    Buffer src_buffer = block->reads[0]->buffer;
    Buffer tgt_buffer = block->writes[0]->buffer;
    padding_constraint[tgt_buffer.get()] = 3;
    TensorIntrin wmma_store = tir::TensorIntrin::Get("wmma_store");

    auto param = wmma_store->implementation->params[0];
    Buffer new_src_buffer = wmma_store->implementation->buffer_map.at(param);
    match_buffers->push_back(MatchBufferRegion(new_src_buffer, block->reads[0]));
    param = wmma_store->implementation->params[1];
    Buffer new_tgt_buffer = wmma_store->implementation->buffer_map.at(param);
    match_buffers->push_back(MatchBufferRegion(new_tgt_buffer, block->writes[0]));

    // todo(jinhongyii): consider non-packed layout situation
    PrimExpr frag_index = floordiv(new_src_buffer->elem_offset, 256) +
                          floordiv(floormod(new_src_buffer->elem_offset, 256), 16);

    auto new_tgt_pointer = Call(runtime::DataType::Handle(), builtin::tvm_access_ptr(),
                                {TypeAnnotation(new_tgt_buffer->dtype), new_tgt_buffer->data,
                                 new_tgt_buffer->elem_offset, new_tgt_buffer->strides[0] * 16, 2});

    return Evaluate(Call(runtime::DataType::Handle(), builtin::tvm_store_matrix_sync(),
                         {new_src_buffer->data, 16, 16, 16, frag_index, new_tgt_pointer,
                          new_tgt_buffer->strides[0], StringImm("row_major")}));
  }

  /**
   * \brief do coalesce load/store
   * @param block the specific block
   * @param mapping the index mapping
   * @param vector_bytes the annotated vectorization bytes
   * @param need_inverse whether need to inverse mapping transform
   * @return the result stmt
   */
  Stmt CoalesceGlobalLoad(Block block, const std::pair<Array<PrimExpr>, Array<Var>>& mapping,
                          int vector_bytes, bool need_inverse = false) {
    Stmt ret = block->body;
    if (need_inverse) {
      std::pair<Array<PrimExpr>, Array<Var>> mapping = GetMapping(block);
      ret = InverseMappingTransform(block, mapping);
    }
    ret = FuseNestLoops(std::move(ret));
    ret = SplitBindVectorize(std::move(ret), vector_bytes);
    return ret;
  }
  /**
   * \brief do padding to the given buffers whose storage scope is "shared"
   * @param buffers the given buffers
   * @param buffer_map the mapping from old buffer to the new padded buffer
   * @return the list of new padded buffers
   */
  Array<Buffer> PadSharedMemory(const Array<Buffer>& buffers, Map<Buffer, Buffer>* buffer_map) {
    Array<Buffer> result;

    for (const Buffer& buffer : buffers) {
      if (buffer.scope() == "shared") {
        int type_factor = 32 / buffer->dtype.bits();
        auto patterns = patterns_[buffer.get()];
        int base_2_bank_size = std::log2(32 * type_factor);
        std::vector<std::vector<int>> base_2_bank(patterns.size(),std::vector<int>(base_2_bank_size));
        
        int n = buffer->shape.size();
        //Step 1. initialize `base_2_bank` with the access pattern of the last dim
        for (int i = 0; i < static_cast<int>(patterns.size()); i++) {
          auto dim_patterns = patterns[i][n - 1];
          for (const Pattern& pattern : dim_patterns) {
            for (int j = pattern.scale; j < pattern.scale + pattern.extent && j < base_2_bank_size;
                 j++) {
              base_2_bank[i][j]++;
            }
          }
        }
        std::vector<int> padding;
        padding.resize(n);
        int constraint = 0;
        if (padding_constraint.count(buffer.get())) {
          constraint = padding_constraint[buffer.get()];
        }
        //Step 2. try out each padding choice to see which has the minimal conflict
        for (int k = n - 2; k >= 0; k--) {
          int min_conflict = INT32_MAX;
          int min_conflict_m = -1;
          int min_pad_size = INT32_MAX;
          for (int m = (k == n - 2) ? constraint : 0; m < base_2_bank_size; m++) {
            int tot_conflict = 0;
            for (int i = 0; i < static_cast<int>(patterns.size()); i++) {
              int conflict = 0;
              auto dim_patterns = patterns[i][k];
              for (const Pattern& pattern : dim_patterns) {
                for (int j = pattern.scale + m; j < pattern.scale + pattern.extent + m; j++) {
                  if (j >= base_2_bank_size) {
                    conflict++;
                  } else {
                    conflict += base_2_bank[i][j];
                  }
                }
              }
              tot_conflict+=std::pow(2,conflict);
            }
            int pad_size =
                (32 + int(std::pow(2, m)) - buffer->shape[k+1].as<IntImmNode>()->value%32) % 32;
            if (tot_conflict < min_conflict || (min_conflict == tot_conflict && pad_size < min_pad_size)) {
              min_conflict_m = m;
              min_conflict = tot_conflict;
              min_pad_size = pad_size;
            }
          }
          for (int i = 0; i < static_cast<int>(patterns.size()); i++) {
            auto dim_patterns = patterns[i][n - 1];
            for (const Pattern& pattern : dim_patterns) {
              for (int j = pattern.scale + min_conflict_m;
                   j < pattern.scale + pattern.extent + min_conflict_m && j < base_2_bank_size;
                   j++) {
                base_2_bank[i][j]++;
              }
            }
          }
          padding[k + 1] = min_pad_size;
        }
        // Step 3. create the new padded buffer
        ObjectPtr<BufferNode> b = make_object<BufferNode>(*buffer.get());
        Array<PrimExpr> strides;
        strides.resize(b->shape.size());
        PrimExpr stride = make_const(b->shape[0].dtype(), 1);
        for (size_t i = b->shape.size(); i != 0; --i) {
          size_t dim = i - 1;
          strides.Set(dim, stride);
          stride = stride * (b->shape[dim] + padding[dim]);
        }
        b->strides = strides;
        Buffer new_buffer(b);
        result.push_back(new_buffer);
        buffer_map->Set(buffer, new_buffer);
      } else {
        result.push_back(buffer);
      }
    }
    return result;
  }
  
  /**
   * \brief replace all occurrence of the old buffer with the new buffer in the stmt
   * @param stmt the stmt to do replacement
   * @param buffer_map the mapping from old buffer to the new buffer
   * @return the stmt after replacement
   */
  Stmt RewriteBufferAccess(Stmt stmt, const Map<Buffer, Buffer>& buffer_map) {
    class Rewriter : public StmtExprMutator {
     public:
      Rewriter(const Map<Buffer, Buffer>& buffer_map) : buffer_map_(buffer_map) {}

     private:
      PrimExpr VisitExpr_(const BufferLoadNode* _op) final {
        BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_op));
        BufferLoadNode* op = load.CopyOnWrite();
        if (buffer_map_.count(op->buffer)) {
          op->buffer = buffer_map_[op->buffer];
        }
        return std::move(load);
      }

      Stmt VisitStmt_(const BufferStoreNode* _op) final {
        BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_op));
        BufferStoreNode* op = store.CopyOnWrite();
        if (buffer_map_.count(op->buffer)) {
          op->buffer = buffer_map_[op->buffer];
        }
        return std::move(store);
      }

      Stmt VisitStmt_(const BlockNode* op) final {
        // To reduce the number of blocks in block sref reuse map, we check whether the block is
        // really mutated (i.e., the old buffer appears in the block). If so, we return the block
        // after mutation. Otherwise we just return the original block.
        bool changed = false;
        // Step 1. Mutate the read region.
        Array<BufferRegion> reads;
        for (const BufferRegion& read : op->reads) {
          if (buffer_map_.count(read->buffer)) {
            changed = true;
            reads.push_back(BufferRegion(buffer_map_[read->buffer], read->region));
          } else {
            reads.push_back(read);
          }
        }
        // Step 2. Mutate the write region.
        Array<BufferRegion> writes;
        for (const BufferRegion& write : op->writes) {
          if (buffer_map_.count(write->buffer)) {
            changed = true;
            writes.push_back(BufferRegion(buffer_map_[write->buffer], write->region));
          } else {
            writes.push_back(write);
          }
        }
        // Step 4. Mutate `match_buffers`. If an old buffer appears as a source of
        // MatchBufferRegion, the storage scope of the target buffer also needs to be set.
        Array<MatchBufferRegion> match_buffers;
        for (const MatchBufferRegion& match_buffer : op->match_buffers) {
          if (buffer_map_.count(match_buffer->source->buffer)) {
            changed = true;
            Buffer new_buffer = buffer_map_[match_buffer->source->buffer];
            match_buffers.push_back(MatchBufferRegion(
                match_buffer->buffer, BufferRegion(new_buffer, match_buffer->source->region)));
          } else {
            match_buffers.push_back(match_buffer);
          }
        }
        // Step 5. Recursively mutate the block.
        Stmt res = StmtMutator::VisitStmt_(op);
        if (res.get() != op) {
          changed = true;
        }

        if (changed) {
          ObjectPtr<BlockNode> block = CopyOnWrite(res.as<BlockNode>());
          block->reads = std::move(reads);
          block->writes = std::move(writes);
          block->match_buffers = std::move(match_buffers);
          return Stmt(block);
        } else {
          return GetRef<Block>(op);
        }
      }
      const Map<Buffer, Buffer>& buffer_map_;
    };
    Rewriter rewriter(buffer_map);
    return rewriter(stmt);
  }

  /**
   * \brief an equivalent of pow(2, scale) * loop_var with loop_var: {min=0, extent=pow(2, extent)}
   */
  struct Pattern {
    int extent;
    int scale;
  };

  /**
   * \brief collect pattern from indices
   */
  class PatternCollector : public StmtExprVisitor {
    void VisitExpr_(const VarNode* op) final {
      int extent = var_range_[GetRef<Var>(op)]->extent.as<IntImmNode>()->value;
      // todo(jinhongyii): check the extent is indeed a power of 2
      int log2_extent = std::log2(extent);
      if (extent > 1) {
        stack_.push({{log2_extent, 0}});
      } else {
        stack_.push({});
      }
    }

    void VisitExpr_(const AddNode* op) final {
      ExprVisitor::VisitExpr_(op);
      std::vector<Pattern> merged_patterns;
      std::vector<Pattern> r = stack_.top();
      stack_.pop();
      std::vector<Pattern> l = stack_.top();
      stack_.pop();
      for (const Pattern& pattern : l) {
        merged_patterns.push_back(pattern);
      }
      for (const Pattern& pattern : r) {
        merged_patterns.push_back(pattern);
      }
      if (merged_patterns.empty()) {
        stack_.push({});
        return;
      }
      std::vector<Pattern> ret;
      ret.push_back(merged_patterns[0]);
      for (int i = 0; i < static_cast<int>(merged_patterns.size()); i++) {
        Pattern prev_pattern = ret.back();
        if (merged_patterns[i].extent + merged_patterns[i].scale == prev_pattern.scale) {
          ret.pop_back();
          ret.push_back(
              {prev_pattern.extent + merged_patterns[i].extent, merged_patterns[i].scale});
        }
      }
      stack_.push(ret);
    }

    void VisitExpr_(const FloorDivNode* op) final {
      ExprVisitor::VisitExpr_(op);
      std::vector<Pattern> inner = stack_.top();
      stack_.pop();
      int lower_factor = std::log2(op->b.as<IntImmNode>()->value);
      std::vector<Pattern> ret;
      for (const Pattern& pattern : inner) {
        if (pattern.scale >= lower_factor) {
          ret.push_back({pattern.extent, pattern.scale - lower_factor});
        } else if (pattern.scale + pattern.extent > lower_factor) {
          ret.push_back({pattern.extent + pattern.scale - lower_factor, 0});
        }
      }
      stack_.push(ret);
    }

    void VisitExpr_(const FloorModNode* op) final {
      ExprVisitor::VisitExpr_(op);
      std::vector<Pattern> inner = stack_.top();
      stack_.pop();
      int extent = std::log2(op->b.as<IntImmNode>()->value);
      std::vector<Pattern> ret;
      for (const Pattern& pattern : inner) {
        if (pattern.scale < extent) {
          if (extent - pattern.scale < pattern.extent) {
            ret.push_back({extent - pattern.scale, pattern.scale});
          } else {
            ret.push_back({pattern.extent, pattern.scale});
          }
        }
      }
      stack_.push(ret);
    }

    void VisitExpr_(const MulNode* op) final {
      ExprVisitor::VisitExpr_(op);
      std::vector<Pattern> inner = stack_.top();
      stack_.pop();
      int scale = std::log2(op->b.as<IntImmNode>()->value);
      std::vector<Pattern> ret;
      for (const Pattern& pattern : inner) {
        ret.push_back({pattern.extent, pattern.scale + scale});
      }
      stack_.push(ret);
    }

   public:
    PatternCollector(const Map<Var, Range>& var_range) : var_range_(var_range) {}

    static std::vector<std::vector<Pattern>> CollectPattern(const Array<PrimExpr>& indices,
                                                            const Map<Var, Range>& var_range) {
      PatternCollector collector(var_range);
      std::vector<std::vector<Pattern>> ret;
      for (const PrimExpr& expr : indices) {
        collector(expr);
        if (collector.stack_.size() == 1) {
          ret.push_back(collector.stack_.top());
          collector.stack_.pop();
        } else {
          ret.push_back({});
        }
      }
      return ret;
    }

    std::stack<std::vector<Pattern>> stack_;
    const Map<Var, Range>& var_range_;
  };
  
  Array<PrimExpr> getWarpAccess(const Array<PrimExpr>& indices,
                                const std::vector<const ForNode*>& warp_access_loops,
                                Map<Var, PrimExpr> substitute_map, Map<Var, Range>* var_range) {
    int prod = 1;
    Array<String> thread_name{"threadIdx.x", "threadIdx.y", "threadIdx.z"};
    bool warp_full = false;
    for (int i = 0; i < thread_name.size() && !warp_full; i++) {
      for (const ForNode* loop : warp_access_loops) {
        if (loop->kind == ForKind::kThreadBinding) {
          String thread_tag = loop->thread_binding.value()->thread_tag;
          if (thread_tag == thread_name[i]) {
            int extent = loop->extent.as<IntImmNode>()->value;
            substitute_map.erase(loop->loop_var);
            if (prod * extent > 32) {
              prod = 32;
              var_range->Set(loop->loop_var, Range::FromMinExtent(loop->min, 32 / prod));
              warp_full = true;
              break;
            } else {
              prod *= extent;
              var_range->Set(loop->loop_var, Range::FromMinExtent(loop->min, extent));
              break;
            }
          }
        }
      }
    }
    for (const ForNode* loop : warp_access_loops) {
      if (loop->kind == ForKind::kVectorized) {
        var_range->Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
        substitute_map.erase(loop->loop_var);
        break;
      }
    }
    Array<PrimExpr> ret;
    arith::Analyzer analyzer;
    for (PrimExpr index : indices) {
      ret.push_back(analyzer.Simplify(Substitute(index, substitute_map)));
    }
    return ret;
  }

  /**
   * \brief analyze the warp access pattern for a given block
   *      An index would be transformed into an array of Pattern
   *      For example, if the block has stmt:
   *      A_shared[ty, tx] = A[ty, tx]
   *      suppose tx is bound to threadIdx.x and ty is bound to threadIdx.y
   *      tx is [0,16), ty is [0,4)
   *      then the pattern would be {{2,0}}, {{4,0}}
   *
   *      if the stmt is A_shared[ty*8+tx%4, tx/4] = A[ty, tx]
   *      then pattern would be {{2,3},{2,0}}, {{2,0}}
   * @param block the given block
   */
  void AnalyzePatternForPadding(Block block) {
    Stmt body = block->body;
    Map<Var, PrimExpr> substitute_map;
    std::vector<const ForNode*> warp_access_loops;
    PrimExpr outer_loop_ct = 1;
    while (const ForNode* loop = body.as<ForNode>()) {
      substitute_map.Set(loop->loop_var, loop->min);
      if (loop->kind == ForKind::kThreadBinding) {
        warp_access_loops.push_back(loop);
      } else if (loop->kind == ForKind::kVectorized) {
        warp_access_loops.push_back(loop);
      }
      outer_loop_ct *= loop->extent;
      body = loop->body;
    }
    for (const ForNode* loop : outer_loops_) {
      substitute_map.Set(loop->loop_var, loop->min);
      outer_loop_ct *= loop->extent;
    }

    Buffer buffer;
    Map<Var, Range> var_range;
    Array<PrimExpr> indices;
    Array<Range> region;
    if (const BufferStoreNode* node = body.as<BufferStoreNode>()) {
      if (node->buffer.scope() == "shared") {
        buffer = node->buffer;
        indices = node->indices;
      } else {
        const BufferLoadNode* buf_load = node->value.as<BufferLoadNode>();
        if (buf_load->buffer.scope() == "shared") {
          buffer = buf_load->buffer;
          indices = buf_load->indices;
        } else {
          return;
        }
      }
    } else {
      if (block->reads[0]->buffer.scope() == "shared") {
        buffer = block->reads[0]->buffer;
        region = block->reads[0]->region;
      } else if (block->writes[0]->buffer.scope() == "shared") {
        buffer = block->writes[0]->buffer;
        region = block->writes[0]->region;
      } else {
        return;
      }
      for (int i = 0; i < static_cast<int>(region.size()); i++) {
        Var var("region" + i);
        indices.push_back(region[i]->min + var);
        var_range.Set(var, Range::FromMinExtent(0, region[i]->extent));
      }
    }
    arith::Analyzer analyzer;
    Array<PrimExpr> substitued_indices =
        getWarpAccess(indices, warp_access_loops, substitute_map, &var_range);
    std::vector<std::vector<Pattern>> patterns =
        PatternCollector::CollectPattern(substitued_indices, var_range);
    patterns_[buffer.get()].push_back(patterns);
    for (const auto& pattern_single_dim : patterns) {
      std::cerr << "{";
      for (const auto& pattern : pattern_single_dim) {
        std::cerr << "{" << pattern.extent << "," << pattern.scale << "}";
      }
      std::cerr << "}, ";
    }
    std::cerr << std::endl;
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    Block block;
    if (op->annotations.count("auto_copy") &&
        is_one(Downcast<PrimExpr>(op->annotations["auto_copy"]))) {
      LOG(INFO) << op->name_hint;
      in_auto_copy_ = true;
      block = runtime::Downcast<Block>(StmtMutator::VisitStmt_(op));
      std::pair<Array<PrimExpr>, Array<Var>> mapping = GetMapping(block);
      block = RelaxThreads(block, mapping);
      int vector_bytes;
      if (block->annotations.count("vector_bytes")) {
        IntImm vec_bytes = Downcast<IntImm>(block->annotations["vector_bytes"]);
        vector_bytes = vec_bytes->value;
      } else {
        vector_bytes = 1;
      }
      BlockNode* n = block.CopyOnWrite();
      if ((src_scope_.rank == runtime::StorageRank::kGlobal &&
           tgt_scope_.rank == runtime::StorageRank::kShared) ||
          (src_scope_.rank == runtime::StorageRank::kShared &&
           tgt_scope_.rank == runtime::StorageRank::kGlobal)) {
        bool need_inverse = src_scope_.rank == runtime::StorageRank::kShared;
        n->body = CoalesceGlobalLoad(block, mapping, vector_bytes, need_inverse);
      } else if ((tgt_scope_.rank == runtime::StorageRank::kWMMAMatrixA ||
                  tgt_scope_.rank == runtime::StorageRank::kWMMAMatrixB)) {
        Array<MatchBufferRegion> match_buffers;
        n->body = RewriteWmmaLoad(block, &match_buffers);
        n->match_buffers = match_buffers;
      } else if (src_scope_.rank == runtime::StorageRank::kWMMAAccumulator) {
        Array<MatchBufferRegion> match_buffers;
        n->body = RewriteWmmaStore(block, &match_buffers);
        n->match_buffers = match_buffers;
      }
      LOG(INFO) << "calc conflict";
      AnalyzePatternForPadding(block);
      LOG(INFO) << "end calc conflict";
      n->alloc_buffers = PadSharedMemory(n->alloc_buffers, &padded_buffer_map_);
    } else {
      block = runtime::Downcast<Block>(StmtMutator::VisitStmt_(op));
      BlockNode* n = block.CopyOnWrite();
      n->alloc_buffers = PadSharedMemory(n->alloc_buffers, &padded_buffer_map_);
    }
    return std::move(block);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    if (in_auto_copy_) {
      if (const BufferLoadNode* buf_load = op->value.as<BufferLoadNode>()) {
        CHECK(op->buffer->dtype == buf_load->buffer->dtype);
        data_bits_ = op->buffer->dtype.bits();
        tgt_scope_ = runtime::StorageScope::Create(op->buffer.scope());
        src_scope_ = runtime::StorageScope::Create(buf_load->buffer.scope());
      }
    }
    return GetRef<BufferStore>(op);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    if (op->thread_binding.defined()) {
      IterVar binding = op->thread_binding.value();
      if (binding->iter_type == kThreadIndex) {
        if (binding->thread_tag == "threadIdx.x") {
          threadIdx_x_ = Downcast<IntImm>(op->extent)->value;
        } else if (binding->thread_tag == "threadIdx.y") {
          threadIdx_y_ = Downcast<IntImm>(op->extent)->value;
        } else if (binding->thread_tag == "threadIdx.z") {
          threadIdx_z_ = Downcast<IntImm>(op->extent)->value;
        }
      }
    }
    outer_loops_.push_back(op);
    Stmt stmt = StmtMutator::VisitStmt_(op);
    outer_loops_.pop_back();
    return stmt;
  }

  bool in_auto_copy_;
  runtime::StorageScope src_scope_;
  runtime::StorageScope tgt_scope_;
  int threadIdx_x_ = 1;
  int threadIdx_y_ = 1;
  int threadIdx_z_ = 1;
  int data_bits_ = -1;
  std::vector<const ForNode*> outer_loops_;
  Map<Buffer, Buffer> padded_buffer_map_;
  std::unordered_map<const BufferNode*, std::vector<std::vector<std::vector<Pattern>>>> patterns_;
  std::unordered_map<const BufferNode*, int> padding_constraint;
};

namespace transform {
Pass LowerAutoCopy() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    AutoCopyMutator mutator;
    n->body = mutator(std::move(f->body));
    n->body = mutator.RewritePaddingBody(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerAutoCopy", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerAutoCopy").set_body_typed(LowerAutoCopy);
}  // namespace transform
}  // namespace tir
}  // namespace tvm