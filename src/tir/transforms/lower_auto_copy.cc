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
 private:
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

  Stmt SplitBindVectorize(Stmt body, int vector_bytes) {
    const ForNode* loop = body.as<ForNode>();
    int tot_threads = threadIdx_x_ * threadIdx_y_ * threadIdx_z_;
    int vector_len = vector_bytes * 8 / data_bits_;
    if (!loop || !is_zero(indexmod(loop->extent, (vector_len * tot_threads)))) {
      if (!loop) {
        LOG(INFO) << 1;
      } else {
        LOG(INFO) << loop->extent << ' ' << (vector_len * tot_threads);
      }
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

  Array<PrimExpr> GetMapping(Block block) {
    Stmt body = block->body;
    while (const ForNode* loop = body.as<ForNode>()) {
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
      result.push_back(analyzer.Simplify(write_index[i] - write_region[i]->min));
    }
    return result;
  }

  Block RelaxThreads(Block block, const Array<PrimExpr>& mapping) {
    Stmt body = block.as<BlockNode>()->body;
    Map<Var, Range> var_range;
    Array<Var> loop_vars;
    while (const ForNode* loop = body.as<ForNode>()) {
      var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
      loop_vars.push_back(loop->loop_var);
      body = loop->body;
    }
    for (const ForNode* loop : outer_loops_) {
      if (loop->kind == ForKind::kThreadBinding) {
        const String& thread_tag = loop->thread_binding.value()->thread_tag;
        if (CanRelaxStorageUndereThread(src_scope_, runtime::ThreadScope::Create(thread_tag)) &&
            CanRelaxStorageUndereThread(tgt_scope_, runtime::ThreadScope::Create(thread_tag))) {
          var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
        }
      } else {
        var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
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
    for (int i = 0; i < n; i++) {
      Var var = loop_vars[i].copy_with_suffix("_relaxed");
      relaxed_loop_vars.push_back(var);
      read_index.push_back(relaxed_read_region[i].min() + var);
    }
    const BufferStoreNode* old_buf_store = TVM_TYPE_AS(old_buf_store, body, BufferStoreNode);
    Map<Var, PrimExpr> substitute_mapping;
    for (int i = 0; i < n; i++) {
      substitute_mapping.Set(loop_vars[i], relaxed_loop_vars[i]);
    }
    for (int i = 0; i < n; i++) {
      write_index.push_back(relaxed_write_region[i].min() +
                            Substitute(mapping[i], substitute_mapping));
    }
    BufferLoad new_buf_load = BufferLoad(read_buffer, read_index);
    BufferStore new_buf_store = BufferStore(write_buffer, new_buf_load, write_index);
    Stmt ret = new_buf_store;
    arith::Analyzer analyzer;
    for (int i = n - 1; i >= 0; i--) {
      PrimExpr extent =
          analyzer.Simplify(relaxed_read_region[i].max() - relaxed_read_region[i].min() + 1);
      ret = For(relaxed_loop_vars[i], 0, extent, ForKind::kSerial, ret);
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

  Stmt InverseMappingTransform(Block block, const Array<PrimExpr>& mapping) {
    Stmt body = block->body;
    Map<Var, Range> var_range;
    Array<PrimExpr> loop_vars;
    while (const ForNode* loop = body.as<ForNode>()) {
      var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
      loop_vars.push_back(loop->loop_var);
      body = loop->body;
    }
    arith::Analyzer analyzer;
    Array<arith::IterSumExpr> iter_map =
        arith::DetectIterMap(mapping, var_range, Bool(true), true, &analyzer);
    CHECK_EQ(iter_map.size(), loop_vars.size());
    Map<Var, PrimExpr> inverse_mapping = arith::InverseAffineIterMap(iter_map, loop_vars);
    Array<PrimExpr> write_index;
    Array<PrimExpr> read_index;
    Array<Var> new_loop_vars;
    int n= loop_vars.size();
    Map<Var, PrimExpr> substitute_map;
    for(int i=0;i<n;i++){
      Var var = runtime::Downcast<Var>(loop_vars[i]).copy_with_suffix("_inverse");
      new_loop_vars.push_back(var);
      substitute_map.Set(runtime::Downcast<Var>(loop_vars[i]),var);
      write_index.push_back(block->writes[0]->region[i]->min+var);
    }
    for (int i = 0; i < n; i++) {
      read_index.push_back(block->reads[0]->region[i]->min+ Substitute
                           (inverse_mapping[Downcast<Var>(loop_vars[i])],substitute_map));
    }
    BufferLoad new_buf_load = BufferLoad(block->reads[0]->buffer, read_index);
    BufferStore new_buf_store = BufferStore(block->writes[0]->buffer, new_buf_load, write_index);
    Stmt ret = new_buf_store;
    for (int i = n - 1; i >= 0; i--) {
      PrimExpr extent = block->writes[0]->region[i]->extent;
      ret = For(new_loop_vars[i], 0, extent, ForKind::kSerial, ret);
    }
    return ret;
  }

  Stmt CoalesceGlobalLoad(Block block, int vector_bytes, bool need_inverse = false) {
    Array<PrimExpr> mapping = GetMapping(block);
    Block relaxed_block = RelaxThreads(block, mapping);
    Stmt ret = relaxed_block->body;
    if (need_inverse) {
      Array<PrimExpr> mapping = GetMapping(relaxed_block);
      ret = InverseMappingTransform(relaxed_block, mapping);
    }
    ret = FuseNestLoops(std::move(ret));
    ret = SplitBindVectorize(std::move(ret), vector_bytes);
    return ret;
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    Block block;
    if (op->annotations.count("auto_copy") &&
        is_one(Downcast<PrimExpr>(op->annotations["auto_copy"]))) {
      in_auto_copy_ = true;
      block = runtime::Downcast<Block>(StmtMutator::VisitStmt_(op));
      BlockNode* n = block.CopyOnWrite();
      if ((src_scope_.rank == runtime::StorageRank::kGlobal &&
           tgt_scope_.rank == runtime::StorageRank::kShared) ||
          (src_scope_.rank == runtime::StorageRank::kShared &&
           tgt_scope_.rank == runtime::StorageRank::kGlobal)) {
        int vector_bytes;
        if (op->annotations.count("vector_bytes")) {
          IntImm vec_bytes = Downcast<IntImm>(n->annotations["vector_bytes"]);
          vector_bytes = vec_bytes->value;
        } else {
          vector_bytes = 1;
        }
        bool need_inverse = src_scope_.rank == runtime::StorageRank::kShared;
        n->body = CoalesceGlobalLoad(block, vector_bytes, need_inverse);
      }
    } else {
      block = runtime::Downcast<Block>(StmtMutator::VisitStmt_(op));
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
};

namespace transform {
Pass LowerAutoCopy() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = AutoCopyMutator()(std::move(f->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerAutoCopy", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerAutoCopy").set_body_typed(LowerAutoCopy);
}  // namespace transform
}  // namespace tir
}  // namespace tvm