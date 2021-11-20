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

using support::NDIntSet;


class AutoCopyMutator : public StmtExprMutator {
 public:
  AutoCopyMutator(std::unordered_map<std::string, int> thread_extent){
    if (thread_extent.count("threadIdx.x")) {
      threadIdx_x_=thread_extent["threadIdx.x"];
    }
    if (thread_extent.count("threadIdx.y")) {
      threadIdx_y_=thread_extent["threadIdx.y"];
    }
    if (thread_extent.count("threadIdx.z")) {
      threadIdx_z_=thread_extent["threadIdx.z"];
    }
  }
  
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
  
  std::pair<Stmt, For> LiftThreadBindingLoops(Stmt stmt){
    std::vector<const ForNode*> normal_loops;
    std::vector<const ForNode*> thread_binding_loops;
    Stmt body = stmt;
    while (const ForNode* loop = body.as<ForNode>()) {
      if (loop->kind == ForKind::kThreadBinding) {
        thread_binding_loops.push_back(loop);
      } else {
        normal_loops.push_back(loop);
      }
      body = loop->body;
    }
    
    for (int i = static_cast<int>(normal_loops.size()) - 1; i >= 0; i--) {
      const ForNode* loop = normal_loops[i];
      body = For(loop->loop_var, loop->min, loop->extent, loop->kind, body, loop->thread_binding,
                 loop->annotations);
    }
    For compute_location;
    for (int i = static_cast<int>(thread_binding_loops.size()) - 1; i >= 0; i--) {
      const ForNode* loop = thread_binding_loops[i];
      body = For(loop->loop_var, loop->min, loop->extent, loop->kind, body, loop->thread_binding,
                 loop->annotations);
      if (i == static_cast<int>(thread_binding_loops.size()) - 1) {
        compute_location=Downcast<For>(body);
      }
    }
    return std::make_pair(body, compute_location);
  }
  
  Stmt CreateLocalStage(Stmt stmt, int vector_bytes){
    Stmt body;
    For compute_location;
    std::tie(body, compute_location) = LiftThreadBindingLoops(std::move(stmt));
    return InsertCacheStage(body ,false, "local", compute_location).first;
  }
  
  Stmt RewriteWmmaLoad(Stmt stmt){
    Array<MatchBufferRegion> match_buffers;
    Stmt body = stmt;
    Map<Var, Range> var_range;
    std::vector<const ForNode*> loops;
    while (const ForNode* loop = body.as<ForNode>()) {
      loops.push_back(loop);
      body = loop->body;
    }
    for (int i = 1; i <= 2; i++) {
      const ForNode* loop = loops[loops.size()-i];
      var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    }
    const BufferStoreNode* buf_store = TVM_TYPE_AS(buf_store, body, BufferStoreNode);
    const BufferLoadNode* buf_load = TVM_TYPE_AS(buf_load, buf_store->value, BufferLoadNode);
    Buffer src_buffer = buf_load->buffer;
    Buffer tgt_buffer = buf_store->buffer;
    padding_constraint[src_buffer.get()] = 3;
    TensorIntrin wmma_load;
    if (tgt_buffer.scope() == "wmma.matrix_a") {
      wmma_load = tir::TensorIntrin::Get("wmma_load_a");
    } else {
      wmma_load = tir::TensorIntrin::Get("wmma_load_b");
    }
    
    auto param = wmma_load->implementation->params[0];
    Buffer new_src_buffer = wmma_load->implementation->buffer_map.at(param);
    auto read_int_set = arith::EvalSet(buf_load->indices, AsIntSet(var_range));
    Array<Range> read_region;
    for (const auto& int_set : read_int_set) {
      read_region.push_back(int_set.CoverRange(Range()));
    }
    match_buffers.push_back(MatchBufferRegion(new_src_buffer, BufferRegion(src_buffer,
                                                                           read_region)));
    param = wmma_load->implementation->params[1];
    Buffer new_tgt_buffer = wmma_load->implementation->buffer_map.at(param);
    auto write_int_set = arith::EvalSet(buf_store->indices, AsIntSet(var_range));
    Array<Range> write_region;
    for (const auto& int_set : write_int_set) {
      write_region.push_back(int_set.CoverRange(Range()));
    }
    match_buffers.push_back(MatchBufferRegion(new_tgt_buffer, BufferRegion(tgt_buffer,
                                                                            write_region)));

    PrimExpr frag_index = floordiv(new_tgt_buffer->elem_offset, 256) +
                          floordiv(floormod(new_tgt_buffer->elem_offset, 256), 16);

    auto new_src_pointer = Call(
        runtime::DataType::Handle(), builtin::tvm_access_ptr(),
        {TypeAnnotation(new_src_buffer->dtype), new_src_buffer->data, new_src_buffer->elem_offset,
         new_src_buffer->strides[new_src_buffer->strides.size() - 2] * 16, 1});
    
    Stmt wmma_body = Evaluate(Call(
        runtime::DataType::Handle(), builtin::tvm_load_matrix_sync(),
        {new_tgt_buffer->data, 16, 16, 16, frag_index, new_src_pointer,
         new_src_buffer->strides[new_src_buffer->strides.size() - 2], StringImm("row_major")}));
    wmma_body =  BlockRealize({},Bool(true), Block({},{},{},"wmma_load",wmma_body,NullOpt, {},
                                             match_buffers,{}));
    for (int i = static_cast<int>(loops.size()) - 3; i >= 0; i--) {
      wmma_body = For(loops[i]->loop_var, loops[i]->min, loops[i]->extent, loops[i]->kind,
                      wmma_body, loops[i]->thread_binding, loops[i]->annotations);
    }
    return wmma_body;
  }
  
  Stmt RewriteWmmaStore(Stmt stmt){
    Array<MatchBufferRegion> match_buffers;
    Stmt body = stmt;
    Map<Var, Range> var_range;
    std::vector<const ForNode*> loops;
    while (const ForNode* loop = body.as<ForNode>()) {
      loops.push_back(loop);
      body = loop->body;
    }
    for (int i = 1; i <= 2; i++) {
      const ForNode* loop = loops[loops.size()-i];
      var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    }
    const BufferStoreNode* buf_store = TVM_TYPE_AS(buf_store, body, BufferStoreNode);
    const BufferLoadNode* buf_load = TVM_TYPE_AS(buf_load, buf_store->value, BufferLoadNode);
    Buffer src_buffer = buf_load->buffer;
    Buffer tgt_buffer = buf_store->buffer;
    padding_constraint[tgt_buffer.get()] = 3;
    TensorIntrin wmma_store = tir::TensorIntrin::Get("wmma_store");
    
    auto param = wmma_store->implementation->params[0];
    Buffer new_src_buffer = wmma_store->implementation->buffer_map.at(param);
    auto read_int_set = arith::EvalSet(buf_load->indices, AsIntSet(var_range));
    Array<Range> read_region;
    for (const auto& int_set : read_int_set) {
      read_region.push_back(int_set.CoverRange(Range()));
    }
    match_buffers.push_back(MatchBufferRegion(new_src_buffer, BufferRegion(src_buffer,
                                                                           read_region)));
    param = wmma_store->implementation->params[1];
    Buffer new_tgt_buffer = wmma_store->implementation->buffer_map.at(param);
    auto write_int_set = arith::EvalSet(buf_store->indices, AsIntSet(var_range));
    Array<Range> write_region;
    for (const auto& int_set : write_int_set) {
      write_region.push_back(int_set.CoverRange(Range()));
    }
    match_buffers.push_back(MatchBufferRegion(new_tgt_buffer, BufferRegion(tgt_buffer,
                                                                            write_region)));

    PrimExpr frag_index = floordiv(new_src_buffer->elem_offset, 256) +
                          floordiv(floormod(new_src_buffer->elem_offset, 256), 16);

    auto new_tgt_pointer = Call(runtime::DataType::Handle(), builtin::tvm_access_ptr(),
                                {TypeAnnotation(new_tgt_buffer->dtype), new_tgt_buffer->data,
                                 new_tgt_buffer->elem_offset, new_tgt_buffer->strides[0] * 16, 2});

    Stmt wmma_body = Evaluate(Call(runtime::DataType::Handle(), builtin::tvm_store_matrix_sync(),
                         {new_src_buffer->data, 16, 16, 16, frag_index, new_tgt_pointer,
                          new_tgt_buffer->strides[0], StringImm("row_major")}));
    wmma_body =  BlockRealize({},Bool(true), Block({},{},{},"wmma_store",wmma_body,NullOpt, {},
                                             match_buffers,{}));
    for (int i = static_cast<int>(loops.size()) - 3; i >= 0; i--) {
      wmma_body = For(loops[i]->loop_var, loops[i]->min, loops[i]->extent, loops[i]->kind,
                      wmma_body, loops[i]->thread_binding, loops[i]->annotations);
    }
    return wmma_body;
  }
  
  /**
   * \brief rewrite wmma load
   * @param block the specific block
   * @param match_buffers the match_buffer to be appended to the new block
   * @return the result stmt
   */
  Stmt SharedToWmma(Stmt stmt) {
    auto pair = TileWmmaBlock(stmt);
    return RewriteWmmaLoad(pair.first);
  }

  /**
   * \brief rewrite wmma store
   * @param block the specific block
   * @param match_buffers the match_buffer to be appended to the new block
   * @return the result stmt
   */
  Stmt WmmaToShared(Stmt stmt) {
    auto pair = TileWmmaBlock(stmt);
    return RewriteWmmaStore(pair.first);
  }
  
  
  std::pair<Stmt, Optional<For>> TileWmmaBlock(Stmt stmt){
    Stmt body = stmt;
    std::vector<const ForNode*> loops;
    while (const ForNode* loop = body.as<ForNode>()) {
      loops.push_back(loop);
      body = loop->body;
    }
    arith::Analyzer analyzer;
    PrimExpr extent_last1 = loops[loops.size() - 1]->extent, extent_last2=loops[loops.size() -
                                                                                  2]->extent;
    
    if (!analyzer.CanProve(floormod(extent_last1, 16) == 0) ||
        !analyzer.CanProve(floormod(extent_last2, 16) == 0)) {
      return std::make_pair(stmt, NullOpt);
    }
    std::vector<Var> new_loop_vars;
    Array<PrimExpr> factor{floordiv(extent_last2,16), floordiv(extent_last1, 16), 16, 16};
    new_loop_vars.reserve(4);
    for (int i = 0; i < 4; i++) {
      new_loop_vars.push_back(loops[loops.size()-(i+1)%2-1]->loop_var.copy_with_suffix
                              (std::to_string(i/2)));
    }
    Map<Var, PrimExpr> substitue_value;
    substitue_value.Set(loops[loops.size() - 2]->loop_var, new_loop_vars[0]*16+new_loop_vars[2]);
    substitue_value.Set(loops[loops.size()-1]->loop_var, new_loop_vars[1]*16+new_loop_vars[3]);
    body = Substitute(body, substitue_value);
    for (int i = 3; i >= 0; i--) {
      body = For(new_loop_vars[i], 0, factor[i], ForKind::kSerial, body);
    }
    For compute_location = Downcast<For>(body);
    for (int i = static_cast<int>(loops.size()) - 3; i>=0;i--) {
      body = For(loops[i]->loop_var, loops[i]->min, loops[i]->extent, loops[i]->kind, body,
                 loops[i]->thread_binding, loops[i]->annotations);
    }
    return std::make_pair(body, compute_location);
  }
  
  std::pair<Stmt, SeqStmt> InsertCacheStage(Stmt stmt, bool is_write_cache, String storage_scope,
                                            Stmt
                                                                                   compute_location){
    class IndexPatternFinder : public ExprVisitor{
     public:
      IndexPatternFinder(const Map<Var, Range>& var_range, Array<PrimExpr>* resulting_index)
          :var_range_(var_range), resulting_index(resulting_index){}
      
      static Array<Array<PrimExpr>> getRankPromotedShape(Array<PrimExpr> indices, const Map<Var, Range>&
                                                                                      var_range,
                                                         Array<PrimExpr>* rewrite_indices){
        Map<Var, arith::IntSet> var_dom = AsIntSet(var_range);
        Array<Array<PrimExpr>> new_shape;
        for (const PrimExpr& expr : indices) {
          IndexPatternFinder extractor(var_range, rewrite_indices);
          arith::IntSet intset = arith::EvalSet(expr, var_dom);
          extractor.mod = intset.max()+1;
          extractor.div = 1;
          extractor.offset =0;
          extractor(expr);
          Array<PrimExpr> access_shape = extractor.access_shape;
          for (int i = static_cast<int>(access_shape.size()) - 1; i >= 1; i--) {
            if (!is_zero(floormod(extractor.offset, access_shape[i]))) {
              return {};
            } else {
              extractor.offset= floordiv(extractor.offset, access_shape[i]);
            }
          }
          access_shape.Set(0, extractor.offset+access_shape[0]);
          new_shape.push_back(access_shape);
        }
        return new_shape;
      }
  
      void VisitExpr_(const VarNode* op) final{
        arith::Analyzer analyzer;
        PrimExpr extent = var_range_[GetRef<Var>(op)]->extent;
        PrimExpr access_iter_range = min(mod, (max(1,floordiv(extent, div))));
        if(!analyzer.CanProveEqual(1, access_iter_range)){
          access_shape.push_back(access_iter_range);
          resulting_index->push_back(floormod(floordiv(GetRef<Var>(op),div), mod));
        }
      }
  
      void VisitExpr_(const FloorDivNode* op) final{
        PrimExpr old_div = div;
        div*=op->b;
        ExprVisitor::VisitExpr_(op);
        div = old_div;
      }
  
      void VisitExpr_(const FloorModNode* op) final{
        PrimExpr old_mod = mod;
        mod=max(1, min(floordiv(op->b,div), mod));
        ExprVisitor::VisitExpr_(op);
        mod = old_mod;
      }
  
      void VisitExpr_(const MulNode* op) final{
        PrimExpr old_mod = mod;
        PrimExpr old_div = div;
        div = max(1, floordiv(div,op->b));
        mod = max(1, floordiv(mod, floordiv(op->b, floordiv(old_div, div))));
        ExprVisitor::VisitExpr_(op);
        mod = old_mod;
        div = old_div;
      }
  
      void VisitExpr_(const AddNode* op) final{
        if (is_const_int(op->b)) {
          offset+= floormod(floordiv(op->b, div),mod);
        }
        ExprVisitor::VisitExpr_(op);
      }
  
      PrimExpr div;
      PrimExpr mod;
      PrimExpr offset;
      Map<Var, Range> var_range_;
      Array<PrimExpr> access_shape;
      Array<PrimExpr>* resulting_index;
    };
    
    class RankPromoter:public StmtExprMutator{
     public:
      static Array<PrimExpr> FlattenNewShape(const Array<Array<PrimExpr>>& new_shape){
        Array<PrimExpr> ret;
        ret.reserve(new_shape.size());
        for (int i = 0; i < static_cast<int>(new_shape.size()); i++) {
          PrimExpr prod = 1;
          for (int j = 0; j < static_cast<int>(new_shape[i].size()); j++) {
            prod*=new_shape[i][j];
          }
          ret.push_back(prod);
        }
        return ret;
      }
      
      static Array<PrimExpr> RewriteIndex(const Array<PrimExpr>& indices, const
                                          Array<Array<PrimExpr>>&
                                              new_shape){
        Array<PrimExpr> new_indices;
        ICHECK_EQ(indices.size(), new_shape.size());
        for (int i = 0; i < static_cast<int>(indices.size()); i++) {
          PrimExpr index = indices[i];
          Array<PrimExpr> dim_convert_index(new_shape[i].size(), 0);
          for (int j = static_cast<int>(new_shape[i].size()) - 1; j >= 0; j--) {
            dim_convert_index.Set(j, floormod(index, new_shape[i][j]));
            index = floordiv(index, new_shape[i][j]);
          }
          for (int j = 0; j < static_cast<int>(new_shape[i].size()); j++) {
            new_indices.push_back(dim_convert_index[j]);
          }
        }
        return new_indices;
      }
      
      
      static Array<PrimExpr> RewriteBackIndex(const Array<PrimExpr>& indices, const
                                              Array<Array<PrimExpr>>&
                                                  new_shape){
        Array<PrimExpr> new_indices;
        int offset =0;
        for (int i = 0; i < static_cast<int>(new_shape.size()); i++) {
          PrimExpr index = 0;
          for (int j = 0; j < static_cast<int>(new_shape[i].size()); j++) {
            index *=new_shape[i][j];
            index+=indices[offset+j];
          }
          new_indices.push_back(index);
          offset+=new_shape[i].size();
        }
        return new_indices;
      }
      RankPromoter(const Buffer& src, const Buffer& dst, const Array<Array<PrimExpr>>& new_shape,
                   const Array<Array<PrimExpr>>& relaxed_new_shape, const Array<Range>& relaxed_region)
          : src_(src),
            dst_(dst),
            new_shape_(new_shape),
            relaxed_new_shape_(relaxed_new_shape),
            relaxed_region_(relaxed_region) {}

      static Stmt RewriteBody(Stmt stmt, const Buffer& src, const Buffer& dst,
                              const Array<Array<PrimExpr>>& new_shape, const
                              Array<Array<PrimExpr>>& relaxed_new_shape,
                              const Array<Range>& relaxed_region){
        RankPromoter promoter(src, dst, new_shape, relaxed_new_shape, relaxed_region);
        return promoter(stmt);
      }
     private:
      Stmt VisitStmt_(const BufferStoreNode* _store) final {
        BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_store));
        if (store->buffer.same_as(src_)) {
          ObjectPtr<BufferStoreNode> new_store = make_object<BufferStoreNode>(*store.get());
          new_store->buffer = dst_;
          new_store->indices = ConvertIndices(new_store->indices);
          return BufferStore(new_store);
        }
        return std::move(store);
      }
      
      PrimExpr VisitExpr_(const BufferLoadNode* _load) final {
        BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_load));
        if (load->buffer.same_as(src_)) {
          ObjectPtr<BufferLoadNode> new_load = make_object<BufferLoadNode>(*load.get());
          new_load->buffer = dst_;
          new_load->indices = ConvertIndices(new_load->indices);
          return BufferLoad(new_load);
        }
        return std::move(load);
      }
      
      Array<PrimExpr> ConvertIndices(const Array<PrimExpr>& indices){
        Array<PrimExpr> rewrite_indices = RewriteIndex(indices, new_shape_);
        arith::Analyzer analyzer;
        for (int i = 0; i < rewrite_indices.size(); i++) {
          rewrite_indices.Set(i, analyzer.Simplify(rewrite_indices[i]- relaxed_region_[i]->min));
        }
        return RewriteBackIndex(rewrite_indices, relaxed_new_shape_);
      }
      
      const Buffer& src_;
      const Buffer& dst_;
      Array<Array<PrimExpr>> new_shape_;
      Array<Array<PrimExpr>> relaxed_new_shape_;
      Array<Range> relaxed_region_;

    };
    
    Stmt body = stmt;
    std::vector<const ForNode*> loops;
    bool is_relax_var = compute_location->IsInstance<BlockNode>();
    Map<Var, Range> relax_var_range;
    Map<Var, Range> all_var_range;
    PrimExpr vector_bytes =-1;
    while (const ForNode* loop = body.as<ForNode>()) {
      if (is_relax_var) {
        relax_var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
      } else {
        loops.push_back(loop);
      }
      all_var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
      if (loop == compute_location.get()) {
        is_relax_var=true;
      }
      if (loop->kind == ForKind::kVectorized) {
        vector_bytes = loop->extent;
      }
      body = loop->body;
    }
    for (const ForNode* loop : outer_loops_) {
      if (loop->kind == ForKind::kThreadBinding) {
        const String& thread_tag = loop->thread_binding.value()->thread_tag;
        if (CanRelaxStorageUndereThread(runtime::StorageScope::Create(storage_scope),
                                        runtime::ThreadScope::Create
                                        (thread_tag))) {
          relax_var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
        }
      }
      all_var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    }
    
    const BufferStoreNode* buf_store = TVM_TYPE_AS(buf_store, body, BufferStoreNode);
    const BufferLoadNode* buf_load = TVM_TYPE_AS(buf_load, buf_store->value, BufferLoadNode);
    Buffer orig_buffer = is_write_cache?buf_store->buffer:buf_load->buffer;
    Array<PrimExpr> indices = is_write_cache? buf_store->indices:buf_load->indices;
    //step 1. get the rank promotion shape
    Array<PrimExpr> rewrite_indices;
    Array<Array<PrimExpr>> new_shape =
        IndexPatternFinder::getRankPromotedShape(indices, all_var_range, &rewrite_indices);
    Region relaxed_region;
    auto relax_var_intset = AsIntSet(relax_var_range);
    arith::Analyzer analyzer;
    analyzer.Bind(all_var_range);
    for (const PrimExpr& index : rewrite_indices) {
      auto int_set = arith::EvalSet(index, relax_var_intset);
      relaxed_region.push_back(Range::FromMinExtent(
          int_set.min(), analyzer.Simplify(int_set.max() - int_set.min() + 1)));
    }
//    step 2. generate loops
    Array<Var> new_loop_vars;
    Array<PrimExpr> orig_buf_indices, new_buf_indices;
    Array<Array<PrimExpr>> relaxed_new_shape;
    for (int i=0;i<static_cast<int>(relaxed_region.size());i++) {
      Var new_loop_var = Var("ax"+ std::to_string(i));
      new_loop_vars.push_back(new_loop_var);
      orig_buf_indices.push_back(relaxed_region[i]->min+new_loop_var);
      new_buf_indices.push_back(new_loop_var);
    }
    relaxed_new_shape.reserve(new_shape.size());
    for (int i = 0, ct = 0; i < static_cast<int>(new_shape.size()); i++) {
      Array<PrimExpr> layer;
      for (int j = 0; j < static_cast<int>(new_shape[i].size()); j++, ct++) {
        layer.push_back(relaxed_region[ct]->extent);
      }
      relaxed_new_shape.push_back(layer);
    }
    Buffer new_buffer = WithScope(orig_buffer, storage_scope);
    BufferNode* buffer_ptr = new_buffer.CopyOnWrite();
    buffer_ptr->shape= RankPromoter::FlattenNewShape(relaxed_new_shape);
    alloc_buffers_.push_back(new_buffer);
    Array<PrimExpr> real_orig_buf_indices =
        RankPromoter::RewriteBackIndex(orig_buf_indices,
                                                                          new_shape);
    Array<PrimExpr> real_new_buf_indices =
        RankPromoter::RewriteBackIndex(new_buf_indices,
                                                                          relaxed_new_shape);
    Stmt generate_body =
        is_write_cache ? BufferStore(orig_buffer, BufferLoad(new_buffer, real_new_buf_indices),
                                     real_orig_buf_indices)
                       : BufferStore(new_buffer, BufferLoad(orig_buffer, real_orig_buf_indices),
                                     real_new_buf_indices);
    for (int i = static_cast<int>(relaxed_region.size()) - 1; i >= 0; i--) {
      if(i==static_cast<int>(relaxed_region.size()) - 1 && !is_const_int(vector_bytes, -1)){
        ICHECK(analyzer.CanProve(vector_bytes==relaxed_region[i]->extent));
        generate_body = For(new_loop_vars[i], 0, relaxed_region[i]->extent, ForKind::kVectorized,
                            generate_body);
      } else {
        generate_body =
            For(new_loop_vars[i], 0, relaxed_region[i]->extent, ForKind::kSerial, generate_body);
      }
    }
    Stmt rewrite_body;
    if (const auto* node = compute_location.as<ForNode>()) {
      rewrite_body = node->body;
    } else if (const auto* node = compute_location.as<BlockNode>()) {
      rewrite_body = node->body;
    } else {
      LOG(FATAL)<<"wrong compute location type";
    }
    rewrite_body = RankPromoter::RewriteBody(rewrite_body, orig_buffer,
                                                  new_buffer, new_shape, relaxed_new_shape,
                                                  relaxed_region);
    SeqStmt insert_location;
    if (is_write_cache) {
      generate_body = insert_location = SeqStmt({rewrite_body, generate_body});
    } else {
       generate_body = insert_location = SeqStmt({generate_body, rewrite_body});
    }
    for (int i = static_cast<int>(loops.size()) - 1; i >= 0; i--) {
      generate_body = For(loops[i]->loop_var, loops[i]->min, loops[i]->extent, loops[i]->kind,
generate_body,loops[i]->thread_binding, loops[i]->annotations);
    }
    return std::make_pair(generate_body, insert_location);
  }
  
  
  class WmmaToGlobalRewriter: public StmtExprMutator{
   public:
    WmmaToGlobalRewriter(const SeqStmtNode* tgt_stmt,
                         AutoCopyMutator* self, int vector_bytes_)
        : tgt_stmt_(tgt_stmt),
          self(self),
          vector_bytes_(vector_bytes_) {}

   private:
    Stmt VisitStmt_(const SeqStmtNode* op) final{
      if (op == tgt_stmt_) {
        ICHECK_EQ(op->seq.size(), 2);
//        LOG(INFO)<<op->seq[0];
        Stmt wmma_to_shared = self->RewriteWmmaStore(op->seq[0]);
        Stmt shared_to_global = self->FuseNestLoops(op->seq[1]);
        shared_to_global = self->SplitBindVectorize(shared_to_global, vector_bytes_);
        return SeqStmt({wmma_to_shared, shared_to_global});
      } else {
        return StmtMutator::VisitStmt_(op);
      }
    }
    
    const SeqStmtNode* tgt_stmt_;
    AutoCopyMutator* self;
    int vector_bytes_;
  };
  
  Stmt WmmaToGlobal(Stmt stmt, int vector_bytes){
    
    Stmt body;
    Optional<For> compute_location;
    
    std::tie(body, compute_location) = TileWmmaBlock(stmt);
    SeqStmt seq;
    std::tie(body, seq) = InsertCacheStage(body, true, "shared.dyn", compute_location.value());
    WmmaToGlobalRewriter rewriter(seq.get(), this, vector_bytes);
    return rewriter(body);
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
   * \param buffers the given buffers
   * \param buffer_map the mapping from old buffer to the new padded buffer
   * \return the list of new padded buffers
   */
  Array<Buffer> PadSharedMemory(const Array<Buffer>& buffers, Map<Buffer, Buffer>* buffer_map) {
    Array<Buffer> result;

    for (const Buffer& buffer : buffers) {
      runtime::StorageScope scope =runtime::StorageScope::Create(buffer.scope());
      if (scope.rank==runtime::StorageRank::kShared) {
        int type_factor = 32 / buffer->dtype.bits();
        auto patterns = patterns_[buffer.get()];
        if (patterns.empty()) {
          result.push_back(buffer);
          continue;
        }
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
  
  static Array<PrimExpr> getWarpAccess(const Array<PrimExpr>& indices,
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
  void AnalyzePatternForPadding(Stmt stmt) {
    
    class PatternAnalyzer:public StmtExprVisitor{
     public:
      PatternAnalyzer(const Map<Var, PrimExpr>& substitute_map,
                     AutoCopyMutator* self)
          : substitute_map_(substitute_map), self(self) {}

     private:
      void VisitStmt_(const ForNode* op)final{
        if (op->kind == ForKind::kVectorized || op->kind == ForKind::kThreadBinding) {
          warp_access_loops_.push_back(op);
        }
        substitute_map_.Set(op->loop_var, op->min);
        StmtExprVisitor::VisitStmt_(op);
        substitute_map_.erase(op->loop_var);
        if (op->kind == ForKind::kVectorized || op->kind == ForKind::kThreadBinding) {
          warp_access_loops_.pop_back();
        }
      }
      
      void VisitStmt_(const BufferStoreNode* op) final{
        runtime::StorageScope scope = runtime::StorageScope::Create(op->buffer.scope());
        if (scope.rank == runtime::StorageRank::kShared) {
          Map<Var, Range> var_range;
          Array<PrimExpr> substitued_indices =
              getWarpAccess(op->indices, warp_access_loops_, substitute_map_, &var_range);
          std::vector<std::vector<Pattern>> patterns =
              PatternCollector::CollectPattern(substitued_indices, var_range);
          self->patterns_[op->buffer.get()].push_back(patterns);
          LOG(INFO)<<op->buffer->name;
          for (const auto& pattern_single_dim : patterns) {
            std::cerr << "{";
            for (const auto& pattern : pattern_single_dim) {
              std::cerr << "{" << pattern.extent << "," << pattern.scale << "}";
            }
            std::cerr << "}, ";
          }
          std::cerr << std::endl;
        }
        StmtExprVisitor::VisitStmt_(op);
      }
      
      void VisitExpr_(const BufferLoadNode* op) final{
        runtime::StorageScope scope = runtime::StorageScope::Create(op->buffer.scope());
        if (scope.rank == runtime::StorageRank::kShared) {
          Map<Var, Range> var_range;
          Array<PrimExpr> substitued_indices =
              getWarpAccess(op->indices, warp_access_loops_, substitute_map_, &var_range);
          std::vector<std::vector<Pattern>> patterns =
              PatternCollector::CollectPattern(substitued_indices, var_range);
          self->patterns_[op->buffer.get()].push_back(patterns);
          LOG(INFO)<<op->buffer->name;
          for (const auto& pattern_single_dim : patterns) {
            std::cerr << "{";
            for (const auto& pattern : pattern_single_dim) {
              std::cerr << "{" << pattern.extent << "," << pattern.scale << "}";
            }
            std::cerr << "}, ";
          }
          std::cerr << std::endl;
        }
        StmtExprVisitor::VisitExpr_(op);
      }
      
      void VisitStmt_(const BlockNode* op) final{
        if (const auto* eval = op->body.as<EvaluateNode>()) {
          if (const auto* call = eval->value.as<CallNode>()) {
            if (call->op == builtin::tvm_load_matrix_sync() ||
                call->op == builtin::tvm_store_matrix_sync()) {
              for (const MatchBufferRegion& r : op->match_buffers) {
                Buffer src_buffer = r->source->buffer;
                runtime::StorageScope scope = runtime::StorageScope::Create(src_buffer.scope());
                if (scope.rank == runtime::StorageRank::kShared) {
                  Region region = r->source->region;
                  Array<PrimExpr> indices;
                  Map<Var, Range> var_range;
                  for (int i = 0; i < static_cast<int>(region.size()); i++) {
                    Var var("region" + std::to_string(i));
                    indices.push_back(region[i]->min + var);
                    var_range.Set(var, Range::FromMinExtent(0, region[i]->extent));
                  }
                  Array<PrimExpr> substitued_indices =
                      getWarpAccess(indices, warp_access_loops_, substitute_map_, &var_range);
                  std::vector<std::vector<Pattern>> patterns =
                      PatternCollector::CollectPattern(substitued_indices, var_range);
                  self->patterns_[src_buffer.get()].push_back(patterns);
                  LOG(INFO)<<src_buffer->name;
                  for (const auto& pattern_single_dim : patterns) {
                    std::cerr << "{";
                    for (const auto& pattern : pattern_single_dim) {
                      std::cerr << "{" << pattern.extent << "," << pattern.scale << "}";
                    }
                    std::cerr << "}, ";
                  }
                  std::cerr << std::endl;
                }
              }
            }
          }
        }
      }
      
      Map<Var, PrimExpr> substitute_map_;
      std::vector<const ForNode*> warp_access_loops_;
      AutoCopyMutator* self;
    };
    
    Map<Var, PrimExpr> substitute_map;
    for (const ForNode* loop : outer_loops_) {
      substitute_map.Set(loop->loop_var, loop->min);
    }
    PatternAnalyzer analyzer(substitute_map, this);
    analyzer(stmt);
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    Block block;
    if (op->annotations.count("auto_copy") &&
        is_one(Downcast<PrimExpr>(op->annotations["auto_copy"]))) {
      in_auto_copy_ = true;
      block = runtime::Downcast<Block>(StmtMutator::VisitStmt_(op));
      std::pair<Array<PrimExpr>, Array<Var>> mapping = GetMapping(block);
      block = RelaxThreads(block, mapping);
      int vector_bytes;
      if (block->annotations.count("vector_bytes")) {
        IntImm vec_bytes = Downcast<IntImm>(block->annotations["vector_bytes"]);
        vector_bytes = vec_bytes->value;
      } else {
        vector_bytes = block->writes[0]->buffer->dtype.bytes();
      }
      BlockNode* n = block.CopyOnWrite();
      if ((src_scope_.rank == runtime::StorageRank::kGlobal &&
           tgt_scope_.rank == runtime::StorageRank::kShared) ||
          (src_scope_.rank == runtime::StorageRank::kShared &&
           tgt_scope_.rank == runtime::StorageRank::kGlobal)) {
        bool need_inverse = src_scope_.rank == runtime::StorageRank::kShared;
        n->body = CoalesceGlobalLoad(block, mapping, vector_bytes, need_inverse);
        if (block->annotations.count("local_stage") && is_one(Downcast<PrimExpr>
            (op->annotations["local_stage"]))) {
          n->body = CreateLocalStage(n->body, vector_bytes);
        }
      } else if ((tgt_scope_.rank == runtime::StorageRank::kWMMAMatrixA ||
                  tgt_scope_.rank == runtime::StorageRank::kWMMAMatrixB)) {
        n->body = SharedToWmma(block->body);
      } else if (src_scope_.rank == runtime::StorageRank::kWMMAAccumulator &&
                 tgt_scope_.rank ==runtime::StorageRank::kShared) {
        n->body = WmmaToShared(block->body);
      } else if(src_scope_.rank == runtime::StorageRank::kWMMAAccumulator &&
                 tgt_scope_.rank ==runtime::StorageRank::kGlobal){
        n->body= WmmaToGlobal(block->body,vector_bytes);
      }
      AnalyzePatternForPadding(block->body);
      for (const Buffer& buffer : alloc_buffers_) {
        n->alloc_buffers.push_back(buffer);
      }
      n->alloc_buffers = PadSharedMemory(n->alloc_buffers, &padded_buffer_map_);
    } else {
      block = runtime::Downcast<Block>(StmtMutator::VisitStmt_(op));
      BlockNode* n = block.CopyOnWrite();
      for (const Buffer& buffer : alloc_buffers_) {
        n->alloc_buffers.push_back(buffer);
      }
      n->alloc_buffers = PadSharedMemory(n->alloc_buffers, &padded_buffer_map_);
    }
    alloc_buffers_.clear();
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
  Array<Buffer> alloc_buffers_;
  std::unordered_map<const BufferNode*, std::vector<std::vector<std::vector<Pattern>>>> patterns_;
  std::unordered_map<const BufferNode*, int> padding_constraint;
};

class ThreadExtentCollector:public StmtVisitor{
 public:
  static std::unordered_map<std::string, int> CollectThreadExtent(const Stmt& stmt){
    ThreadExtentCollector collector;
    collector(stmt);
    return collector.thread_extent_;
  }
  
 private:
  void VisitStmt_(const BlockNode* op) final{
    if (op->annotations.count("warp_execution") && is_one
        (Downcast<PrimExpr>(op->annotations["warp_execution"]))) {
      thread_extent_["threadIdx.x"] = 32;
    }
    StmtVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const ForNode* op) final{
    if (op->thread_binding.defined() && op->thread_binding.value()->iter_type==kThreadIndex) {
      thread_extent_[op->thread_binding.value()->thread_tag] = op->extent.as<IntImmNode>()->value;
    }
    StmtVisitor::VisitStmt_(op);
  }
  
  std::unordered_map<std::string, int> thread_extent_;
};

namespace transform {
Pass LowerAutoCopy() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    AutoCopyMutator mutator(ThreadExtentCollector::CollectThreadExtent(n->body));
    n->body = mutator(std::move(n->body));
    n->body = mutator.RewritePaddingBody(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerAutoCopy", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerAutoCopy").set_body_typed(LowerAutoCopy);
}  // namespace transform
}  // namespace tir
}  // namespace tvm