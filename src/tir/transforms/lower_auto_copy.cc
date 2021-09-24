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
 *        Currently only support 2D memory movement
 * \file lower_auto_copy.cc
 */

namespace tvm {
namespace tir {

class AutoCopyMutator : public StmtExprMutator {
 public:
  Stmt RewritePaddingBody(Stmt stmt){
    return RewriteBufferAccess(stmt, padded_buffer_map_);
  }
  
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

//  void NormalizeBuffer(Stmt stmt, Buffer buffer){
//    if (!buffer->strides.empty()) {
//      return;
//    }
//    Array<PrimExpr> new_shape;
//    for (int i = 0; i < static_cast<int>(buffer->shape.size()); i++) {
//      if (!is_one(buffer->shape[i]) ) {
//        new_shape.push_back(buffer->shape[i]);
//      }
//    }
//    if (new_shape.size() == buffer->shape.size()) {
//      return;
//    }
//    ObjectPtr<BufferNode> n = make_object<BufferNode>(*buffer.get());
//
//  }
  
  
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
      if(!is_zero(pattern)) {
        result.push_back(pattern);
      }
    }
    // todo(jinhongyii): is there some cases where the size is different?
    ICHECK(result.size() == loop_vars.size());
    return std::make_pair(result, loop_vars);
  }

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
      PrimExpr ext = analyzer.Simplify(relaxed_read_region[i].max()-relaxed_read_region[i].min()+1);
      if(is_one(ext)){
        read_index.push_back(relaxed_read_region[i].min());
      } else {
        Var var = loop_vars[0].copy_with_suffix("_relaxed"+ std::to_string(j++));
        relaxed_loop_vars.push_back(var);
        if(is_one(read_region[i]->extent)) {
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
    for (int i = 0, j=0; i < static_cast<int>(write_region.size()); i++) {
      PrimExpr new_bound = Substitute(write_region[i]->min,substitute_mapping);
      arith::IntSet relaxed_bound = arith::EvalSet(new_bound, var_dom);
      if (is_one(write_region[i]->extent)) {
        write_index.push_back(relaxed_bound.min());
      } else {
        write_index.push_back( new_bound+Substitute(mapping_pattern[j++],
                       substitute_mapping));
      }
    }
    BufferLoad new_buf_load = BufferLoad(read_buffer, read_index);
    BufferStore new_buf_store = BufferStore(write_buffer, new_buf_load, write_index);
    Stmt ret = new_buf_store;
    for (int i = static_cast<int>(relaxed_loop_vars.size()) - 1; i >= 0; i--) {
      PrimExpr loop_extent =extent[relaxed_loop_vars[i]];
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
    for (int i = 0, j=0; i < static_cast<int>(block->writes[0]->region.size()); i++) {
      if(is_zero(block->writes[0]->region[i]->extent)){
        write_index.push_back(block->writes[0]->region[i]->min);
      } else {
        Var var = runtime::Downcast<Var>(loop_vars[j]).copy_with_suffix("_inverse");
        new_loop_vars.push_back(var);
        substitute_map.Set(runtime::Downcast<Var>(loop_vars[j++]), var);
        write_index.push_back(block->writes[0]->region[i]->min + var);
      }
    }
    for (int i = 0, j=0; i < static_cast<int>(block->reads[0]->region.size()); i++) {
      if(is_zero(block->reads[0]->region[i]->extent)){
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

  Stmt RewriteWmmaLoad(Stmt stmt){
    Stmt body = stmt;
    std::vector<int> index;
    arith::Analyzer analyzer;
    int i=0;
    while (const ForNode* loop = body.as<ForNode>()) {
      if (analyzer.CanProveEqual(loop->extent, 1)) {
        continue;
      } else if (analyzer.CanProveEqual(loop->extent, 16)) {
        index.push_back(i);
      } else {
        return stmt;
      }
      i++;
    }
    if (index.size() != 16) {
      return stmt;
    }
    
  }

  Stmt RewriteWmmaStore(Stmt stmt){
  
  }
  
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
  
  Array<Buffer> PadSharedMemory(const Array<Buffer>& buffers, Map<Buffer, Buffer>* buffer_map){
    Array<Buffer> result;
    
    for (const Buffer& buffer : buffers) {
      if (buffer.scope() == "shared") {
        int type_factor = 32/buffer->dtype.bits();
        int padding_space[]{0,type_factor,2*type_factor,4*type_factor,8*type_factor,16*type_factor};
        int min_conflict_i;
        int64_t min_conflict=INT64_MAX;
        const std::vector<int64_t>& conflict = conflicts[buffer.get()];
        if (conflict.empty()) {
          continue;
        }
        ICHECK(fabs(std::pow(6,buffer->shape.size()-1)-conflict.size())<=1e-5);
        for (int i = 0; i < conflict.size(); i++) {
          if (min_conflict > conflict[i]) {
            min_conflict = conflict[i];
            min_conflict_i = i;
          }
        }
        std::vector<int> padding = getPadding(padding_space, min_conflict_i, buffer);
        ObjectPtr<BufferNode> n = make_object<BufferNode>(*buffer.get());
        Array<PrimExpr> strides;
        strides.resize(n->shape.size());
        PrimExpr stride = make_const(n->shape[0].dtype(), 1);
        for (size_t i = n->shape.size(); i != 0; --i) {
          size_t dim = i - 1;
          strides.Set(dim, stride);
          stride = stride * (n->shape[dim]+padding[dim]);
        }
        n->strides=strides;
        Buffer new_buffer(n);
        result.push_back(new_buffer);
        buffer_map->Set(buffer, new_buffer);
      } else {
        result.push_back(buffer);
      }
    }
    return result;
  }

  Stmt RewriteBufferAccess(Stmt stmt, const Map<Buffer, Buffer>& buffer_map){
    class Rewriter : public StmtExprMutator {
     public:
      Rewriter(const Map<Buffer, Buffer>& buffer_map):buffer_map_(buffer_map){}
     private:
      PrimExpr VisitExpr_(const BufferLoadNode* _op) final{
        BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_op));
        BufferLoadNode* op = load.CopyOnWrite();
        if (buffer_map_.count(op->buffer)) {
          op->buffer = buffer_map_[op->buffer];
        }
        return std::move(load);
      }
      
      Stmt VisitStmt_(const BufferStoreNode* _op) final{
        BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_op));
        BufferStoreNode* op = store.CopyOnWrite();
        if (buffer_map_.count(op->buffer)) {
          op->buffer = buffer_map_[op->buffer];
        }
        return std::move(store);
      }
      
      Stmt VisitStmt_(const BlockNode* op) final {
        // To reduce the number of blocks in block sref reuse map, we check whether the block is really
        // mutated (i.e., the old buffer appears in the block). If so, we return the block after
        // mutation. Otherwise we just return the original block.
        bool changed = false;
        // Step 1. Mutate the read region.
        Array<BufferRegion> reads;
        for (const BufferRegion& read : op->reads) {
          if(buffer_map_.count(read->buffer)){
            changed = true;
            reads.push_back(BufferRegion(buffer_map_[read->buffer], read->region));
          } else {
            reads.push_back(read);
          }
        }
        // Step 2. Mutate the write region.
        Array<BufferRegion> writes;
        for (const BufferRegion& write : op->writes) {
          if(buffer_map_.count(write->buffer)){
            changed = true;
            writes.push_back(BufferRegion(buffer_map_[write->buffer], write->region));
          } else {
            writes.push_back(write);
          }
        }
        // Step 4. Mutate `match_buffers`. If an old buffer appears as a source of MatchBufferRegion,
        // the storage scope of the target buffer also needs to be set.
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

  
  inline PrimExpr GetFlattenedIndices(Array<PrimExpr> indices, Buffer buffer, const std::vector<int>& padding){
    PrimExpr ret=0;
    for (int i = 0;i<static_cast<int>(indices.size());i++) {
      ret*=(buffer->shape[i]+padding[i]);
      ret+=indices[i];
    }
    return ret;
  }

  int CalcSharedMemoryBankConflict(Array<PrimExpr> indices, Buffer buffer,
                                   const std::vector<const ForNode*>& warp_access_loops,
                                   Map<Var, PrimExpr> substitute_map,
                                   const std::vector<int>& padding) {
    Var tx_var, ty_var, tz_var, vectorize_var;
    int prod = std::min(32, threadIdx_x_* threadIdx_y_* threadIdx_z_);
    int vectorize=1;
    for (const ForNode* loop : warp_access_loops) {
      if (loop->kind == ForKind::kVectorized) {
        vectorize_var = loop->loop_var;
        vectorize = loop->extent.as<IntImmNode>()->value;
        prod*=vectorize;
        continue;
      }
      String thread_tag = loop->thread_binding.value()->thread_tag;
      if (thread_tag == "threadIdx.x") {
        tx_var = loop->loop_var;
      } else if (thread_tag == "threadIdx.y") {
        ty_var = loop->loop_var;
      } else if (thread_tag == "threadIdx.z") {
        tz_var = loop->loop_var;
      }
    }
    PrimExpr flattened_index =  GetFlattenedIndices(indices, buffer, padding);
    int bank[32]{0};
    for (int i = 0; i < prod; i++) {
      if (vectorize_var.defined()) {
        substitute_map.Set(vectorize_var, i% vectorize);
      }
      if (tx_var.defined()) {
        substitute_map.Set(tx_var, i /vectorize % threadIdx_x_);
      }
      if (ty_var.defined()) {
        substitute_map.Set(ty_var, i / (vectorize*threadIdx_x_) % threadIdx_y_);
      }
      if (tz_var.defined()) {
        substitute_map.Set(tz_var, i / (vectorize*threadIdx_x_* threadIdx_y_) );
      }

      arith::Analyzer analyzer;
      int type_factor = 32/buffer->dtype.bits();
      ICHECK(type_factor!=0);
      PrimExpr substituted_access = analyzer.Simplify(indexmod(indexdiv(Substitute(flattened_index,
                                                                   substitute_map),type_factor),
                                                               32));
      int access_bank = substituted_access.as<IntImmNode>()->value;
      bank[access_bank]++;
    }
    int conflict = 0;
    for (int i = 0; i < 32; i++) {
      conflict = std::max(conflict, bank[i]);
    }
    return conflict;
  }
  
  inline std::vector<int> getPadding(int* padding_space, int idx, Buffer buffer){
    std::vector<int> padding;
    padding.resize(buffer->shape.size());
    for (int j = static_cast<int>(buffer->shape.size()) - 1; j >= 1; j--) {
      padding[j] = padding_space[idx%6];
      idx/=6;
    }
    padding[0] = 0;
    return std::move(padding);
  }
  
  
  void getConflictForAllPaddingSize(Stmt stmt){
    Stmt body = stmt;
    Map<Var, PrimExpr> substitute_map;
    std::vector<const ForNode*> warp_access_loops;
    int vectorize = 1;
    PrimExpr outer_loop_ct=1;
    while (const ForNode* loop = body.as<ForNode>()) {
      substitute_map.Set(loop->loop_var, loop->min);
      if (loop->kind == ForKind::kThreadBinding) {
        warp_access_loops.push_back(loop);
      } else if (loop->kind == ForKind::kVectorized) {
        vectorize = loop->extent.as<IntImmNode>()->value;
        warp_access_loops.push_back(loop);
      }
      outer_loop_ct *=loop->extent;
      body = loop->body;
    }
    for (const ForNode* loop : outer_loops_) {
      substitute_map.Set(loop->loop_var, loop->min);
      outer_loop_ct*=loop->extent;
    }
    const BufferStoreNode* node = TVM_TYPE_AS(node, body, BufferStoreNode);
    Buffer buffer;
    Array<PrimExpr> indices;
    if (node->buffer.scope() == "shared") {
      buffer=node->buffer;
      indices = node->indices;
    } else  {
      const BufferLoadNode* buf_load = node->value.as<BufferLoadNode>();
      if (buf_load->buffer.scope() == "shared") {
        buffer = buf_load->buffer;
        indices = buf_load->indices;
      } else {
        return;
      }
    }

    int type_factor = 32/buffer->dtype.bits();
    int padding_space[]{0,type_factor,2*type_factor,4*type_factor,8*type_factor,16*type_factor};
    int space_size = 1;
    for (int i = 0; i < buffer->shape.size()-1; i++) {
      space_size*=6;
    }
    std::vector<int64_t> result;
    arith::Analyzer analyzer;
    for (int i = 0; i < space_size; i++) {
      if (padding_space[i % 6] < vectorize && padding_space[i % 6] > 0) {
        result.push_back(2147483647);
        continue;
      }
      std::vector<int> padding = getPadding(padding_space, i, buffer);
      int64_t outer_extents = analyzer.Simplify(outer_loop_ct).as<IntImmNode>()->value;
      result.push_back(
          outer_extents *
          (CalcSharedMemoryBankConflict(indices, buffer, warp_access_loops, substitute_map, padding) /
               vectorize -1));
    }
    if (conflicts.count(buffer.get())) {
      std::vector<int64_t>& conflict = conflicts[buffer.get()];
      ICHECK(conflict.size() ==result.size());
      for (int i = 0; i < conflict.size(); i++) {
        conflict[i]+=result[i];
      }
    } else {
      conflicts[buffer.get()] = result;
    }
    for (const auto& e : conflicts[buffer.get()]) {
      std::cout<<e<<" ";
    }
    std::cout<<std::endl;
  }
  
  Stmt VisitStmt_(const BlockNode* op) final {
    Block block;
    if (op->annotations.count("auto_copy") &&
        is_one(Downcast<PrimExpr>(op->annotations["auto_copy"]))) {
      in_auto_copy_ = true;
      block = runtime::Downcast<Block>(StmtMutator::VisitStmt_(op));
      std::pair<Array<PrimExpr>, Array<Var>> mapping = GetMapping(block);
      block = RelaxThreads(block, mapping);
      BlockNode* n = block.CopyOnWrite();
      if ((src_scope_.rank == runtime::StorageRank::kGlobal &&
           tgt_scope_.rank == runtime::StorageRank::kShared) ||
          (src_scope_.rank == runtime::StorageRank::kShared &&
           tgt_scope_.rank == runtime::StorageRank::kGlobal)) {
        int vector_bytes;
        if (block->annotations.count("vector_bytes")) {
          IntImm vec_bytes = Downcast<IntImm>(block->annotations["vector_bytes"]);
          vector_bytes = vec_bytes->value;
        } else {
          vector_bytes = 1;
        }
        bool need_inverse = src_scope_.rank == runtime::StorageRank::kShared;
        n->body = CoalesceGlobalLoad(block, mapping, vector_bytes, need_inverse);
        getConflictForAllPaddingSize(n->body);
        n->alloc_buffers = PadSharedMemory(n->alloc_buffers, &padded_buffer_map_);
      }
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
  std::unordered_map<const BufferNode*, std::vector<int64_t>> conflicts;
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