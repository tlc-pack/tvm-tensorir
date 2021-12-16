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
#include "rewrite_rule.h"

namespace tvm{
namespace tir{
/*!
 * \brief lift all the thread binding loops
 * \param stmt the top loop
 * \return a pair. The first is the transformed stmt.
 *         The second is the lowest thread binding loop.
 */
std::pair<Stmt, For> LiftThreadBindingLoops(Stmt stmt) {
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
      compute_location = Downcast<For>(body);
    }
  }
  return std::make_pair(body, compute_location);
}

/*!
 * \brief Analyze the access pattern for rank promotion
 */
class IndexPatternFinder : public ExprVisitor {
 public:
  IndexPatternFinder(const Map<Var, Range>& var_range, Array<PrimExpr>* resulting_index)
      : var_range_(var_range), resulting_index_(resulting_index) {}
    
  static Array<Array<PrimExpr>> getRankPromotedShape(Array<PrimExpr> indices,
                                                     const Map<Var, Range>& var_range,
                                                     Array<PrimExpr>* rewrite_indices) {
    Map<Var, arith::IntSet> var_dom = AsIntSet(var_range);
    Array<Array<PrimExpr>> new_shape;
    for (const PrimExpr& expr : indices) {
      IndexPatternFinder extractor(var_range, rewrite_indices);
      arith::IntSet intset = arith::EvalSet(expr, var_dom);
      extractor.mod_ = intset.max() + 1;
      extractor.div_ = 1;
      extractor.offset_ = 0;
      extractor(expr);
      Array<PrimExpr> access_shape = extractor.access_shape_;
      for (int i = static_cast<int>(access_shape.size()) - 1; i >= 1; i--) {
        if (!is_zero(floormod(extractor.offset_, access_shape[i]))) {
          return {};
        } else {
          extractor.offset_ = floordiv(extractor.offset_, access_shape[i]);
        }
      }
      access_shape.Set(0, extractor.offset_ + access_shape[0]);
      new_shape.push_back(access_shape);
    }
    return new_shape;
  }
  
  void VisitExpr_(const VarNode* op) final {
    arith::Analyzer analyzer;
    PrimExpr extent = var_range_[GetRef<Var>(op)]->extent;
    PrimExpr access_iter_range = min(mod_, (max(1, floordiv(extent, div_))));
    if (!analyzer.CanProveEqual(1, access_iter_range)) {
      access_shape_.push_back(access_iter_range);
      resulting_index_->push_back(floormod(floordiv(GetRef<Var>(op), div_), mod_));
    }
  }

  void VisitExpr_(const FloorDivNode* op) final {
    PrimExpr old_div = div_;
    div_ *= op->b;
    ExprVisitor::VisitExpr_(op);
    div_ = old_div;
  }

  void VisitExpr_(const FloorModNode* op) final {
    PrimExpr old_mod = mod_;
    mod_ = max(1, min(floordiv(op->b, div_), mod_));
    ExprVisitor::VisitExpr_(op);
    mod_ = old_mod;
  }

  void VisitExpr_(const MulNode* op) final {
    PrimExpr old_mod = mod_;
    PrimExpr old_div = div_;
    div_ = max(1, floordiv(div_, op->b));
    mod_ = max(1, floordiv(mod_, floordiv(op->b, floordiv(old_div, div_))));
    ExprVisitor::VisitExpr_(op);
    mod_ = old_mod;
    div_ = old_div;
  }

  void VisitExpr_(const AddNode* op) final {
    if (is_const_int(op->b)) {
      offset_ += floormod(floordiv(op->b, div_), mod_);
    }
    ExprVisitor::VisitExpr_(op);
  }

  PrimExpr div_;
  PrimExpr mod_;
  PrimExpr offset_;
  Map<Var, Range> var_range_;
  Array<PrimExpr> access_shape_;
  Array<PrimExpr>* resulting_index_;
};

/*!
 * \brief Utilities to perform rank promotion
 */
class RankPromoter : public StmtExprMutator {
 public:
  static Array<PrimExpr> FlattenNewShape(const Array<Array<PrimExpr>>& new_shape) {
    Array<PrimExpr> ret;
    ret.reserve(new_shape.size());
    for (int i = 0; i < static_cast<int>(new_shape.size()); i++) {
      PrimExpr prod = 1;
      for (int j = 0; j < static_cast<int>(new_shape[i].size()); j++) {
        prod *= new_shape[i][j];
      }
      ret.push_back(prod);
    }
    return ret;
  }

  static Array<PrimExpr> RewriteIndex(const Array<PrimExpr>& indices,
                                      const Array<Array<PrimExpr>>& new_shape) {
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

  static Array<PrimExpr> RewriteBackIndex(const Array<PrimExpr>& indices,
                                          const Array<Array<PrimExpr>>& new_shape) {
    Array<PrimExpr> new_indices;
    int offset = 0;
    for (int i = 0; i < static_cast<int>(new_shape.size()); i++) {
      PrimExpr index = 0;
      for (int j = 0; j < static_cast<int>(new_shape[i].size()); j++) {
        index *= new_shape[i][j];
        index += indices[offset + j];
      }
      new_indices.push_back(index);
      offset += new_shape[i].size();
    }
    return new_indices;
  }
  RankPromoter(const Buffer& src, const Buffer& dst, const Array<Array<PrimExpr>>& new_shape,
               const Array<Array<PrimExpr>>& relaxed_new_shape,
               const Array<Range>& relaxed_region)
      : src_(src),
        dst_(dst),
        new_shape_(new_shape),
        relaxed_new_shape_(relaxed_new_shape),
        relaxed_region_(relaxed_region) {}

  static Stmt RewriteBody(Stmt stmt, const Buffer& src, const Buffer& dst,
                          const Array<Array<PrimExpr>>& new_shape,
                          const Array<Array<PrimExpr>>& relaxed_new_shape,
                          const Array<Range>& relaxed_region) {
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

  Array<PrimExpr> ConvertIndices(const Array<PrimExpr>& indices) {
    Array<PrimExpr> rewrite_indices = RewriteIndex(indices, new_shape_);
    arith::Analyzer analyzer;
    for (int i = 0; i < static_cast<int>(rewrite_indices.size()); i++) {
      rewrite_indices.Set(i, analyzer.Simplify(rewrite_indices[i] - relaxed_region_[i]->min));
    }
    return RewriteBackIndex(rewrite_indices, relaxed_new_shape_);
  }

  const Buffer& src_;
  const Buffer& dst_;
  Array<Array<PrimExpr>> new_shape_;
  Array<Array<PrimExpr>> relaxed_new_shape_;
  Array<Range> relaxed_region_;
};

/*!
 * \brief Insert a cache stage to the compute location
 * \param stmt the stmt
 * \param is_write_cache whether to write a read cache or write cache
 * \param storage_scope the storage scope of the new cache 
 * \param compute_location the compute location. 
 * \param outer_loops the outer loops of this stmt
 * \param alloc_buffer the new cache block
 * \return a pair. The first is the stmt after transformation.
 *         The second is the SeqStmt that contains 2 stages (one original and another inserted).
 */
std::pair<Stmt, SeqStmt> InsertCacheStage(Stmt stmt, bool is_write_cache, String storage_scope,
                                          For compute_location, const Array<For>& outer_loops,
                                          Buffer* alloc_buffer) {
  Stmt body = stmt;
  std::vector<const ForNode*> loops;
  bool need_relax = !compute_location.defined();
  Map<Var, Range> relax_var_range;
  Map<Var, Range> all_var_range;
  PrimExpr vector_bytes = -1;
  // Step 1. Perform rank promotion on the buffer access, turning a strided-changing dimension into
  // several contiguous-changing dimensions
  // Step 1.1 collect loop var range for rank promotion
  while (const ForNode* loop = body.as<ForNode>()) {
    if (need_relax) {
      relax_var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    } else {
      loops.push_back(loop);
    }
    all_var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    if (loop == compute_location.get()) {
      need_relax = true;
    }
    if (loop->kind == ForKind::kVectorized) {
      vector_bytes = loop->extent;
    }
    body = loop->body;
  }
  for (const For& loop : outer_loops) {
    if (loop->kind == ForKind::kThreadBinding) {
      const String& thread_tag = loop->thread_binding.value()->thread_tag;
      if (CanRelaxStorageUnderThread(runtime::StorageScope::Create(storage_scope),
                                     runtime::ThreadScope::Create(thread_tag))) {
        relax_var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
      }
    }
    all_var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
  }

  const BufferStoreNode* buf_store = TVM_TYPE_AS(buf_store, body, BufferStoreNode);
  const BufferLoadNode* buf_load = TVM_TYPE_AS(buf_load, buf_store->value, BufferLoadNode);
  Buffer orig_buffer = is_write_cache ? buf_store->buffer : buf_load->buffer;
  Array<PrimExpr> indices = is_write_cache ? buf_store->indices : buf_load->indices;
  // Step 1.2 get the new shape and new access indices after rank promotion
  Array<PrimExpr> rewrite_indices;
  Array<Array<PrimExpr>> new_shape =
      IndexPatternFinder::getRankPromotedShape(indices, all_var_range, &rewrite_indices);
  // Step 2. relax the access region after rank promotion
  Region relaxed_region;
  auto relax_var_intset = AsIntSet(relax_var_range);
  arith::Analyzer analyzer;
  analyzer.Bind(all_var_range);
  for (const PrimExpr& index : rewrite_indices) {
    auto int_set = arith::EvalSet(index, relax_var_intset);
    relaxed_region.push_back(Range::FromMinExtent(
        int_set.min(), analyzer.Simplify(int_set.max() - int_set.min() + 1)));
  }
  // Step 3. generate the data copy bodies
  // preparation work
  Array<Var> new_loop_vars;
  Array<PrimExpr> orig_buf_indices, new_buf_indices;
  Array<Array<PrimExpr>> relaxed_new_shape;
  for (int i = 0; i < static_cast<int>(relaxed_region.size()); i++) {
    Var new_loop_var = Var("ax" + std::to_string(i));
    new_loop_vars.push_back(new_loop_var);
    orig_buf_indices.push_back(relaxed_region[i]->min + new_loop_var);
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
  // Step 3.1 create a buffer for the cache
  Buffer new_buffer = WithScope(orig_buffer, storage_scope);
  BufferNode* buffer_ptr = new_buffer.CopyOnWrite();
  buffer_ptr->shape = RankPromoter::FlattenNewShape(relaxed_new_shape);
  *alloc_buffer = new_buffer;
  Array<PrimExpr> real_orig_buf_indices =
      RankPromoter::RewriteBackIndex(orig_buf_indices, new_shape);
  Array<PrimExpr> real_new_buf_indices =
      RankPromoter::RewriteBackIndex(new_buf_indices, relaxed_new_shape);
  // Step 3.2 generate a body that writes to the cache
  Stmt generate_body =
      is_write_cache ? BufferStore(orig_buffer, BufferLoad(new_buffer, real_new_buf_indices),
                                   real_orig_buf_indices)
                     : BufferStore(new_buffer, BufferLoad(orig_buffer, real_orig_buf_indices),
                                   real_new_buf_indices);
  for (int i = static_cast<int>(relaxed_region.size()) - 1; i >= 0; i--) {
    if (i == static_cast<int>(relaxed_region.size()) - 1 && !is_const_int(vector_bytes, -1)) {
      ICHECK(analyzer.CanProve(vector_bytes == relaxed_region[i]->extent));
      generate_body = For(new_loop_vars[i], 0, relaxed_region[i]->extent, ForKind::kVectorized,
                          generate_body);
    } else {
      generate_body =
          For(new_loop_vars[i], 0, relaxed_region[i]->extent, ForKind::kSerial, generate_body);
    }
  }
  // Step 3.3 rewrite the original body to load from cache
  Stmt rewrite_body;
  if (compute_location.defined()) {
    rewrite_body = compute_location->body;
  } else {
    rewrite_body = stmt;
  }
  rewrite_body = RankPromoter::RewriteBody(rewrite_body, orig_buffer, new_buffer, new_shape,
                                           relaxed_new_shape, relaxed_region);
  SeqStmt insert_location;
  if (is_write_cache) {
    generate_body = insert_location = SeqStmt({rewrite_body, generate_body});
  } else {
    generate_body = insert_location = SeqStmt({generate_body, rewrite_body});
  }
  for (int i = static_cast<int>(loops.size()) - 1; i >= 0; i--) {
    generate_body = For(loops[i]->loop_var, loops[i]->min, loops[i]->extent, loops[i]->kind,
                        generate_body, loops[i]->thread_binding, loops[i]->annotations);
  }
  return std::make_pair(generate_body, insert_location);
}

Stmt CreateLocalStage::Rewrite(const Stmt& stmt, const Map<String, ObjectRef>& constraints, Map<String, ObjectRef>*
                                                                                                output) const {
  Stmt body;
  For compute_location;
  std::tie(body, compute_location) = LiftThreadBindingLoops(std::move(stmt));
  Array<For> outer_loops = Downcast<Array<For>>(constraints["outer_loops"]);
  Buffer cache_buffer;
  Stmt after_caching =  InsertCacheStage(body, false, "local", compute_location, outer_loops,
                                  &cache_buffer)
      .first;
  output->Set("alloc_buffer", cache_buffer);
  return after_caching;
}

}// namespace tir
}// namespace tvm

