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

#include <string>

#include "../utils.h"

namespace tvm {
namespace tir {

using support::NDIntSet;

bool HasBuffer(const Array<BufferRegion>& buffer_regions, const Buffer& buffer) {
 for (const BufferRegion& buffer_region : buffer_regions) {
   if (buffer_region->buffer.same_as(buffer)) {
     return true;
   }
 }
 return false;
}

void RelaxBufferRegions(const Array<BufferRegion>& buffer_regions,
                       const Buffer& buffer,                    //
                       const Map<Var, Range>& var_dom,  //
                       const Map<Var, Range>& full_var_dom,
                       const Map<Var, PrimExpr>& bindings,      //
                       std::vector<NDIntSet>* relaxed_regions) {
 for (const BufferRegion& buffer_region : buffer_regions) {
   if (buffer_region->buffer.same_as(buffer)) {
     LOG(INFO)<<var_dom;
     LOG(INFO)<<full_var_dom;
     LOG(INFO)<<Substitute(buffer_region->region, bindings);
     arith::Analyzer analyzer;
     analyzer.Bind(full_var_dom);
     Array<Range> simplified_region;
     for (const Range& range : Substitute(buffer_region->region, bindings)) {
       simplified_region.push_back(Range::FromMinExtent(analyzer.Simplify(range->min),
                                                        analyzer.Simplify(range->extent)));
     }
     LOG(INFO)<<simplified_region;
     Array<arith::IntSet> relaxed_region =
         arith::EvalSet(simplified_region, AsIntSet(var_dom));
     LOG(INFO)<<relaxed_region;
     relaxed_regions->push_back({relaxed_region.begin(), relaxed_region.end()});
   }
 }
}

class ScopeReplacer : public StmtMutator {
public:
 static Block Replace(const BlockNode* scope_block, const Buffer& dst, const ForNode* old_loop,
                      const ForNode* new_loop) {
   ObjectPtr<BlockNode> new_scope_block = make_object<BlockNode>(*scope_block);
   new_scope_block->body = ScopeReplacer(old_loop, new_loop)(std::move(new_scope_block->body));
   new_scope_block->alloc_buffers.push_back(dst);
   return Block(new_scope_block);
 }

private:
 explicit ScopeReplacer(const ForNode* old_loop, const ForNode* new_loop)
     : old_loop_(old_loop), new_loop_(new_loop), found_(false) {}

 Stmt VisitStmt(const Stmt& stmt) final { return found_ ? stmt : StmtMutator::VisitStmt(stmt); }
 Stmt VisitStmt_(const BlockNode* block) final { return GetRef<Block>(block); }
 Stmt VisitStmt_(const ForNode* loop) final {
   if (loop == old_loop_) {
     found_ = true;
     return GetRef<For>(new_loop_);
   }
   return StmtMutator::VisitStmt_(loop);
 }

 const ForNode* old_loop_;
 const ForNode* new_loop_;
 bool found_;
};

Array<PrimExpr> RewriteBackIndex(const Array<PrimExpr>& indices, const Array<Array<PrimExpr>>&
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


Array<PrimExpr> RewriteIndex(const Array<PrimExpr>& indices, const Array<Array<PrimExpr>>&
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

Region RewriteRegion(const Region& region, const Array<Array<PrimExpr>>& new_shape){
  Array<PrimExpr> indices;
  arith::Analyzer analyzer;
  for (int i = 0; i < static_cast<int>(region.size()); i++) {
    indices.push_back(region[i]->min);
  }
  Array<PrimExpr> rewrite_indices = RewriteIndex(indices, new_shape);
  int offset = 0;
  Array<PrimExpr> rewrite_extent(rewrite_indices.size(), 1);
  bool rank_promotion_success_=true;
  for (int i = 0; i < static_cast<int>(region.size()); i++) {
    PrimExpr extent_remain = region[i]->extent;
    for (int j = static_cast<int>(new_shape[i].size()) - 1; j >= 0; j--) {
      if (analyzer.CanProve(extent_remain > 1)) {
        if (!analyzer.CanProve(floormod(extent_remain, new_shape[i][j]) == 0) ||
            !analyzer.CanProve(rewrite_indices[offset+j] == 0)) {
          rank_promotion_success_ = false;
          break;
        }
        rewrite_extent.Set(offset+j, new_shape[i][j]);
        extent_remain = floordiv(extent_remain, new_shape[i][j]);
      } else if (analyzer.CanProve(extent_remain == 1)) {
        break;
      } else {
        rank_promotion_success_ = false;
      }
    }
    offset+=new_shape[i].size();
  }
  if (!rank_promotion_success_) {
    LOG(INFO)<<"fail";
    return {};
  }
  Region new_region;
  for (int i = 0; i < static_cast<int>(rewrite_indices.size()); i++) {
    new_region.push_back(Range::FromMinExtent(rewrite_indices[i], rewrite_extent[i]));
  }
  return new_region;
}

Array<BufferRegion> RewriterBufferRegions(Array<BufferRegion> buffer_regions, Buffer src,
                                          Buffer dst,
                                          const Array<Array<PrimExpr>>& new_shape){
  Array<BufferRegion> ret;
  for (int i = 0; i < static_cast<int>(buffer_regions.size()); i++) {
    if (buffer_regions[i]->buffer.same_as(src)) {
      ret.push_back(BufferRegion(dst, RewriteRegion(buffer_regions[i]->region, new_shape)));
    } else {
      ret.push_back(buffer_regions[i]);
    }
  }
  return ret;
}

class BufferReplacer : public StmtExprMutator {
public:
 explicit BufferReplacer(const Buffer& src, const Buffer& dst, Map<Block, Block>* block_sref_reuse,
                         const Array<Array<PrimExpr>> new_shape)
     : src_(src), dst_(dst), block_sref_reuse_(block_sref_reuse), new_shape_(new_shape) {}

private:
 Stmt VisitStmt_(const BufferStoreNode* _store) final {
   BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_store));
   if (store->buffer.same_as(src_)) {
     ObjectPtr<BufferStoreNode> new_store = make_object<BufferStoreNode>(*store.get());
     new_store->buffer = dst_;
     if (!new_shape_.empty()) {
       new_store->indices = RewriteIndex(new_store->indices, new_shape_);
     }
     return BufferStore(new_store);
   }
   return store;
 }

 PrimExpr VisitExpr_(const BufferLoadNode* _load) final {
   BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_load));
   if (load->buffer.same_as(src_)) {
     ObjectPtr<BufferLoadNode> new_load = make_object<BufferLoadNode>(*load.get());
     new_load->buffer = dst_;
     if (!new_shape_.empty()) {
       new_load->indices = RewriteIndex(new_load->indices, new_shape_);
     }
     return BufferLoad(new_load);
   }
   return load;
 }

 Stmt VisitStmt_(const BlockNode* _block) final {
   Block old_block = GetRef<Block>(_block);
   Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(_block));
   ObjectPtr<BlockNode> new_block = make_object<BlockNode>(*block.get());
   if (!new_shape_.empty()) {
     new_block->reads = RewriterBufferRegions(new_block->reads, src_, dst_, new_shape_);
     new_block->writes = RewriterBufferRegions(new_block->writes, src_, dst_, new_shape_);
   } else {
     new_block->reads = ReplaceBuffer(new_block->reads, src_, dst_);
     new_block->writes = ReplaceBuffer(new_block->writes, src_, dst_);
   }

   block_sref_reuse_->Set(old_block, Block(new_block));
   return Block(new_block);
 }

 const Buffer& src_;
 const Buffer& dst_;
 Map<Block, Block>* block_sref_reuse_;
 Array<Array<PrimExpr>> new_shape_;
};


class PatternExtractor: public ExprVisitor{
 public:
  PatternExtractor(const Map<Var, Range>& var_range):var_range_(var_range){}
  
  static Array<Array<PrimExpr>> getRankPromotedShape(Array<PrimExpr> indices, const Map<Var, Range>&
                                                                                  var_range){
    Map<Var, arith::IntSet> var_dom = AsIntSet(var_range);
    Array<Array<PrimExpr>> new_shape;
    for (const PrimExpr& expr : indices) {
      PatternExtractor extractor(var_range);
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
};

struct ReadWriteAtImpl {
 template <bool is_read>
 static StmtSRef Main(ScheduleState self, const StmtSRef& loop_sref, const StmtSRef& block_sref,
                      int buffer_index, const String& storage_scope,
                      Map<String, ObjectRef> annotations) {
   const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
   Buffer src =
       GetNthAccessBuffer(self, GetRef<Block>(block), buffer_index, /*is_write=*/!is_read);
   Array<StmtSRef> loop_srefs = GetLoops(block_sref);
   Map<Var, Range> var_range;
   for (const StmtSRef& loop_sref : loop_srefs) {
     const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
     var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
   }
   Array<Range> region = is_read?block->reads[buffer_index]->region:block->writes[buffer_index]->region;
   Map<Var, PrimExpr> binding;
   const tir::BlockRealize& realize = tir::GetBlockRealize(self, block_sref);
   for (int i = 0; i < static_cast<int>(realize->iter_values.size()); i++) {
     binding.Set(block->iter_vars[i]->var,realize->iter_values[i]);
   }
   Array<PrimExpr> indices;
   for (const Range& range : region) {
     indices.push_back(Substitute(range->min, binding));
   }
   Array<Array<PrimExpr>> new_shape = PatternExtractor::getRankPromotedShape(indices, var_range);
   LOG(INFO)<<"new_shape: "<<new_shape;
   Buffer dst_buffer = WithScope(src, storage_scope);
   auto dst =  dst_buffer.CopyOnWrite();
   dst->shape.clear();
   for (int i = 0; i < static_cast<int>(new_shape.size()); i++) {
     for (const PrimExpr& extent : new_shape[i]) {
       dst->shape.push_back(extent);
     }
   }
   ReadWriteAtImpl impl(self, loop_sref, src, dst_buffer, annotations, new_shape);
   std::pair<For, BlockRealize> new_loop_block;
   new_loop_block =
       impl.MakeLoopAndBlock<is_read>(src->name + "_" + storage_scope);
   if (!impl.rank_promotion_success_) {
     dst_buffer = WithScope(src, storage_scope);
     ReadWriteAtImpl impl(self, loop_sref, src, dst_buffer, annotations);
     new_loop_block =
         impl.MakeLoopAndBlock<is_read>(src->name + "_" + storage_scope);
   }
//   dst_buffer = WithScope(src, storage_scope);
//   ReadWriteAtImpl impl(self, loop_sref, src, dst_buffer, annotations);
//   std::pair<For, BlockRealize> new_loop_block =
//       impl.MakeLoopAndBlock<is_read>(src->name + "_" + storage_scope);
   StmtSRef result_block_sref =
       impl.ReplaceScopeBlock(new_loop_block.first.get(), new_loop_block.second->block.get());
   impl.UpdateBlockInfo(result_block_sref);
   return result_block_sref;
 }

private:
 Array<BufferRegion> RewriteBufferRegions_(Array<BufferRegion> buffer_regions, Buffer src,
                                           Buffer dst,
                                           const Array<Array<PrimExpr>>& new_shape){
   Array<BufferRegion> ret = RewriterBufferRegions(buffer_regions, src, dst, new_shape);
   rank_promotion_success_=!ret.empty();
   return ret;
   
 }

 
 StmtSRef ReplaceScopeBlock(const ForNode* new_loop, const BlockNode* new_block) {
   StmtSRef scope_root_sref = GetScopeRoot(self_, loop_sref_,
                                           /*require_stage_pipeline=*/false,
                                           /*require_subtree_compact_dataflow=*/false);
   const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_block, scope_root_sref);
   Block new_scope_block = ScopeReplacer::Replace(scope_block, dst_, loop_, new_loop);
   block_sref_reuse_.Set(GetRef<Block>(scope_block), new_scope_block);
   self_->Replace(scope_root_sref, new_scope_block, block_sref_reuse_);
   return self_->stmt2ref.at(new_block);
 }

 void UpdateBlockInfo(const StmtSRef& new_block_sref) {
   BlockInfo& block_info = self_->block_info[new_block_sref];
   block_info.affine_binding = false;
   block_info.region_cover = true;
   block_info.scope->stage_pipeline = true;
 }
 


 template <bool is_read>
 std::pair<For, BlockRealize> MakeLoopAndBlock(const String& new_block_name_hint) {
   Array<Stmt> subtrees = AsArray(loop_->body);
   int n_subtrees = subtrees.size();
   LOG(INFO)<<n_subtrees;
   runtime::StorageScope scope = runtime::StorageScope::Create(dst_.scope());
   std::vector<NDIntSet> relaxed_regions;
   std::vector<int> r_pos;
   std::vector<int> w_pos;
   relaxed_regions.reserve(n_subtrees);
   r_pos.reserve(n_subtrees);
   w_pos.reserve(n_subtrees);
   // Step 1. Iterate over all subtrees
   for (int i = 0; i < n_subtrees; ++i) {
     bool r_visited = false;
     bool w_visited = false;
     auto f_visit = [this, &relaxed_regions, &r_visited, &w_visited,
                     &scope](const ObjectRef& obj) -> bool {
       const BlockRealizeNode* realize = obj.as<BlockRealizeNode>();
       if (realize == nullptr) {
         return true;
       }
       const BlockNode* block = realize->block.get();
       bool has_r = HasBuffer(block->reads, src_);
       bool has_w = HasBuffer(block->writes, src_);
       r_visited = r_visited || has_r;
       w_visited = w_visited || has_w;
       if (is_read ? has_r : has_w) {
         if(!new_shape_.empty()) {
           LOG(INFO)<<RewriteBufferRegions_(block->reads, src_, dst_, new_shape_);
           RelaxBufferRegions(
               /*buffer_regions=*/
                   is_read ? RewriteBufferRegions_(block->reads, src_, dst_, new_shape_)
                           :  RewriteBufferRegions_(block->writes, src_, dst_, new_shape_),
               /*buffer=*/dst_,
               /*var_dom=*/
               LoopDomainOfSRefTreePath(
                   /*low_inclusive=*/GetRef<StmtSRef>(self_->stmt2ref.at(block)->parent),
                   /*high_exclusive=*/loop_sref_,
                   /*extra_relax_scope=*/scope),
               /*full_var_dom=*/
               LoopDomainOfSRefTreePath(
                   /*low_inclusive=*/GetRef<StmtSRef>(self_->stmt2ref.at(block)->parent),
                   /*high_exclusive=*/GetScopeRoot(self_,self_->stmt2ref.at(block), false, false)),
               /*bindings=*/GetBindings(GetRef<BlockRealize>(realize)),
               /*relaxed_regions=*/&relaxed_regions);
         } else {
           RelaxBufferRegions(
               /*buffer_regions=*/is_read ? block->reads : block->writes,
               /*buffer=*/src_,
               /*var_dom=*/
               LoopDomainOfSRefTreePath(
                   /*low_inclusive=*/GetRef<StmtSRef>(self_->stmt2ref.at(block)->parent),
                   /*high_exclusive=*/loop_sref_,
                   /*extra_relax_scope=*/scope),
               /*full_var_dom=*/
               LoopDomainOfSRefTreePath(
                   /*low_inclusive=*/GetRef<StmtSRef>(self_->stmt2ref.at(block)->parent),
                   /*high_exclusive=*/GetScopeRoot(self_,self_->stmt2ref.at(block), false, false)),
               /*bindings=*/GetBindings(GetRef<BlockRealize>(realize)),
               /*relaxed_regions=*/&relaxed_regions);
         }
       }
       return false;
     };
     PreOrderVisit(subtrees[i], f_visit);
     if (r_visited) {
       r_pos.push_back(i);
     }
     if (w_visited) {
       w_pos.push_back(i);
     }
   }
   if (!rank_promotion_success_) {
     return std::make_pair(For(),BlockRealize());
   }
   // Step 2. Calculate `insert_pos` and [st, ed) for buffer replacement
   int insert_pos = -1, st = -1, ed = -1;
   if (is_read) {
     ICHECK(!r_pos.empty());
     // No write after the first read
     ICHECK(w_pos.empty() || w_pos.back() < r_pos.front());
     // Can be inserted at [0, r_pos.front()], i.e. before the first read
     insert_pos = r_pos.front();
     // Buffer reads in [insert_pos, +oo) is rewritten
     st = insert_pos;
     ed = n_subtrees;
   } else {
     ICHECK(!w_pos.empty());
     // No read after the last write
     ICHECK(r_pos.empty() || r_pos.back() <= w_pos.back());
     // Can be inserted into (w_pos.back(), +oo), i.e. after the last write
     insert_pos = w_pos.back() + 1;
     st = 0;
     ed = insert_pos;
   }
   // Step 3. Calculate `domain`, the domain of buffer access
   NDIntSet relaxed = support::NDIntSetUnion(relaxed_regions);
   int ndim = relaxed.size();
   Array<Range> domain;
   domain.reserve(ndim);
   for (int i = 0; i < ndim; ++i) {
     const arith::IntSet& int_set = relaxed[i];
     PrimExpr min = analyzer_->Simplify(int_set.min());
     PrimExpr extent = analyzer_->Simplify(int_set.max() + 1 - min);
     domain.push_back(Range::FromMinExtent(min, extent));
   }
   // Step 4. Insert the auto copy block and replace buffers
   BufferReplacer replacer(src_, dst_, &block_sref_reuse_, new_shape_);
   for (int i = st; i < ed; ++i) {
     Stmt stmt = subtrees[i];
     subtrees.Set(i, Stmt(nullptr));
     subtrees.Set(i, replacer(std::move(stmt)));
   }
   BlockRealize realize = is_read ? MakeBlock(src_, dst_, domain, new_block_name_hint, is_read)
                                  : MakeBlock(dst_, src_, domain, new_block_name_hint, is_read);
   subtrees.insert(subtrees.begin() + insert_pos, realize);
   ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loop_);
   new_loop->body = SeqStmt(std::move(subtrees));
   return {For(new_loop), realize};
 }

 BlockRealize MakeBlock(const Buffer& copy_from, const Buffer& copy_to, const Array<Range>& domain,
                        const String& name_hint, bool is_read) const {
   int n = domain.size();
   std::vector<Var> loop_vars;
   loop_vars.reserve(n);
   for (int i = 0; i < n; ++i) {
     loop_vars.push_back(Var("ax" + std::to_string(i)));
   }
   Array<PrimExpr> indices;
   indices.reserve(n);
   for (int i = 0; i < n; ++i) {
     indices.push_back(domain[i]->min + loop_vars[i]);
   }
   Stmt stmt;
   Array<PrimExpr> rewrite_back_index;
   LOG(INFO)<<indices;
   if (!new_shape_.empty()) {
     rewrite_back_index = RewriteBackIndex(indices, new_shape_);
     LOG(INFO)<<rewrite_back_index;
     if (is_read) {
       stmt = BufferStore(copy_to, /*value=*/BufferLoad(copy_from, rewrite_back_index),
                          /*indices=*/indices);
     } else {
       stmt = BufferStore(copy_to, /*value=*/BufferLoad(copy_from, indices),
                          /*indices=*/rewrite_back_index);
     }
   } else{
     stmt = BufferStore(copy_to, /*value=*/BufferLoad(copy_from, indices), /*indices=*/indices);
   }
   Map<Var, Range> var_range;
   for (int i = n - 1; i >= 0; --i) {
     stmt = For(loop_vars[i], Integer(0), domain[i]->extent, ForKind::kSerial, stmt);
     var_range.Set(loop_vars[i], Range::FromMinExtent(0, domain[i]->extent));
   }
   Array<BufferRegion> reads, writes;
   if (new_shape_.empty()) {
     reads = {BufferRegion(copy_from, domain)};
     writes = {BufferRegion(copy_to, domain)};
   } else {
     Array<Range> rewrite_back_domain;
     for (const PrimExpr& index : rewrite_back_index) {
       arith::IntSet intset = arith::EvalSet(index, AsIntSet(var_range));
       rewrite_back_domain.push_back(Range::FromMinExtent(intset.min(), analyzer_->Simplify
                                                                        (intset.max()-intset.min()+1)));
     }
     if (is_read) {
       reads={BufferRegion(copy_from, rewrite_back_domain)};
       writes = {BufferRegion(copy_to, domain)};
     } else {
       reads = {BufferRegion(copy_from, domain)};
       writes = {BufferRegion(copy_to, rewrite_back_domain)};
     }
   }
   return BlockRealize(
       /*values=*/{},
       /*predicate=*/const_true(),
       Block(/*iter_vars*/ {},
             /*reads=*/reads,
             /*writes=*/writes,
             /*name_hint=*/name_hint,  //
             /*body=*/std::move(stmt),
             /*init=*/NullOpt,
             /*alloc_buffers=*/{},
             /*match_buffers=*/{},
             /*annotations=*/annotations_));
 }

 explicit ReadWriteAtImpl(ScheduleState self, const StmtSRef& loop_sref, const Buffer& src,
                          const Buffer& dst, Map<String, ObjectRef> annotations, const
                                                                                     Array<Array<PrimExpr>>& new_shape)
     : self_(self),
       loop_sref_(loop_sref),
       loop_(nullptr),
       src_(src),
       dst_(dst),
       annotations_(annotations),
       block_sref_reuse_(),
       analyzer_(std::make_unique<arith::Analyzer>()),
        new_shape_(new_shape){
   loop_ = TVM_SREF_TO_FOR(loop_, loop_sref);
 }
 
 explicit ReadWriteAtImpl(ScheduleState self, const StmtSRef& loop_sref, const Buffer& src,
                         const Buffer& dst, Map<String, ObjectRef> annotations)
    : self_(self),
      loop_sref_(loop_sref),
      loop_(nullptr),
      src_(src),
      dst_(dst),
      annotations_(annotations),
      block_sref_reuse_(),
      analyzer_(std::make_unique<arith::Analyzer>()){
  loop_ = TVM_SREF_TO_FOR(loop_, loop_sref);
}

 ScheduleState self_;
 const StmtSRef& loop_sref_;
 const ForNode* loop_;
 const Buffer& src_;
 const Buffer& dst_;
 Map<String, ObjectRef> annotations_;
 Map<Block, Block> block_sref_reuse_;
 std::unique_ptr<arith::Analyzer> analyzer_;
 Array<Array<PrimExpr>> new_shape_;
 bool rank_promotion_success_ = true;
};

StmtSRef ReadAt(ScheduleState self, const StmtSRef& loop_sref, const StmtSRef& block_sref,
               int read_buffer_index, const String& storage_scope) {
 return ReadWriteAtImpl::Main<true>(self, loop_sref, block_sref, read_buffer_index, storage_scope,
                                    {{"auto_copy", Integer(1)}});
}

StmtSRef WriteAt(ScheduleState self, const StmtSRef& loop_sref, const StmtSRef& block_sref,
                int write_buffer_index, const String& storage_scope) {
 return ReadWriteAtImpl::Main<false>(self, loop_sref, block_sref, write_buffer_index,
                                     storage_scope, {{"auto_copy", Integer(1)}});
}

/******** Instruction Registration ********/

struct ReadAtTraits : public UnpackedInstTraits<ReadAtTraits> {
 static constexpr const char* kName = "ReadAt";
 static constexpr bool kIsPure = false;

private:
 static constexpr size_t kNumInputs = 2;
 static constexpr size_t kNumAttrs = 2;
 static constexpr size_t kNumDecisions = 0;

 StmtSRef ReadAt(ScheduleState self, const StmtSRef& loop_sref, const StmtSRef& block_sref,
                 int buffer_index, const String& storage_scope);
 static BlockRV UnpackedApplyToSchedule(Schedule sch, LoopRV loop, BlockRV block,
                                        Integer read_buffer_index, String storage_scope) {
   return sch->ReadAt(loop, block, read_buffer_index->value, storage_scope);
 }

 static String UnpackedAsPython(Array<String> outputs, String loop, String block,
                                Integer read_buffer_index, String storage_scope) {
   PythonAPICall py("read_at");
   py.Input("loop", loop);
   py.Input("block", block);
   py.Input("read_buffer_index", read_buffer_index->value);
   py.Input("storage_scope", storage_scope);
   py.SingleOutput(outputs);
   return py.Str();
 }

 template <typename>
 friend struct ::tvm::tir::UnpackedInstTraits;
};

struct WriteAtTraits : public UnpackedInstTraits<WriteAtTraits> {
 static constexpr const char* kName = "WriteAt";
 static constexpr bool kIsPure = false;

private:
 static constexpr size_t kNumInputs = 2;
 static constexpr size_t kNumAttrs = 2;
 static constexpr size_t kNumDecisions = 0;

 static BlockRV UnpackedApplyToSchedule(Schedule sch, LoopRV loop, BlockRV block,
                                        Integer write_buffer_index, String storage_scope) {
   return sch->WriteAt(loop, block, write_buffer_index->value, storage_scope);
 }

 static String UnpackedAsPython(Array<String> outputs, String loop, String block,
                                Integer write_buffer_index, String storage_scope) {
   PythonAPICall py("write_at");
   py.Input("loop", loop);
   py.Input("block", block);
   py.Input("write_buffer_index", write_buffer_index->value);
   py.Input("storage_scope", storage_scope);
   py.SingleOutput(outputs);
   return py.Str();
 }

 template <typename>
 friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(ReadAtTraits);
TVM_REGISTER_INST_KIND_TRAITS(WriteAtTraits);

}  // namespace tir
}  // namespace tvm
