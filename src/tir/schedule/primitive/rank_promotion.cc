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
#include "../utils.h"

namespace tvm {
namespace tir {

class AccessPatternExtractor: public ExprVisitor{
 public:
  AccessPatternExtractor(const Map<Var, Range>& var_range):var_range_(var_range){}
  
  static Array<Array<PrimExpr>> getRankPromotedShape(Array<PrimExpr> indices, const Map<Var, Range>&
      var_range){
    Map<Var, arith::IntSet> var_dom = AsIntSet(var_range);
    Array<Array<PrimExpr>> new_shape;
    for (const PrimExpr& expr : indices) {
      AccessPatternExtractor extractor(var_range);
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
    LOG(INFO)<<"in floordiv mod:"<<mod<<" div:"<<div;
    ExprVisitor::VisitExpr_(op);
    div = old_div;
  }
  
  void VisitExpr_(const FloorModNode* op) final{
    PrimExpr old_mod = mod;
    mod=max(1, min(floordiv(op->b,div), mod));
    LOG(INFO)<<"in floormod mod:"<<mod<<" div:"<<div;
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

/*!
 * \brief Find the defining site of the buffer in the given block and its ancestors
 * \param block_sref The block sref
 * \param buffer The buffer
 * \return The defining site of the buffer and whether the buffer is allocated (otherwise the
 *         buffer is from match_buffer).
 */
std::pair<StmtSRef, bool> GetBufferAllocSite(const StmtSRef& block_sref, const Buffer& buffer) {
  // Climb up along the sref tree, and find the block where `buffer` is in alloc_buffers or
  // match_buffers.
  const StmtSRefNode* defining_site_sref = block_sref.get();
  while (defining_site_sref != nullptr) {
    const auto* block = defining_site_sref->StmtAs<BlockNode>();
    // If this sref is not a block sref, skip it.
    if (block == nullptr) {
      defining_site_sref = defining_site_sref->parent;
      continue;
    }
    // Try to find the buffer in `allloc_buffers`
    for (const Buffer& alloc_buffer : block->alloc_buffers) {
      if (buffer.same_as(alloc_buffer)) {
        return {GetRef<StmtSRef>(defining_site_sref), true};
      }
    }
    // We do not allow the buffer being defined in `match_buffer`.
    for (const MatchBufferRegion match_buffer : block->match_buffers) {
      if (buffer.same_as(match_buffer)) {
        return {GetRef<StmtSRef>(defining_site_sref), false};
      }
    }
    defining_site_sref = defining_site_sref->parent;
  }
  // If we cannot find the defining site block, it means that the buffer must be in the function's
  // buffer_map, which isn't an intermediate buffer. In this case we should report error.
  LOG(FATAL)
      << "ValueError: The buffer is expected to be an intermediate buffer defined in some block";
  throw;
}

/*!
 * \brief A helper mutator which recursively mutates the old buffer and read/write index
 *         and collects the block sref reuse information for the following replacement.
 */
class RankPromotionBufferRewriter : StmtExprMutator {
 public:
  /*!
   * \param allocate_site The block where `old_buffer` was allocated.
   * \param old_buffer The old buffer
   * \param new_shape the shape of the new buffer
   * \param block_sref_reuse The block sref reuse map to be updated
   * \return The new block after the mutation
   */
  static Optional<Block> Mutate(const Block& allocate_site, const Buffer& old_buffer,
                      const Array<Array<PrimExpr>>& new_shape, Map<Block, Block>*
                          block_sref_reuse) {
    auto new_buffer = make_object<BufferNode>(*old_buffer.get());
    new_buffer->shape.clear();
    for (int i = 0; i < static_cast<int>(new_shape.size()); i++) {
      for (const PrimExpr& extent : new_shape[i]) {
        new_buffer->shape.push_back(extent);
      }
    }
    RankPromotionBufferRewriter mutator(old_buffer, Buffer(new_buffer), block_sref_reuse, new_shape);
    Stmt new_block = mutator.VisitStmt(allocate_site);
    if (mutator.success_) {
      return Downcast<Block>(new_block);
    } else {
      return NullOpt;
    }
  }
  
 private:
  RankPromotionBufferRewriter(const Buffer& old_buffer, Buffer new_buffer,
                      Map<Block, Block>* block_sref_reuse, const Array<Array<PrimExpr>>& new_shape)
      : block_sref_reuse_(block_sref_reuse), new_shape_(new_shape) {
    buffer_map_[old_buffer.get()] = std::move(new_buffer);
  }

  Array<PrimExpr> RewriteIndex(const Array<PrimExpr>& indices, const Array<Array<PrimExpr>>&
      new_shape,
                               const Array<PrimExpr>& old_shape){
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
  
  Region RewriteRegion(const Region& region, const Array<Array<PrimExpr>>& new_shape,
                               const Array<PrimExpr>& old_shape){
    Array<PrimExpr> indices;
    arith::Analyzer analyzer;
    for (int i = 0; i < static_cast<int>(region.size()); i++) {
      indices.push_back(region[i]->min);
    }
    Array<PrimExpr> rewrite_indices = RewriteIndex(indices, new_shape, old_shape);
    int offset = 0;
    Array<PrimExpr> rewrite_extent(rewrite_indices.size(), 1);
    for (int i = 0; i < static_cast<int>(region.size()); i++) {
      PrimExpr extent_remain = region[i]->extent;
      for (int j = static_cast<int>(new_shape[i].size()) - 1; j >= 0; j--) {
        if (analyzer.CanProve(extent_remain > 1)) {
          if (!analyzer.CanProve(floormod(extent_remain, new_shape[i][j]) == 0) ||
              !analyzer.CanProve(rewrite_indices[offset+j] == 0)) {
            success_ = false;
            break;
          }
          rewrite_extent.Set(offset+j, new_shape[i][j]);
          extent_remain = floordiv(extent_remain, new_shape[i][j]);
        } else if (analyzer.CanProve(extent_remain == 1)) {
          break;
        } else {
          success_ = false;
        }
      }
      offset+=new_shape[i].size();
    }
    if (!success_) {
      return {};
    }
    Region new_region;
    for (int i = 0; i < static_cast<int>(rewrite_indices.size()); i++) {
      new_region.push_back(Range::FromMinExtent(rewrite_indices[i], rewrite_extent[i]));
    }
    return new_region;
  }
  
  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr res = ExprMutator::VisitExpr_(op);
    op = res.as<BufferLoadNode>();
    ICHECK(op);
    auto it = buffer_map_.find(op->buffer.get());
    if (it != buffer_map_.end()) {
      ObjectPtr<BufferLoadNode> ptr = make_object<BufferLoadNode>(*op);
      ptr->buffer = it->second;
      ptr->indices = RewriteIndex(ptr->indices, new_shape_, it->first->shape);
      return PrimExpr(ptr);
    } else {
      return res;
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt res = StmtMutator::VisitStmt_(op);
    auto it = buffer_map_.find(op->buffer.get());
    if (it != buffer_map_.end()) {
      ObjectPtr<BufferStoreNode> ptr = CopyOnWrite(res.as<BufferStoreNode>());
      ptr->buffer = it->second;
      ptr->indices = RewriteIndex(ptr->indices, new_shape_, it->first->shape);
      return Stmt(ptr);
    } else {
      return res;
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    // To reduce the number of blocks in block sref reuse map, we check whether the block is really
    // mutated (i.e., the old buffer appears in the block). If so, we return the block after
    // mutation. Otherwise we just return the original block.
    bool changed = false;
    // Step 1. Mutate the read region.
    Array<BufferRegion> reads;
    for (const BufferRegion& read : op->reads) {
      auto it = buffer_map_.find(read->buffer.get());
      if (it != buffer_map_.end()) {
        changed = true;
        Region new_region = RewriteRegion(read->region, new_shape_, it->first->shape);
        reads.push_back(BufferRegion(it->second, new_region));
      } else {
        reads.push_back(read);
      }
    }
    // Step 2. Mutate the write region.
    Array<BufferRegion> writes;
    for (const BufferRegion& write : op->writes) {
      auto it = buffer_map_.find(write->buffer.get());
      if (it != buffer_map_.end()) {
        changed = true;
        Region new_region = RewriteRegion(write->region, new_shape_, it->first->shape);
        writes.push_back(BufferRegion(it->second, new_region));
      } else {
        writes.push_back(write);
      }
    }
    // Step 3. Mutate `alloc_buffers` for the old buffer allocated in this block.
    Array<Buffer> alloc_buffers;
    for (const Buffer& buffer : op->alloc_buffers) {
      auto it = buffer_map_.find(buffer.get());
      if (it != buffer_map_.end()) {
        changed = true;
        alloc_buffers.push_back(it->second);
      } else {
        alloc_buffers.push_back(buffer);
      }
    }
    // Step 4. Mutate `match_buffers`.
    Array<MatchBufferRegion> match_buffers;
    for (const MatchBufferRegion& match_buffer : op->match_buffers) {
      auto it = buffer_map_.find(match_buffer->source->buffer.get());
      if (it != buffer_map_.end()) {
        changed = true;
        Region src_region = match_buffer->source->region;
        Region rewrite_src_region = RewriteRegion(src_region, new_shape_, it->first->shape);
        
        ObjectPtr<BufferNode> new_target_buffer =
            make_object<BufferNode>(*match_buffer->buffer.get());
        Array<PrimExpr> rewrite_tgt_buffer_shape;
        for (int i = 0; i < static_cast<int>(rewrite_src_region.size()); i++) {
          rewrite_tgt_buffer_shape.push_back(rewrite_src_region[i]->extent);
        }
        new_target_buffer->shape = rewrite_tgt_buffer_shape;
        buffer_map_[match_buffer->buffer.get()] = Buffer(new_target_buffer);
        match_buffers.push_back(MatchBufferRegion(
            Buffer(new_target_buffer), BufferRegion(it->second, rewrite_src_region)));
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
      block->alloc_buffers = std::move(alloc_buffers);
      block->match_buffers = std::move(match_buffers);
      block_sref_reuse_->Set(GetRef<Block>(op), Block(block));
      return Stmt(block);
    } else {
      return GetRef<Block>(op);
    }
  }
  /*! \brief A mapping which maps old buffers to new buffers, including the buffers defined in
   *         MatchBufferRegion.*/
  std::unordered_map<const BufferNode*, Buffer> buffer_map_;
  /*! \brief The block sref reuse map for the following replacement */
  Map<Block, Block>* block_sref_reuse_;
  Array<Array<PrimExpr>> new_shape_;
  bool success_ = true;
};

bool checkValidity(const Array<Array<PrimExpr>>& new_shape, const Array<PrimExpr>& old_shape){
  arith::Analyzer analyzer;
  PrimExpr new_prod = 1, old_prod=1;
  for (int i = 0; i < static_cast<int>(new_shape.size()); i++) {
    for (const PrimExpr& extent : new_shape[i]) {
      new_prod*=extent;
    }
  }
  for (const PrimExpr& extent : old_shape) {
    old_prod*=extent;
  }
  return analyzer.CanProve(new_prod==old_prod);
}

void PromoteRank(ScheduleState self, const StmtSRef& block_sref, int write_buffer_index) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  CheckAffineBinding(self, GetRef<Block>(block));
  if(const BufferStoreNode* buf_store = block->body.as<BufferStoreNode>()){
    Buffer buffer = block->writes[write_buffer_index]->buffer;
    Array<StmtSRef> loop_srefs = GetLoops(block_sref);
    Map<Var, Range> var_range;
    for (const StmtSRef& loop_sref : loop_srefs) {
      const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
      var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    }
    Array<PrimExpr> indices = buf_store->indices;
    const tir::BlockRealize& realize = tir::GetBlockRealize(self, block_sref);
    Map<Var, PrimExpr> binding;
    for (int i = 0; i < static_cast<int>(realize->iter_values.size()); i++) {
       binding.Set(block->iter_vars[i]->var,realize->iter_values[i]);
    }
    for (int i = 0; i < static_cast<int>(indices.size()); i++) {
      indices.Set(i, Substitute(indices[i], binding));
    }
    Array<Array<PrimExpr>> new_shape = AccessPatternExtractor::getRankPromotedShape(indices,
                                                                                    var_range);
    LOG(INFO)<<new_shape;
    if(!checkValidity(new_shape, buffer->shape)){
      return;
    }
    Map<Block, Block> block_sref_reuse;
    StmtSRef allocate_site_sref;
    bool is_alloc;
    std::tie(allocate_site_sref, is_alloc) = GetBufferAllocSite(block_sref, buffer);
    const BlockNode* allocate_site = TVM_SREF_TO_BLOCK(allocate_site, allocate_site_sref);
    // We do not allow the buffer being defined in `match_buffer`.
    CHECK(is_alloc) << "ValueError: Set the storage scope of a buffer defined in MatchBufferRegion is"
                       " not allowed. You might want to set the storage scope of its source buffer if"
                       " you really want to change its storage scope.";
    Optional<Block> rewrite_body = RankPromotionBufferRewriter::Mutate(GetRef<Block>(allocate_site), buffer,
                                                                       new_shape,
                                           &block_sref_reuse);
    if (rewrite_body.defined()) {
      self->Replace(allocate_site_sref, rewrite_body.value(), block_sref_reuse);
    }
  }
  
}

}  // namespace tir
}  // namespace tvm