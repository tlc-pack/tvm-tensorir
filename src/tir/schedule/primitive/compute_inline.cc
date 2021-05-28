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

static const char prim_compute_inline[] = "compute-inline";
static const char prim_reverse_compute_inline[] = "reverse_compute-inline";

class NonSingleReaderWriterError : public ScheduleError {
 public:
  explicit NonSingleReaderWriterError(const char* prim, IRModule mod, bool is_read, Block block)
      : prim_(prim), mod_(mod), is_read_(is_read), block_(std::move(block)) {}

  String primitive() const final { return prim_; }

  IRModule mod() const final { return mod_; }

  String FastErrorString() const final {
    return is_read_ ? "ScheduleError: The block is allowed to read only a single buffer region"
                    : "ScheduleError: The block is allowed to write only a single buffer region";
  }

  String DetailRenderTemplate() const final {
    if (is_read_) {
      int k = block_->reads.size();
      return "The block is only allowed to read a single buffer region, but it reads " +
             std::to_string(k) + " region(s): {0}";
    } else {
      int k = block_->writes.size();
      return "The block is only allowed to write a single buffer region, but it writes " +
             std::to_string(k) + " region(s): {0}";
    }
  }

  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

  static Buffer CheckRead(const char* prim, const ScheduleState& self, const Block& block) {
    if (block->reads.size() != 1) {
      throw NonSingleReaderWriterError(prim, self->mod, true, block);
    }
    return block->reads[0]->buffer;
  }

  static Buffer CheckWrite(const char* prim, const ScheduleState& self, const Block& block) {
    if (block->writes.size() != 1) {
      throw NonSingleReaderWriterError(prim, self->mod, false, block);
    }
    return block->writes[0]->buffer;
  }

  const char* prim_;
  IRModule mod_;
  bool is_read_;
  Block block_;
};

class BodyAnalysisError : public ScheduleError {
 public:
  explicit BodyAnalysisError(const char* prim, IRModule mod, Block block)
      : prim_(prim), mod_(mod), block_(std::move(block)) {}

  String primitive() const final { return prim_; }

  IRModule mod() const final { return mod_; }

  String FastErrorString() const final {
    return "ScheduleError: The block cannot be inlined because its body is illegal";
  }

  String DetailRenderTemplate() const final {
    if (prim_ == String(prim_compute_inline)) {
      return R"(The body of the inlined block should be in form of
    'A[i, j, k, ...] = f(i, j, k, ...)',
where the indices on the left are distinct atomic variables,
and there should not no variables other than the index variables)";
    } else if (prim_ == String(prim_reverse_compute_inline)) {
      return R"(The body of the inlined block should be in form of
    `B[...] = g(i, j, k, A[i, j, k, ...] ...)`,
where A is the only buffer the block consumes, whose indices are distinct atomic variables,
and there should not no variables other than the index variables)";
    } else {
      ICHECK(false) << "not reachable";
      throw;
    }
  }

  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

  String prim_;
  IRModule mod_;
  Block block_;
};

class OnlyLeafError : public ScheduleError {
 public:
  explicit OnlyLeafError(const char* prim, IRModule mod, Block leaf_block, Block scope_root)
      : prim_(prim),
        mod_(mod),
        leaf_block_(std::move(leaf_block)),
        scope_root_(std::move(scope_root)) {}

  String primitive() const final { return prim_; }

  IRModule mod() const final { return mod_; }

  String FastErrorString() const final {
    return "ScheduleError: Cannot remove the only leaf in the scope";
  }

  String DetailRenderTemplate() const final {
    return "Block {0} is the only leaf in the scope {1}, which cannot be removed; Otherwise the "
           "scope will be empty.";
  }

  Array<ObjectRef> LocationsOfInterest() const final { return {leaf_block_, scope_root_}; }

  const char* prim_;
  IRModule mod_;
  Block leaf_block_;
  Block scope_root_;
};

class NonSingleProducerError : public ScheduleError {
 public:
  explicit NonSingleProducerError(const char* prim, IRModule mod, Block block)
      : prim_(prim), mod_(mod), block_(std::move(block)) {}

  String primitive() const final { return prim_; }

  IRModule mod() const final { return mod_; }

  String FastErrorString() const final {
    return "ScheduleError: The consumer block to be inlined is required to have only a single "
           "producer block, and the producer block should be a complete block who has only a "
           "single consumer";
  }

  String DetailRenderTemplate() const final {
    return "The consumer block {0} to be inlined is required to have only a single "
           "producer block, and the producer block should be a complete block who has only a "
           "single consumer";
  }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

  const char* prim_;
  IRModule mod_;
  Block block_;

  static void Check(const ScheduleState& self, const StmtSRef& consumer_block_sref,
                    const StmtSRef& scope_root_sref) {
    BlockScope scope = self->GetBlockScope(scope_root_sref);
    Array<Dependency> producers = scope->GetDepsByDst(consumer_block_sref);
    if (producers.size() == 1 && producers[0]->kind == DepKind::kRAW) {
      const StmtSRef& producer_block_sref = producers[0]->src;
      if (IsCompleteBlock(self, producer_block_sref, scope_root_sref)) {
        Array<Dependency> consumers = scope->GetDepsBySrc(producer_block_sref);
        if (consumers.size() == 1) {
          return;
        }
      }
    }
    const BlockNode* block = TVM_SREF_TO_BLOCK(block, consumer_block_sref);
    throw NonSingleProducerError(prim_reverse_compute_inline, self->mod, GetRef<Block>(block));
  }
};

/*!
 * \brief Construct a new AST, with a specific sref tree leaf removed.
 * The ancestors who have only a single child will be removed too.
 * \param leaf_sref The block/loop sref to the sref tree leaf to be removed
 * \param src_stmt The root of the subtree where the replacement begins
 * \param tgt_stmt The root of the subtree after the replacement
 * \return A boolean indicating if the search succeeds
 */
bool WithLeafRemoved(const StmtSRef& leaf_sref, Stmt* src_stmt, Stmt* tgt_stmt) {
  // Go upwards until find an ancestor with more than two children
  const StmtNode* last_stmt = leaf_sref->stmt;
  StmtSRefNode* sref = leaf_sref->parent;
  for (;; last_stmt = sref->stmt, sref = sref->parent) {
    if (const auto* loop = sref->StmtAs<ForNode>()) {
      if (const auto* seq = loop->body.as<SeqStmtNode>()) {
        if (seq->size() > 1) {
          break;
        }
      }
    } else {
      break;
    }
  }
  if (const auto* block = sref->StmtAs<BlockNode>()) {
    if (const auto* seq = block->body.as<SeqStmtNode>()) {
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*block);
      n->body = SeqStmtRemove(GetRef<SeqStmt>(seq), GetRef<Stmt>(last_stmt));
      *src_stmt = GetRef<Stmt>(block);
      *tgt_stmt = Stmt(std::move(n));
      return true;
    }
  }
  if (const auto* loop = sref->StmtAs<ForNode>()) {
    if (const auto* seq = loop->body.as<SeqStmtNode>()) {
      ObjectPtr<ForNode> n = make_object<ForNode>(*loop);
      n->body = SeqStmtRemove(GetRef<SeqStmt>(seq), GetRef<Stmt>(last_stmt));
      *src_stmt = GetRef<Stmt>(loop);
      *tgt_stmt = Stmt(std::move(n));
      return true;
    }
  }
  return false;
}

/*!
 * \brief Extracts expressions that loads from a specific buffer
 * \param buffer The buffer to be loaded from
 * \param from The BufferStore statement to be extracted from
 * \return A list of `BufferLoad` expressions
 */
std::vector<const BufferLoadNode*> ExtractBufferLoad(const Buffer& buffer,
                                                     const BufferStoreNode* from) {
  struct Extractor : public ExprVisitor {
    void VisitExpr_(const BufferLoadNode* load) final {
      if (load->buffer.get() == buffer) {
        result.push_back(load);
      }
    }
    const BufferNode* buffer;
    std::vector<const BufferLoadNode*> result;
  } extractor;
  extractor.buffer = buffer.get();
  for (const PrimExpr& expr : from->indices) {
    extractor(expr);
  }
  extractor(from->value);
  return std::move(extractor.result);
}

/*!
 * \brief The base class of the inliner, which handles:
 * 1) Substitute a subtree with the specific block being inlined
 * 2) Update the block signature to reflect the changes of read/write/allocated buffers
 * 3) Maintain a list of index variables and their substition of the buffer being inlined
 */
class BaseInliner : public StmtExprMutator {
 protected:
  explicit BaseInliner(const Buffer& inlined_buffer, const Block& inlined_block,
                       const StmtSRef& scope_root_sref)
      : inlined_buffer_(inlined_buffer),
        inlined_store_(inlined_block->body.as<BufferStoreNode>()),
        scope_root_sref_(scope_root_sref) {
    AddAccessedBuffers(inlined_block.get());
  }

  Stmt VisitStmt_(const ForNode* loop) final {
    if (src_stmt.get() == loop) {
      loop = tgt_stmt.as<ForNode>();
      ICHECK(loop != nullptr);
    }
    return StmtExprMutator::VisitStmt_(loop);
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    AddAccessedBuffers(block);
    Block src_block = GetRef<Block>(block);
    if (src_block.same_as(src_stmt)) {
      block = tgt_stmt.as<BlockNode>();
      ICHECK(block != nullptr);
    }
    Block tgt_block = Downcast<Block>(StmtExprMutator::VisitStmt_(block));
    tgt_block = UpdateBlockSignature(src_block, std::move(tgt_block));
    block_reuse.Set(src_block, tgt_block);
    return std::move(tgt_block);
  }

  /*!
   * \brief Check if the indices are atomic distinct variables and the access is n-dimensional.
   * If so, set `self->idx_vars_` properly.
   * \param indices The indices to be extracted
   * \param expected_ndim The expected ndim of the access
   * \return A boolean flag indicating if the check is successful
   */
  bool SetIndexVars(const Array<PrimExpr>& indices, int expected_ndim) {
    int n = indices.size();
    if (n != expected_ndim) {
      // Failure: dimension mismatch
      return false;
    }
    std::vector<const VarNode*> result;
    result.reserve(n);
    for (const PrimExpr& i : indices) {
      if (const auto* var = i.as<VarNode>()) {
        result.push_back(var);
      } else {
        // Failure: indexing expression is not a variable
        return false;
      }
    }
    using DistinctSet = std::unordered_set<const VarNode*>;
    int n_distinct = DistinctSet(result.begin(), result.end()).size();
    if (n != n_distinct) {
      // Failure: indexing variables are not distinct
      return false;
    }
    if (idx_vars_.empty()) {
      idx_vars_ = std::move(result);
    } else if (!support::ArraySame(idx_vars_, result)) {
      // Failure: indexing variables are not consitent in different BufferLoads
      return false;
    }
    return true;
  }

  /*!
   * \brief Set the mapping of index substitution `self->idx_sub_`
   * \param indices The expressions that the corresponding index variables are replaced to
   */
  void SetIndexSubstitution(const Array<PrimExpr>& indices) {
    ICHECK_EQ(indices.size(), idx_vars_.size());
    int n = idx_vars_.size();
    idx_sub_.reserve(n);
    for (int i = 0; i < n; ++i) {
      idx_sub_[idx_vars_[i]] = indices[i];
    }
  }

 private:
  void AddAccessedBuffers(const BlockNode* block) {
    for (const BufferRegion& buffer_region : block->reads) {
      const Buffer& buffer = buffer_region->buffer;
      buffer_var_map_.Set(buffer->data, buffer);
    }
    for (const BufferRegion& buffer_region : block->writes) {
      const Buffer& buffer = buffer_region->buffer;
      buffer_var_map_.Set(buffer->data, buffer);
    }
    for (const Buffer& buffer : block->alloc_buffers) {
      buffer_var_map_.Set(buffer->data, buffer);
    }
  }

  Block UpdateBlockSignature(Block src_block, Block tgt_block) {
    bool is_scope_root = src_block.get() == scope_root_sref_->stmt;
    // Step 1. Update `BlockNode::alloc_buffers`
    Array<Buffer> alloc_buffers;
    if (is_scope_root) {
      alloc_buffers.reserve(tgt_block->alloc_buffers.size());
      for (const Buffer& alloc_buffer : tgt_block->alloc_buffers) {
        if (!alloc_buffer.same_as(inlined_buffer_)) {
          alloc_buffers.push_back(alloc_buffer);
        }
      }
    } else {
      alloc_buffers = std::move(tgt_block->alloc_buffers);
    }
    // Step 2. Update `BlockNode::reads` and `BlockNode::writes`
    Array<BufferRegion> reads = std::move(tgt_block->reads);
    Array<BufferRegion> writes = std::move(tgt_block->writes);
    if (!is_scope_root) {
      Array<Array<BufferRegion>> inspected = GetBlockAccessRegion(tgt_block, buffer_var_map_);
      reads = std::move(inspected[0]);
      writes = std::move(inspected[1]);
    }
    // Step 3. Assemble the result
    BlockNode* n = tgt_block.CopyOnWrite();
    n->reads = std::move(reads);
    n->writes = std::move(writes);
    n->alloc_buffers = std::move(alloc_buffers);
    return tgt_block;
  }

 protected:
  /*! \brief The buffer to be inlined */
  Buffer inlined_buffer_{nullptr};
  /*! \brief The body of the block to be inlined */
  const BufferStoreNode* inlined_store_{nullptr};
  /*! \brief The scope root */
  StmtSRef scope_root_sref_{nullptr};
  /*! \brief Maps a buffer's data field to itself */
  Map<Var, Buffer> buffer_var_map_;
  /*! \brief The indices used for indexing the buffer to be inlined */
  std::vector<const VarNode*> idx_vars_;
  /*! \brief The mapping to substitute index variables to PrimExprs */
  std::unordered_map<const VarNode*, PrimExpr> idx_sub_;

 public:
  /*! \brief The Stmt to be replaced when removing the leaf block */
  Stmt src_stmt{nullptr};
  /*! \brief The Stmt to be replaced to when removing the leaf block */
  Stmt tgt_stmt{nullptr};
  /*! \brief The reuse mapping of block srefs */
  Map<Block, Block> block_reuse;
};

/*!
 * \brief Helper to inline the producer block into its consumer(s)
 * The derived class implements the following functionalities:
 * 1) Substitute `BufferLoad` on the buffer to be inlined
 * to its value calculation in the producer block
 * 2) Analyze the producer block to determine the remapping of index variables
 */
class ComputeInliner : public BaseInliner {
 public:
  explicit ComputeInliner(const Buffer& inlined_buffer, const Block& producer_block,
                          const StmtSRef& scope_root_sref)
      : BaseInliner(inlined_buffer, producer_block, scope_root_sref) {}

  bool AnalyzeBody(const Block& producer_block) {
    if (inlined_store_ == nullptr) {
      return false;
    }
    int n_vars = UndefinedVars(GetRef<Stmt>(inlined_store_), {}).size();
    if (!SetIndexVars(inlined_store_->indices, n_vars)) {
      return false;
    }
    return true;
  }

 private:
  using BaseInliner::VisitExpr_;
  using BaseInliner::VisitStmt_;

  PrimExpr VisitExpr_(const BufferLoadNode* _load) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_load));
    if (!load->buffer.same_as(inlined_buffer_)) {
      return std::move(load);
    }
    return ReplaceInlinedBuffer(std::move(load));
  }

  PrimExpr ReplaceInlinedBuffer(BufferLoad load) {
    SetIndexSubstitution(load->indices);
    return Substitute(inlined_store_->value, idx_sub_);
  }
};

/*!
 * \brief Helper to inline the consumer block into its producer
 * The derived class implements the following functionalities:
 * 1) Analyze the consumer block to determine the remapping of index variables
 * 2) Substitute `BufferStore` of the buffer to be inlined,
 * replacing it with direct writing to the buffer that consumer writes
 */
class ReverseComputeInliner : public BaseInliner {
  class Substituter : public StmtExprMutator {
   public:
    explicit Substituter(ReverseComputeInliner* self) : self_(self) {}

   private:
    PrimExpr VisitExpr_(const VarNode* var) final {
      auto it = self_->idx_sub_.find(var);
      ICHECK(it != self_->idx_sub_.end());
      return (*it).second;
    }

    PrimExpr VisitExpr_(const BufferLoadNode* _load) final {
      BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_load));
      return load->buffer.same_as(self_->inlined_buffer_) ? self_->producer_rhs_ : load;
    }

    ReverseComputeInliner* self_;
  };

 public:
  explicit ReverseComputeInliner(const Buffer& inlined_buffer, const Block& consumer_block,
                                 const StmtSRef& scope_root_sref)
      : BaseInliner(inlined_buffer, consumer_block, scope_root_sref) {}

  bool AnalyzeBody(const Block& consumer_block) {
    if (inlined_store_ == nullptr) {
      // Failure: block body is not BufferStore
      return false;
    }
    std::vector<const BufferLoadNode*> loads = ExtractBufferLoad(inlined_buffer_, inlined_store_);
    if (loads.size() == 0) {
      // Failure: no BufferLoad from the `inlined_buffer_`
      return false;
    }
    int n_vars = UndefinedVars(GetRef<BufferStore>(inlined_store_), {}).size();
    for (const BufferLoadNode* load : loads) {
      if (!SetIndexVars(load->indices, n_vars)) {
        // Failure: incorrect of inconsistent index vars
        return false;
      }
    }
    return true;
  }

 private:
  using BaseInliner::VisitExpr_;
  using BaseInliner::VisitStmt_;

  Stmt VisitStmt_(const BufferStoreNode* _store) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_store));
    if (!store->buffer.same_as(inlined_buffer_)) {
      return std::move(store);
    }
    return ReplaceInlinedBuffer(std::move(store));
  }

  Stmt ReplaceInlinedBuffer(BufferStore producer) {
    SetIndexSubstitution(producer->indices);
    producer_rhs_ = producer->value;
    return Substituter(this)(GetRef<BufferStore>(inlined_store_));
  }

  /*! \brief The RHS value of the producer's BufferStore statement */
  PrimExpr producer_rhs_{nullptr};
};

void ComputeInline(ScheduleState self, const StmtSRef& producer_block_sref) {
  const char* prim = prim_compute_inline;
  const BlockNode* _producer_block = TVM_SREF_TO_BLOCK(_producer_block, producer_block_sref);
  Block producer_block = GetRef<Block>(_producer_block);
  Buffer inlined_buffer = NonSingleReaderWriterError::CheckWrite(prim, self, producer_block);
  // Step 1. Get the scope block
  StmtSRef scope_root_sref = CheckScopeStagePipeline(prim, self, producer_block_sref);
  // Step 2. Check completeness
  CheckCompleteBlock(prim, self, producer_block_sref, scope_root_sref);
  // Step 3. Analyze the block body
  ComputeInliner inliner(inlined_buffer, producer_block, scope_root_sref);
  if (!inliner.AnalyzeBody(producer_block)) {
    throw BodyAnalysisError(prim, self->mod, producer_block);
  }
  // Step 4. Create a plan that removes the leaf block to be inlined
  if (!WithLeafRemoved(producer_block_sref, &inliner.src_stmt, &inliner.tgt_stmt)) {
    const BlockNode* scope_root = TVM_SREF_TO_BLOCK(scope_root, scope_root_sref);
    throw OnlyLeafError(prim, self->mod, producer_block, GetRef<Block>(scope_root));
  }
  // Step 5. Create an AST where the leaf `producer_block_sref` points to is removed,
  // and update other blocks who read from the removed block
  Stmt tgt_stmt = inliner(GetRef<Stmt>(scope_root_sref->stmt));
  // Step 6. Do the real mutation on the AST and the sref tree in the schedule state
  self->Replace(scope_root_sref, tgt_stmt, inliner.block_reuse);
}

void ReverseComputeInline(ScheduleState self, const StmtSRef& consumer_block_sref) {
  const char* prim = prim_reverse_compute_inline;
  const BlockNode* _consumer_block = TVM_SREF_TO_BLOCK(_consumer_block, consumer_block_sref);
  Block consumer_block = GetRef<Block>(_consumer_block);
  Buffer inlined_buffer = NonSingleReaderWriterError::CheckRead(prim, self, consumer_block);
  // Step 1. Get the scope block
  StmtSRef scope_root_sref = CheckScopeStagePipeline(prim, self, consumer_block_sref);
  // Step 2. Check completeness
  CheckCompleteBlock(prim, self, consumer_block_sref, scope_root_sref);
  // Step 3. Check if the consumer has a single complete producer
  NonSingleProducerError::Check(self, consumer_block_sref, scope_root_sref);
  // Step 4. Analyze the block body
  ReverseComputeInliner inliner(inlined_buffer, consumer_block, scope_root_sref);
  if (!inliner.AnalyzeBody(consumer_block)) {
    throw BodyAnalysisError(prim, self->mod, consumer_block);
  }
  // Step 5. Create a plan that removes the leaf block to be inlined
  if (!WithLeafRemoved(consumer_block_sref, &inliner.src_stmt, &inliner.tgt_stmt)) {
    const BlockNode* scope_root = TVM_SREF_TO_BLOCK(scope_root, scope_root_sref);
    throw OnlyLeafError(prim, self->mod, consumer_block, GetRef<Block>(scope_root));
  }
  // Step 6. Create an AST where the leaf `consumer_block_sref` points to is removed,
  // and update other blocks who read from the removed block
  Stmt tgt_stmt = inliner(GetRef<Stmt>(scope_root_sref->stmt));
  // Step 7. Do the real mutation on the AST and the sref tree in the schedule state
  self->Replace(scope_root_sref, tgt_stmt, inliner.block_reuse);
}

}  // namespace tir
}  // namespace tvm
