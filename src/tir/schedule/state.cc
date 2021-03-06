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
#include "./utils.h"

namespace tvm {
namespace tir {

/**************** Utility functions ****************/

/*!
 * \brief Set the `StmtSRefNode::seq_index` field for stmt
 * \param self The schedule class
 * \param stmt The statement, or the realize node of the statement whose sref to be set
 * \param seq_index The seq_index to be set
 */
void SetSeqIndex(ScheduleStateNode* self, const Stmt& stmt, int seq_index) {
  if (const auto* realize = stmt.as<BlockRealizeNode>()) {
    const BlockNode* block = realize->block.get();
    ICHECK(self->stmt2ref.count(block));
    self->stmt2ref.at(block)->seq_index = seq_index;
  } else if (const auto* block = stmt.as<BlockNode>()) {
    ICHECK(self->stmt2ref.count(block));
    self->stmt2ref.at(block)->seq_index = seq_index;
  } else if (const auto* loop = stmt.as<ForNode>()) {
    ICHECK(self->stmt2ref.count(loop));
    self->stmt2ref.at(loop)->seq_index = seq_index;
  } else {
    // do nothing
  }
}

/*!
 * \brief Update seq_index of the children of a SeqStmt
 * \param self The schedule class
 * \param seq_stmt The SeqStmt whose childrens needs updating
 */
void SetSeqIndex(ScheduleStateNode* self, const SeqStmtNode* seq_stmt) {
  int i = 0;
  for (const Stmt& stmt : seq_stmt->seq) {
    SetSeqIndex(self, stmt, i);
    ++i;
  }
}

/*!
 * \brief Update the sref information on the schedule class, as well as the statement of sref itself
 * More specifically, update
 *  `sref->stmt` to `new_stmt`
 *  `self->stmt2ref`, remove the old statement that sref points to, and add the new statement
 * \param self The schedule class to be updated
 * \param sref The sref to be updated
 * \param new_stmt The statement that replaces the statement inside the sref
 */
void UpdateSRef(ScheduleStateNode* self, StmtSRefNode* sref, const StmtNode* new_stmt) {
  ICHECK(new_stmt->IsInstance<BlockNode>() || new_stmt->IsInstance<ForNode>());
  const StmtNode* old_stmt = sref->stmt;
  ICHECK_NE(new_stmt, old_stmt);
  self->stmt2ref[new_stmt] = GetRef<StmtSRef>(sref);
  self->stmt2ref.erase(sref->stmt);
  sref->stmt = new_stmt;
}

/**************** Creation ****************/

/*! \brief A helper class to create a new ScheduleStateNode */
class StateCreator : private StmtVisitor {
 public:
  /*!
   * \brief The entry function
   * \param self The schedule state to be completed
   */
  static ObjectPtr<ScheduleStateNode> Create(IRModule mod, bool debug_mode) {
    ObjectPtr<ScheduleStateNode> n = make_object<ScheduleStateNode>();
    ScheduleStateNode* self = n.get();
    // Set `n->mod`
    n->mod = std::move(mod);
    // Set `n->debug_mode`
    n->debug_mode = debug_mode;
    // Set `n->stmt2ref` and `n->scopes`
    StateCreator creator(self);
    for (const auto& kv : n->mod->functions) {
      const BaseFunc& base_func = kv.second;
      if (const auto* func = base_func.as<PrimFuncNode>()) {
        creator.VisitStmt(func->body);
      }
    }
    return n;
  }

 private:
  explicit StateCreator(ScheduleStateNode* self)
      : self_(self), srefs_{}, realizes_{}, block_frames_{} {}

  /*!
   * \brief Add a new statement to the stack, which becomes the current scope
   * \param stmt A loop statement or a block statement
   */
  StmtSRef PushSRef(const StmtNode* stmt) {
    if (srefs_.empty()) {
      srefs_.push_back(StmtSRef(stmt, nullptr, -1, false));
    } else {
      StmtSRefNode* parent = srefs_.back().operator->();
      srefs_.push_back(StmtSRef(stmt, parent, -1, false));
    }
    return srefs_.back();
  }

  /*! \brief Pop the top of the scope and record it in stmt2ref map */
  StmtSRef PopSRef() {
    StmtSRef sref = std::move(srefs_.back());
    self_->stmt2ref[sref->stmt] = sref;
    srefs_.pop_back();
    return sref;
  }

  void VisitStmt_(const BlockNode* block) final {
    PushSRef(block);
    block_frames_.emplace_back();
    // `stmt->init` is not visited
    VisitStmt(block->body);
    StmtSRef sref = PopSRef();
    // Set up `binding_valid`
    Map<Var, Range> loop_var_ranges;
    for (StmtSRefNode* parent = sref->parent; parent && parent->stmt->IsInstance<ForNode>();
         parent = parent->parent) {
      const auto* loop = static_cast<const ForNode*>(parent->stmt);
      loop_var_ranges.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    }
    sref->binding_valid =
        ValidateBlockBinding(GetRef<BlockRealize>(realizes_.back()), loop_var_ranges);
    // Collect `scopes` info
    self_->scopes.Set(sref, BlockScope(std::move(block_frames_.back().leaf_blocks)));
    block_frames_.pop_back();
    // Update parent scope if exists
    if (!block_frames_.empty()) {
      block_frames_.back().leaf_blocks.push_back(sref);
    }
  }

  void VisitStmt_(const ForNode* loop) final {
    PushSRef(loop);
    VisitStmt(loop->body);
    PopSRef();
  }

  void VisitStmt_(const BlockRealizeNode* realize) final {
    realizes_.push_back(realize);
    VisitStmt(realize->block);
    realizes_.pop_back();
  }

  void VisitStmt_(const SeqStmtNode* seq_stmt) final {
    // Set `seq_index` information for SeqStmtNode
    StmtVisitor::VisitStmt_(seq_stmt);
    SetSeqIndex(self_, seq_stmt);
  }

  /*! \brief A stack frame gathering the block scope information */
  struct BlockFrame {
    /*! \brief The leaf blocks of the current scope. */
    Array<StmtSRef> leaf_blocks;
  };

  /*! \brief The result stmt2ref */
  ScheduleStateNode* self_;
  /*! \brief The stack frame used to indicate the current scope */
  std::vector<StmtSRef> srefs_;
  /*! \brief The BlockRealize in the ancestors */
  std::vector<const BlockRealizeNode*> realizes_;
  /*! \brief The stack frames of blocks in the DFS visit. */
  std::vector<BlockFrame> block_frames_;
};

/**************** Constructor ****************/

ScheduleState::ScheduleState(IRModule mod, bool debug_mode) {
  data_ = StateCreator::Create(mod, debug_mode);
  // Verify the region cover
  ScheduleState self = GetRef<ScheduleState>(get());
  for (const auto& it : self->scopes) {
    const StmtSRef& sref = it.first;
    VerifyRegionCover(self, sref);
  }
}

ScheduleState::ScheduleState(PrimFunc func, bool debug_mode)
    : ScheduleState(IRModule({{GlobalVar("main"), func}}), debug_mode) {}

/**************** Replace ****************/

/*!
 * \brief Record the different sref reuse types in the replacement
 *
 * 1) Intact: the subtree appears as the same object on both `src_stmt` and `tgt_stmt`,
 * which, given the immutability of the IR, means the entire subtree is unchanged,
 * and we do not need to recurse into the subtree.
 *
 * 2) Loop/Block sref reuse: for two different objects (`src`, `tgt`),
 * which are both loops or both blocks,
 * there is correspondence between them,
 * which makes us to reuse the sref pointing to `src`, and changes it to point to `tgt`.
 */
struct ReuseInfo {
  /*! \brief Kind 1. Intact reuse */
  std::unordered_set<const StmtNode*> intact;
  /*! \brief Kind 2.1. Loop sref reuse */
  std::unordered_set<const VarNode*> loop_sref_reuse;
  /*! \brief Kind 2.2. Block sref reuse.
   * Maps an old Block in `src_stmt` to a new block in `tgt_stmt`
   */
  std::unordered_map<const BlockNode*, const BlockNode*> block_sref_reuse;
};

/*!
 * \brief A helper visitor used in `SRefUpdater`,
 * which collects two cases of reusing srefs:
 *
 * 1) Intact: the subtree represented by `intact` appears on both old and new IR.
 * Given the immutability of the IR, we can quickly decide that the entire subtree is unchanged,
 * which means we do not need to visit into the subtree of the old statement.
 *
 * 2) Reused block/loop: for two different objects (`src`, `tgt`),
 * which are both loops or both blocks,
 * and there is correspondence between them,
 * which makes us to reuse the sref pointing to `src`, and changes it to point to `tgt`,
 *
 * \sa SRefUpdater
 */
class ReuseCollector : public StmtVisitor {
 public:
  static ReuseInfo Collect(ScheduleStateNode* self, const Stmt& tgt_stmt,
                           const Map<Block, Block>& block_sref_reverse_reuse) {
    ReuseCollector collector(self);
    collector.VisitStmt(tgt_stmt);
    ReuseInfo result;
    result.intact = {collector.intact_.begin(), collector.intact_.end()};
    result.loop_sref_reuse = {collector.loop_vars_.begin(), collector.loop_vars_.end()};
    for (const auto& kv : block_sref_reverse_reuse) {
      const Block& new_block = kv.first;
      const Block& old_block = kv.second;
      result.block_sref_reuse.emplace(old_block.get(), new_block.get());
    }
    return result;
  }

 private:
  explicit ReuseCollector(ScheduleStateNode* self) : self_(self) {}

  void VisitStmt_(const ForNode* op) final {
    if (self_->stmt2ref.count(op)) {
      intact_.push_back(op);
    } else {
      // Collect loop vars for detecting reuse of loop sref
      loop_vars_.push_back(op->loop_var.get());
      StmtVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const BlockNode* op) final {
    if (self_->stmt2ref.count(op)) {
      intact_.push_back(op);
    } else {
      StmtVisitor::VisitStmt_(op);
    }
  }

  ScheduleStateNode* self_;
  std::vector<const StmtNode*> intact_;
  std::vector<const VarNode*> loop_vars_;
};

/*!
 * \brief A helper visitor used in `SRefUpdater`,
 * which removes the stale srefs that are useless after the replacement.
 *
 * It uses the reuse information previously collected to
 * 1) delete those srefs that are not reused.
 * 2) return the sref objects that are loop/block sref reuses, but not intact reuses
 * \sa SRefUpdater
 */
class SRefTreePruner : public StmtVisitor {
 public:
  /*!
   * \brief The entry function
   * \param self The schedule class
   * \param info The reuse info about loop reuses and intact reuse
   * \param src_stmt The `src_stmt` where stale srefs to be removed
   * \return Mapping from the reuse elements to reused srefs, more specifically:
   * 1) Loop reuse: maps a loop var to the reused sref
   * 2) Block reuse: maps a block stmt to the reused sref,
   * where the block comes from the subtree of `tgt_stmt`
   * 3) Intact reuse: not returned
   */
  static std::unordered_map<const Object*, StmtSRef> Prune(ScheduleStateNode* self,
                                                           const ReuseInfo& reuse_info,
                                                           const Stmt& src_stmt) {
    SRefTreePruner pruner(self, reuse_info);
    pruner.VisitStmt(src_stmt);
    return std::move(pruner.reused_srefs_);
  }

 private:
  explicit SRefTreePruner(ScheduleStateNode* self, const ReuseInfo& reuse_info)
      : self_(self), reuse_info_(reuse_info) {}

  void VisitStmt_(const ForNode* op) final {
    if (reuse_info_.intact.count(op)) {
      return;
    }
    auto it = self_->stmt2ref.find(op);
    ICHECK(it != self_->stmt2ref.end())
        << "IndexError: Cannot find correpsonding StmtSRef for the loop:\n"
        << GetRef<For>(op);
    StmtSRef& sref = it->second;
    // Detect reuse
    const VarNode* loop_var = op->loop_var.get();
    if (reuse_info_.loop_sref_reuse.count(loop_var)) {
      // sref can be reused
      reused_srefs_.emplace(loop_var, std::move(sref));
    } else {
      sref->stmt = nullptr;
      sref->parent = nullptr;
      sref->seq_index = -1;
    }
    // erase the statement
    self_->stmt2ref.erase(it);
    // detect recursively
    VisitStmt(op->body);
  }

  void VisitStmt_(const BlockNode* op) final {
    if (reuse_info_.intact.count(op)) {
      return;
    }
    auto it = self_->stmt2ref.find(op);
    ICHECK(it != self_->stmt2ref.end())
        << "IndexError: Cannot find correpsonding StmtSRef for the block:\n"
        << GetRef<Block>(op);
    StmtSRef& sref = it->second;
    // Detect reuse
    auto reuse_it = reuse_info_.block_sref_reuse.find(op);
    if (reuse_it != reuse_info_.block_sref_reuse.end()) {
      // sref can be reused
      reused_srefs_.emplace(reuse_it->second, std::move(sref));
    } else {
      sref->stmt = nullptr;
      sref->parent = nullptr;
      sref->seq_index = -1;
      self_->scopes.erase(sref);
    }
    // erase the statement
    self_->stmt2ref.erase(it);
    // detect recursively
    // op->init is omitted
    VisitStmt(op->body);
  }

  ScheduleStateNode* self_;
  const ReuseInfo& reuse_info_;
  /*!
   * \brief Reused srefs:
   * 1) loop var -> StmtSRef
   * 2) block stmt -> StmtSRef, where the block comes from the subtree of `tgt_stmt`
   */
  std::unordered_map<const Object*, StmtSRef> reused_srefs_;
};

/*!
 * \brief Update the sref in the `tgt_stmt` given the reuse information
 *
 * After being updated, in the `tgt_stmt` subtree,
 * 1) all `parent`s are correct
 * 2) all `seq_index`s are correct, except for the root
 * 3) all `stmt`s are correct, except for the root
 */
class SRefUpdater : public StmtVisitor {
 public:
  static void Update(ScheduleStateNode* self, StmtSRefNode* root_parent,
                     const std::unordered_map<const Object*, StmtSRef>& reused_srefs,
                     const Stmt& tgt_stmt) {
    SRefUpdater(self, root_parent, reused_srefs).VisitStmt(tgt_stmt);
  }

 private:
  explicit SRefUpdater(ScheduleStateNode* self, StmtSRefNode* root_parent,
                       const std::unordered_map<const Object*, StmtSRef>& reused_srefs)
      : self_(GetRef<ScheduleState>(self)), parents_{root_parent}, reused_srefs_(reused_srefs) {}

  void VisitStmt_(const ForNode* op) final {
    StmtSRef& sref = self_->stmt2ref[op];
    // Detect intact
    if (sref.defined()) {
      sref->parent = parents_.back();
      sref->seq_index = -1;
      return;
    }
    // Detect reuse
    auto it = reused_srefs_.find(op->loop_var.get());
    if (it != reused_srefs_.end()) {
      // Update `stmt2ref[op]` to `reused_srefs_[op->loop_var]`
      sref = it->second;
      sref->stmt = op;
      sref->parent = parents_.back();
      sref->seq_index = -1;
    } else {
      sref = StmtSRef(op, parents_.back(), /*seq_index=*/-1, /*binding_valid=*/true);
    }
    // Recursive visit
    parents_.push_back(sref.get());
    VisitStmt(op->body);
    parents_.pop_back();
  }

  void VisitStmt_(const BlockNode* op) final {
    StmtSRef& sref = self_->stmt2ref[op];
    // Detect intact
    if (sref.defined()) {
      sref->parent = parents_.back();
      sref->seq_index = -1;
      return;
    }
    // Detect reuse
    auto it = reused_srefs_.find(op);
    if (it != reused_srefs_.end()) {
      // Update `stmt2ref[op]` to `reused_srefs_[op]`
      sref = it->second;
      sref->stmt = op;
      sref->parent = parents_.back();
      sref->seq_index = -1;
    } else {
      sref = StmtSRef(op, parents_.back(), /*seq_index=*/-1, /*binding_valid=*/true);
    }
    // Recursive visit
    parents_.push_back(sref.get());
    VisitStmt(op->body);
    parents_.pop_back();
    // Additionally, need to update the scope because the block is changed
    self_->scopes.Set(sref, BlockScope(tir::GetChildBlocks(self_, sref)));
  }

  void VisitStmt_(const SeqStmtNode* seq_stmt) final {
    StmtVisitor::VisitStmt_(seq_stmt);
    SetSeqIndex(self_.get(), seq_stmt);
  }

  ScheduleState self_;
  std::vector<StmtSRefNode*> parents_;
  const std::unordered_map<const Object*, StmtSRef>& reused_srefs_;
};

/*!
 * \brief A helper that
 * 1) returns a new copy of `child_sref->parent->stmt`,
 * where `child_sref->stmt` is replaced with `child_tgt_stmt`.
 * 2) Then it points `child_sref` to `child_tgt_stmt`.
 * 3) Finally it makes the subtree of the srefs pointing to the returned AST correct,
 * except for the root.
 */
class ChildReplacer : private StmtMutator {
 public:
  static Stmt Mutate(const StmtNode* parent_stmt, const StmtNode* child_src_stmt,
                     const Stmt& child_tgt_stmt, int seq_index, bool allow_copy_on_write) {
    // Check the invariant
    ICHECK(child_src_stmt->IsInstance<BlockNode>() ||  //
           child_src_stmt->IsInstance<ForNode>());
    ICHECK(child_tgt_stmt->IsInstance<BlockNode>() ||  //
           child_tgt_stmt->IsInstance<ForNode>() ||    //
           child_tgt_stmt->IsInstance<BlockRealizeNode>());
    ChildReplacer replacer(child_src_stmt, child_tgt_stmt, seq_index);
    replacer.allow_copy_on_write_ = allow_copy_on_write;
    // Step 1. Copy-on-write the `parent_stmt` and extract its `body`,
    // where `body` means the body of either a block or a loop
    Stmt* body = nullptr;
    Stmt result = replacer.CopyOnWriteWithBody(parent_stmt, &body);
    // Step 2. Mutate the `result->body`, searching for `child_old_stmt`
    // and replace it with `child_tgt_stmt`
    *body = replacer.VisitStmt(*body);
    return result;
  }

 private:
  explicit ChildReplacer(const StmtNode* src_stmt, const Stmt& tgt_stmt, int seq_index)
      : src_stmt_(src_stmt), tgt_stmt_(tgt_stmt), seq_index_(seq_index) {}

  Stmt VisitStmt(const Stmt& stmt) final {
    if (stmt.get() == src_stmt_) {
      // if the statement matches the replace tgt_stmt
      // just return the tgt_stmt
      return tgt_stmt_;
    } else {
      return StmtMutator::VisitStmt(stmt);
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final { return GetRef<Stmt>(op); }
  Stmt VisitStmt_(const ForNode* op) final { return GetRef<Stmt>(op); }

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    int i = this->seq_index_;
    int n = op->seq.size();
    if (0 <= i && i < n) {
      const Stmt& stmt = op->seq[i];
      Optional<Stmt> new_stmt = NullOpt;
      // `stmt` can be Loop or BlockRealize
      // `src_stmt` can be Loop or Block
      // so the match from `stmt` to `src_stmt` can be
      // 1) Loop -> Loop
      // 2) BlockRealize -> Block
      if (stmt.get() == this->src_stmt_) {
        // Case 1. src_stmt is Loop, stmt is Loop
        new_stmt = tgt_stmt_;
      } else if (const auto* realize = stmt.as<BlockRealizeNode>()) {
        // Case 2. stmt is BlockRealize, src_stmt is Block
        if (realize->block.get() == src_stmt_) {
          const auto* tgt_block = TVM_TYPE_AS(tgt_block, tgt_stmt_, BlockNode);
          ObjectPtr<BlockRealizeNode> new_realize = make_object<BlockRealizeNode>(*realize);
          new_realize->block = GetRef<Block>(tgt_block);
          new_stmt = BlockRealize(std::move(new_realize));
        }
      }
      // Move new_stmt to position i
      if (new_stmt.defined()) {
        ObjectPtr<SeqStmtNode> new_seq_stmt = CopyOnWrite(op);
        new_seq_stmt->seq.Set(i, new_stmt.value());
        return SeqStmt(std::move(new_seq_stmt));
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt CopyOnWriteWithBody(const StmtNode* stmt, Stmt** body) {
    if (stmt->IsInstance<BlockNode>()) {
      auto* block = const_cast<BlockNode*>(static_cast<const BlockNode*>(stmt));
      ObjectPtr<BlockNode> new_block = CopyOnWrite(block);
      *body = &new_block->body;
      return Block(std::move(new_block));
    } else if (stmt->IsInstance<ForNode>()) {
      auto* loop = const_cast<ForNode*>(static_cast<const ForNode*>(stmt));
      ObjectPtr<ForNode> new_loop = CopyOnWrite(loop);
      *body = &new_loop->body;
      return For(std::move(new_loop));
    }
    LOG(FATAL) << "TypeError: Unexpected type: " << stmt->GetTypeKey();
    throw;
  }

  const StmtNode* src_stmt_;
  const Stmt& tgt_stmt_;
  int seq_index_;
};

void ScheduleStateNode::Replace(const tir::StmtSRef& _src_sref, const Stmt& tgt_stmt,
                                const Map<Block, Block>& block_sref_reuse) {
  {
    const StmtNode* src_stmt = _src_sref->stmt;
    bool input_correct =
        (src_stmt->IsInstance<ForNode>() && tgt_stmt->IsInstance<ForNode>()) ||
        (src_stmt->IsInstance<ForNode>() && tgt_stmt->IsInstance<BlockRealizeNode>()) ||
        (src_stmt->IsInstance<BlockNode>() && tgt_stmt->IsInstance<BlockNode>());
    if (!input_correct) {
      LOG(FATAL) << "TypeError: src_stmt has type: " << src_stmt->GetTypeKey()
                 << ". tgt_stmt has type: " << tgt_stmt->GetTypeKey() << ".\nsrc_stmt:\n"
                 << GetRef<Stmt>(src_stmt) << "\ntgt_stmt:\n"
                 << tgt_stmt;
    }
  }
  // Rule out the case that no replacement happens
  if (_src_sref->stmt == tgt_stmt.get()) {
    return;
  }
  // Reset sref as a new sref so that its content won't be affected by subsequent changes
  StmtSRef src_sref(_src_sref->stmt, _src_sref->parent, _src_sref->seq_index,
                    /*binding_valid=*/false);
  Stmt src_stmt = GetRef<Stmt>(src_sref->stmt);
  // Step 1. Create all the nodes needed for the new sref tree.
  //   The `SRefCreator` visits the AST `tgt_stmt`, creating new nodes along the way.
  //   It deals with 3 cases:
  //
  //   Case 1.1: Visiting a node already present in the old AST
  //     It means we can skip the entire subtree, leaving it untouched
  //     Mark those nodes as `intact`
  //   Case 1.2: Can somehow infer the node being visited is mutated from the old AST, including
  //     (a) The loop var appears in the old AST
  //     (b) The node is explicitly present in `block_sref_reuse`
  //     It means we need to retain and reuse the sref, so that those srefs users hold won't expire
  //     Mark those nodes as `reuse`
  //   Case 1.3: It is a completely new node
  //     Create a new node for it
  //
  // After this step
  // 1) all `parent`s are correct
  // 2) all `seq_index`s are correct, except for the root
  // 3) all `stmt`s are correct, except for the root
  {
    // Step 1.1. Collect info for different kinds of reuses
    // 1) intact
    // 2) loop/block reuse
    ReuseInfo reuse_info = ReuseCollector::Collect(this, tgt_stmt, block_sref_reuse);
    // Step 1.2. Collect loop/block reuse to their corresponding srefs
    // and remove those srefs in the `src_stmt` that are no longer used after replacement
    std::unordered_map<const Object*, StmtSRef> reused_srefs =
        SRefTreePruner::Prune(this, reuse_info, src_stmt);
    // Step 1.3. Update the sref tree, inserting newly created srefs and properly handle reused
    // srefs in `tgt_stmt`
    SRefUpdater::Update(this, /*root_parent=*/src_sref->parent, reused_srefs, tgt_stmt);
  }
  // Step 2. Set the ancestors' children properly
  //   Iteratively visit the ancestors, creating new ones whose `body`s are properly fixed.
  //   The visit stops when all the ancestors are uniquely referenced, i.e. can mutate inplace.
  //   Along the way, because we create a new ancestor path,
  //   we need to update those sref points from old ancestors to newly created ones
  // `num_copy_steps` is the maximum number of hops until we need to copy
  // To reach a node that can be mutated in-place, it needs `num_copy_steps + 1` hops
  int num_copy_steps = -1;
  // The PrimFunc that `src_sref` belongs to, and its corresponding GlobalVar
  const PrimFuncNode* g_func = nullptr;
  GlobalVar g_var;
  {
    const StmtSRefNode* p = src_sref.get();
    for (int i = 0;; ++i) {
      if (!p->stmt->unique()) {
        num_copy_steps = i;
      }
      if (p->parent != nullptr) {
        p = p->parent;
        continue;
      }
      // Find `g_func` and `g_var`
      for (const auto& kv : this->mod->functions) {
        const GlobalVar& var = kv.first;
        const BaseFunc& base_func = kv.second;
        if (const auto* func = base_func.as<PrimFuncNode>()) {
          if (const auto* realize = func->body.as<BlockRealizeNode>()) {
            if (realize->block.get() == p->stmt) {
              g_var = var;
              g_func = func;
              break;
            }
          }
        }
      }
      ICHECK(g_var.defined() && g_func != nullptr);
      // If the function itself is not unique, then we assume the root is not unique
      if (this->mod.unique() && this->mod->functions.unique() && g_func->unique()) {
        // all the way up beyond the AST is unique, then do nothing
      } else {
        // something is not unique, then do one more step
        num_copy_steps = i + 1;
      }
      break;
    }
  }
  // Loop invariant:
  //
  // Before step `i`:
  // 1) `child_sref` is `src_sref` going up by `i` steps
  // 2) `child_tgt_stmt` is the subtree that `child_sref` should correspond to after replacement
  // 3) except for the subtree root, srefs that point to the subtree of `child_tgt_stmt` are correct
  // 4) for the subtree root of `child_tgt_stmt`, `child_sref` has not pointed to it yet
  // 5) `tgt_stmt` is of type Loop, Block or BlockRealize
  //
  // During step `i`:
  // 1) Create `parent_stmt` that corresponds to `child_sref->parent
  // 2) Point `child_sref` to `child_tgt_stmt`
  // 3) `tgt_stmt` is of type Loop or Block
  StmtSRefNode* child_sref = src_sref.get();
  Stmt child_tgt_stmt = std::move(tgt_stmt);
  for (int i = 0; i <= num_copy_steps && child_sref->parent != nullptr; ++i) {
    bool parent_unique = (i == num_copy_steps);
    // replacing `child_sref->stmt` to `child_tgt_stmt`.
    const StmtNode* parent_stmt = child_sref->parent->stmt;
    const StmtNode* child_src_stmt = child_sref->stmt;
    // Step 2.1. Link `child_sref` to `child_tgt_stmt`
    if (i == 0) {
      // The `seq_index` of the root of `tgt_stmt` is set as -1, which might be incorrect
      SetSeqIndex(this, child_tgt_stmt, child_sref->seq_index);
    } else {
      // Point `child_sref` to `child_tgt_stmt`
      UpdateSRef(this, child_sref, child_tgt_stmt.get());
    }
    // Step 2.2. Create `new_parent_stmt`, by mutating the body of `parent_stmt`,
    Stmt new_parent_stmt = ChildReplacer::Mutate(parent_stmt, child_src_stmt, child_tgt_stmt,
                                                 /*seq_index=*/child_sref->seq_index,
                                                 /*allow_copy_on_write=*/parent_unique);
    // Step 2.3. Go to next parent
    if (parent_unique) {
      // If the node can be directly mutated inplace,
      // then there is no need to update its parent and the function
      break;
    }
    child_tgt_stmt = std::move(new_parent_stmt);
    child_sref = child_sref->parent;
  }
  // Step 3. Handle the case that we mutate the root
  if (child_sref->parent == nullptr) {
    // From the loop invariant, upon exit, while its subtree is properly set,
    // `child_sref` is not properly to `child_tgt_stmt` yet.
    if (src_sref->parent != nullptr) {
      // Not replacing a root
      UpdateSRef(this, child_sref, child_tgt_stmt.get());
    }
    // Update the body of the `this->mod`
    IRModuleNode* new_mod = this->mod.CopyOnWrite();
    MapNode* new_map = new_mod->functions.CopyOnWrite();
    PrimFunc ref_new_func = Downcast<PrimFunc>(std::move(new_map->at(g_var)));
    ICHECK(ref_new_func.get() == g_func);
    PrimFuncNode* new_func = ref_new_func.CopyOnWrite();
    new_mod->functions.Set(g_var, ref_new_func);
    // Assign `child_tgt_stmt`, which is a Block, to the root block
    const auto* realize = TVM_TYPE_AS(realize, g_func->body, BlockRealizeNode);
    const auto* child_block = TVM_TYPE_AS(child_block, child_tgt_stmt, BlockNode);
    ObjectPtr<BlockRealizeNode> new_realize = make_object<BlockRealizeNode>(*realize);
    new_realize->block = GetRef<Block>(child_block);
    new_func->body = BlockRealize(std::move(new_realize));
    this->mod = GetRef<IRModule>(new_mod);
  }
  if (this->debug_mode) {
    VerifySRefTree(GetRef<ScheduleState>(this));
  }
}

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(ScheduleStateNode);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleState")
    .set_body_typed([](ObjectRef obj, bool debug_mode) {
      if (const auto* func = obj.as<PrimFuncNode>()) {
        return ScheduleState(GetRef<PrimFunc>(func), debug_mode);
      }
      if (const auto* mod = obj.as<IRModuleNode>()) {
        return ScheduleState(GetRef<IRModule>(mod), debug_mode);
      }
      LOG(FATAL) << "TypeError: Expects `IRModule` or `PrimFunc`, but gets: " << obj->GetTypeKey();
      throw;
    });
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleStateReplace")
    .set_body_method<ScheduleState>(&ScheduleStateNode::Replace);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleStateGetSRef")
    .set_body_typed([](ScheduleState self, Stmt stmt) -> Optional<StmtSRef> {
      auto it = self->stmt2ref.find(stmt.get());
      return it != self->stmt2ref.end() ? it->second : Optional<StmtSRef>(NullOpt);
    });
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleStateGetScope")
    .set_body_typed([](ScheduleState self, StmtSRef block_sref) -> Optional<BlockScope> {
      auto it = self->scopes.find(block_sref);
      return it != self->scopes.end() ? (*it).second : Optional<BlockScope>(NullOpt);
    });

}  // namespace tir
}  // namespace tvm
