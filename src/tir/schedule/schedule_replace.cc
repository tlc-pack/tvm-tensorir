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
#include "./schedule_common.h"

namespace tvm {
namespace tir {

/*!
 * \brief Set the `StmtSRefNode::seq_index` field for stmt
 * \param self The schedule class
 * \param stmt The statement, or the realize node of the statement whose sref to be set
 * \param seq_index The seq_index to be set
 */
void SetSeqIndex(ScheduleNode* self, const Stmt& stmt, int seq_index) {
  if (const auto* realize = stmt.as<BlockRealizeNode>()) {
    const BlockNode* block = realize->block.get();
    CHECK(self->stmt2ref.count(block));
    self->stmt2ref.at(block)->seq_index = seq_index;
  } else if (const auto* block = stmt.as<BlockNode>()) {
    CHECK(self->stmt2ref.count(block));
    self->stmt2ref.at(block)->seq_index = seq_index;
  } else if (const auto* loop = stmt.as<LoopNode>()) {
    CHECK(self->stmt2ref.count(loop));
    self->stmt2ref.at(loop)->seq_index = seq_index;
  } else if (stmt->IsInstance<IfThenElseNode>() || stmt->IsInstance<BufferStoreNode>() ||
             stmt->IsInstance<EvaluateNode>()) {
    // do nothing
  } else {
    LOG(FATAL) << "TypeError: Unexpected type: " << stmt->GetTypeKey();
    throw;
  }
}

/*!
 * \brief Update the sref information on the schedule class, as well as the statement of sref itself
 * More specifically, update
 *  `sref->stmt` to `new_stmt`
 *  `sch->stmt2ref`, remove the old statement that sref points to, and add the new statement
 * \param sch The schedule class to be updated
 * \param sref The sref to be updated
 * \param new_stmt The statement that replaces the statement inside the sref
 */
void UpdateSRef(ScheduleNode* sch, StmtSRefNode* sref, const StmtNode* new_stmt) {
  CHECK(new_stmt->IsInstance<BlockNode>() || new_stmt->IsInstance<LoopNode>());
  const StmtNode* old_stmt = sref->stmt;
  CHECK_NE(new_stmt, old_stmt);
  sch->stmt2ref[new_stmt] = GetRef<StmtSRef>(sref);
  sch->stmt2ref.erase(sref->stmt);
  sref->stmt = new_stmt;
}

/*!
 * \brief To a specific sref, count the number of steps to its highest non-unique ancestor.
 *
 * If the sref itself is the highest, then the result is 0.
 * If the parent of the sref is the highest, then the result is 1.
 * ...
 */
int StepsHighestNonUniqueAncestor(const StmtSRef& sref, bool func_is_unique) {
  int result = -1;
  int i = 0;
  for (const StmtSRefNode* ptr = sref.get(); ptr != nullptr; ptr = ptr->parent, ++i) {
    if (!ptr->stmt->unique()) {
      result = i;
    }
  }
  // If the function itself is not unique, then we assume the root is not unique
  if (!func_is_unique) {
    result = i;
  }
  return result;
}

/*!
 * \brief For a new AST, the SRefCreator creates 3 kinds of srefs
 * 1) Reused: it means we found a correspondence between a stmt to an old one
 * (although they are not the same object), and thus decide to reuse that old sref
 * 2) Intact: it means we found an old statement, i.e. the same object,
 * it means we do not even need to visit into the subtree of the old statement,
 * which is intact.
 * 3) New: a complete new stmt which has a completely new sref
 */
struct CreationInfo {
  /*! \brief The srefs that are reused */
  std::unordered_set<StmtSRef, ObjectPtrHash, ObjectPtrEqual> reused;
  /*! \brief The srefs whose subtrees that are unchanged and their parents */
  std::unordered_map<StmtSRef, StmtSRefNode*, ObjectPtrHash, ObjectPtrEqual> intact;
};

/*!
 * \brief A helper that creates new srefs for newly-added blocks and loops.
 *
 * Algorithm:
 *   1) Recursively visit the AST to be replaced to
 *   2) If a node is already tracked in `ScheduleNode::stmt2ref`,
 *   then stop recursion because the entire subtree has been properly tracked.
 *   In this case, set `used_border_parent_` of this node to its parent recorded in the recursion,
 *   3) If not, it means we need to either reuse an old sref or create a new sref
 *   (a) If the loop/block to be replaced proves to be a subtitute of an old one,
 *   then reuse the existing sref to make sure it won't expire on users' side
 *   (b) Otherwise, create a new sref
 *
 * Change:
 *   `ScheduleNode::stmt2ref` and `ScheduleNode::scopes`.
 *
 * Assumption:
 *  If the root is block/loop, then it is not intact
 *
 * Effect of the visitor:
 *   1) For the block/loop root:
 *   - `StmtSRefNode::parent` is set properly
 *   - `StmtSRefNode::seq_index` is set to -1
 *   2) For internal nodes:
 *   - `StmtSRefNode::parent` is set properly
 *   - `StmtSRefNode::seq_index` is set properly
 */
class SRefCreator : public StmtVisitor {
 public:
  static CreationInfo Create(ScheduleNode* self, const Map<Block, Block>& block_reuse,
                             StmtSRefNode* parent, const Stmt& new_stmt) {
    // For each loop var, find its corresponding sref
    // `block_reuse` and `loop_reuse` work together providing information to detect sref reuse
    std::unordered_map<const VarNode*, StmtSRef> loop_reuse;
    loop_reuse.reserve(self->stmt2ref.size());
    for (const auto& iter : self->stmt2ref) {
      const StmtNode* stmt = iter.first;
      const StmtSRef& sref = iter.second;
      if (stmt->IsInstance<tir::LoopNode>()) {
        const LoopNode* loop = static_cast<const LoopNode*>(stmt);
        loop_reuse.emplace(loop->loop_var.get(), sref);
      }
    }
    // Then construct a SRefCreator
    SRefCreator creator(self, block_reuse, loop_reuse, parent);
    creator(new_stmt);
    return std::move(creator.result);
  }

 private:
  explicit SRefCreator(ScheduleNode* self,                                              //
                       const Map<Block, Block>& block_reuse,                            //
                       const std::unordered_map<const VarNode*, StmtSRef>& loop_reuse,  //
                       StmtSRefNode* parent)
      : self(self), block_reuse(block_reuse), loop_reuse(loop_reuse), parents({parent}) {}

  void VisitStmt_(const LoopNode* op) override {
    StmtSRef& sref = self->stmt2ref[op];
    StmtSRefNode* parent = parents.back();
    // Case 1. The subtree has been tracked by the stmt2ref
    if (sref.defined()) {
      sref->seq_index = -1;
      result.intact.emplace(sref, parent);
      return;
    }
    // Case 2. We are replace an existing loop,
    // reuse the existing sref so that users don't get an expired one
    auto it = loop_reuse.find(op->loop_var.get());
    if (it != loop_reuse.end()) {
      sref = it->second;
      sref->stmt = op;
      sref->parent = parent;
      sref->seq_index = -1;
      result.reused.insert(sref);
    } else {
      // Case 3. Replacing an existing loop with a new one
      sref = StmtSRef(op, parent, /*seq_index=*/-1, /*binding_valid=*/true);
    }
    parents.push_back(sref.get());
    VisitStmt(op->body);
    parents.pop_back();
  }

  void VisitStmt_(const BlockNode* op) override {
    StmtSRef& sref = self->stmt2ref[op];
    StmtSRefNode* parent = parents.back();
    // Case 1. The subtree has been tracked by the stmt2ref
    if (sref.defined()) {
      sref->seq_index = -1;
      result.intact.emplace(sref, parent);
      return;
    }
    // Case 2. We are replace an existing block,
    // reuse the existing sref so that users don't get an expired one
    auto it = block_reuse.find(GetRef<Block>(op));
    if (it != block_reuse.end()) {
      sref = self->stmt2ref.at((*it).second.get());
      sref->stmt = op;
      sref->parent = parent;
      sref->seq_index = -1;
      result.reused.insert(sref);
    } else {
      // Case 3. Replacing an existing block with a new one
      sref = StmtSRef(op, parent, /*seq_index=*/-1, /*binding_valid=*/true);
    }
    parents.push_back(sref.get());
    VisitStmt(op->body);
    parents.pop_back();
    // Additionally, need to update the scope because the block is changed
    UpdateScope(op, self->stmt2ref, &self->scopes);
  }

  void VisitStmt_(const SeqStmtNode* op) override {
    StmtVisitor::VisitStmt_(op);
    // Update seq_index of children
    for (int i = 0, n = op->seq.size(); i < n; ++i) {
      SetSeqIndex(self, op->seq[i], i);
    }
  }

  ScheduleNode* self;
  const Map<Block, Block>& block_reuse;
  const std::unordered_map<const VarNode*, StmtSRef>& loop_reuse;
  std::vector<StmtSRefNode*> parents;

  CreationInfo result;
};

/*!
 * \brief A helper that
 * 1) returns a new copy of `child_sref->parent->stmt`,
 * where `child_sref->stmt` is replaced with `child_stmt`.
 * 2) Then it points `child_sref` to `child_stmt`.
 * 3) Finally it makes the subtree of the srefs pointing to the returned AST correct,
 * except for the root.
 */
class ChildReplacer : private StmtMutator {
 public:
  static Stmt Mutate(ScheduleNode* self, StmtSRefNode* child_sref, const Stmt& child_stmt,
                     bool allow_copy_on_write, bool update_child_sref) {
    Stmt* body = nullptr;
    Stmt result{nullptr};
    // Step 1. Copy-on-write the `parent_stmt` and extract its `body`
    const StmtNode* parent_stmt = child_sref->parent->stmt;
    if (parent_stmt->IsInstance<BlockNode>()) {
      const auto* block = static_cast<const BlockNode*>(parent_stmt);
      ObjectPtr<BlockNode> new_block = CopyOnWrite(block, allow_copy_on_write);
      body = &new_block->body;
      result = Block(std::move(new_block));
    } else if (parent_stmt->IsInstance<LoopNode>()) {
      const auto* loop = static_cast<const LoopNode*>(parent_stmt);
      ObjectPtr<LoopNode> new_loop = CopyOnWrite(loop, allow_copy_on_write);
      body = &new_loop->body;
      result = Loop(std::move(new_loop));
    } else {
      LOG(FATAL) << "TypeError: Unexpected type: " << parent_stmt->GetTypeKey();
      throw;
    }
    // Step 2. Mutate the `result->body`, searching for `child_sref->stmt`
    // and replace it with `child_stmt`
    *body = ChildReplacer(child_sref->stmt, child_stmt, allow_copy_on_write).VisitStmt(*body);
    // Step 3. Link `child_sref` to `child_stmt`
    if (update_child_sref) {
      UpdateSRef(self, child_sref, child_stmt.get());
    }
    // Step 4. Make `seq_index` correct for the subtree (except for the root)
    ResetSeqIndex(self, *body);
    return result;
  }

 private:
  static void ResetSeqIndex(ScheduleNode* self, const Stmt& stmt) {
    class SeqIndexResetter : public StmtVisitor {
     public:
      explicit SeqIndexResetter(ScheduleNode* self) : self(self) {}
      void VisitStmt_(const BlockNode* op) override { self->stmt2ref.at(op)->seq_index = -1; }
      void VisitStmt_(const LoopNode* op) override { self->stmt2ref.at(op)->seq_index = -1; }
      void VisitStmt_(const SeqStmtNode* op) override {
        int i = 0;
        for (const Stmt& stmt : op->seq) {
          SetSeqIndex(self, stmt, i);
          ++i;
        }
      }
      ScheduleNode* self;
    };
    (SeqIndexResetter(self))(stmt);
  }

  explicit ChildReplacer(const StmtNode* src_stmt, const Stmt& tgt_stmt, bool allow_copy_on_write)
      : src_stmt(src_stmt), tgt_stmt(tgt_stmt) {
    this->allow_copy_on_write_ = allow_copy_on_write;
  }

  Stmt VisitStmt(const Stmt& stmt) override {
    if (stmt.get() == src_stmt) {
      // if the statement matches the replace tgt_stmt
      // just return the tgt_stmt
      return tgt_stmt;
    } else {
      return StmtMutator::VisitStmt(stmt);
    }
  }

  Stmt VisitStmt_(const BlockNode* op) override { return GetRef<Stmt>(op); }
  Stmt VisitStmt_(const LoopNode* op) override { return GetRef<Stmt>(op); }

  const StmtNode* src_stmt;
  const Stmt& tgt_stmt;

  template <typename TNode>
  static ObjectPtr<TNode> CopyOnWrite(const TNode* node, bool allow_copy_on_write) {
    return allow_copy_on_write ? runtime::GetObjectPtr<TNode>(const_cast<TNode*>(node))
                               : runtime::make_object<TNode>(*node);
  }
};

/*!
 * \brief A helper that removes stale srefs that are useless after the replacement
 *
 * Algorithm:
 *   1) Recursively visit the AST to be replaced
 *   2) If a node is already marked as `intact subtree`,
 *   it means it won't be affected,
 *   so we set its parent and return.
 *   3) If a node is not reused, then set its `stmt` and `parent` fields to nullptr,
 *   indicating that it has expired.
 *
 * Change:
 *   `ScheduleNode::stmt2ref` and `ScheduleNode::scopes`.
 */
class SRefRemover : public StmtVisitor {
 public:
  static void Remove(ScheduleNode* self, const CreationInfo& info, const Stmt& stmt) {
    SRefRemover remover(self, info);
    remover(stmt);
  }

 private:
  explicit SRefRemover(ScheduleNode* self, const CreationInfo& info) : self(self), info(info) {}

  bool CheckIntactSubtree(const StmtSRef& sref) const {
    auto itr = info.intact.find(sref);
    if (itr == info.intact.end()) {
      return false;
    }
    sref->parent = itr->second;
    return true;
  }

  bool CheckReused(const StmtSRef& sref) const {
    if (info.reused.count(sref)) {
      return true;
    }
    sref->stmt = nullptr;
    sref->parent = nullptr;
    return false;
  }

  void VisitStmt_(const LoopNode* op) override {
    StmtSRef sref = self->stmt2ref.at(op);
    if (CheckIntactSubtree(sref)) {
      return;
    }
    CheckReused(sref);
    self->stmt2ref.erase(op);
    VisitStmt(op->body);
  }

  void VisitStmt_(const BlockNode* op) override {
    StmtSRef sref = self->stmt2ref.at(op);
    if (CheckIntactSubtree(sref)) {
      return;
    }
    if (!CheckReused(sref)) {
      self->scopes.erase(sref);
    }
    self->stmt2ref.erase(op);
    VisitStmt(op->body);
  }

  ScheduleNode* self;
  CreationInfo info;
};

void ScheduleNode::Replace(const StmtSRef& _src_sref, const Stmt& tgt_stmt,
                           const Map<Block, Block>& block_reuse) {
  {
    const StmtNode* src_stmt = _src_sref->stmt;
    bool input_correct =
        (src_stmt->IsInstance<LoopNode>() && tgt_stmt->IsInstance<LoopNode>()) ||
        (src_stmt->IsInstance<LoopNode>() && tgt_stmt->IsInstance<BlockRealizeNode>()) ||
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
  const StmtNode* root_stmt = this->root->stmt;
  // Step 1. Create all the nodes needed for the new sref tree.
  //   The `SRefCreator` visits the AST `tgt_stmt`, creating new nodes along the way.
  //   It deals with 3 cases:
  //
  //   Case 1.1: Visiting a node already present in the old AST
  //     It means we can skip the entire subtree, leaving it untouched
  //     Mark those nodes as `intact`
  //   Case 1.2: Can somehow infer the node being visited is mutated from the old AST, including
  //     (a) The loop var appears in the old AST
  //     (b) The node is explicitly present in `block_reuse`
  //     It means we need to retain and reuse the sref, so that those srefs users hold won't expire
  //     Mark those nodes as `reuse`
  //   Case 1.3: It is a completely new node
  //     Create a new node for it
  //
  // After creation, it is guaranteed that
  //   all the srefs in the AST `tgt_stmt` have proper `parent`s, except for those `intact` nodes.
  CreationInfo creation_info = SRefCreator::Create(this, block_reuse, src_sref->parent, tgt_stmt);
  // Step 2. Set the ancestors' children properly
  //   Iteratively visit the ancestors, creating new ones whose `body`s are properly fixed.
  //   The visit stops when all the ancestors are uniquely referenced, i.e. can mutate inplace.
  //   Along the way, because we create a new ancestor path,
  //   we need to update those sref points from old ancestors to newly created ones
  // `num_copy_steps` is the maximum number of hops until we need to copy
  // To reach a node that can be mutated in-place, it needs `num_copy_steps + 1` hops
  int num_copy_steps =
      StepsHighestNonUniqueAncestor(src_sref, /*func_is_unique=*/this->func.unique());
  // Loop invariant:
  //
  // Before step `i`:
  // 1) `child_sref` is `src_sref` going up by `i` steps
  // 2) `child_stmt` is the subtree that `child_sref` should correspond to after replacement
  // 3) except for the subtree root, all srefs that point to the subtree of `child_stmt` are correct
  // 4) for the subtree root of `child_stmt`, `child_sref` has not pointed to it yet
  // 5) `tgt_stmt` is of type Loop, Block or BlockRealize
  //
  // During step `i`:
  // 1) Create `parent_stmt` that corresponds to `child_sref->parent
  // 2) Point `child_sref` to `child_stmt`
  // 3) `tgt_stmt` is of type Loop or Block
  StmtSRefNode* child_sref = src_sref.get();
  Stmt child_stmt = std::move(tgt_stmt);
  for (int i = 0; i <= num_copy_steps && child_sref->stmt != root_stmt; ++i) {
    bool parent_unique = (i == num_copy_steps);
    // Step 2.1. Create `parent_stmt`, by mutating the body of `parent_sref->stmt`,
    // replacing `child_sref->stmt` to `child_stmt`.
    // Step 2.2. Link `child_sref` to `child_stmt`
    Stmt parent_stmt =
        ChildReplacer::Mutate(this, child_sref, child_stmt,
                              /*allow_copy_on_write=*/parent_unique, /*update_child_sref=*/i != 0);
    // Step 2.3. Go to next parent
    if (parent_unique) {
      // If the node can be directly mutated inplace,
      // then there is no need to update its parent and the function
      break;
    }
    child_stmt = std::move(parent_stmt);
    child_sref = child_sref->parent;
  }
  // Step 3. Remove the statements from the old AST that are not used any more
  // i.e. those are not reused or intact
  SRefRemover::Remove(this, creation_info, src_stmt);
  // Step 4. Handle the case that we mutate the root
  if (child_sref->stmt == root_stmt) {
    // From the loop invariant, upon exit, while its subtree is properly set,
    // `child_sref` is not properly to `child_stmt` yet.
    if (src_sref->stmt == root_stmt) {
      // Replacing the root
      this->root = this->stmt2ref.at(child_stmt.get());
    } else {
      // Replacing a non-root
      UpdateSRef(this, this->root.get(), child_stmt.get());
    }
    // Update the body of the `this->func`
    PrimFuncNode* new_func = this->func.CopyOnWrite();
    // Assign `child_stmt`, which is a Block, to the root block
    const auto* realize = TVM_TYPE_AS(realize, func->body, BlockRealizeNode);
    const auto* child_block = TVM_TYPE_AS(child_block, child_stmt, BlockNode);
    ObjectPtr<BlockRealizeNode> new_realize = make_object<BlockRealizeNode>(*realize);
    new_realize->block = GetRef<Block>(child_block);
    new_func->body = BlockRealize(std::move(new_realize));
    this->func = GetRef<PrimFunc>(new_func);
  }
  // TODO(@junrushao1994): provide a configurable way to turn it on
  // this->ValidateSRef();
}

struct Internal {
  static void Replace(Schedule self, StmtSRef src_sref, Stmt tgt_stmt,
                      Optional<Map<Block, Block>> block_reuse) {
    return self->Replace(src_sref, tgt_stmt, block_reuse.value_or({}));
  }
};

TVM_REGISTER_GLOBAL("tir.schedule.Replace").set_body_typed(Internal::Replace);

}  // namespace tir
}  // namespace tvm
