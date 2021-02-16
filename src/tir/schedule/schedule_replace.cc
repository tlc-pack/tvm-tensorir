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
 * \brief Update the body of the PrimFunc
 * \param func The PrimFunc to be updated
 * \param new_body The new body to be updated to
 * \return The new PrimFunc
 */
PrimFunc UpdatePrimFunc(PrimFunc* func, const Stmt& new_body) {
  const auto* realize = (*func)->body.as<BlockRealizeNode>();
  const auto* block = new_body.as<BlockNode>();
  CHECK(realize);
  CHECK(block);
  ObjectPtr<BlockRealizeNode> new_realize = make_object<BlockRealizeNode>(*realize);
  PrimFuncNode* new_func = func->CopyOnWrite();
  new_realize->block = GetRef<Block>(block);
  new_func->body = BlockRealize(new_realize);
  return GetRef<PrimFunc>(new_func);
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
  if (!func_is_unique) {
    result = i;
  }
  return result;
}

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
 */
class SRefCreator : public StmtVisitor {
 public:
  explicit SRefCreator(ScheduleNode* self, const Map<Block, Block>& block_sref_map,
                       StmtSRefNode* parent)
      : self(self), parents({parent}), block_sref_map(block_sref_map) {
    // Set `loop_var2sref` properly
    loop_var2sref.reserve(self->stmt2ref.size());
    for (const auto& iter : self->stmt2ref) {
      const StmtNode* stmt = iter.first;
      const StmtSRef& sref = iter.second;
      if (stmt->IsInstance<tir::LoopNode>()) {
        const LoopNode* loop = static_cast<const LoopNode*>(stmt);
        loop_var2sref.emplace(loop->loop_var.get(), sref);
      }
    }
  }

  void VisitStmt_(const LoopNode* op) override {
    StmtSRef& sref = self->stmt2ref[op];
    StmtSRefNode* parent = parents.back();
    // Case 1. The subtree has been tracked by the stmt2ref
    if (sref.defined()) {
      used_border_parent_[sref] = parent;
      return;
    }
    // Case 2. We are replace an existing loop,
    // reuse the existing sref so that users don't get an expired one
    auto it = loop_var2sref.find(op->loop_var.get());
    if (it != loop_var2sref.end()) {
      sref = it->second;
      sref->stmt = op;
      sref->parent = parent;
      reuse_sref_.insert(sref);
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
      used_border_parent_[sref] = parent;
      return;
    }
    // Case 2. We are replace an existing block,
    // reuse the existing sref so that users don't get an expired one
    auto it = block_sref_map.find(GetRef<Block>(op));
    if (it != block_sref_map.end()) {
      sref = self->stmt2ref.at((*it).second.get());
      sref->stmt = op;
      sref->parent = parent;
      reuse_sref_.insert(sref);
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

  ScheduleNode* self;
  std::vector<StmtSRefNode*> parents;
  const Map<Block, Block>& block_sref_map;
  std::unordered_map<const VarNode*, StmtSRef> loop_var2sref;

  std::unordered_set<StmtSRef, ObjectPtrHash, ObjectPtrEqual> reuse_sref_;
  std::unordered_map<StmtSRef, StmtSRefNode*, ObjectPtrHash, ObjectPtrEqual> used_border_parent_;
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
  explicit SRefRemover(ScheduleNode* self,
                       std::unordered_map<StmtSRef, StmtSRefNode*, ObjectPtrHash, ObjectPtrEqual>&&
                           used_border_parent,
                       std::unordered_set<StmtSRef, ObjectPtrHash, ObjectPtrEqual>&& reuse_sref)
      : self(self), reuse_sref_(reuse_sref), used_border_parent_(used_border_parent) {}

  bool CheckIntactSubtree(const StmtSRef& sref) const {
    auto itr = used_border_parent_.find(sref);
    if (itr == used_border_parent_.end()) {
      return false;
    }
    sref->parent = itr->second;
    return true;
  }

  bool CheckReused(const StmtSRef& sref) const {
    if (reuse_sref_.count(sref)) {
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

 private:
  ScheduleNode* self;
  std::unordered_set<StmtSRef, ObjectPtrHash, ObjectPtrEqual> reuse_sref_;
  std::unordered_map<StmtSRef, StmtSRefNode*, ObjectPtrHash, ObjectPtrEqual> used_border_parent_;
};

class SubReplacer : protected StmtMutator {
 public:
  explicit SubReplacer(StmtSRefNode* sref, const Stmt& target,
                       std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref)
      : src_sref(sref), tgt_stmt(target), stmt2ref_(stmt2ref), is_first_stmt(true) {}

  /*!
   * \brief mutate weakref
   * \param weakref The statement to be mutated.
   * \param allow_copy_on_write Whether we allow copy on write in the weakref.
   *        That means weakref is only referenced once, and all its
   *        parents are also only referenced once.
   * \return The result of the mutation.
   */
  Stmt operator()(const StmtNode* weakref, bool allow_copy_on_write) {
    this->allow_copy_on_write_ = allow_copy_on_write;
    Stmt stmt{nullptr};
    if (weakref->IsInstance<LoopNode>()) {
      return StmtMutator::VisitStmt_(static_cast<const LoopNode*>(weakref));
    } else if (weakref->IsInstance<BlockNode>()) {
      return StmtMutator::VisitStmt_(static_cast<const BlockNode*>(weakref));
    }
    LOG(FATAL) << "Unreachable";
    throw;
  }

  Stmt VisitStmt(const Stmt& stmt) override {
    if (stmt.get() == src_sref->stmt) {
      // if the statement matches the replace target
      // just return the target stmt
      return tgt_stmt;
    } else {
      return StmtMutator::VisitStmt(stmt);
    }
  }

  Stmt VisitStmt_(const BlockNode* op) override {
    if (is_first_stmt) {
      is_first_stmt = false;
      // It is okay to visit the init block now, because it won't take any effect
      return StmtMutator::VisitStmt_(op);
    } else {
      return GetRef<Stmt>(op);
    }
  }

  Stmt VisitStmt_(const LoopNode* op) override {
    if (is_first_stmt) {
      is_first_stmt = false;
      return StmtMutator::VisitStmt_(op);
    } else {
      return GetRef<Stmt>(op);
    }
  }

  Stmt VisitStmt_(const SeqStmtNode* stmt) override {
    int64_t seq_index = src_sref->seq_index;
    // fast path
    if (seq_index >= 0 && is_son(stmt->seq[seq_index], src_sref->stmt)) {
      auto n = CopyOnWrite(stmt);
      if (tgt_stmt->IsInstance<SeqStmtNode>()) {
        // note that nested SeqStmt is not allowed, so we flatten target here
        const Array<Stmt>& target_seq = tgt_stmt.as<SeqStmtNode>()->seq;
        n->seq.erase(n->seq.begin() + seq_index);
        n->seq.insert(n->seq.begin() + seq_index, target_seq.begin(), target_seq.end());
        for (size_t i = 0; i < target_seq.size(); i++)
          (*stmt2ref_)[target_seq[i].get()]->seq_index = i + seq_index;
      } else {
        n->seq.Set(seq_index, tgt_stmt);
      }
      return Stmt(n);
    } else {
      return StmtMutator::VisitStmt_(stmt);
    }
  }

 private:
  // target is Block/Loop, But son of SeqStmt may be the BlockRealize
  static bool is_son(const Stmt& son, const StmtNode* parent) {
    if (son.as<LoopNode>()) {
      return son.get() == parent;
    } else {
      const auto* ptr = son.as<BlockRealizeNode>();
      CHECK(ptr != nullptr);
      return ptr->block.get() == parent;
    }
  }

  StmtSRefNode* src_sref;
  const Stmt& tgt_stmt;
  std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref_;

  bool is_first_stmt;
};

void ScheduleNode::Replace(StmtSRef sref, Stmt tgt_stmt, Map<Block, Block> block_sref_map) {
  // Reset sref as a new sref so that its content won't be affected by subsequent changes
  sref = StmtSRef(sref->stmt, sref->parent);
  Stmt old_stmt = GetRef<Stmt>(sref->stmt);
  const StmtNode* root_stmt = this->root->stmt;
  // Create SRef tree for the incoming target Stmt
  // Initialize old SRef remover
  SRefCreator creator(this, block_sref_map, sref->parent);
  creator(tgt_stmt);
  SRefRemover remover(this, std::move(creator.used_border_parent_), std::move(creator.reuse_sref_));
  // The maximum number of hops until we don't need to copy
  int num_copy_steps = StepsHighestNonUniqueAncestor(sref, /*func_is_unique=*/this->func.unique());
  StmtSRefNode* src_sref = sref.get();
  for (int i = 0; i <= num_copy_steps && src_sref->stmt != root_stmt; ++i) {
    bool parent_is_uniquely_referenced = (i == num_copy_steps);
    StmtSRefNode* parent_sref = src_sref->parent;
    // Replace `src_sref` with target and return a new parent Stmt)
    Stmt new_stmt = SubReplacer(src_sref, tgt_stmt, &this->stmt2ref)(parent_sref->stmt,
                                                                     parent_is_uniquely_referenced);
    if (i != 0) {
      UpdateSRef(this, src_sref, tgt_stmt.get());
    }
    tgt_stmt = new_stmt;
    if (parent_is_uniquely_referenced) {
      // If the node can be directly mutated inplace,
      // then there is no need to update its parent and the function
      break;
    }
    src_sref = parent_sref;
  }
  remover(old_stmt);
  if (src_sref->stmt == root_stmt) {
    if (sref->stmt == root_stmt) {
      // Replacing the root is easier
      this->root = this->stmt2ref[tgt_stmt.get()];
    } else {
      // Replacing a non-root
      UpdateSRef(this, this->root.get(), tgt_stmt.get());
    }
    this->func = UpdatePrimFunc(&this->func, tgt_stmt);
  }
}

struct Internal {
  static void Replace(Schedule self, StmtSRef sref, Stmt target,
                      Optional<Map<Block, Block>> block_sref_map) {
    return self->Replace(sref, target, block_sref_map.value_or({}));
  }
};

TVM_REGISTER_GLOBAL("tir.schedule.Replace").set_body_typed(Internal::Replace);

}  // namespace tir
}  // namespace tvm
