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

struct Info {
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
 */
class SRefCreator : public StmtVisitor {
 public:
  static Info Create(ScheduleNode* self, const Map<Block, Block>& block_sref_map,
                     StmtSRefNode* parent, const Stmt& new_stmt) {
    // For each loop var, find its corresponding sref
    // `block_sref_map` and `loop_var2sref` work together providing information to detect sref reuse
    std::unordered_map<const VarNode*, StmtSRef> loop_var2sref;
    loop_var2sref.reserve(self->stmt2ref.size());
    for (const auto& iter : self->stmt2ref) {
      const StmtNode* stmt = iter.first;
      const StmtSRef& sref = iter.second;
      if (stmt->IsInstance<tir::LoopNode>()) {
        const LoopNode* loop = static_cast<const LoopNode*>(stmt);
        loop_var2sref.emplace(loop->loop_var.get(), sref);
      }
    }
    // Then construct a SRefCreator
    SRefCreator creator(self, block_sref_map, loop_var2sref, parent);
    creator(new_stmt);
    return std::move(creator.result);
  }

 private:
  explicit SRefCreator(ScheduleNode* self,                                                 //
                       const Map<Block, Block>& block_sref_map,                            //
                       const std::unordered_map<const VarNode*, StmtSRef>& loop_var2sref,  //
                       StmtSRefNode* parent)
      : self(self),
        block_sref_map(block_sref_map),
        loop_var2sref(loop_var2sref),
        parents({parent}) {}

  void VisitStmt_(const LoopNode* op) override {
    StmtSRef& sref = self->stmt2ref[op];
    StmtSRefNode* parent = parents.back();
    // Case 1. The subtree has been tracked by the stmt2ref
    if (sref.defined()) {
      result.intact.emplace(sref, parent);
      return;
    }
    // Case 2. We are replace an existing loop,
    // reuse the existing sref so that users don't get an expired one
    auto it = loop_var2sref.find(op->loop_var.get());
    if (it != loop_var2sref.end()) {
      sref = it->second;
      sref->stmt = op;
      sref->parent = parent;
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
      result.intact.emplace(sref, parent);
      return;
    }
    // Case 2. We are replace an existing block,
    // reuse the existing sref so that users don't get an expired one
    auto it = block_sref_map.find(GetRef<Block>(op));
    if (it != block_sref_map.end()) {
      sref = self->stmt2ref.at((*it).second.get());
      sref->stmt = op;
      sref->parent = parent;
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

  ScheduleNode* self;
  const Map<Block, Block>& block_sref_map;
  const std::unordered_map<const VarNode*, StmtSRef>& loop_var2sref;
  std::vector<StmtSRefNode*> parents;

  Info result;
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
  static void Remove(ScheduleNode* self, const Info& info, const Stmt& stmt) {
    SRefRemover remover(self, info);
    remover(stmt);
  }

 private:
  explicit SRefRemover(ScheduleNode* self, const Info& info) : self(self), info(info) {}

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
  Info info;
};

class ParentMutator : protected StmtMutator {
 public:
  static Stmt Mutate(ScheduleNode* self, StmtSRefNode* src_sref, const Stmt& tgt_stmt,
                     bool allow_copy_on_write) {
    ParentMutator mutator(self, src_sref, tgt_stmt);
    mutator.allow_copy_on_write_ = allow_copy_on_write;
    const StmtNode* parent_stmt = src_sref->parent->stmt;
    if (parent_stmt->IsInstance<LoopNode>()) {
      return mutator.VisitStmt_(static_cast<const LoopNode*>(parent_stmt));
    } else if (parent_stmt->IsInstance<BlockNode>()) {
      return mutator.VisitStmt_(static_cast<const BlockNode*>(parent_stmt));
    }
    LOG(FATAL) << "TypeError: Unknown type: " << (parent_stmt ? "None" : parent_stmt->GetTypeKey());
    throw;
  }

 private:
  explicit ParentMutator(ScheduleNode* self, StmtSRefNode* src_sref, const Stmt& tgt_stmt)
      : self(self), src_sref(src_sref), tgt_stmt(tgt_stmt), is_first_stmt(true) {}

  Stmt VisitStmt(const Stmt& stmt) override {
    if (stmt.get() == src_sref->stmt) {
      // if the statement matches the replace tgt_stmt
      // just return the tgt_stmt
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
    int i = src_sref->seq_index;
    // fast path
    if (i >= 0 && is_same(stmt->seq[i], src_sref->stmt)) {
      ObjectPtr<SeqStmtNode> n = CopyOnWrite(stmt);
      if (const auto* tgt_stmt_seq = tgt_stmt.as<SeqStmtNode>()) {
        const Array<Stmt>& target_seq = tgt_stmt_seq->seq;
        n->seq.erase(n->seq.begin() + i);
        n->seq.insert(n->seq.begin() + i, target_seq.begin(), target_seq.end());
        for (const Stmt& stmt : target_seq) {
          self->stmt2ref[stmt.get()]->seq_index = i++;
        }
      } else {
        n->seq.Set(i, tgt_stmt);
      }
      return Stmt(n);
    } else {
      return StmtMutator::VisitStmt_(stmt);
    }
  }

 private:
  static bool is_same(const Stmt& son, const StmtNode* expected) {
    if (son->IsInstance<LoopNode>()) {
      return son.get() == expected;
    } else {
      const auto* ptr = son.as<BlockRealizeNode>();
      CHECK(ptr != nullptr);
      return ptr->block.get() == expected;
    }
  }

  ScheduleNode* self;
  StmtSRefNode* src_sref;
  const Stmt& tgt_stmt;

  bool is_first_stmt;
};

void ScheduleNode::Replace(StmtSRef sref, Stmt tgt_stmt, const Map<Block, Block>& block_sref_map) {
  // Reset sref as a new sref so that its content won't be affected by subsequent changes
  sref = StmtSRef(sref->stmt, sref->parent);
  Stmt src_stmt = GetRef<Stmt>(sref->stmt);
  const StmtNode* root_stmt = this->root->stmt;
  // Create sref tree for the incoming tgt_stmt
  Info creation_info = SRefCreator::Create(this, block_sref_map, sref->parent, tgt_stmt);
  // The maximum number of hops until we don't need to copy
  int num_copy_steps = StepsHighestNonUniqueAncestor(sref, /*func_is_unique=*/this->func.unique());
  StmtSRefNode* src_sref = sref.get();
  for (int i = 0; i <= num_copy_steps && src_sref->stmt != root_stmt; ++i) {
    bool parent_is_uniquely_referenced = (i == num_copy_steps);
    // Within the child/children of `src_sref->parent`,
    // replace `src_sref` with `tgt_stmt`,
    // and return the new parent stmt
    Stmt new_parent_stmt =
        ParentMutator::Mutate(this, src_sref, tgt_stmt, parent_is_uniquely_referenced);
    if (i != 0) {
      // If `i == 0`, `src_sref` is `sref`, a local temporary object and we should not update it
      UpdateSRef(this, src_sref, tgt_stmt.get());
    }
    tgt_stmt = new_parent_stmt;
    if (parent_is_uniquely_referenced) {
      // If the node can be directly mutated inplace,
      // then there is no need to update its parent and the function
      break;
    }
    src_sref = src_sref->parent;
  }
  SRefRemover::Remove(this, creation_info, src_stmt);
  if (src_sref->stmt == root_stmt) {
    if (sref->stmt == root_stmt) {
      // Replacing the root is easier
      this->root = this->stmt2ref[tgt_stmt.get()];
    } else {
      // Replacing a non-root
      UpdateSRef(this, this->root.get(), tgt_stmt.get());
    }
    // Update the body of the `this->func`
    this->func = UpdatePrimFunc(&this->func, tgt_stmt);
  }
}

struct Internal {
  static void Replace(Schedule self, StmtSRef sref, Stmt tgt_stmt,
                      Optional<Map<Block, Block>> block_sref_map) {
    return self->Replace(sref, tgt_stmt, block_sref_map.value_or({}));
  }
};

TVM_REGISTER_GLOBAL("tir.schedule.Replace").set_body_typed(Internal::Replace);

}  // namespace tir
}  // namespace tvm
