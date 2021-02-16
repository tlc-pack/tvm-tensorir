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

class SubReplacer : protected StmtMutator {
 public:
  SubReplacer(StmtSRefNode* sref, const Stmt& target,
              std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref)
      : sref_(sref), target_(target), stmt2ref_(stmt2ref) {}
  /*!
   * \brief mutate weakref
   * \param weakref The statement to be mutated.
   * \param allow_copy_on_write Whether we allow copy on write in the weakref.
   *        That means weakref is only referenced once, and all its
   *        parents are also only referenced once.
   * \return The result of the mutation.
   */
  Stmt operator()(const StmtNode* weakref, bool allow_copy_on_write) {
    std::swap(allow_copy_on_write, allow_copy_on_write_);
    if (allow_copy_on_write_) {
      CHECK(weakref->unique()) << GetRef<Stmt>(weakref);
    }
    Stmt stmt;
    if (weakref->IsInstance<LoopNode>()) {
      stmt = StmtMutator::VisitStmt_(Downcast<Loop>(GetRef<Stmt>(weakref)).get());
    } else if (weakref->IsInstance<BlockNode>()) {
      stmt = StmtMutator::VisitStmt_(Downcast<Block>(GetRef<Stmt>(weakref)).get());
    } else {
      LOG(FATAL) << "StmtSRef only points to Block or Loop";
    }
    std::swap(allow_copy_on_write, allow_copy_on_write_);
    if (allow_copy_on_write) {
      CHECK(stmt.operator->() == weakref);
    }
    return stmt;
  }

  Stmt VisitStmt(const Stmt& stmt) final {
    if (stmt.get() == sref_->stmt) {
      // if the statement matches the replace target
      // just return the target stmt
      return target_;
    } else {
      return StmtMutator::VisitStmt(stmt);
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final { return VisitSRefStmt(op); }

  Stmt VisitStmt_(const LoopNode* op) final { return VisitSRefStmt(op); }

  Stmt VisitStmt_(const SeqStmtNode* stmt) final {
    int64_t seq_index = sref_->seq_index;
    // fast path
    if (seq_index >= 0 && is_son(stmt->seq[seq_index], sref_->stmt)) {
      auto n = CopyOnWrite(stmt);
      if (target_->IsInstance<SeqStmtNode>()) {
        // note that nested SeqStmt is not allowed, so we flatten target here
        const Array<Stmt>& target_seq = target_.as<SeqStmtNode>()->seq;
        n->seq.erase(n->seq.begin() + seq_index);
        n->seq.insert(n->seq.begin() + seq_index, target_seq.begin(), target_seq.end());
        for (size_t i = 0; i < target_seq.size(); i++)
          (*stmt2ref_)[target_seq[i].operator->()]->seq_index = i + seq_index;
      } else {
        n->seq.Set(seq_index, target_);
      }
      return Stmt(n);
    } else {
      return StmtMutator::VisitStmt_(stmt);
    }
  }

 private:
  template <typename T>
  Stmt VisitSRefStmt(const T* op) {
    if (sref_scope_counter_ > 0) {
      return GetRef<Stmt>(op);
    } else {
      ++sref_scope_counter_;
      return StmtMutator::VisitStmt_(op);
    }
  }

  // target is Block/Loop, But son of SeqStmt may be the BlockRealize
  static bool is_son(const Stmt& son, const StmtNode* target) {
    if (son.as<LoopNode>()) {
      return son.get() == target;
    } else {
      const auto* ptr = son.as<BlockRealizeNode>();
      CHECK(ptr != nullptr);
      return ptr->block.get() == target;
    }
  }

  // Node that this counter works for faster visiting.
  // We guarantee that each visit will only visit Schedulable
  // Stmt Node (BlockNode and LoopNode) once, the parent node.
  // As for its children, they can be either replaced or remain unchanged
  int sref_scope_counter_{0};
  StmtSRefNode* sref_;
  const Stmt& target_;
  std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref_;
};

/*!
 * \brief create schedulable reference during Schedule.Replace
 * \note This Visitor will create schedulable reference corresponding
 *       AST node in target stmt.
 */
class SRefCreator : public StmtVisitor {
 public:
  SRefCreator(ScheduleNode* self, const Map<Block, Block>& block_sref_map, StmtSRefNode* parent)
      : self(self), block_sref_map(block_sref_map), parents({parent}) {
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

  void VisitStmt_(const LoopNode* op) final {
    StmtSRef& sref = self->stmt2ref[op];
    if (sref.defined()) {
      used_border_parent_[sref] = parents.back();
      return;
    }
    // Create corresponding StmtSRef
    // note that we only create the StmtSRef whose node is not
    // in the AST and reuse those StmtSRef when node is in the AST.
    sref = CreateLoopSRef(op);
    parents.push_back(sref.operator->());
    VisitStmt(op->body);
    parents.pop_back();
  }

  void VisitStmt_(const BlockNode* op) final {
    StmtSRef& sref = self->stmt2ref[op];
    if (sref.defined()) {
      used_border_parent_[sref] = parents.back();
      return;
    }
    // Create corresponding StmtSRef
    // note that we only create the StmtSRef whose node is not
    // in the AST and reuse those StmtSRef when node is in the AST.
    sref = CreateBlockSRef(op);
    parents.push_back(sref.operator->());
    VisitStmt(op->body);
    parents.pop_back();
    UpdateScope(op, self->stmt2ref, &self->scopes);
  }

 private:
  StmtSRef CreateLoopSRef(const LoopNode* op) {
    StmtSRefNode* parent = parents.back();
    auto it = loop_var2sref.find(op->loop_var.get());
    if (it != loop_var2sref.end()) {
      StmtSRef reuse_sref = it->second;
      reuse_sref->stmt = op;
      reuse_sref->parent = parent;
      reuse_sref_.insert(reuse_sref);
      return reuse_sref;
    }
    return StmtSRef(op, parent, /*seq_index=*/-1, /*binding_valid=*/true);
  }

  StmtSRef CreateBlockSRef(const BlockNode* op) {
    StmtSRefNode* parent = parents.back();
    auto it = block_sref_map.find(GetRef<Block>(op));
    if (it != block_sref_map.end()) {
      StmtSRef reuse_sref = self->stmt2ref.at((*it).second.get());
      reuse_sref->stmt = op;
      reuse_sref->parent = parent;
      reuse_sref_.insert(reuse_sref);
      return reuse_sref;
    }
    return StmtSRef(op, parent, /*seq_index=*/-1, /*binding_valid=*/true);
  }

  friend class ScheduleNode;
  ScheduleNode* self;
  Map<Block, Block> block_sref_map;
  std::unordered_map<const VarNode*, StmtSRef> loop_var2sref;

  std::vector<StmtSRefNode*> parents;
  std::unordered_set<StmtSRef, ObjectPtrHash, ObjectPtrEqual> reuse_sref_;
  std::unordered_map<StmtSRef, StmtSRefNode*, ObjectPtrHash, ObjectPtrEqual> used_border_parent_;
};

/*!
 * \brief remove useless schedulable reference during Schedule.Replace
 * \note The Schedule.Replace will remove nodes from AST. This visitor will help to
 *       remove their schedulable reference.
 */
class SRefRemover : public StmtVisitor {
 public:
  SRefRemover(std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref,
              std::unordered_map<StmtSRef, StmtSRefNode*, ObjectPtrHash, ObjectPtrEqual>&&
                  used_border_parent,
              std::unordered_map<StmtSRef, Scope, ObjectPtrHash, ObjectPtrEqual>* block_scopes,
              std::unordered_set<StmtSRef, ObjectPtrHash, ObjectPtrEqual>&& reuse_sref)
      : reuse_sref_(reuse_sref),
        used_border_parent_(used_border_parent),
        stmt2ref_(stmt2ref),
        block_scopes_(block_scopes) {}

  void VisitStmt_(const LoopNode* op) final { VisitSRefStmt(op); }

  void VisitStmt_(const BlockNode* op) final { VisitSRefStmt(op); }

 private:
  template <typename T>
  void VisitSRefStmt(const T* op) {
    const auto* stmt_ptr = GetRef<Stmt>(op).operator->();
    // Remove useless StmtSRef until the border
    CHECK(stmt2ref_->count(stmt_ptr));
    StmtSRef sref = stmt2ref_->at(stmt_ptr);
    if (used_border_parent_.count(sref) == 0) {
      // If we will reuse the sref later, we don't remove it
      if (reuse_sref_.count(sref) == 0) {
        sref->stmt = nullptr;
        sref->parent = nullptr;
        if (stmt_ptr->template IsInstance<BlockNode>()) {
          block_scopes_->erase(sref);
        }
      }
      stmt2ref_->erase(stmt_ptr);
      VisitStmt(op->body);
    } else {
      sref->parent = used_border_parent_.at(sref);
    }
  }

  std::unordered_set<StmtSRef, ObjectPtrHash, ObjectPtrEqual> reuse_sref_;
  std::unordered_map<StmtSRef, StmtSRefNode*, ObjectPtrHash, ObjectPtrEqual> used_border_parent_;
  std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref_;
  std::unordered_map<StmtSRef, Scope, ObjectPtrHash, ObjectPtrEqual>* block_scopes_;
};

void ScheduleNode::Replace(StmtSRef sref, Stmt target, Map<Block, Block> block_sref_map) {
  // Note that old_ref is only a temporary SRef
  Stmt old_stmt = GetRef<Stmt>(sref->stmt);
  StmtSRef old_ref = StmtSRef(sref->stmt, sref->parent);
  const StmtNode* root_stmt = this->root->stmt;
  // Create SRef tree for the incoming target Stmt
  SRefCreator creator(this, block_sref_map, old_ref->parent);
  creator(target);
  // Initialize old SRef remover
  SRefRemover remover(&stmt2ref, std::move(creator.used_border_parent_), &scopes,
                      std::move(creator.reuse_sref_));
  // num_copy_steps: maximum number of hops until we don't need to copy
  int curr_step = 0;
  int num_copy_steps = -1;
  // Find the highest non-unique Stmt
  for (const StmtSRefNode* ptr = old_ref.operator->(); ptr != nullptr;
       ptr = ptr->parent, ++curr_step) {
    if (!ptr->stmt->unique()) {
      num_copy_steps = curr_step;
    }
  }
  if (!func.unique()) {
    num_copy_steps = curr_step;
  }
  // Update the function body
  curr_step = 0;
  for (StmtSRefNode* ptr = old_ref.operator->(); ptr->stmt != root_stmt;
       ptr = ptr->parent, ++curr_step) {
    StmtSRefNode* parent = ptr->parent;
    // parent_step = current_step + 1
    // if parent_step <= num_copy_step, then it implies
    // that parent is not unique and we need to copy
    bool parent_is_uniquely_referenced = curr_step + 1 > num_copy_steps;
    // replace ptr(son of parent->node) with target and return a new parent Stmt)
    Stmt new_stmt =
        SubReplacer(ptr, target, &stmt2ref)(parent->stmt, parent_is_uniquely_referenced);
    if (curr_step != 0) {
      UpdateSRef(this, ptr, target.get());
    }
    if (parent_is_uniquely_referenced) {
      CHECK(new_stmt.get() == parent->stmt);
      // if one node has been direct write, there is no need to
      // update its parent and the function
      remover(old_stmt);
      return;
    }
    target = new_stmt;
  }
  remover(old_stmt);
  if (old_ref->stmt == root_stmt) {
    // The replace point is root, we directly use the sref tree created by SRefCreator
    root = stmt2ref[target.operator->()];
  } else {
    // Otherwise we reuse root sref
    UpdateSRef(this, root.operator->(), target.get());
  }
  this->func = UpdatePrimFunc(&func, target);
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
