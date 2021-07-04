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

void UpdateSRef(ScheduleNode* sch, StmtSRefNode* sref, const Stmt& stmt) {
  CHECK(stmt->IsInstance<BlockNode>() || stmt->IsInstance<LoopNode>());
  sch->stmt2ref[stmt.operator->()] = GetRef<StmtSRef>(sref);
  sch->stmt2ref.erase(sref->stmt);
  sref->stmt = stmt.operator->();
}

PrimFunc UpdateFuncBody(const PrimFuncNode* func, const Stmt& new_body) {
  CHECK(func->body.as<BlockRealizeNode>());
  CHECK(new_body->IsInstance<BlockNode>());

  if (func->unique()) {
    auto root_br = const_cast<BlockRealizeNode*>(func->body.as<BlockRealizeNode>());
    root_br->block = Downcast<Block>(new_body);
    return GetRef<PrimFunc>(func);
  } else {
    auto n_br = make_object<BlockRealizeNode>(*(func->body.as<BlockRealizeNode>()));
    n_br->block = Downcast<Block>(new_body);
    auto n_func = make_object<PrimFuncNode>(*func);
    n_func->body = Stmt(n_br);
    return PrimFunc(n_func);
  }
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

class LoopCollector : public StmtVisitor {
 public:
  explicit LoopCollector(std::unordered_map<const StmtNode*, StmtSRef>* stmt2_ref)
      : stmt2ref_(stmt2_ref) {}

  void VisitStmt_(const LoopNode* op) final {
    loop_var2sref[op->loop_var.get()] = stmt2ref_->at(op);
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BlockNode* op) final {
    // do not collect loops inside block's init
    this->VisitStmt(op->body);
  }

 private:
  friend class ScheduleNode;
  std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref_;
  std::unordered_map<const VarNode*, StmtSRef> loop_var2sref;
};

/*!
 * \brief create schedulable reference during Schedule.Replace
 * \note This Visitor will create schedulable reference corresponding
 *       AST node in target stmt.
 */
class SRefCreator : public StmtVisitor {
 public:
  SRefCreator(std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref,
              std::unordered_map<const VarNode*, StmtSRef>&& loop_var2ref,
              std::unordered_map<StmtSRef, Scope, ObjectPtrHash, ObjectPtrEqual>* block_scopes,
              Map<Block, Block>&& block_sref_map, StmtSRefNode* parent)
      : parent_(parent),
        stmt2ref_(stmt2ref),
        loop_var2ref_(loop_var2ref),
        block_scopes_(block_scopes),
        block_sref_map_(block_sref_map) {}

  void VisitStmt_(const LoopNode* op) final { VisitSRefStmt(op); }

  void VisitStmt_(const BlockNode* op) final { VisitSRefStmt(op); }

 private:
  StmtSRef CreateNewSRef(const StmtNode* stmt_ptr) {
    if (stmt_ptr->IsInstance<LoopNode>()) {
      const auto* op = GetRef<Stmt>(stmt_ptr).as<LoopNode>();
      auto it = loop_var2ref_.find(op->loop_var.get());
      if (it != loop_var2ref_.end()) {
        StmtSRef reuse_sref = it->second;
        reuse_sref->stmt = stmt_ptr;
        reuse_sref->parent = parent_;
        reuse_sref_.insert(reuse_sref);
        return reuse_sref;
      }
    } else if (block_sref_map_.defined()) {
      Block block = Downcast<Block>(GetRef<Stmt>(stmt_ptr));
      auto it = block_sref_map_.find(block);
      if (it != block_sref_map_.end()) {
        StmtSRef reuse_sref = stmt2ref_->at((*it).second.as<BlockNode>());
        reuse_sref->stmt = stmt_ptr;
        reuse_sref->parent = parent_;
        reuse_sref_.insert(reuse_sref);
        return reuse_sref;
      }
    }
    StmtSRef sref = StmtSRef(stmt_ptr, parent_);
    sref->binding_valid = true;
    return sref;
  }

  template <typename T>
  void VisitSRefStmt(const T* op) {
    const auto* stmt_ptr = GetRef<Stmt>(op).operator->();
    if (stmt2ref_->count(stmt_ptr) == 0) {
      // Create corresponding StmtSRef
      // note that we only create the StmtSRef whose node is not
      // in the AST and reuse those StmtSRef when node is in the AST.
      StmtSRef ref = CreateNewSRef(stmt_ptr);
      (*stmt2ref_)[stmt_ptr] = ref;
      StmtSRefNode* current = ref.operator->();
      std::swap(current, parent_);
      VisitStmt(op->body);
      std::swap(current, parent_);
      if (stmt_ptr->template IsInstance<BlockNode>()) {
        UpdateScope(stmt_ptr, *stmt2ref_, block_scopes_);
      }
    } else {
      // Mark the border of reused StmtSRef
      used_border_parent_[stmt2ref_->at(stmt_ptr)] = parent_;
    }
  }

  friend class ScheduleNode;
  StmtSRefNode* parent_{nullptr};
  std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref_;
  std::unordered_map<const VarNode*, StmtSRef> loop_var2ref_;
  std::unordered_map<StmtSRef, Scope, ObjectPtrHash, ObjectPtrEqual>* block_scopes_;
  Map<Block, Block> block_sref_map_;

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

void ScheduleNode::Replace(StmtSRef ref, Stmt target, Map<Block, Block> block_sref_map) {
  // Note that old_ref is only a temporary SRef
  StmtSRef old_ref = StmtSRef(ref->stmt, ref->parent);
  auto root_node = root->stmt;
  const Stmt& old_stmt = GetRef<Stmt>(ref->stmt);
  // Collect loop_var to Loop mapping under old stmt
  LoopCollector collector(&stmt2ref);
  collector(old_stmt);
  // Create SRef tree for the incoming target Stmt
  SRefCreator creator(&stmt2ref, std::move(collector.loop_var2sref), &scopes,
                      std::move(block_sref_map), old_ref->parent);
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
  if (!func.unique()) num_copy_steps = curr_step;
  // Update the function body
  curr_step = 0;
  for (StmtSRefNode* ptr = old_ref.operator->(); ptr->stmt != root_node;
       ptr = ptr->parent, ++curr_step) {
    StmtSRefNode* parent = ptr->parent;
    // parent_step = current_step + 1
    // if parent_step <= num_copy_step, then it implies
    // that parent is not unique and we need to copy
    bool parent_is_uniquely_referenced = curr_step + 1 > num_copy_steps;
    // replace ptr(son of parent->node) with target and return a new parent Stmt)
    Stmt new_stmt =
        SubReplacer(ptr, target, &stmt2ref)(parent->stmt, parent_is_uniquely_referenced);
    if (curr_step != 0) UpdateSRef(this, ptr, target);
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
  if (old_ref->stmt == root_node) {
    // The replace point is root, we directly use the sref tree created by SRefCreator
    root = stmt2ref[target.operator->()];
  } else {
    // Otherwise we reuse root sref
    UpdateSRef(this, root.operator->(), target);
  }
  func = UpdateFuncBody(func.operator->(), target);
}

}  // namespace tir
}  // namespace tvm
