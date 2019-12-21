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

#include <tvm/te/schedule.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include "../cow_stmt_mutator.h"

namespace tvm {
namespace te {

/*! \brief The tool to create schedule */
class ScheduleCreator : public IRMutator {
 public:
  explicit ScheduleCreator(std::unordered_map<const StmtNode*, StmtSRef>* stmt_map)
      : stmt_map_(stmt_map) {}

  Stmt Mutate_(const te::BlockNode* op, const Stmt& s) final {
    return MutateSRefStmt(op, s);
  }

  Stmt Mutate_(const te::LoopNode* op, const Stmt& s) final {
    return MutateSRefStmt(op, s);
  }

  void FlattenBlock(const ir::Block* op, std::vector<Stmt>* seq) {
    if (const auto block = op->first.as<ir::Block>()) {
      FlattenBlock(block, seq);
    } else {
      seq->push_back(Mutate(op->first));
    }
    if (const auto block = op->rest.as<ir::Block>()) {
      FlattenBlock(block, seq);
    } else {
      seq->push_back(Mutate(op->rest));
    }
  }

  Stmt Mutate_(const ir::Block* op, const Stmt& s) final {
    std::vector<Stmt> new_stmt;
    FlattenBlock(op, &new_stmt);
    return te::SeqStmt(new_stmt);
  }

 private:
  template <typename T>
  Stmt MutateSRefStmt(const T* op, const Stmt& s) {
    StmtSRef sref_node(nullptr, parent_scope_);
    auto tmp = sref_node.operator->();

    std::swap(parent_scope_, tmp);
    Stmt new_stmt = IRMutator::Mutate_(op, s);
    std::swap(parent_scope_, tmp);

    sref_node->node = new_stmt.operator->();
    (*stmt_map_)[sref_node->node] = sref_node;
    return new_stmt;
  }

  std::unordered_map<const StmtNode*, StmtSRef>* stmt_map_;
  StmtSRefNode* parent_scope_{nullptr};
};

class SubReplacer : protected COWStmtMutator {
 public:
  SubReplacer(StmtSRefNode* sref, const Stmt& target)
      : sref_(sref), target_(target) {}
  /*!
   * \brief mutate weakref
   * \param weakref The statement to be mutated.
   * \param allow_copy_on_write Whether we allow copy on write in the weakref.
   *        That means weakref is only referenced once, and all its
   *        parents are also only referenced once.
   * \return The result of the mutation.
   */
  Stmt operator()(const StmtNode* weakref,
                  bool allow_copy_on_write) {
    std::swap(allow_copy_on_write, allow_copy_on_write_);
    auto n = runtime::GetObjectPtr<StmtNode>(const_cast<StmtNode*>(weakref));
    if (allow_copy_on_write_) {
      // We have just get a reference of n, so if the original
      // object is unique, the use_count will be 2
      CHECK_EQ(n.use_count(), 2U);
    }
    Stmt stmt = Stmt(n);
    stmt = VisitStmt(stmt);
    std::swap(allow_copy_on_write, allow_copy_on_write_);
    if (allow_copy_on_write) {
      CHECK(stmt.operator->() == weakref);
    }
    return stmt;
  }

  Stmt VisitStmt(const Stmt& stmt) final {
    if (stmt.get() == sref_->node) {
      // if the statement matches the replace target
      // just return the target stmt
      return target_;
    } else {
      return COWStmtMutator::VisitStmt(stmt);
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    return VisitSRefStmt(op);
  }

  Stmt VisitStmt_(const LoopNode* op) final {
    return VisitSRefStmt(op);
  }

  Stmt VisitStmt_(const SeqStmtNode* stmt) final {
    int64_t seq_index = sref_->seq_index;
    // fast path
    if (seq_index >= 0 &&
        (*stmt)[seq_index].get() == sref_->node) {
      auto n = CopyOnWrite(stmt);
      n->seq.Set(seq_index, target_);
      return Stmt(n);
    } else {
      return COWStmtMutator::VisitStmt_(stmt);
    }
  }

 private:
  template <typename T>
  Stmt VisitSRefStmt(const T* op) {
    if (sref_scope_counter_ > 0) {
      return GetRef<Stmt>(op);
    } else {
      ++sref_scope_counter_;
      return COWStmtMutator::VisitStmt_(op);
    }
  }

  int sref_scope_counter_{0};
  StmtSRefNode* sref_;
  const Stmt& target_;
};

Function UpdateFuncBody(FunctionNode* func, Stmt new_body) {
  if (func->unique()) {
    func->body = std::move(new_body);
    return GetRef<Function>(func);
  } else {
    auto n = make_object<FunctionNode>(*func);
    n->body = std::move(new_body);
    return Function(n);
  }
}

/*!
 * \brief update schedulable reference during Schedule.Replace
 * \note This Visitor will parse the AST twice. The first time, it
 *       will create references needed by the target statment. Then
 *       the second one will delete all useless reference whose stmt
 *       has been removed from the AST.
 */
class ChildUpdater : public IRVisitor {
 public:
  explicit ChildUpdater(Schedule schedule, StmtSRefNode* parent)
      : parent_(parent), schedule_(std::move(schedule)) {}

  /*! The first parsing, create references*/
  void CreateChildSRef(const Stmt& target) {
    create = true;
    Visit(target);

  }

  /*! The second parsing, remove useless references*/
  void RemoveSRef(StmtSRef ref) {
    create = false;
    Visit(GetRef<Stmt>(ref->node));
  }

  void Visit_(const LoopNode* op) final {
    VisitSRefStmt(op);
  }

  void Visit_(const BlockNode* op) final {
    VisitSRefStmt(op);
  }

 private:
  template <typename T>
  void VisitSRefStmt(const T* op) {
    const auto* stmt_ptr = GetRef<Stmt>(op).operator->();
    if (create) {
      if (schedule_->stmt2ref.count(stmt_ptr) == 0) {
        // Create corresponding StmtSRef
        StmtSRef ref = StmtSRef(stmt_ptr, parent_);
        schedule_->stmt2ref[stmt_ptr] = ref;
        auto current = ref.operator->();
        std::swap(current, parent_);
        Visit(op->body);
        std::swap(current, parent_);
      } else {
        // Mark the border of reused StmtSRef
        used_border_.insert(schedule_->stmt2ref.at(stmt_ptr));
      }
    } else {
      // Remove useless StmtSRef until the border
      CHECK(schedule_->stmt2ref.count(stmt_ptr));
      StmtSRef sref = schedule_->stmt2ref.at(stmt_ptr);
      if (used_border_.count(sref) == 0) {
        sref->node = nullptr;
        sref->parent = nullptr;
        schedule_->stmt2ref.erase(stmt_ptr);
        Visit(op->body);
      }
    }
  }

  StmtSRefNode* parent_;
  Schedule schedule_;
  // The Visitor does the first parsing when create is true
  // and does the second one when it is false
  bool create{true};
  std::unordered_set<StmtSRef, NodeHash, NodeEqual> used_border_;
};

void Schedule::Replace(StmtSRef ref, Stmt target) {
  ChildUpdater updater(*this, ref->parent);
  updater.CreateChildSRef(target);
  StmtSRef origin_ref = ref;
  // num_copy_steps: maximum number of hops until we don't need to copy
  int curr_step = 0;
  int num_copy_steps = -1;
  ScheduleNode* self = operator->();

  for (StmtSRefNode* ptr = ref.operator->(); ptr != self->root.get();
       ptr = ptr->parent, ++curr_step) {
    if (ptr->node->unique()) {
      num_copy_steps = curr_step;
    }
  }
  if (!self->func.unique()) num_copy_steps = curr_step;

  curr_step = 0;
  for (StmtSRefNode* ptr = ref.operator->(); ptr != self->root.get();
       ptr = ptr->parent, ++curr_step) {
    StmtSRefNode* parent = ptr->parent;
    bool allow_direct_write = curr_step + 1 > num_copy_steps;

    Stmt new_stmt = SubReplacer(ptr, target)(parent->node, allow_direct_write);
    UpdateSRef(ptr, target);
    if (allow_direct_write) {
      CHECK(new_stmt.get() == parent->node);
      // if one node has been direct write, there is no need to
      // update its parent and the function
      updater.RemoveSRef(origin_ref);
      return;
    }
    target = new_stmt;
  }
  updater.RemoveSRef(origin_ref);
  UpdateSRef(self->root.operator->(), target);
  self->func = UpdateFuncBody(self->func.operator->(), target);
}

Schedule Schedule::Create(const Function& func) {
  std::unordered_map<const StmtNode*, StmtSRef> stmt_map;
  ScheduleCreator creator(&stmt_map);
  Stmt new_stmt = creator.Mutate(func->body);
  Function new_func = Function(func->params, func->buffer_map, func->name, new_stmt);
  CHECK(func->body.as<BlockNode>());
  auto n = make_node<ScheduleNode>();
  n->func = std::move(new_func);
  n->stmt2ref = std::move(stmt_map);
  n->root = n->stmt2ref[n->func->body.operator->()];
  return Schedule(n);
}

StmtSRef::StmtSRef(const StmtNode* node, StmtSRefNode* parent, int64_t seq_index) {
  auto n = make_node<StmtSRefNode>();
  n->node = node;
  n->parent = parent;
  n->seq_index = seq_index;
  data_ = std::move(n);
}

void Schedule::UpdateSRef(StmtSRefNode* sref, const Stmt& stmt) {
  ScheduleNode* self = operator->();
  self->stmt2ref[stmt.operator->()] = GetRef<StmtSRef>(sref);
  self->stmt2ref.erase(sref->node);
  sref->node = stmt.operator->();
}

TVM_REGISTER_NODE_TYPE(ScheduleNode);
TVM_REGISTER_NODE_TYPE(StmtSRefNode);

}  // namespace te
}  // namespace tvm
