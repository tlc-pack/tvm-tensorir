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
#include "../cow_stmt_mutator.h"

namespace tvm {
namespace te {

/*! \brief The tool to create schedule */
class ScheduleCreator : public IRMutator {
 public:
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

  const std::unordered_map<const StmtNode*, StmtSRef> GetStmtMap() const {
    return stmt_map_;
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
    stmt_map_[sref_node->node] = sref_node;
    return new_stmt;
  }

  std::unordered_map<const StmtNode*, StmtSRef> stmt_map_;
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

 protected:
  template <typename T>
  Stmt VisitSRefStmt(const T* op) {
    if (sref_scope_counter_ > 0) {
      return GetRef<Stmt>(op);
    } else {
      ++sref_scope_counter_;
      return COWStmtMutator::VisitStmt_(op);
    }
  }

  /*! \brief */
  int sref_scope_counter_{0};
  StmtSRefNode* sref_;
  const Stmt& target_;
};

Function UpdateFuncBody(const Function& func, Stmt new_body) {
  auto n = make_object<FunctionNode>(*(func.operator->()));
  n->body = std::move(new_body);
  return Function(n);
}

void Schedule::Replace(StmtSRef ref, Stmt target) {
  UpdateChildren(target, ref->parent);
  // num_copy_steps: maximum number of hops until we don't need to copy
  int curr_step = 0;
  int num_copy_steps = -1;
  ScheduleNode* self = operator->();
  const auto& func = self->func;

  for (StmtSRefNode* ptr = ref.operator->(); ptr != self->root.get();
       ptr = ptr->parent, ++curr_step) {
    if (ptr->node->unique()) {
      num_copy_steps = curr_step;
    }
  }
  if (!func.unique()) num_copy_steps = curr_step;

  curr_step = 0;
  for (StmtSRefNode* ptr = ref.operator->(); ptr != self->root.get();
       ptr = ptr->parent, ++curr_step) {
    StmtSRefNode* parent = ptr->parent;
    bool allow_direct_write = curr_step + 1 > num_copy_steps;

    Stmt new_stmt = SubReplacer(ptr, target)(parent->node, allow_direct_write);
    UpdateSRef(ptr, target);
    if (allow_direct_write) {
      CHECK(new_stmt.get() == parent->node);
      break;
    }
    target = new_stmt;
  }
  if (curr_step + 1 <= num_copy_steps) {
    UpdateSRef(self->root.operator->(), target);
  }

  if (!func.unique()) {
    self->func = UpdateFuncBody(func, target);
  }
}

template <typename F>
void IterChildren(const Stmt& stmt, F fupdate) {
  Stmt body;
  if (const auto* block = stmt.as<BlockNode>()) {
    body = block->body;
  } else if (const auto* loop = stmt.as<LoopNode>()) {
    body = loop->body;
  } else {
    return;
  }
  // Don't support ir::Block in schedule
  CHECK(!body.as<ir::Block>());
  if (const auto* seq = body.as<SeqStmtNode>()) {
    for (const auto& child : seq->seq) {
      fupdate(child);
    }
  } else {
    fupdate(body);
  }
}

void Schedule::UpdateChildren(const Stmt& stmt, StmtSRefNode* parent) {
  const auto* stmt_ptr = stmt.operator->();
  if (operator->()->stmt2ref.count(stmt_ptr) == 0) {
    StmtSRef ref = StmtSRef(stmt_ptr, parent);
    operator->()->stmt2ref[stmt_ptr] = ref;
    IterChildren(stmt, [this, &ref](const Stmt& s) { return UpdateChildren(s, ref.operator->()); });
  }
}

Schedule Schedule::Create(const Function& func) {
  ScheduleCreator creator;
  Stmt new_stmt = creator.Mutate(func->body);
  Function new_func = Function(func->params, func->buffer_map, func->name, new_stmt);
  CHECK(func->body.as<BlockNode>());
  auto n = make_node<ScheduleNode>();
  n->func = std::move(new_func);
  n->stmt2ref = creator.GetStmtMap();
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
