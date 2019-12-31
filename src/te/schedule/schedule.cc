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

class BlockFlattener : public IRMutator {
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
};

/*! \brief The tool to create schedule */
class ScheduleCreator : public IRVisitor {
 public:
  explicit ScheduleCreator(std::unordered_map<const StmtNode*, StmtSRef>* stmt_map)
      : stmt_map_(stmt_map) {}

  void Visit_(const te::BlockNode* op) override {
    return VisitSRefStmt(op);
  }

  void Visit_(const te::LoopNode* op) override {
    return VisitSRefStmt(op);
  }

  void Visit_(const ir::Block* op) override {
    LOG(FATAL) << "ir::Block is not allowed in te functions";
  }

 protected:
  template <typename T>
  void VisitSRefStmt(const T* op) {
    StmtSRef sref_node(op, parent_scope_);
    auto tmp = sref_node.operator->();

    std::swap(parent_scope_, tmp);
    IRVisitor::Visit_(op);
    std::swap(parent_scope_, tmp);

    (*stmt_map_)[sref_node->node] = sref_node;
  }

  std::unordered_map<const StmtNode*, StmtSRef>* stmt_map_;
  StmtSRefNode* parent_scope_{nullptr};
};

class DependencyAnalyzer : public ScheduleCreator {
 public:
  DependencyAnalyzer(std::unordered_map<const StmtNode*, StmtSRef>* stmt_map,
                     std::unordered_map<StmtSRef, Scope, NodeHash, NodeEqual>* block_scopes)
      : ScheduleCreator(stmt_map), block_scopes_(block_scopes) {}

  void Visit_(const BlockNode* op) final {
    Scope scope(make_object<ScopeNode>());

    std::swap(current_scope_, scope);
    ScheduleCreator::Visit_(op);
    std::swap(current_scope_, scope);

    StmtSRef block_sref = stmt_map_->at(op);
    (*block_scopes_)[block_sref] = scope;

    // Update tensor write map
    auto& write_map = current_scope_->write_map;
    for (const auto& write : op->writes) {
      Array<StmtSRef> array;
      if (write_map.count(write->buffer)) {
        array = write_map.at(write->buffer);
      }
      array.push_back(block_sref);
      write_map[write->buffer] = array;
    }
    // Update dependency graph
    for (const auto& read : op->reads) {
      const auto& read_buffer = read->buffer;
      if (write_map.count(read_buffer)) {
        // The block depends on every block who write a input tensor
        for (const auto& write_block : write_map[read_buffer]) {
          current_scope_.AddEdge(write_block, block_sref);
        }
      }
    }
  }

 private:
  std::unordered_map<StmtSRef, Scope, NodeHash, NodeEqual>* block_scopes_;
  Scope current_scope_{make_object<ScopeNode>()};
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
      CHECK_EQ(n.use_count(), 2U) << GetRef<Stmt>(weakref);
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

  // Node that this counter works for faster visiting.
  // We guarantee that each visit will only visit Schedulable
  // Stmt Node (BlockNode and LoopNode) once, the parent node.
  // As for its children, they can be either replaced or remain unchanged
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
 * \brief remove useless schedulable reference during Schedule.Replace
 * \note The Schedule.Replace will remove nodes from AST. This visitor will help to
 *       remove their schedulable reference.
 */
class SRefRemover : public IRVisitor {
 public:
  SRefRemover(std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref,
              std::unordered_set<StmtSRef, NodeHash, NodeEqual>&& used_border)
      : used_border_(used_border), stmt2ref_(stmt2ref) {}

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
    // Remove useless StmtSRef until the border
    CHECK(stmt2ref_->count(stmt_ptr));
    StmtSRef sref = stmt2ref_->at(stmt_ptr);
    if (used_border_.count(sref) == 0) {
      sref->node = nullptr;
      sref->parent = nullptr;
      stmt2ref_->erase(stmt_ptr);
      Visit(op->body);
    }
  }

  std::unordered_set<StmtSRef, NodeHash, NodeEqual> used_border_;
  std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref_;
};

/*!
 * \brief create schedulable reference during Schedule.Replace
 * \note This Visitor will create schedulable reference corresponding
 *       AST node in target stmt.
 */
class SRefCreator : public IRVisitor {
 public:
  SRefCreator(std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref,
              StmtSRefNode* parent)
      : parent_(parent), stmt2ref_(stmt2ref) {}

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
    if (stmt2ref_->count(stmt_ptr) == 0) {
      // Create corresponding StmtSRef
      // note that we only create the StmtSRef whose node is not
      // in the AST and reuse those StmtSRef when node is in the AST.
      StmtSRef ref = StmtSRef(stmt_ptr, parent_);
      (*stmt2ref_)[stmt_ptr] = ref;
      auto current = ref.operator->();
      std::swap(current, parent_);
      Visit(op->body);
      std::swap(current, parent_);
    } else {
      // Mark the border of reused StmtSRef
      used_border_.insert(stmt2ref_->at(stmt_ptr));
    }
  }

  friend class Schedule;
  StmtSRefNode* parent_;
  std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref_;
  std::unordered_set<StmtSRef, NodeHash, NodeEqual> used_border_;
};

void Schedule::Replace(StmtSRef ref, Stmt target) {
  ScheduleNode* self = operator->();
  SRefCreator creator(&self->stmt2ref, ref->parent);
  creator.Visit(target);
  SRefRemover remover(&self->stmt2ref, std::move(creator.used_border_));
  StmtSRef origin_ref = ref;
  // num_copy_steps: maximum number of hops until we don't need to copy
  int curr_step = 0;
  int num_copy_steps = -1;

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
    bool allow_direct_write = curr_step - 1 > num_copy_steps;

    Stmt new_stmt = SubReplacer(ptr, target)(parent->node, allow_direct_write);
    UpdateSRef(ptr, target);
    if (allow_direct_write) {
      CHECK(new_stmt.get() == parent->node);
      // if one node has been direct write, there is no need to
      // update its parent and the function
      remover.Visit(GetRef<Stmt>(origin_ref->node));
      return;
    }
    target = new_stmt;
  }
  remover.Visit(GetRef<Stmt>(origin_ref->node));
  UpdateSRef(self->root.operator->(), target);
  self->func = UpdateFuncBody(self->func.operator->(), target);
}

Schedule Schedule::Create(const Function& func) {
  Stmt new_stmt = BlockFlattener().Mutate(func->body);

  std::unordered_map<const StmtNode*, StmtSRef> stmt_map;
  std::unordered_map<StmtSRef, Scope, NodeHash, NodeEqual> block_scopes;

  DependencyAnalyzer dependency_analyzer(&stmt_map, &block_scopes);
  dependency_analyzer.Visit(new_stmt);
  Function new_func = Function(func->params, func->buffer_map, func->name, new_stmt);
  CHECK(func->body.as<BlockNode>());
  auto n = make_node<ScheduleNode>();
  n->func = std::move(new_func);
  n->stmt2ref = std::move(stmt_map);
  n->root = n->stmt2ref[n->func->body.operator->()];
  n->scopes_ = block_scopes;
  return Schedule(n);
}

void Schedule::UpdateSRef(StmtSRefNode* sref, const Stmt& stmt) {
  ScheduleNode* self = operator->();
  self->stmt2ref[stmt.operator->()] = GetRef<StmtSRef>(sref);
  self->stmt2ref.erase(sref->node);
  sref->node = stmt.operator->();
}

Array<StmtSRef> Schedule::GetBlock(const std::string& tag, StmtSRef scope) const {
  if (!scope.defined()) {
    scope = operator->()->root;
  }
  CHECK(GetRef<Stmt>(scope->node).as<BlockNode>());
  Array<StmtSRef> ret;
  for (const auto& block : Blocks(scope)) {
    if (GetRef<Stmt>(block->node).as<BlockNode>()->tag == tag) {
      ret.push_back(block);
    }
  }
  return ret;
}

Array<StmtSRef> Schedule::GetBlock(const Buffer& buffer, StmtSRef scope) const {
  if (!scope.defined()) {
    scope = operator->()->root;
  }
  CHECK(GetRef<Stmt>(scope->node).as<BlockNode>());
  CHECK_GT(operator->()->scopes_.count(scope), 0);
  const auto& write_map = operator->()->scopes_.at(scope)->write_map;
  if (write_map.count(buffer)) {
    return write_map.at(buffer);
  } else {
    return Array<StmtSRef>();
  }
}

Array<StmtSRef> Schedule::Blocks(StmtSRef scope) const {
  if (!scope.defined()) {
    scope = operator->()->root;
  }
  CHECK(GetRef<Stmt>(scope->node).as<BlockNode>());
  CHECK_GT(operator->()->scopes_.count(scope), 0);
  const auto& write_map = operator->()->scopes_.at(scope)->write_map;
  Array<StmtSRef> ret;
  for (const auto& x : write_map) {
    for (const auto& block : x.second) {
      ret.push_back(block);
    }
  }
  return ret;
}

StmtSRef::StmtSRef(const StmtNode* node, StmtSRefNode* parent, int64_t seq_index) {
  auto n = make_node<StmtSRefNode>();
  n->node = node;
  n->parent = parent;
  n->seq_index = seq_index;
  data_ = std::move(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<StmtSRefNode>([](const ObjectRef& node, IRPrinter* p) {
  const auto* op = static_cast<const StmtSRefNode*>(node.get());
  if (const auto* loop = GetRef<Stmt>(op->node).as<LoopNode>()) {
    p->PrintIndent();
    p->stream << "for ";
    p->Print(loop->loop_var);
    p->stream << " = ";
    p->Print(loop->min);
    p->stream << " to ";
    p->Print(loop->extent);
  } else {
    p->Print(Downcast<Block>(GetRef<Stmt>(op->node)));
  }
});

TVM_REGISTER_NODE_TYPE(ScheduleNode);
TVM_REGISTER_NODE_TYPE(StmtSRefNode);

}  // namespace te
}  // namespace tvm
