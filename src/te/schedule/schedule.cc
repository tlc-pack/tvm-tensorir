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
#include <tvm/ir_functor_ext.h>
#include <tvm/ir_pass.h>
#include <tvm/attrs.h>

namespace tvm {
namespace te {

/*! \brief The tool to create schedule */
class ScheduleCreator : public StmtVisitor {
 public:
  explicit ScheduleCreator(std::unordered_map<const StmtNode*, StmtSRef>* stmt_map)
      : stmt_map_(stmt_map) {}

  void VisitStmt_(const te::BlockNode* op) override {
    return VisitSRefStmt(op);
  }

  void VisitStmt_(const te::LoopNode* op) override {
    return VisitSRefStmt(op);
  }

 protected:
  template <typename T>
  void VisitSRefStmt(const T* op) {
    StmtSRef sref_node(op, parent_scope_);
    auto tmp = sref_node.get();

    std::swap(parent_scope_, tmp);
    StmtVisitor::VisitStmt_(op);
    std::swap(parent_scope_, tmp);

    (*stmt_map_)[sref_node->node] = sref_node;
  }

  std::unordered_map<const StmtNode*, StmtSRef>* stmt_map_;
  StmtSRefNode* parent_scope_{nullptr};
};

class DependencyAnalyzer : public ScheduleCreator {
 public:
  DependencyAnalyzer(std::unordered_map<const StmtNode*, StmtSRef>* stmt_map,
                     std::unordered_map<StmtSRef, Scope, ObjectHash, ObjectEqual>* block_scopes)
      : ScheduleCreator(stmt_map), block_scopes_(block_scopes) {}

  void VisitStmt_(const BlockNode* op) final {
    Scope scope(make_object<ScopeNode>());

    std::swap(current_scope_, scope);
    ScheduleCreator::VisitStmt_(op);
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
  std::unordered_map<StmtSRef, Scope, ObjectHash, ObjectEqual>* block_scopes_;
  Scope current_scope_{make_object<ScopeNode>()};
};

class SubReplacer : protected StmtMutator {
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
    if (allow_copy_on_write_) {
      CHECK(weakref->unique()) << GetRef<Stmt>(weakref);
    }
    Stmt stmt = VisitStmt(GetRef<Stmt>(weakref));
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
      return StmtFunctor::VisitStmt(stmt);
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
class SRefRemover : public StmtVisitor {
 public:
  SRefRemover(std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref,
              std::unordered_set<StmtSRef, ObjectHash, ObjectEqual>&& used_border)
      : used_border_(used_border), stmt2ref_(stmt2ref) {}

  void VisitStmt_(const LoopNode* op) final {
    VisitSRefStmt(op);
  }

  void VisitStmt_(const BlockNode* op) final {
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
      VisitStmt(op->body);
    }
  }

  std::unordered_set<StmtSRef, ObjectHash, ObjectEqual> used_border_;
  std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref_;
};

/*!
 * \brief create schedulable reference during Schedule.Replace
 * \note This Visitor will create schedulable reference corresponding
 *       AST node in target stmt.
 */
class SRefCreator : public StmtVisitor {
 public:
  SRefCreator(std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref,
              StmtSRefNode* parent)
      : parent_(parent), stmt2ref_(stmt2ref) {}

  void VisitStmt_(const LoopNode* op) final {
    VisitSRefStmt(op);
  }

  void VisitStmt_(const BlockNode* op) final {
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
      auto current = ref.get();
      std::swap(current, parent_);
      VisitStmt(op->body);
      std::swap(current, parent_);
    } else {
      // Mark the border of reused StmtSRef
      used_border_.insert(stmt2ref_->at(stmt_ptr));
    }
  }

  friend class Schedule;
  StmtSRefNode* parent_;
  std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref_;
  std::unordered_set<StmtSRef, ObjectHash, ObjectEqual> used_border_;
};

void Schedule::Replace(StmtSRef ref, Stmt target) {
  ScheduleNode* self = operator->();
  SRefCreator creator(&self->stmt2ref, ref->parent);
  creator(target);
  SRefRemover remover(&self->stmt2ref, std::move(creator.used_border_));
  StmtSRef origin_ref = ref;
  // num_copy_steps: maximum number of hops until we don't need to copy
  int curr_step = 0;
  int num_copy_steps = -1;

  for (StmtSRefNode* ptr = ref.get(); ptr != nullptr;
       ptr = ptr->parent, ++curr_step) {
    if (!ptr->node->unique()) {
      num_copy_steps = curr_step;
    }
  }
  if (!self->func.unique()) num_copy_steps = curr_step;

  // Update the function body
  curr_step = 0;
  for (StmtSRefNode* ptr = ref.get(); ptr != self->root.get();
       ptr = ptr->parent, ++curr_step) {
    StmtSRefNode* parent = ptr->parent;
    // parent_step = current_step + 1
    // if parent_step <= num_copy_step, then it implies
    // that parent is not unique and we need to copy
    bool parent_is_uniquely_referenced = curr_step + 1 > num_copy_steps;
    Stmt new_stmt = SubReplacer(ptr, target)(parent->node, parent_is_uniquely_referenced);
    UpdateSRef(ptr, target);
    if (parent_is_uniquely_referenced) {
      CHECK(new_stmt.get() == parent->node);
      // if one node has been direct write, there is no need to
      // update its parent and the function
      remover(GetRef<Stmt>(origin_ref->node));
      return;
    }
    target = new_stmt;
  }
  remover(GetRef<Stmt>(origin_ref->node));
  UpdateSRef(self->root.operator->(), target);
  self->func = UpdateFuncBody(self->func.operator->(), target);
}

Schedule Schedule::Create(Function func) {
  std::unordered_map<const StmtNode*, StmtSRef> stmt_map;
  std::unordered_map<StmtSRef, Scope, ObjectHash, ObjectEqual> block_scopes;

  DependencyAnalyzer dependency_analyzer(&stmt_map, &block_scopes);
  dependency_analyzer(func->body);
  CHECK(func->body.as<BlockNode>());
  auto n = make_object<ScheduleNode>();
  n->func = std::move(func);
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

StmtSRef Schedule::GetScope(StmtSRef node) const {
  while (node.defined()) {
    node = GetRef<StmtSRef>(node->parent);
    if (GetRef<Stmt>(node->node).as<BlockNode>()) {
      return node;
    }
  }
  LOG(FATAL) << "Cannot find a father block";
  return StmtSRef();
}

Array<Stmt> Schedule::GetChildren(const Stmt& stmt) {
  Stmt body;
  if (const auto* block = stmt.as<BlockNode>()) {
    body = block->body;
  } else if (const auto* loop = stmt.as<LoopNode>()) {
    body = loop->body;
  } else {
    return Array<Stmt>();
  }
  if (const auto* seq = body.as<SeqStmtNode>()) {
    return seq->seq;
  } else {
    return Array<Stmt>{body};
  }
}

Array<StmtSRef> Schedule::GetLoopsInScope(const StmtSRef& block) const {
  Array<StmtSRef> ret;
  StmtSRef sref = GetRef<StmtSRef>(block->parent);
  while (!GetRef<Stmt>(sref->node).as<BlockNode>()) {
    if (GetRef<Stmt>(sref->node).as<LoopNode>()) {
      ret.push_back(sref);
    }
    sref = GetRef<StmtSRef>(sref->parent);
  }
  return Array<StmtSRef>(ret.rbegin(), ret.rend());
}

StmtSRef Schedule::fuse(const StmtSRef& outer, const StmtSRef& inner) {
  // Can only fuse neighbor loop without any extra branches.
  // Future enhancement: this condition can be eliminated by lifting all siblings of inner
  // as the children of the father of outer
  const auto* outer_loop = GetRef<Stmt>(outer->node).as<LoopNode>();
  const auto* inner_loop = GetRef<Stmt>(inner->node).as<LoopNode>();
  CHECK(outer_loop != nullptr && inner_loop != nullptr);

  CHECK(inner->parent == outer.get());
  auto outer_children = GetChildren(GetRef<Stmt>(outer_loop));
  CHECK(outer_children.size() == 1 && outer_children[0].get() == inner_loop);
  // Check both loops are in the same scope
  CHECK_EQ(GetScope(outer), GetScope(inner));

  // Currently, can not fuse Loops with annotations
  if (!outer_loop->annotations.empty() || !inner_loop->annotations.empty()) {
    LOG(FATAL) << "InvalidSchedule: " << "Cannot fuse loops that already has annotations";
  }

  Expr min = 0;
  Expr extent = outer_loop->extent * inner_loop->extent;

  Var fused_var = outer_loop->loop_var.copy_with_suffix(
      "." + inner_loop->loop_var.get()->name_hint + ".fused");

  auto vmap = [&](const Variable* v) -> Expr {
    if (GetRef<Var>(v).same_as(outer_loop->loop_var)) {
      return floordiv(fused_var, inner_loop->extent) + outer_loop->min;
    } else if (GetRef<Var>(v).same_as(inner_loop->loop_var)) {
      return floormod(fused_var, inner_loop->extent) + inner_loop->min;
    } else {
      return NullValue<Expr>();
    }
  };

  Loop fused_node = Loop(
      fused_var, min, extent, outer_loop->annotations,
      SubstituteInScope(inner_loop->body, vmap));

  // relink
  Replace(outer, fused_node);

  return operator->()->stmt2ref[fused_node.operator->()];
}

class PredicateUpdater : public StmtMutator {
 public:
  explicit PredicateUpdater(Expr predicate) : predicate_(std::move(predicate)) {}

  Stmt VisitStmt_(const BlockNode* op) final {
    auto n = CopyOnWrite(op);
    n->predicate = n->predicate && predicate_;
    return Stmt(n);
  }
 private:
  Expr predicate_;
};

Array<StmtSRef> Schedule::split(const StmtSRef& node, const Expr& factor) {
  const auto* loop = GetRef<Stmt>(node->node).as<LoopNode>();
  Var outer_var = loop->loop_var.copy_with_suffix(".outer");
  Var inner_var = loop->loop_var.copy_with_suffix(".inner");

  Expr outer_min = loop->min;
  Expr outer_extent = floordiv(loop->extent + factor - 1, factor);

  Expr inner_min = 0;
  Expr inner_extent = factor;

  auto vmap = [&](const Variable* v) -> Expr {
    if (GetRef<Var>(v).same_as(loop->loop_var)) {
      return outer_var * factor + inner_var;
    } else {
      return NullValue<Expr>();
    }
  };

  Map<Var, Range> vrange;
  vrange.Set(outer_var, Range::make_by_min_extent(outer_min, outer_extent));
  vrange.Set(inner_var, Range::make_by_min_extent(inner_min, inner_extent));
  Expr predicate = Simplify(outer_var * factor + inner_var < loop->extent, vrange);
  Stmt new_stmt = PredicateUpdater(predicate)(SubstituteInScope(loop->body, vmap));

  Loop inner_loop(inner_var, inner_min, inner_extent, loop->annotations, new_stmt);
  Loop outer_loop(outer_var, outer_min, outer_extent, loop->annotations, inner_loop);

  // relink
  Replace(node, outer_loop);

  StmtSRef inner_sref = operator->()->stmt2ref[inner_loop.as<StmtNode>()];
  StmtSRef outer_sref = operator->()->stmt2ref[outer_loop.as<StmtNode>()];
  return Array<StmtSRef>{outer_sref, inner_sref};
}

StmtSRef::StmtSRef(const StmtNode* node, StmtSRefNode* parent, int64_t seq_index) {
  auto n = make_object<StmtSRefNode>();
  n->node = node;
  n->parent = parent;
  n->seq_index = seq_index;
  data_ = std::move(n);
}

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<StmtSRefNode>([](const ObjectRef& node, NodePrinter* p) {
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
