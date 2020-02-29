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

#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace tir {

/*! \brief The tool to create schedule */
class ScheduleCreator : public StmtVisitor {
 public:
  explicit ScheduleCreator(std::unordered_map<const StmtNode*, StmtSRef>* stmt_map)
      : stmt_map_(stmt_map) {}

  void VisitStmt_(const BlockNode* op) override {
    VisitSRefStmt(op);
  }

  void VisitStmt_(const LoopNode* op) override {
    VisitSRefStmt(op);
  }

  void VisitStmt_(const SeqStmtNode* op) override {
    StmtVisitor::VisitStmt_(op);
    for (size_t index = 0; index < op->seq.size(); index++) {
      if (op->seq[index]->IsInstance<BlockRealizeNode>()) {
        (*stmt_map_)[op->seq[index].as<BlockRealizeNode>()->block.operator->()]->seq_index = index;
      } else {
        (*stmt_map_)[op->seq[index].operator->()]->seq_index = index;
      }
    }
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
    if (seq_index >= 0 && is_son(stmt->seq[seq_index], sref_->node)) {
      auto n = CopyOnWrite(stmt);
      if (target_->IsInstance<SeqStmtNode>()) {
        // note that nested SeqStmt is not allowed, so we flatten target here
        const Array<Stmt>& target_seq = target_.as<SeqStmtNode>()->seq;
        n->seq.Erase(n->seq.begin() + seq_index);
        n->seq.Insert(n->seq.begin() + seq_index, target_seq.begin(), target_seq.end());
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
      const auto *ptr = son.as<BlockRealizeNode>();
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

Function UpdateFuncBody(FunctionNode* func, const Stmt& new_body) {
  CHECK(func->body.as<BlockRealizeNode>());
  CHECK(new_body->IsInstance<BlockNode>());

  if (func->unique()) {
    auto root_br = const_cast<BlockRealizeNode*>(func->body.as<BlockRealizeNode>());
    root_br->block = Downcast<Block>(new_body);
    return GetRef<Function>(func);
  } else {
    auto n_br = make_object<BlockRealizeNode>(*(func->body.as<BlockRealizeNode>()));
    n_br->block = Downcast<Block>(new_body);
    auto n_func = make_object<FunctionNode>(*func);
    n_func->body = Stmt(n_br);
    return Function(n_func);
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
      std::unordered_map<StmtSRef, StmtSRefNode*, ObjectHash, ObjectEqual>&& used_border_parent,
      std::unordered_set<StmtSRef, ObjectHash, ObjectEqual>&& reuse_sref)
      : reuse_sref_(reuse_sref), used_border_parent_(used_border_parent), stmt2ref_(stmt2ref) {}

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
    if (used_border_parent_.count(sref) == 0) {
      // If we will reuse the sref later, we don't remove it
      if (reuse_sref_.count(sref) == 0) {
        sref->node = nullptr;
        sref->parent = nullptr;
      }
      stmt2ref_->erase(stmt_ptr);
      VisitStmt(op->body);
    } else {
      sref->parent = used_border_parent_.at(sref);
    }
  }

  std::unordered_set<StmtSRef, ObjectHash, ObjectEqual> reuse_sref_;
  std::unordered_map<StmtSRef, StmtSRefNode*, ObjectHash, ObjectEqual> used_border_parent_;
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
              std::unordered_map<const VarNode*, StmtSRef>&& loop_var2ref,
              Map<Block, Block>&& block_sref_map,
              StmtSRefNode* parent)
      : parent_(parent), stmt2ref_(stmt2ref),
        loop_var2ref_(loop_var2ref), block_sref_map_(block_sref_map) {}

  void VisitStmt_(const LoopNode* op) final {
    VisitSRefStmt(op);
  }

  void VisitStmt_(const BlockNode* op) final {
    VisitSRefStmt(op);
  }

 private:
  StmtSRef CreateNewSRef(const StmtNode* stmt_ptr) {
    if (stmt_ptr->IsInstance<LoopNode>()) {
      const auto* op = GetRef<Stmt>(stmt_ptr).as<LoopNode>();
      auto it = loop_var2ref_.find(op->loop_var.get());
      if (it != loop_var2ref_.end()) {
        StmtSRef reuse_sref = it->second;
        reuse_sref->node = stmt_ptr;
        reuse_sref->parent = parent_;
        reuse_sref_.insert(reuse_sref);
        return reuse_sref;
      }
    } else if (block_sref_map_.defined()) {
      Block block = Downcast<Block>(GetRef<Stmt>(stmt_ptr));
      auto it = block_sref_map_.find(block);
      if (it != block_sref_map_.end()) {
        StmtSRef reuse_sref = stmt2ref_->at((*it).second.as<BlockNode>());
        reuse_sref->node = stmt_ptr;
        reuse_sref->parent = parent_;
        reuse_sref_.insert(reuse_sref);
        return reuse_sref;
      }
    }
    return StmtSRef(stmt_ptr, parent_);
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
      auto current = ref.get();
      std::swap(current, parent_);
      VisitStmt(op->body);
      std::swap(current, parent_);
    } else {
      // Mark the border of reused StmtSRef
      used_border_parent_[stmt2ref_->at(stmt_ptr)] = parent_;
    }
  }

  friend class Schedule;
  StmtSRefNode* parent_;
  std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref_;
  std::unordered_map<const VarNode*, StmtSRef> loop_var2ref_;
  Map<Block, Block> block_sref_map_;

  std::unordered_set<StmtSRef, ObjectHash, ObjectEqual> reuse_sref_;
  std::unordered_map<StmtSRef, StmtSRefNode*, ObjectHash, ObjectEqual> used_border_parent_;
};

class IRSubstitueInScope : public StmtExprMutator {
 public:
  explicit IRSubstitueInScope(
      std::function<PrimExpr(const VarNode*)> fmap)
      : fmap_(std::move(fmap)) {}

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = fmap_(op);
    if (it.defined()) {
      return it;
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    auto fmutate = [this](const PrimExpr& e) { return this->VisitExpr(e); };
    Array<PrimExpr> v = op->binding_values;
    v.MutateByApply(fmutate);
    PrimExpr pred = this->VisitExpr(op->predicate);
    if (v.same_as(op->binding_values) && pred.same_as(op->predicate)) {
      return GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->binding_values = std::move(v);
      n->predicate = std::move(pred);
      return Stmt(n);
    }
  }

 private:
  const std::function<PrimExpr(const VarNode*)> fmap_;
};

Stmt Schedule::SubstituteInScope(const Stmt& stmt,
                                 const std::function<PrimExpr(const VarNode*)>& value_func) {
  return IRSubstitueInScope(value_func)(stmt);
}

class LoopCollector : public StmtVisitor {
 public:
  explicit LoopCollector(std::unordered_map<const StmtNode*, StmtSRef>* stmt2_ref)
      : stmt2ref_(stmt2_ref) {}

  void VisitStmt_(const LoopNode* op) final {
    loop_var2sref[op->loop_var.get()] = (*stmt2ref_)[op];
    StmtVisitor::VisitStmt_(op);
  }

 private:
  friend class Schedule;
  std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref_;
  std::unordered_map<const VarNode*, StmtSRef> loop_var2sref;
};

void Schedule::Replace(StmtSRef ref, Stmt target, Map<Block, Block> block_sref_map) {
  ScheduleNode* self = operator->();
  // Note that old_ref is only a temporary SRef
  StmtSRef old_ref = StmtSRef(ref->node, ref->parent);
  const Stmt& old_stmt = GetRef<Stmt>(ref->node);
  // Collect loop_var to Loop mapping under old stmt
  LoopCollector collector(&self->stmt2ref);
  collector(old_stmt);
  // Create SRef tree for the incoming target Stmt
  SRefCreator creator(&self->stmt2ref, std::move(collector.loop_var2sref),
      std::move(block_sref_map), old_ref->parent);
  creator(target);
  // Initialize old SRef remover
  SRefRemover remover(&self->stmt2ref,
      std::move(creator.used_border_parent_), std::move(creator.reuse_sref_));
  // num_copy_steps: maximum number of hops until we don't need to copy
  int curr_step = 0;
  int num_copy_steps = -1;
  // Find the highest non-unique Stmt
  for (StmtSRefNode* ptr = old_ref.get(); ptr != nullptr; ptr = ptr->parent, ++curr_step) {
    if (!ptr->node->unique()) {
      num_copy_steps = curr_step;
    }
  }
  if (!self->func.unique()) num_copy_steps = curr_step;
  // Update the function body
  curr_step = 0;
  for (StmtSRefNode* ptr = old_ref.get(); ptr->node != self->root->node;
      ptr = ptr->parent, ++curr_step) {
    StmtSRefNode* parent = ptr->parent;
    // parent_step = current_step + 1
    // if parent_step <= num_copy_step, then it implies
    // that parent is not unique and we need to copy
    bool parent_is_uniquely_referenced = curr_step + 1 > num_copy_steps;
    // replace ptr(son of parent->node) with target and return a new parent Stmt)
    Stmt new_stmt = SubReplacer(ptr, target, &self->stmt2ref)
        (parent->node, parent_is_uniquely_referenced);
    if (curr_step != 0) UpdateSRef(ptr, target);
    if (parent_is_uniquely_referenced) {
      CHECK(new_stmt.get() == parent->node);
      // if one node has been direct write, there is no need to
      // update its parent and the function
      remover(old_stmt);
      return;
    }
    target = new_stmt;
  }
  remover(old_stmt);
  if (old_ref->node == self->root->node) {
    // The replace point is root, we directly use the sref tree created by SRefCreator
    self->root = self->stmt2ref[target.operator->()];
  } else {
    // Otherwise we reuse root sref
    UpdateSRef(self->root.operator->(), target);
  }
  self->func = UpdateFuncBody(self->func.operator->(), target);
}

Schedule Schedule::Create(Function func) {
  std::unordered_map<const StmtNode*, StmtSRef> stmt_map;
  std::unordered_map<StmtSRef, Scope, ObjectHash, ObjectEqual> block_scopes;

  DependencyAnalyzer dependency_analyzer(&stmt_map, &block_scopes);
  dependency_analyzer(func->body);
  const auto* op = func->body.as<BlockRealizeNode>();
  CHECK(op != nullptr);
  auto n = make_object<ScheduleNode>();
  n->func = std::move(func);
  n->stmt2ref = std::move(stmt_map);
  n->root = n->stmt2ref[op->block.as<StmtNode>()];
  n->scopes_ = block_scopes;
  return Schedule(n);
}

void Schedule::UpdateSRef(StmtSRefNode* sref, const Stmt& stmt) {
  CHECK(stmt->IsInstance<BlockNode>() || stmt->IsInstance<LoopNode>());
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
  std::unordered_set<StmtSRef, ObjectHash, ObjectEqual> collect;
  for (const auto& x : write_map) {
    for (const auto& block : x.second) {
      collect.insert(block);
    }
  }
  Array<StmtSRef> ret;
  for (const auto& block : collect)
      ret.push_back(block);
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

/*! \note Nested SeqStmt is not allowed in schedule. */
Array<Stmt> Schedule::GetChildren(const Stmt& stmt, bool keep_realize) {
  Stmt body;
  if (const auto* block = stmt.as<BlockNode>()) {
    body = block->body;
  } else if (const auto* loop = stmt.as<LoopNode>()) {
    body = loop->body;
  } else {
    return Array<Stmt>();
  }
  if (const auto* seq = body.as<SeqStmtNode>()) {
    Array<Stmt> ret;
    for (const Stmt& child : seq->seq)
      if (child->IsInstance<BlockRealizeNode>() && !keep_realize) {
        ret.push_back(child.as<BlockRealizeNode>()->block);
      } else {
        ret.push_back(child);
      }
    return ret;
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
  // Equivalence
  // - The total repeat number has not changed for each direct child block.
  // - The execution order has not changed. (The block executes with the same
  //   args and the same order with before.)

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

  PrimExpr min = 0;
  PrimExpr extent = outer_loop->extent * inner_loop->extent;

  Var fused_var = outer_loop->loop_var.copy_with_suffix(
      "_" + inner_loop->loop_var.get()->name_hint + "_fused");

  auto vmap = [&](const VarNode* v) -> PrimExpr {
    if (GetRef<Var>(v).same_as(outer_loop->loop_var)) {
      return floordiv(fused_var, inner_loop->extent) + outer_loop->min;
    } else if (GetRef<Var>(v).same_as(inner_loop->loop_var)) {
      return floormod(fused_var, inner_loop->extent) + inner_loop->min;
    } else {
      return NullValue<PrimExpr>();
    }
  };

  Loop fused_node = Loop(
      fused_var, min, extent, outer_loop->annotations,
      SubstituteInScope(inner_loop->body, vmap));

  // relink
  this->Replace(outer, fused_node);

  return operator->()->stmt2ref[fused_node.operator->()];
}

class PredicateUpdater : public StmtMutator {
 public:
  explicit PredicateUpdater(PrimExpr predicate) : predicate_(std::move(predicate)) {}

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    auto n = CopyOnWrite(op);
    n->predicate = n->predicate && predicate_;
    return Stmt(n);
  }
 private:
  PrimExpr predicate_;
};

Array<StmtSRef> Schedule::split(const StmtSRef& node,
                                const PrimExpr& nparts,
                                const PrimExpr& factor) {
  // Equivalence
  // - The total repeat number has not changed for each direct child block with updating predicate.
  // - The execution order has not changed. (The block executes with the same
  //   args and the same order with before.)

  const auto* loop = GetRef<Stmt>(node->node).as<LoopNode>();

  // Currently, can not split Loops with annotations
  if (!loop->annotations.empty()) {
    LOG(FATAL) << "InvalidSchedule: " << "Cannot split loops that already has annotations";
  }

  Var outer_var = loop->loop_var.copy_with_suffix("_outer");
  Var inner_var = loop->loop_var.copy_with_suffix("_inner");

  const PrimExpr& outer_min = loop->min;
  const PrimExpr& outer_extent = nparts;

  const PrimExpr& inner_min = 0;
  const PrimExpr& inner_extent = factor;

  auto vmap = [&](const VarNode* v) -> PrimExpr {
    if (GetRef<Var>(v).same_as(loop->loop_var)) {
      return outer_var * factor + inner_var;
    } else {
      return NullValue<PrimExpr>();
    }
  };

  arith::Analyzer analyzer;
  analyzer.Bind(outer_var, Range::make_by_min_extent(outer_min, outer_extent));
  analyzer.Bind(inner_var, Range::make_by_min_extent(inner_min, inner_extent));
  PrimExpr predicate = analyzer.Simplify(outer_var * factor + inner_var < loop->extent);
  Stmt new_stmt = PredicateUpdater(predicate)(SubstituteInScope(loop->body, vmap));

  Loop inner_loop(inner_var, inner_min, inner_extent, loop->annotations, new_stmt);
  Loop outer_loop(outer_var, outer_min, outer_extent, loop->annotations, inner_loop);

  // relink
  this->Replace(node, outer_loop);

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

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<StmtSRefNode>([](const ObjectRef& node, ReprPrinter* p) {
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

// schedule
TVM_REGISTER_GLOBAL("tir.schedule.CreateSchedule")
.set_body_typed(Schedule::Create);

TVM_REGISTER_GLOBAL("tir.schedule.Replace")
.set_body_method(&Schedule::Replace);

TVM_REGISTER_GLOBAL("tir.schedule.GetStmtSRef")
.set_body_typed<StmtSRef(Schedule, Stmt)>(
[](Schedule schedule, Stmt stmt) {
return schedule->stmt2ref.at(stmt.operator->());
});

TVM_REGISTER_GLOBAL("tir.schedule.GetStmt")
.set_body_typed<Stmt(StmtSRef)>(
[](StmtSRef sref) {
return GetRef<Stmt>(sref->node);
});

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleBlocks")
.set_body_method(&Schedule::Blocks);

TVM_REGISTER_GLOBAL("tir.schedule.GetBlocksFromTag")
.set_body_typed<Array<StmtSRef>(Schedule, std::string, StmtSRef)>(
[](Schedule schedule, std::string tag, StmtSRef scope) {
return schedule.GetBlock(tag, scope);
});

TVM_REGISTER_GLOBAL("tir.schedule.GetBlocksFromBuffer")
.set_body_typed<Array<StmtSRef>(Schedule, Buffer, StmtSRef)>(
[](Schedule schedule, Buffer buffer, StmtSRef scope) {
return schedule.GetBlock(buffer, scope);
});

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetLoopsInScope")
.set_body_method(&Schedule::GetLoopsInScope);

// schedule primitive
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleFuse")
.set_body_method(&Schedule::fuse);

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleSplitByFactor")
.set_body_typed<Array<StmtSRef>(Schedule, StmtSRef, PrimExpr)>(
[](Schedule schedule, StmtSRef node, PrimExpr factor) {
const auto* loop = GetRef<Stmt>(node->node).as<LoopNode>();
return schedule.split(node, floordiv(loop->extent + factor - 1, factor), factor);
});

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleSplitByNParts")
.set_body_typed<Array<StmtSRef>(Schedule, StmtSRef, PrimExpr)>(
[](Schedule schedule, StmtSRef node, PrimExpr nparts) {
const auto* loop = GetRef<Stmt>(node->node).as<LoopNode>();
return schedule.split(node, nparts, floordiv(loop->extent + nparts - 1, nparts));
});

// dependency graph
TVM_REGISTER_GLOBAL("tir.schedule.GetSuccessors")
.set_body_typed<Array<StmtSRef>(Schedule, StmtSRef, StmtSRef)>(
[](Schedule schedule, StmtSRef scope, StmtSRef block) {
return schedule->scopes_[scope].GetSuccessors(block);
});

TVM_REGISTER_GLOBAL("tir.schedule.GetPredecessors")
.set_body_typed<Array<StmtSRef>(Schedule, StmtSRef, StmtSRef)>(
[](Schedule schedule, StmtSRef scope, StmtSRef block) {
return schedule->scopes_[scope].GetPredecessors(block);
});

}  // namespace tir
}  // namespace tvm
