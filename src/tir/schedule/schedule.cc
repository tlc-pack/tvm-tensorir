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
#include <tvm/tir/ir_pass.h>
#include "schedule_common.h"

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

class DependencyAnalyzer : public StmtVisitor {
 public:
  DependencyAnalyzer(const std::unordered_map<const StmtNode*, StmtSRef>& stmt_map,
                     std::unordered_map<StmtSRef, Scope, ObjectHash, ObjectEqual>* block_scopes,
                     bool recursive = true)
      : stmt_map_(stmt_map), block_scopes_(block_scopes), recursive_(recursive) {}

  void VisitStmt_(const BlockNode* op) final {
    StmtSRef block_sref = stmt_map_.at(op);

    if (recursive_ || first_visit_) {
      first_visit_ = false;
      Scope scope(make_object<ScopeNode>());
      std::unordered_map<Buffer, Array<StmtSRef>, ObjectHash, ObjectEqual> read_map_;
      std::swap(current_scope_, scope);
      std::swap(current_read_map_, read_map_);

      StmtVisitor::VisitStmt_(op);

      std::swap(current_scope_, scope);
      std::swap(current_read_map_, read_map_);
      (*block_scopes_)[block_sref] = scope;
    }

    // Update tensor write map
    auto& write_map = current_scope_->write_map;
    for (const auto& write : op->writes) {
      Array<StmtSRef> array;
      auto it = write_map.find(write->buffer);
      if (it != write_map.end()) {
        array = it->second;
      }
      array.push_back(block_sref);
      write_map[write->buffer] = array;
    }

    // Update tensor read map
    for (const auto& read : op->reads) {
      Array<StmtSRef> array;
      auto it = current_read_map_.find(read->buffer);
      if (it != current_read_map_.end()) {
        array = it->second;
      }
      array.push_back(block_sref);
      current_read_map_[read->buffer] = array;
    }

    // Update dependency graph
    for (const auto& read : op->reads) {
      const auto& read_buffer = read->buffer;
      // RAW dependency: The block depends on every block which writes a input tensor
      auto it = write_map.find(read_buffer);
      if (it != write_map.end()) {
        for (const auto& write_block : it->second) {
          current_scope_.AddEdge(write_block, block_sref, DepType::kRAW);
        }
      }
    }
    for (const auto& write : op->writes) {
      const auto& write_buffer = write->buffer;

      // WAW dependency: The block must execute after every block
      // which writes the same tensor before
      auto it = write_map.find(write_buffer);
      if (it != write_map.end()) {
        for (const auto& write_block : it->second) {
          current_scope_.AddEdge(write_block, block_sref, DepType::kWAW);
        }
      }

      // WAR dependency: The block must execute after every block
      // which reads the output tensor before
      it = current_read_map_.find(write_buffer);
      if (it != current_read_map_.end()) {
        for (const auto& read_block : it->second) {
          if (!read_block.same_as(block_sref)) {
            LOG(FATAL) << "WAR dependency is not allowed for now.";
          }
          current_scope_.AddEdge(read_block, block_sref, DepType::kWAR);
        }
      }
    }
  }

 private:
  const std::unordered_map<const StmtNode*, StmtSRef>& stmt_map_;
  std::unordered_map<StmtSRef, Scope, ObjectHash, ObjectEqual>* block_scopes_;
  Scope current_scope_{make_object<ScopeNode>()};
  // read map is only used for dependency analyse
  std::unordered_map<Buffer, Array<StmtSRef>, ObjectHash, ObjectEqual> current_read_map_;

  // analysis current scope or the whole subtree
  const bool recursive_;
  bool first_visit_{true};
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
              std::unordered_map<StmtSRef,
                                 StmtSRefNode*,
                                 ObjectHash,
                                 ObjectEqual>&& used_border_parent,
              std::unordered_map<StmtSRef, Scope, ObjectHash, ObjectEqual>* block_scopes,
              std::unordered_set<StmtSRef, ObjectHash, ObjectEqual>&& reuse_sref)
      : reuse_sref_(reuse_sref), used_border_parent_(used_border_parent),
        stmt2ref_(stmt2ref), block_scopes_(block_scopes) {}

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

  std::unordered_set<StmtSRef, ObjectHash, ObjectEqual> reuse_sref_;
  std::unordered_map<StmtSRef, StmtSRefNode*, ObjectHash, ObjectEqual> used_border_parent_;
  std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref_;
  std::unordered_map<StmtSRef, Scope, ObjectHash, ObjectEqual>* block_scopes_;
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
              std::unordered_map<StmtSRef, Scope, ObjectHash, ObjectEqual>* block_scopes,
              Map<Block, Block>&& block_sref_map,
              StmtSRefNode* parent)
      : parent_(parent),
        stmt2ref_(stmt2ref),
        loop_var2ref_(loop_var2ref),
        block_scopes_(block_scopes),
        block_sref_map_(block_sref_map) {}

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
      if (stmt_ptr->template IsInstance<BlockNode>()) {
        DependencyAnalyzer(*stmt2ref_, block_scopes_, false)(GetRef<Stmt>(stmt_ptr));
      }
    } else {
      // Mark the border of reused StmtSRef
      used_border_parent_[stmt2ref_->at(stmt_ptr)] = parent_;
    }
  }

  friend class ScheduleNode;
  StmtSRefNode* parent_;
  std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref_;
  std::unordered_map<const VarNode*, StmtSRef> loop_var2ref_;
  std::unordered_map<StmtSRef, Scope, ObjectHash, ObjectEqual>* block_scopes_;
  Map<Block, Block> block_sref_map_;

  std::unordered_set<StmtSRef, ObjectHash, ObjectEqual> reuse_sref_;
  std::unordered_map<StmtSRef, StmtSRefNode*, ObjectHash, ObjectEqual> used_border_parent_;
};

class LoopCollector : public StmtVisitor {
 public:
  explicit LoopCollector(std::unordered_map<const StmtNode*, StmtSRef>* stmt2_ref)
      : stmt2ref_(stmt2_ref) {}

  void VisitStmt_(const LoopNode* op) final {
    loop_var2sref[op->loop_var.get()] = (*stmt2ref_)[op];
    StmtVisitor::VisitStmt_(op);
  }

 private:
  friend class ScheduleNode;
  std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref_;
  std::unordered_map<const VarNode*, StmtSRef> loop_var2sref;
};

void ScheduleNode::Replace(StmtSRef ref, Stmt target, Map<Block, Block> block_sref_map) {
  // Note that old_ref is only a temporary SRef
  StmtSRef old_ref = StmtSRef(ref->node, ref->parent);
  auto root_node = root->node;
  const Stmt& old_stmt = GetRef<Stmt>(ref->node);
  // Collect loop_var to Loop mapping under old stmt
  LoopCollector collector(&stmt2ref);
  collector(old_stmt);
  // Create SRef tree for the incoming target Stmt
  SRefCreator creator(&stmt2ref, std::move(collector.loop_var2sref), &scopes_,
                      std::move(block_sref_map), old_ref->parent);
  creator(target);
  // Initialize old SRef remover
  SRefRemover remover(&stmt2ref, std::move(creator.used_border_parent_),
                      &scopes_, std::move(creator.reuse_sref_));
  // num_copy_steps: maximum number of hops until we don't need to copy
  int curr_step = 0;
  int num_copy_steps = -1;
  // Find the highest non-unique Stmt
  for (StmtSRefNode* ptr = old_ref.get(); ptr != nullptr; ptr = ptr->parent, ++curr_step) {
    if (!ptr->node->unique()) {
      num_copy_steps = curr_step;
    }
  }
  if (!func.unique()) num_copy_steps = curr_step;
  // Update the function body
  curr_step = 0;
  for (StmtSRefNode* ptr = old_ref.get(); ptr->node != root_node;
       ptr = ptr->parent, ++curr_step) {
    StmtSRefNode* parent = ptr->parent;
    // parent_step = current_step + 1
    // if parent_step <= num_copy_step, then it implies
    // that parent is not unique and we need to copy
    bool parent_is_uniquely_referenced = curr_step + 1 > num_copy_steps;
    // replace ptr(son of parent->node) with target and return a new parent Stmt)
    Stmt new_stmt = SubReplacer(ptr, target, &stmt2ref)
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
  if (old_ref->node == root_node) {
    // The replace point is root, we directly use the sref tree created by SRefCreator
    root = stmt2ref[target.operator->()];
  } else {
    // Otherwise we reuse root sref
    UpdateSRef(root.operator->(), target);
  }
  func = UpdateFuncBody(func.operator->(), target);
}

Schedule ScheduleNode::Create(Function function) {
  std::unordered_map<const StmtNode*, StmtSRef> stmt_map;
  std::unordered_map<StmtSRef, Scope, ObjectHash, ObjectEqual> block_scopes;
  ScheduleCreator schedule_creator(&stmt_map);
  schedule_creator(function->body);
  DependencyAnalyzer dependency_analyzer(stmt_map, &block_scopes);
  dependency_analyzer(function->body);
  const auto* op = function->body.as<BlockRealizeNode>();
  CHECK(op != nullptr);
  auto n = make_object<ScheduleNode>();
  n->func = function;
  n->stmt2ref = std::move(stmt_map);
  n->root = n->stmt2ref[op->block.as<StmtNode>()];
  n->scopes_ = block_scopes;
  n->ValidateLoops(function);
  for (const auto& it : block_scopes) {
    CHECK(it.first->node->IsInstance<BlockNode>());
    n->CheckRegionCover(it.first);
  }
  return Schedule(n);
}

void ScheduleNode::UpdateSRef(StmtSRefNode* sref, const Stmt& stmt) {
  CHECK(stmt->IsInstance<BlockNode>() || stmt->IsInstance<LoopNode>());
  stmt2ref[stmt.operator->()] = GetRef<StmtSRef>(sref);
  stmt2ref.erase(sref->node);
  sref->node = stmt.operator->();
}

Array<StmtSRef> ScheduleNode::GetBlock(const std::string& tag, StmtSRef scope) const {
  if (!scope.defined()) {
    scope = root;
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

Array<StmtSRef> ScheduleNode::GetBlock(const Buffer& buffer, StmtSRef scope) const {
  if (!scope.defined()) {
    scope = root;
  }
  CHECK(GetRef<Stmt>(scope->node).as<BlockNode>());
  CHECK_GT(scopes_.count(scope), 0);
  const auto& write_map = scopes_.at(scope)->write_map;
  if (write_map.count(buffer)) {
    return write_map.at(buffer);
  } else {
    return Array<StmtSRef>();
  }
}

Array<StmtSRef> ScheduleNode::Blocks(StmtSRef scope) const {
  if (!scope.defined()) {
    scope = root;
  }
  CHECK(GetRef<Stmt>(scope->node).as<BlockNode>());
  CHECK_GT(scopes_.count(scope), 0);
  const auto& write_map = scopes_.at(scope)->write_map;
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

StmtSRef ScheduleNode::GetScope(StmtSRef node) const {
  while (node.defined()) {
    node = GetRef<StmtSRef>(node->parent);
    if (GetRef<Stmt>(node->node).as<BlockNode>()) {
      return node;
    }
  }
  LOG(FATAL) << "Cannot find a father block";
  return StmtSRef();
}

Array<StmtSRef> ScheduleNode::GetLoopsInScope(const StmtSRef& block) const {
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

bool ScheduleNode::IsCompactDataFlow(const StmtSRef& sub_tree) const {
  StmtSRef scope_sref = GetScope(sub_tree);
  const Scope& scope = scopes_.at(scope_sref);
  std::unordered_set<StmtSRef, ObjectHash, ObjectEqual> child_blocks;
  ChildBlockGatherer(this, &child_blocks)(GetRef<Stmt>(sub_tree->node));
  for (const auto& block : child_blocks) {
    if (!scope.IsComplete(block)) return false;
  }
  return true;
}

bool ScheduleNode::ValidateSRef() const {
  SRefValidator(this)(func->body);
  return true;
}

StmtSRef ScheduleNode::fuse(const StmtSRef& outer, const StmtSRef& inner) {
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

  return stmt2ref[fused_node.operator->()];
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

Array<StmtSRef> ScheduleNode::split(const StmtSRef& node,
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
  PrimExpr predicate = outer_var * factor + inner_var < loop->extent;
  Stmt new_stmt = SubstituteInScope(loop->body, vmap);
  if (!analyzer.CanProve(predicate))
    new_stmt = PredicateUpdater(predicate)(new_stmt);

  Loop inner_loop(inner_var, inner_min, inner_extent, loop->annotations, new_stmt);
  Loop outer_loop(outer_var, outer_min, outer_extent, loop->annotations, inner_loop);

  // relink
  this->Replace(node, outer_loop);

  StmtSRef inner_sref = stmt2ref[inner_loop.as<StmtNode>()];
  StmtSRef outer_sref = stmt2ref[outer_loop.as<StmtNode>()];
  return Array<StmtSRef>{outer_sref, inner_sref};
}

class AnnotationUpdater : public StmtMutator {
 public:
  explicit AnnotationUpdater(Annotation annotation) : annotation_(std::move(annotation)) {}

 private:
  Stmt VisitStmt_(const LoopNode* op) override {
    auto n = CopyOnWrite(op);
    n->annotations.push_back(std::move(annotation_));
    return Stmt(n);
  }

  Annotation annotation_;
};

void ScheduleNode::vectorize(const StmtSRef& node) {
  /*!
   * Check:
   * - 1. check the block under is complete block or reduction block
   * - 2. check `input_loop` is bound and only bound to `data_par` block_vars
   * - 3. check the loops of reduction blocks are validatable
   * Mutate:
   * - 4. set Annotation on the loop
   * Proof:
   * We prove by showing that there are no data flows between `input_loop=i` and`input_loop=j`,
   * and we show this by induction on the number of blocks.
   *
   * If there is only one block below
   * - The block is complete. All the instances are independent of each other.
   * - The block is reduction. `input_loop` bound and only bound to `data_par` blocks + loops of
   * reduction blocks are validatable => instances of `input_loop=i` will write different positions
   * with instances of `input_loop=j`, hence they are independent.
   *
   * If there's a new block coming in. Consider its instances under `input_loop=i`.
   * - If the producer is complete. Producer instances under `input_loop=j` may write the positions
   * that new instances under `input_loop=i`  may read, but it will read the same value produced by
   * the producer under `input_loop=i` since it's complete.
   * - If the producer is reduction. Producer instances under `input_loop=j` will never write the
   * positions that new instances under `input_loop=j` may read. Hence no data flow.
   */

  const auto* loop = DowncastPtr<LoopNode>(node->node);
  CHECK(loop != nullptr) << "Vectorize expect a loop";
  // Currently, can not vectorize Loops with annotations
  if (!loop->annotations.empty()) {
    LOG(FATAL) << "InvalidSchedule: " << "Cannot vectorize loop that already has annotations";
  }
  // Now vectorize only support:
  //   1. All the blocks are complete below
  //   2. A single block below the loop
  // TODO(bohan): support reduction later
  if (!IsCompactDataFlow(node)) {
    auto children = GetChildren(GetRef<Stmt>(loop), true);
    CHECK(children.size() == 1 && children[0]->IsInstance<BlockRealizeNode>());
    const BlockRealize& br = Downcast<BlockRealize>(children[0]);
    CHECK(stmt2ref[br->block.operator->()]->binding_valid) << "Vectorize expect valid bindings";
    for (size_t i = 0; i < br->binding_values.size(); ++i) {
      if (br->block->iter_vars[i]->iter_type != IterVarType::kDataPar
          && RelatedWithVar(loop->loop_var, br->binding_values[i])) {
        LOG(FATAL) << "The loop is related with non-datapar block vars";
      }
    }
  }
  Annotation annotation = Annotation(attr::loop_type, StringImmNode::make("vectorize"));
  Stmt new_stmt = AnnotationUpdater(annotation)(GetRef<Stmt>(loop));
  this->Replace(node, new_stmt);
}

void ScheduleNode::unroll(const StmtSRef& node) {
  // Equivalence : Unroll is trivial
  const auto* loop = DowncastPtr<LoopNode>(node->node);
  CHECK(loop != nullptr) << "Unroll expect a loop";
  // Currently, can not unroll Loops with annotations
  if (!loop->annotations.empty()) {
    LOG(FATAL) << "InvalidSchedule: " << "Cannot unroll loop that already has annotations";
  }
  Annotation annotation = Annotation(tir::attr::loop_type, StringImmNode::make("unroll"));
  Stmt new_stmt = AnnotationUpdater(annotation)(GetRef<Stmt>(loop));
  this->Replace(node, new_stmt);
}

StmtSRef::StmtSRef(const StmtNode* node, StmtSRefNode* parent, int64_t seq_index) {
  auto n = make_object<StmtSRefNode>();
  n->node = node;
  n->parent = parent;
  n->seq_index = seq_index;
  n->binding_valid = false;
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
.set_body_typed(ScheduleNode::Create);

TVM_REGISTER_GLOBAL("tir.schedule.Replace")
.set_body_typed<void(Schedule, StmtSRef, Stmt, Map<Block, Block>)>(
    [](Schedule schedule, StmtSRef ref, Stmt target, Map<Block, Block> block_sref_map) {
      return schedule->Replace(ref, target, block_sref_map);
    });

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
.set_body_typed<Array<StmtSRef>(Schedule, StmtSRef)>(
    [](Schedule schedule, StmtSRef scope) {
      return schedule->Blocks(scope);
    });

TVM_REGISTER_GLOBAL("tir.schedule.GetBlocksFromTag")
.set_body_typed<Array<StmtSRef>(Schedule, std::string, StmtSRef)>(
    [](Schedule schedule, std::string tag, StmtSRef scope) {
      return schedule->GetBlock(tag, scope);
    });

TVM_REGISTER_GLOBAL("tir.schedule.GetBlocksFromBuffer")
.set_body_typed<Array<StmtSRef>(Schedule, Buffer, StmtSRef)>(
    [](Schedule schedule, Buffer buffer, StmtSRef scope) {
      return schedule->GetBlock(buffer, scope);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetLoopsInScope")
.set_body_typed<Array<StmtSRef>(Schedule, StmtSRef)>(
    [](Schedule schedule, StmtSRef scope) {
      return schedule->GetLoopsInScope(scope);
    });

// schedule primitive
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleFuse")
.set_body_typed<StmtSRef(Schedule, StmtSRef, StmtSRef)>(
    [](Schedule schedule, StmtSRef outer, StmtSRef inner) {
      return schedule->fuse(outer, inner);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleSplitByFactor")
.set_body_typed<Array<StmtSRef>(Schedule, StmtSRef, PrimExpr)>(
    [](Schedule schedule, StmtSRef node, PrimExpr factor) {
      const auto* loop = GetRef<Stmt>(node->node).as<LoopNode>();
      return schedule->split(node, floordiv(loop->extent + factor - 1, factor), factor);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleSplitByNParts")
.set_body_typed<Array<StmtSRef>(Schedule, StmtSRef, PrimExpr)>(
    [](Schedule schedule, StmtSRef node, PrimExpr nparts) {
      const auto* loop = GetRef<Stmt>(node->node).as<LoopNode>();
      return schedule->split(node, nparts, floordiv(loop->extent + nparts - 1, nparts));
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleReorder")
.set_body_typed<void(Schedule, Array<StmtSRef>)>(
    [](Schedule schedule, Array<StmtSRef> order) {
      return schedule->reorder(order);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleComputeAt")
.set_body_typed<void(Schedule, StmtSRef, StmtSRef)>(
    [](Schedule schedule, StmtSRef block_sref, StmtSRef loop_sref) {
      return schedule->compute_at(block_sref, loop_sref);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleVectorize")
.set_body_typed<void(Schedule, StmtSRef)>(
    [](Schedule schedule, StmtSRef node) {
      return schedule->vectorize(node);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleUnroll")
.set_body_typed<void(Schedule, StmtSRef)>(
    [](Schedule schedule, StmtSRef node) {
      return schedule->unroll(node);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleCacheWrite")
.set_body_typed<StmtSRef(Schedule, Buffer, std::string)>(
    [](Schedule schedule, Buffer buffer, std::string scope) {
      return schedule->cache_write(buffer, scope);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleCacheRead")
.set_body_typed<StmtSRef(Schedule, Buffer, std::string)>(
    [](Schedule schedule, Buffer buffer, std::string scope) {
      return schedule->cache_read(buffer, scope);
    });

// dependency graph
TVM_REGISTER_GLOBAL("tir.schedule.GetSuccessors")
.set_body_typed<Array<DepEdge>(Schedule, StmtSRef, StmtSRef)>(
    [](Schedule schedule, StmtSRef scope, StmtSRef block) {
      return schedule->scopes_[scope].GetSuccessors(block);
    });

TVM_REGISTER_GLOBAL("tir.schedule.GetPredecessors")
.set_body_typed<Array<DepEdge>(Schedule, StmtSRef, StmtSRef)>(
    [](Schedule schedule, StmtSRef scope, StmtSRef block) {
      return schedule->scopes_[scope].GetPredecessors(block);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ValidateSRef")
.set_body_typed<bool(Schedule)>(
    [](Schedule schedule) {
      return schedule->ValidateSRef();
    });

}  // namespace tir
}  // namespace tvm
