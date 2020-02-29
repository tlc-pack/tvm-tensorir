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
#include <tvm/ir/attrs.h>
#include <tvm/arith/analyzer.h>
#include <queue>
#include <utility>

namespace tvm {
namespace tir {

/*! \brief Helper class to generate vmap lambda to be used in SubstituteInScope */
std::function<PrimExpr(const VarNode*)> vmap_generator(Var new_var, Var old_var) {
  return [&](const VarNode* v) -> PrimExpr {
    if (GetRef<Var>(v).same_as(old_var)) {
      return new_var;
    } else {
      return NullValue<PrimExpr>();
    }
  };
}

template <typename T>
class ChildGather : public StmtVisitor {
 public:
  ChildGather() = default;

  void VisitStmt(const Stmt& n) final {
    if (n->IsInstance<BlockNode>() && counter++ > 0) return;
    StmtVisitor::VisitStmt(n);
  }

  void VisitStmt_(const T* op) final {
    ret.push_back(op);
    StmtVisitor::VisitStmt_(op);
  }

  std::vector<const T*> ret;

 private:
  int counter = 0;
};

template <typename T>
std::vector<const T*> Schedule::GetUnderSRef(const StmtSRef& top) const {
  ChildGather<T> child_gather;
  child_gather(GetRef<Stmt>(top->node));
  return std::move(child_gather.ret);
}

/*! \brief Helper class to detect whether a PrimExpr is related with var */
class VarRelatedDetector : public ExprVisitor {
 public:
  explicit VarRelatedDetector(const Var& var) : var_(var) {}

  void VisitExpr_(const VarNode* op) final {
    related_ |= GetRef<Var>(op).same_as(var_);
  }

  bool related_{false};

 private:
  const Var& var_;
};

/*! \brief Wrapper function for VarRelatedDetector */
bool RelatedWithVar(const Var& var, const PrimExpr& expr) {
  VarRelatedDetector detector(var);
  detector(expr);
  return detector.related_;
}

std::pair<std::vector<Stmt>, size_t> Schedule::DecomposeLoop(
    const StmtSRefNode* now_sref, const StmtSRefNode* bottom_sref,
    const std::unordered_map<const StmtSRefNode*, const StmtSRefNode*>* successor) {
  // Decompose(Loop(before -> target -> after))
  // = Loop(before) -> Loop(Decompose(target)) -> Loop(after)

  const auto* now = GetRef<Stmt>(now_sref->node).as<LoopNode>();
  const auto* bottom = GetRef<Stmt>(bottom_sref->node).as<LoopNode>();

  std::pair<std::vector<Stmt>, int> ret;
  ret.second = 0;
  if (now == bottom) {
    // Reach bottom
    ret.first.push_back(Loop(make_object<LoopNode>(*bottom)));
    return ret;
  }

  int rename_counter = 0;
  Array<Stmt> children = GetChildren(GetRef<Stmt>(now), true);

  // Initialize target
  Loop target = Downcast<Loop>(GetRef<Stmt>(successor->at(now_sref)->node));

  // Generate a loop for stmts before target loop if necessary
  Array<Stmt> before;
  for (Stmt stmt : children) {
    if (stmt.same_as(target)) break;
    before.push_back(stmt);
  }
  if (!before.empty()) {
    Var before_var = Var(now->loop_var->name_hint + std::to_string(rename_counter++));
    auto vmap = vmap_generator(before_var, now->loop_var);
    Stmt before_body = before.size() == 1 ? before[0] : SeqStmt(before);
    ret.first.push_back(Loop(before_var, now->min, now->extent, now->annotations,
                             SubstituteInScope(before_body, vmap)));
    ret.second += 1;
  }

  // Generate loops for target loop
  auto decomposed_target = DecomposeLoop(successor->at(now_sref), bottom_sref, successor);
  for (size_t i = 0; i < decomposed_target.first.size(); i++) {
    Stmt loop = decomposed_target.first[i];
    Var new_var = Var(now->loop_var->name_hint + std::to_string(rename_counter++));
    if (i == decomposed_target.second) rename_counter--;
    auto vmap = vmap_generator(new_var, now->loop_var);
    ret.first.push_back(Loop(new_var, now->min, now->extent, now->annotations,
                             i == decomposed_target.second ? loop : SubstituteInScope(loop, vmap)));
  }
  ret.second += decomposed_target.second;

  // Generate a loop for stmts after target loop if necessary
  Array<Stmt> after;
  for (auto it = children.rbegin(); it != children.rend(); ++it) {
    if ((*it).same_as(target)) break;
    after.push_back(*it);
  }
  if (!after.empty()) {
    Var after_var = Var(now->loop_var->name_hint + std::to_string(rename_counter));
    Stmt after_body = after.size() == 1 ? after[0] : SeqStmt(after);
    auto vmap = vmap_generator(after_var, now->loop_var);
    ret.first.push_back(Loop(after_var, now->min, now->extent, now->annotations,
                             SubstituteInScope(after_body, vmap)));
  }
  return ret;
}

bool Schedule::DetectLoopReorderable(const StmtSRef& loop) {
  const auto* loop_ptr = GetRef<Stmt>(loop->node).as<LoopNode>();
  CHECK(loop_ptr);
  std::vector<const BlockRealizeNode*> blocks = GetUnderSRef<BlockRealizeNode>(loop);
  for (const auto* block_realize_ptr : blocks) {
    const auto* block_ptr = block_realize_ptr->block.as<BlockNode>();
    CHECK_EQ(block_realize_ptr->binding_values.size(), block_ptr->iter_vars.size());
    for (size_t i = 0; i < block_ptr->iter_vars.size(); i++) {
      IterVarType var_type = block_ptr->iter_vars[i]->iter_type;
      if (var_type != kDataPar && var_type != kThreadIndex && var_type != kCommReduce
          && RelatedWithVar(loop_ptr->loop_var, block_realize_ptr->binding_values[i]))
        return false;
    }
  }
  return true;
}

void Schedule::reorder(const Array<StmtSRef>& order) {
  // Equivalence
  // - The equivalence is based on the fact that if a loop is kDataPar/kCommReduce/kThreadIndex
  // then for (i) { S[i]->T[i]->U[i]; } is equivalent with
  // for (i) {S[i]} -> for (i) {T[i]} -> for (i) {U[i]}
  // - We recursively transform the original loop into a collection
  // of equivalent simple loops(single branch), and we reorder the target one.

  // Check iter_type and loops are mutually different
  std::unordered_set<StmtSRef, ObjectHash, ObjectEqual> seen_loop;
  for (StmtSRef loop_sref : order) {
    CHECK(GetRef<Stmt>(loop_sref->node).as<LoopNode>())
      << "Order has to be a list a Loops";
    CHECK(Schedule::DetectLoopReorderable(loop_sref))
      << "Cannot reorder Loop("
      << GetRef<Stmt>(loop_sref->node).as<LoopNode>()->loop_var << ")";
    CHECK_EQ(seen_loop.count(loop_sref), 0)
      << "Same Loop can not appear more than once "
      << GetRef<Stmt>(loop_sref->node).as<LoopNode>()->loop_var;
    seen_loop.insert(loop_sref);
  }
  // Check these loops are in the same line
  std::vector<const LoopNode*> all_loops = GetUnderSRef<LoopNode>(GetScope(order[0]));
  // successor[LoopA] = LoopB
  // means the loops need reordering under LoopA are all under LoopB
  // where LoopB is a direct son of LoopA
  std::unordered_map<const StmtSRefNode*, const StmtSRefNode*> successor;
  for (auto it = all_loops.rbegin(); it != all_loops.rend(); ++it) {
    StmtSRef now = this->operator->()->stmt2ref.at(*it);
    if (seen_loop.count(now) || successor.count(now.get())) {
      const StmtSRefNode* parent = now->parent;
      CHECK(successor.count(parent) == 0 || successor.at(parent) == now.get())
        << "The loops have to be in the same line";
      successor[parent] = now.get();
    }
  }
  // Check these loops are in the same scope(Block)
  for (const LoopNode* loop : all_loops) {
    StmtSRef sref = this->operator->()->stmt2ref.at(loop);
    if (seen_loop.count(sref)) {
      seen_loop.erase(sref);
    }
  }
  CHECK(seen_loop.empty()) << "Loops have to be under the same scope";
  for (StmtSRef loop_sref : order)
    seen_loop.insert(loop_sref);

  // Reorder
  // top and bottom denote the range of loops need reordering
  const StmtSRefNode* top = nullptr, * bottom = nullptr;
  for (const LoopNode* loop : all_loops) {
    StmtSRef sref = this->operator->()->stmt2ref.at(loop);
    if (seen_loop.count(sref)) {
      top = sref.get();
      break;
    }
  }
  for (auto it = all_loops.rbegin(); it != all_loops.rend(); ++it) {
    StmtSRef sref = this->operator->()->stmt2ref.at(*it);
    if (seen_loop.count(sref)) {
      bottom = sref.get();
      break;
    }
  }
  // at first we decompose the loop into multiple loops to enable reorder with branches
  std::pair<std::vector<Stmt>, size_t> res = DecomposeLoop(top, bottom, &successor);
  // reorder the res.second-th Loop, which is the target loop
  const StmtSRefNode* old_loop = top;
  const auto* new_loop = res.first[res.second].as<LoopNode>();
  for (int index = 0;;) {
    // decide which loop to copy
    const LoopNode* copy = nullptr;
    if (seen_loop.count(GetRef<StmtSRef>(old_loop))) {
      copy = GetRef<Stmt>(order[index++]->node).as<LoopNode>();
    } else {
      copy = GetRef<Stmt>(old_loop->node).as<LoopNode>();
    }
    // mutate the generated loop
    auto n = runtime::GetObjectPtr<LoopNode>(const_cast<LoopNode*>(new_loop));
    // The loop sref in target loop ought to be reused, so we copy the loop var
    n->loop_var = copy->loop_var;
    n->min = copy->min;
    n->extent = copy->extent;
    n->annotations = copy->annotations;
    // next level
    if (old_loop == bottom)
      break;
    old_loop = successor.at(old_loop);
    new_loop = (new_loop->body).as<LoopNode>();
  }

  this->Replace(GetRef<StmtSRef>(top), res.first.size() == 1 ? res.first[0] : SeqStmt(res.first));
}

}  // namespace tir
}  // namespace tvm
