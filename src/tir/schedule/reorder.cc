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
#include "schedule_common.h"

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
  size_t counter = 0;
};

/*! \brief Gather all the type T nodes under top, which are in the same scope with top */
template <typename T>
std::vector<const T*> GatherChild(const StmtNode* top) {
  ChildGather<T> child_gather;
  child_gather(GetRef<Stmt>(top));
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

bool DetectLoopReorderable(const LoopNode* loop_ptr) {
  std::vector<const BlockRealizeNode*> blocks = GatherChild<BlockRealizeNode>(loop_ptr);
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

/*! \brief Wrap a new Loop outside body, substitute the loop var at the same time */
Loop NewLoopWrapper(const Stmt& body, const LoopNode* loop, const std::string& suffix) {
  auto node = make_object<LoopNode>(*loop);
  node->loop_var = std::move(Var(loop->loop_var->name_hint + suffix));
  node->body = SubstituteInScope(body, vmap_generator(node->loop_var, loop->loop_var));
  return Loop(node);
}

/*!
 * \brief Decompose loop into a collection of equivalent loops,
 *   Decompose(Loop(Before->Target->After)) = Loop(Before)->Loop(Decompose(Target))->Loop(After)
 *   e.g. Decompose(A-(C,D,Target,E,F)) = (A-(C,D)) - (A-Target) - (A-(E,F))
 */
std::pair<std::vector<Stmt>, size_t> DecomposeLoop(
    const StmtSRefNode* now_sref, const StmtSRefNode* bottom_sref,
    const std::unordered_map<const StmtSRefNode*, const StmtSRefNode*>* successor) {
  std::pair<std::vector<Stmt>, size_t> ret;
  const auto* now = DowncastPtr<LoopNode>(now_sref->node);
  const auto* bottom = DowncastPtr<LoopNode>(bottom_sref->node);
  ret.second = 0;
  if (now == bottom) {
    // Reach bottom
    ret.first.push_back(bottom->body);
    return ret;
  }
  // collect before and after
  const Loop& target = Downcast<Loop>(GetRef<Stmt>(successor->at(now_sref)->node));
  bool meet_target = false;
  std::vector<Stmt> before, after;
  for (const Stmt& stmt : GetChildren(GetRef<Stmt>(now), true)) {
    if (stmt.same_as(target)) {
      meet_target = true;
    } else if (!meet_target) {
      before.push_back(stmt);
    } else {
      after.push_back(stmt);
    }
  }
  // Loop(before)
  size_t rename_counter = 0;
  if (!before.empty()) {
    ret.first.push_back(
        NewLoopWrapper(SeqStmt::Flatten(before), now, std::to_string(rename_counter++)));
    ret.second += 1;
  }
  // Loop(target) note that we don't wrap the target body
  auto decomposed_target = DecomposeLoop(successor->at(now_sref), bottom_sref, successor);
  for (size_t i = 0; i < decomposed_target.first.size(); i++)
    if (i != decomposed_target.second) {
      ret.first.push_back(
          NewLoopWrapper(decomposed_target.first[i], now, std::to_string(rename_counter++)));
    } else {
      rename_counter--;
      ret.first.push_back(decomposed_target.first[i]);
    }
  ret.second += decomposed_target.second;
  // Loop(after)
  if (!after.empty())
    ret.first.push_back(
        NewLoopWrapper(SeqStmt::Flatten(after), now, std::to_string(rename_counter)));
  return ret;
}

/*!
 * \brief generate reordered loops recursively from top to bottom according to order
 *        Generate :: Int -> Loop
 *        Generate k = k < n ? Loop_i_k(Generate k+1) : Loop_i_n(body_target)
 * \param old_loop the original loop
 * \param bottom the bottom loop to be reordered
 * \param target_body the body of the generated bottom loop
 * \param order the order of reordered loops
 * \param index the index of order
 * \param seen_loop used to judge whether old_loop is in order
 * \param successor used to climb down the original loop tree
 * \return the generated reordered loops
 */
Stmt ReorderTarget(const StmtSRefNode* old_loop, const StmtSRefNode* bottom,
                   const Stmt& target_body,
                   const Array<StmtSRef>& order, size_t index,
                   const std::unordered_set<StmtSRef, ObjectHash, ObjectEqual>& seen_loop,
                   const std::unordered_map<const StmtSRefNode*, const StmtSRefNode*>& successor) {
  size_t new_index = index;
  // The order list maybe incomplete, so we may copy the old_loop rather than order
  const LoopNode* copy = seen_loop.count(GetRef<StmtSRef>(old_loop)) ?
      DowncastPtr<LoopNode>(order[new_index++]->node) : DowncastPtr<LoopNode>(old_loop->node);
  auto n = make_object<LoopNode>(*copy);
  if (old_loop == bottom) {
    // bottom loop
    n->body = target_body;
  } else {
    // reorder recursively
    n->body = ReorderTarget(successor.at(old_loop), bottom,
                            target_body, order, new_index, seen_loop, successor);
  }
  return Stmt(n);
}

void Schedule::reorder(const Array<StmtSRef>& order) {
  // Equivalence
  // - The equivalence is based on the fact that if a loop is kDataPar/kCommReduce/kThreadIndex
  // then for (i) { S[i]->T[i]->U[i]; } is equivalent with
  // for (i) {S[i]} -> for (i) {T[i]} -> for (i) {U[i]}
  // - We recursively transform the original loop into a collection
  // of equivalent simple loops(single branch), and we reorder the target one.

  // Check
  // 1. check iter_type are valid and loops are mutually different
  std::unordered_set<StmtSRef, ObjectHash, ObjectEqual> seen_loop;
  for (const StmtSRef& loop_sref : order) {
    const auto* loop = DowncastPtr<LoopNode>(loop_sref->node);
    CHECK(loop) << "Order has to be a list a Loops";
    CHECK(DetectLoopReorderable(loop)) << "Cannot reorder Loop(" << loop->loop_var << ")";
    CHECK_EQ(seen_loop.count(loop_sref), 0) << "Same Loop can not appear more than once ";
    seen_loop.insert(loop_sref);
  }
  // 2. check these loops are in the same line
  // The algorithm now is to scan the inverse DFS order of the whole loop tree in the scope.
  // For some Loop x, it is potentially in the reorder range if
  //   - x is in the reorder list
  //   - exactly 1 son y of x is potentially in the reorder range
  //     (If there are more, then the loops are not in the same line).
  //     Put (x,y) in the map.
  // After the inverse DFS, we can know how to catch the loop line by the map.
  std::vector<const LoopNode*> all_loops = GatherChild<LoopNode>(GetScope(order[0])->node);
  std::unordered_map<const StmtSRefNode*, const StmtSRefNode*> successor;
  // top and bottom denotes the range of loops need reordering
  const StmtSRefNode* top = nullptr, * bottom = nullptr;
  for (auto it = all_loops.rbegin(); it != all_loops.rend(); ++it) {
    StmtSRef now = this->operator->()->stmt2ref.at(*it);
    if (seen_loop.count(now) || successor.count(now.get())) {
      const StmtSRefNode* parent = now->parent;
      CHECK(successor.count(parent) == 0 || successor.at(parent) == now.get())
        << "The loops have to be in the same line";
      successor[parent] = now.get();
      if (bottom == nullptr) bottom = now.get();
      top = now.get();
      seen_loop.erase(now);
    }
  }
  // 3. check these loops are in the same scope(Block)
  CHECK(seen_loop.empty()) << "Loops have to be under the same scope";
  for (const StmtSRef& loop_sref : order)
    seen_loop.insert(loop_sref);

  // Reorder
  // 1. at first we decompose the loop into multiple loops to enable reorder with branches
  std::pair<std::vector<Stmt>, size_t> res = DecomposeLoop(top, bottom, &successor);
  // 2. reorder the res.second-th loop, which is the target loop
  res.first[res.second] =
      ReorderTarget(top, bottom, res.first[res.second], order, 0, seen_loop, successor);
  this->Replace(GetRef<StmtSRef>(top), SeqStmt::Flatten(res.first));
}

}  // namespace tir
}  // namespace tvm
