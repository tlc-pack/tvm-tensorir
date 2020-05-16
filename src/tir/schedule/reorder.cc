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
#include <queue>
#include <utility>
#include "schedule_common.h"

namespace tvm {
namespace tir {

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
    n->body = DowncastPtr<LoopNode>(old_loop->node)->body;
  } else {
    // reorder recursively
    n->body = ReorderTarget(successor.at(old_loop), bottom, order, new_index, seen_loop, successor);
  }
  return Stmt(n);
}

void ScheduleNode::reorder(const Array<StmtSRef>& order) {
  /*!
   * Check:
   *    - check loops are in the same line and are single-branch
   *    - the block below has all its block_var to be data_par or reduce.
   *
   * Mutate:
   *    - reorder the loops
   */

  // Check
  // 1. check loops are mutually different
  std::unordered_set<StmtSRef, ObjectHash, ObjectEqual> seen_loop;
  for (const StmtSRef& loop_sref : order) {
    const auto* loop = DowncastPtr<LoopNode>(loop_sref->node);
    CHECK(loop) << "Order has to be a list a Loops";
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
  // If x is potentially in the reorder range, check x is single branch
  // After the inverse DFS, we can know how to catch the loop line by the map.
  std::vector<const LoopNode*> all_loops = GatherChild<LoopNode>(GetScope(order[0])->node);
  std::unordered_map<const StmtSRefNode*, const StmtSRefNode*> successor;
  // top and bottom denotes the range of loops need reordering
  const StmtSRefNode* top = nullptr, * bottom = nullptr;
  for (auto it = all_loops.rbegin(); it != all_loops.rend(); ++it) {
    StmtSRef now = stmt2ref.at(*it);
    if (seen_loop.count(now) || successor.count(now.get())) {
      const StmtSRefNode* parent = now->parent;
      CHECK(successor.count(parent) == 0 || successor.at(parent) == now.get())
        << "reorder expects the loops have to be in the same line";
      successor[parent] = now.get();
      if (bottom == nullptr) bottom = now.get();
      if (seen_loop.count(now)) top = now.get();
      seen_loop.erase(now);
    }
  }
  // 3. check these loops are in the same scope(Block)
  CHECK(seen_loop.empty()) << "reorder expects loops to be under the same scope";
  for (const StmtSRef& loop_sref : order) {
    seen_loop.insert(loop_sref);
  }
  // 4. check these loops are single-branch
  const StmtSRefNode* now = top;
  for (; ;) {
    const auto& children = GetChildren(GetRef<Stmt>(now->node));
    CHECK_EQ(children.size(), 1) << "reorder expects the loops to be single-branch";
    now = stmt2ref[children[0].operator->()].operator->();
    if (now->node->IsInstance<BlockNode>()) break;
  }
  // 5. the block below has all its block_var to be data_par or reduce
  const auto* block = DowncastPtr<BlockNode>(now->node);
  CHECK(block);
  for (const auto & iter_var : block->iter_vars) {
    IterVarType var_type = iter_var->iter_type;
    CHECK(var_type == kDataPar || var_type == kThreadIndex || var_type == kCommReduce)
      << "reorder expects block var to be data_par or reduce";
  }
  // Reorder
  const auto& res = ReorderTarget(top, bottom, order, 0, seen_loop, successor);
  this->Replace(GetRef<StmtSRef>(top), res);
}

}  // namespace tir
}  // namespace tvm
