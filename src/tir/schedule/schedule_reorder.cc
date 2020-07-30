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

std::vector<const StmtSRefNode*> GetLoopsPostOrder(const StmtSRef& root_sref,
                                                   const ScheduleNode* sch) {
  std::vector<const StmtSRefNode*> loops;
  // Gather all the loops under parent_block
  PreOrderVisit(root_sref->GetStmt<BlockNode>()->body, [&loops, sch](const ObjectRef& node) {
    // Stops at a new BlockNode
    if (node->IsInstance<BlockNode>()) {
      return false;
    }
    // Collects every LoopNode
    if (const auto* loop = node.as<LoopNode>()) {
      loops.push_back(sch->stmt2ref.at(loop).get());
    }
    return true;
  });
  // Reverse to get bottom-up order
  std::reverse(loops.begin(), loops.end());
  return loops;
}

void ScheduleNode::reorder(const Array<StmtSRef>& order) {
  /*
   * Check:
   * - check loops are in the same line and are single-branch
   * - the block below has all its block_var to be data_par or reduce.
   * Mutate:
   * - reorder the loops
   */
  CHECK(!order.empty()) << "ValueError: 'reorder' expects 'order' to be an non-empty list";
  // Check 1. type checks and uniqueness check
  std::unordered_set<const StmtSRefNode*> loops;
  for (const StmtSRef& loop_sref : order) {
    // type check
    const auto* loop = loop_sref->GetStmt<LoopNode>();
    CHECK(loop) << "TypeError: 'reorder' expects an array of loops, but get type: "
                << loop_sref->stmt->GetTypeKey();
    // uniqueness check
    const StmtSRefNode* loop_sref_ptr = loop_sref.get();
    CHECK_EQ(loops.count(loop_sref_ptr), 0U)
        << "ValueError: 'reorder' expects an array of unique array, but get duplicate: "
        << GetRef<Stmt>(loop_sref->stmt);
    loops.insert(loop_sref_ptr);
  }
  // Check 2. Loops are in the same line
  // The algorithm now is to scan the inverse DFS order of the whole loop tree in the scope.
  // For some Loop x, it is potentially in the reorder range if
  //   - x is in the reorder list
  //   - exactly 1 son y of x is potentially in the reorder range
  //     (If there are more, then the loops are not in the same line).
  //     Put (x, y) in the map.
  // If x is potentially in the reorder range, check x is single branch
  // After the inverse DFS, we can know how to catch the loop line by the map.
  // Top and bottom denotes the range of loops need reordering
  const StmtSRefNode* top = nullptr;
  const StmtSRefNode* bottom = nullptr;
  // Maps a parent to its child
  std::unordered_map<const StmtSRefNode*, const StmtSRefNode*> successor;
  // Gather all the loops under parent_block
  int n_loops_not_found = order.size();
  for (const StmtSRefNode* loop : GetLoopsPostOrder(GetParentBlockSRef(order[0]), this)) {
    bool is_in_reorder_list = loops.count(loop);
    bool has_inner_loop = successor.count(loop);
    if (is_in_reorder_list || has_inner_loop) {
      const StmtSRefNode* parent = loop->parent;
      // If the successor of `parent` exists, then it is not the current loop
      CHECK(!successor.count(parent))
          << "ValueError: 'reorder' expects the loops be in the same line";
      successor[parent] = loop;
      // `bottom` is the first loop encountered
      if (bottom == nullptr) {
        bottom = loop;
      }
      // `top` is the last loop encountered
      if (is_in_reorder_list) {
        top = loop;
        --n_loops_not_found;
      }
    }
  }
  // Check 3. Loops are in the same block scope
  CHECK_EQ(n_loops_not_found, 0)
      << "ValueError: 'reorder' expects loops to be under the same block scope";
  // Check 4. Loops are single-branch
  const BlockNode* block = nullptr;
  for (const StmtSRefNode* loop = top; !(block = loop->GetStmt<BlockNode>());) {
    Array<Stmt> children = GetChildren(GetRef<Stmt>(loop->stmt));
    CHECK_EQ(children.size(), 1) << "ValueError: 'reorder' expects the loops to be single-branch";
    loop = stmt2ref[children[0].get()].get();
  }
  // Check 5. the block below has all its block_var to be data_par or reduce
  for (const IterVar& iter_var : block->iter_vars) {
    IterVarType kind = iter_var->iter_type;
    // TODO(@junrushao1994): remove kThreadIndex
    CHECK(kind == kDataPar || kind == kCommReduce || kind == kThreadIndex)
        << "ValueError: 'reorder' expects block var to be data paralell or reduction";
  }
  std::function<Stmt(const StmtSRefNode*, int index)> f_reorder =
      [&bottom, &loops, &successor, &order, &f_reorder](const StmtSRefNode* loop,
                                                        int index) -> Stmt {
    // The order list maybe incomplete, so we may copy the old_loop rather than order
    const LoopNode* copy =
        loops.count(loop) ? order[index++]->GetStmt<LoopNode>() : loop->GetStmt<LoopNode>();
    ObjectPtr<LoopNode> n = make_object<LoopNode>(*copy);
    if (loop == bottom) {
      // bottom loop
      n->body = loop->GetStmt<LoopNode>()->body;
    } else {
      // reorder recursively
      n->body = f_reorder(successor.at(loop), index);
    }
    return Stmt(n);
  };
  this->Replace(GetRef<StmtSRef>(top), f_reorder(top, 0));
}

}  // namespace tir
}  // namespace tvm
