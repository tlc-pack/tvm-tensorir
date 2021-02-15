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
#include <tvm/arith/iter_affine_map.h>

#include "./schedule_common.h"

namespace tvm {
namespace tir {

/*! \brief Append a new predicate to the each children of type BlockRealize */
class PredicateUpdater : public StmtMutator {
 public:
  /*!
   * \brief Constructor
   * \param predicate The predicate to be apppend to BlockRealizeNode
   */
  explicit PredicateUpdater(const PrimExpr& predicate) : predicate_(predicate) {}
  // For each direct child of type BlockRealizeNode, append the predicate
  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    // We do not recursively do this
    ObjectPtr<BlockRealizeNode> n = make_object<BlockRealizeNode>(*realize);
    n->predicate = n->predicate && predicate_;
    return BlockRealize(n);
  }

 private:
  /*! \brief The predicate to be added */
  const PrimExpr& predicate_;
};

class BlockRealizeRewriter : public StmtExprMutator {
 public:
  explicit BlockRealizeRewriter(
      const std::unordered_map<Var, Range, ObjectPtrHash, ObjectPtrEqual>& loop_map) {
    loop_map_.insert(loop_map.begin(), loop_map.end());
  }

  Stmt VisitStmt_(const LoopNode* op) final {
    loop_map_[op->loop_var] = Range::FromMinExtent(op->min, op->extent);
    Stmt res = StmtMutator::VisitStmt_(op);
    loop_map_.erase(op->loop_var);
    return res;
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    auto v =
        arith::IterMapSimplify(op->block->iter_vars, op->binding_values, loop_map_, op->predicate);
    if (v.same_as(op->binding_values)) {
      return GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->binding_values = std::move(v);
      return Stmt(n);
    }
  }

 private:
  std::unordered_map<Var, Range, ObjectPtrHash, ObjectPtrEqual> loop_map_;
};

Stmt RewriteBindings(const Stmt& stmt, const Array<StmtSRef>& loops) {
  std::unordered_map<Var, Range, ObjectPtrHash, ObjectPtrEqual> loop_map;
  for (const auto& sref : loops) {
    const auto* loop = sref->GetStmt<LoopNode>();
    loop_map[loop->loop_var] = Range::FromMinExtent(loop->min, loop->extent);
  }
  BlockRealizeRewriter rewriter(loop_map);
  return rewriter(stmt);
}

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
      loops.push_back(sch->stmt2ref.at(loop).operator->());
    }
    return true;
  });
  // Reverse to get bottom-up order
  std::reverse(loops.begin(), loops.end());
  return loops;
}

Array<StmtSRef> ScheduleNode::split(const StmtSRef& loop_sref, const PrimExpr& nparts,
                                    const PrimExpr& factor) {
  // Equivalence
  // - The total repeat number has not changed for each direct child block with updating predicate.
  // - The execution order has not changed. (The block executes with the same args and the same
  // order with before.)
  const auto* loop = loop_sref->GetStmt<LoopNode>();
  CHECK(loop != nullptr) << "TypeError: 'split' expects a loop, but get type: "
                         << loop_sref->stmt->GetTypeKey();
  // Currently, can not split Loops with annotations
  CHECK(loop->annotations.empty())
      << "ValueError: 'split' expects loops without annotation, but 'loop' has: "
      << loop->annotations;
  arith::Analyzer analyzer;
  CHECK(analyzer.CanProve(loop->min == 0))
      << "ValueError: Only support loop starting with 0 for now";
  // Step 1. Replace all occurrence of the original loop var with new variables
  Var outer_var = loop->loop_var.copy_with_suffix("_outer");
  Var inner_var = loop->loop_var.copy_with_suffix("_inner");
  // TODO(@junrushao1994): use Optional<PrimExpr> instead
  Stmt new_loop_body = SubstituteInScope(loop->body, [&](const VarNode* v) -> PrimExpr {
    if (v == loop->loop_var.get()) {
      return outer_var * factor + inner_var;
    } else {
      return NullValue<PrimExpr>();
    }
  });
  // Step 2. Update predicate to guard the loop
  PrimExpr outer_min = 0;
  PrimExpr outer_extent = nparts;
  PrimExpr inner_min = 0;
  PrimExpr inner_extent = factor;
  analyzer.Bind(outer_var, Range::FromMinExtent(outer_min, outer_extent));
  analyzer.Bind(inner_var, Range::FromMinExtent(inner_min, inner_extent));
  PrimExpr predicate = outer_var * factor + inner_var < loop->extent;
  if (!analyzer.CanProve(predicate)) {
    new_loop_body = PredicateUpdater(predicate)(new_loop_body);
  }
  // Step 3. Generate two nested loops to replace the original loop
  Loop inner_loop(inner_var, inner_min, inner_extent, loop->annotations, new_loop_body);
  Loop outer_loop(outer_var, outer_min, outer_extent, loop->annotations, inner_loop);
  outer_loop = Downcast<Loop>(RewriteBindings(outer_loop, GetLoopsInScope(loop_sref)));
  this->Replace(loop_sref, outer_loop);
  return {stmt2ref.at(outer_loop.get()), stmt2ref.at(outer_loop->body.get())};
}

StmtSRef ScheduleNode::fuse(const StmtSRef& outer_sref, const StmtSRef& inner_sref) {
  // Equivalence
  // - The total repeat number has not changed for each direct child block.
  // - The execution order has not changed. (The block executes with the same
  //   args and the same order with before.)

  // Can only fuse neighbor loop without any extra branches.
  // Future enhancement: this condition can be eliminated by lifting all siblings of inner
  // as the children of the father of outer
  const auto* outer = outer_sref->GetStmt<LoopNode>();
  const auto* inner = inner_sref->GetStmt<LoopNode>();
  CHECK(outer != nullptr) << "TypeError: 'fuse' expects 'outer' as a loop, but get type: "
                          << outer_sref->stmt->GetTypeKey();
  CHECK(inner != nullptr) << "TypeError: 'fuse' expects 'inner' as a loop, but get type: "
                          << inner_sref->stmt->GetTypeKey();
  CHECK(outer->annotations.empty())
      << "ValueError: 'fuse' expects loops without annotation, but 'outer' has: "
      << outer->annotations;
  CHECK(inner->annotations.empty())
      << "ValueError: 'fuse' expects loops without annotation, but 'inner' has: "
      << inner->annotations;
  // Step 1. Check `inner` is the only children of `outer` and they are in the same scope
  CHECK(inner_sref->parent == outer_sref.get())
      << "ValueError: 'fuse' expects 'outer' to be parent of 'inner'";
  Array<Stmt> outer_children = GetChildren(GetRef<Stmt>(outer));
  CHECK(outer_children.size() == 1 && outer_children[0].get() == inner)
      << "ValueError: 'fuse' expects 'inner' to be the only child of 'outer'";
  CHECK(GetParentBlockSRef(outer_sref).get() == GetParentBlockSRef(inner_sref).get())
      << "ValueError: 'fuse' expects 'inner' and 'outer' to be in the same block scope";
  // Step 2. Create fused loop var and replace the loop var used in inner and outer loop
  arith::Analyzer analyzer;
  CHECK(analyzer.CanProve(inner->min == 0))
      << "ValueError: Only support inner loop starting with 0 for now";
  CHECK(analyzer.CanProve(outer->min == 0))
      << "ValueError: Only support outer loop starting with 0 for now";
  Var fused_var = outer->loop_var.copy_with_suffix("_" + inner->loop_var->name_hint + "_fused");
  Stmt new_loop_body = SubstituteInScope(inner->body, [&](const VarNode* v) -> PrimExpr {
    if (GetRef<Var>(v).same_as(outer->loop_var)) {
      return floordiv(fused_var, inner->extent) + outer->min;
    } else if (GetRef<Var>(v).same_as(inner->loop_var)) {
      return floormod(fused_var, inner->extent) + inner->min;
    } else {
      return NullValue<PrimExpr>();
    }
  });
  // Step 3. Generate a loop to replace the original two nested loops
  PrimExpr fused_min = 0;
  PrimExpr fused_extent = analyzer.Simplify(outer->extent * inner->extent);
  Loop fused_loop = Loop(fused_var, fused_min, fused_extent, outer->annotations, new_loop_body);
  fused_loop = Downcast<Loop>(RewriteBindings(fused_loop, GetLoopsInScope(outer_sref)));
  this->Replace(outer_sref, fused_loop);
  return stmt2ref.at(fused_loop.get());
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
    const StmtSRefNode* loop_sref_ptr = loop_sref.operator->();
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
    loop = stmt2ref[children[0].get()].operator->();
  }
  // Check 5. the block below has all its block_var to be data_par or reduce
  for (const IterVar& iter_var : block->iter_vars) {
    IterVarType kind = iter_var->iter_type;
    // TODO(@junrushao1994): remove kThreadIndex
    CHECK(kind == kDataPar || kind == kCommReduce || kind == kThreadIndex)
        << "ValueError: 'reorder' expects block var to be data parallel or reduction";
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

struct Internal {
  static StmtSRef Fuse(Schedule self, StmtSRef outer, StmtSRef inner) {
    return self->fuse(outer, inner);
  }
  static Array<StmtSRef> SplitByFactor(Schedule self, StmtSRef loop_sref, PrimExpr factor) {
    const auto* loop = loop_sref->GetStmt<LoopNode>();
    return self->split(loop_sref, floordiv(loop->extent + factor - 1, factor), factor);
  }
  static Array<StmtSRef> SplitByNParts(Schedule self, StmtSRef loop_sref, PrimExpr nparts) {
    const auto* loop = loop_sref->GetStmt<LoopNode>();
    return self->split(loop_sref, nparts, floordiv(loop->extent + nparts - 1, nparts));
  }
  static void Reorder(Schedule self, Array<StmtSRef> order) { self->reorder(order); }
};

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleFuse").set_body_typed(Internal::Fuse);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleSplitByFactor").set_body_typed(Internal::SplitByFactor);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleSplitByNParts").set_body_typed(Internal::SplitByNParts);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleReorder").set_body_typed(Internal::Reorder);

}  // namespace tir
}  // namespace tvm
