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
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt_functor.h>

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
  explicit PredicateUpdater(const PrimExpr& predicate) : predicate(predicate) {}
  // For each direct child of type BlockRealizeNode, append the predicate
  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    // We do not recursively do this
    ObjectPtr<BlockRealizeNode> n = make_object<BlockRealizeNode>(*realize);
    n->predicate = n->predicate && predicate;
    return BlockRealize(n);
  }
  /*! \brief The predicate to be added */
  const PrimExpr& predicate;
};

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
  this->Replace(loop_sref, outer_loop);
  return {stmt2ref.at(outer_loop.get()), stmt2ref.at(inner_loop.get())};
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
      << "ValueError: Only support outter loop starting with 0 for now";
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
  PrimExpr fused_extent = outer->extent * inner->extent;
  Loop fused_loop = Loop(fused_var, fused_min, fused_extent, outer->annotations, new_loop_body);
  this->Replace(outer_sref, fused_loop);
  return stmt2ref.at(fused_loop.get());
}

}  // namespace tir
}  // namespace tvm
