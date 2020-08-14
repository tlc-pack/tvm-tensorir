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

#include <tvm/arith/int_set.h>
#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt_functor.h>

#include "schedule_common.h"

namespace tvm {
namespace tir {

class CheckOneLine : public StmtVisitor {
 public:
  void VisitStmt_(const SeqStmtNode* op) final { legal = false; }
  bool legal = true;
};

StmtSRef ScheduleNode::blockize(const StmtSRef& sref) {
  /*!
   * Check:
   *   - The sub AST is one-line with only one block
   *
   * Mutate:
   *   - extra block var from the only block
   *   - Update block binding
   */
  const auto* l = sref->GetStmt<LoopNode>();
  CHECK(l) << "Only support blockize a loop for now";
  CheckOneLine checker;
  checker(GetRef<Stmt>(sref->stmt));
  CHECK(checker.legal) << "Only one line subtree can be blockize";

  Array<StmtSRef> child_blocks = GetChildBlocks(sref);
  CHECK_EQ(child_blocks.size(), 1);
  const auto& block_sref = *(child_blocks.begin());
  const auto& block_realize = GetBlockRealize(block_sref);
  const auto& inner_block = block_realize->block;

  std::vector<const LoopNode*> loops;
  std::unordered_map<const VarNode*, arith::IntSet> vmap;
  auto now = block_sref;
  while (now != sref) {
    now = GetRef<StmtSRef>(now->parent);
    const auto* loop = now->GetStmt<LoopNode>();
    CHECK(loop);
    loops.push_back(loop);
    vmap[loop->loop_var.get()] = arith::IntSet::FromRange(Range(loop->min, loop->extent));
  }

  // Update AST
  Array<IterVar> iter_vars;
  for (const auto& iter_var : inner_block->iter_vars) {
    iter_vars.push_back(
        IterVar(iter_var->dom, iter_var->var.copy_with_suffix(""), iter_var->iter_type));
  }

  Array<PrimExpr> values;
  std::unordered_map<Var, PrimExpr, ObjectHash, ObjectEqual> var_map;
  for (size_t i = 0; i < inner_block->iter_vars.size(); ++i) {
    const auto& var = iter_vars[i]->var;
    const auto& value = block_realize->binding_values[i];
    auto expr = arith::EvalSet(value, vmap).min();
    values.push_back(expr);
    var_map[var] = expr;
  }

  arith::Analyzer analyzer;
  MatchingSimplifier matching_simplifier(var_map, &analyzer);

  // Update inner block realize
  Array<PrimExpr> new_bindings;
  for (const auto& value : block_realize->binding_values) {
    new_bindings.push_back(matching_simplifier(value));
  }
  auto n = make_object<BlockRealizeNode>(*block_realize.get());
  n->binding_values = std::move(new_bindings);

  // Regenerate loops
  Stmt s = BlockRealize(n);
  for (const auto& loop : loops) {
    auto loop_node = make_object<LoopNode>(*loop);
    loop_node->body = s;
    s = Loop(loop_node);
  }

  // Calculate new block region
  std::vector<TensorRegion> reads, writes;
  RelaxRegion(block_sref, GetRef<StmtSRef>(sref->parent), &reads, &writes);

  auto rewrite_region = [&matching_simplifier](std::vector<TensorRegion> *regions) {
    for (size_t i = 0; i < regions->size(); ++i) {
      auto tensor_region = (*regions)[i];
      Region region;
      for (const auto& range : tensor_region->region) {
        region.push_back(Range::FromMinExtent(matching_simplifier(range->min),
                                              matching_simplifier(range->extent)));
      }
      (*regions)[i] = TensorRegion(tensor_region->buffer, region);
    }
  };
  rewrite_region(&reads);
  rewrite_region(&writes);

  auto outer_block = Block(iter_vars, reads, writes, s, Array<BufferAllocate>(),
                           Array<Annotation>(), "blockized_" + inner_block->tag);

  auto outer_realize = BlockRealize(values, IntImm(DataType::Bool(), 1), outer_block);
  this->Replace(sref, outer_realize);
  // Check loop binding
  // TODO(Siyuan): enhance validation
  this->ValidateLoops();

  return this->stmt2ref.at(outer_block.get());
}
}  // namespace tir
}  // namespace tvm
