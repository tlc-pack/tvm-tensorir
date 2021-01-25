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

#include "../../arith/pattern_match.h"
#include "./schedule_common.h"

namespace tvm {
namespace tir {

bool CheckOneLine(const Stmt& s) {
  bool legal = true, meet_block = false;
  PostOrderVisit(s, [&legal, &meet_block](const ObjectRef& obj) {
    if (obj->IsInstance<SeqStmtNode>() && !meet_block) {
      legal = false;
    } else if (obj->IsInstance<BlockRealizeNode>()) {
      meet_block = true;
    }
  });
  return legal;
}

Array<arith::DivisionForm> TrivialSubspaceDivision(const Array<IterVar>& iter_vars,
                                                   const Array<PrimExpr>& bindings,
                                                   const std::vector<Var>& outer_loops,
                                                   const std::vector<Var>& inner_loops,
                                                   const PrimExpr& predicate) {
  if (!is_one(predicate)) return {};
  std::vector<arith::DivisionForm> res;
  for (size_t i = 0; i < bindings.size(); ++i) {
    bool outer = StmtExprContainsVar(bindings[i], outer_loops);
    bool inner = StmtExprContainsVar(bindings[i], inner_loops);
    if (outer && !inner) {
      res.emplace_back(arith::IterSumExpr({}, bindings[i]), iter_vars[i]->dom->extent,
                       arith::IterSumExpr({}, 0), 1);
    } else if (inner && !outer) {
      res.emplace_back(arith::IterSumExpr({}, 0), 1, arith::IterSumExpr({}, bindings[i]),
                       iter_vars[i]->dom->extent);
    } else {
      return {};
    }
  }
  return res;
}

StmtSRef ScheduleNode::blockize(const StmtSRef& loop_sref, const String& exec_scope) {
  /*!
   * Check:
   *   - The sub AST is one-line with only one block
   *
   * Mutate:
   *   - extra block var from the only block
   *   - Update block binding
   */
  const auto* loop = loop_sref->GetStmt<LoopNode>();
  CHECK(loop) << "TypeError: Only support blockize a loop for now, but get type: "
              << loop_sref->stmt->GetTypeKey();
  // check there exists no SeqStmt under loop
  CHECK(CheckOneLine(GetRef<Stmt>(loop))) << "ValueError: Only one line subtree can be blockize";
  // get the inner Block, BlockRealize and StmtSRef
  Array<StmtSRef> child_blocks = GetChildBlocks(loop_sref);
  CHECK_EQ(child_blocks.size(), 1) << "ValueError: Only one line subtree can be blockize";
  StmtSRef block_sref = child_blocks[0];
  BlockRealize block_realize = GetBlockRealize(block_sref);
  Block block = block_realize->block;
  // collect loops inside/outside loop_sref
  std::vector<const LoopNode*> outer_loops, inner_loops;
  std::vector<Var> outer_iters, inner_iters;
  std::unordered_map<Var, Range, ObjectPtrHash, ObjectPtrEqual> iters;
  bool inner = true;
  for (StmtSRef current_sref = block_sref;;) {
    current_sref = GetRef<StmtSRef>(current_sref->parent);
    if (!current_sref.defined()) break;
    const auto* current_loop = current_sref->GetStmt<LoopNode>();
    if (!current_loop) break;
    if (inner) {
      inner_loops.push_back(current_loop);
      inner_iters.push_back(current_loop->loop_var);
    } else {
      outer_loops.push_back(current_loop);
      outer_iters.push_back(current_loop->loop_var);
    }
    iters[current_loop->loop_var] = Range::FromMinExtent(current_loop->min, current_loop->extent);
    if (current_sref == loop_sref) inner = false;
  }
  arith::Analyzer analyzer;
  auto division = arith::SubspaceDivision(block_realize->binding_values, iters, inner_iters,
                                          block_realize->predicate, &analyzer);
  if (division.empty()) {
    // It is possible to blockize if we can not do perfect subspace division if we can divide
    // the block var bindings into two categories
    // 1. The binding covers no inner loop var
    // 2. The binding covers only inner loop vars
    division = TrivialSubspaceDivision(block->iter_vars, block_realize->binding_values, outer_iters,
                                       inner_iters, block_realize->predicate);
  }
  CHECK(!division.empty()) << "ValueError: The bindings of the block below can not be blockized";
  // Generate a new inner block
  arith::IterVarMapConverter converter(&analyzer);
  Array<IterVar> inner_block_vars, outer_block_vars;
  Array<PrimExpr> inner_bindings, outer_bindings;
  std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> block_var_no;
  std::unordered_map<Var, Range, ObjectPtrHash, ObjectPtrEqual> bv_iters;
  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    if (division[i]->IsOuter()) {
      // extract this iter var to outer block directly
      outer_bindings.push_back(converter.Convert(division[i]->outer));
      outer_block_vars.push_back(iter_var);
      bv_iters[iter_var->var] = Range::FromMinExtent(iter_var->var, 1);
    } else {
      const IterVar outer_var(Range::FromMinExtent(0, division[i]->outer_extent),
                              iter_var->var.copy_with_suffix("o"), iter_var->iter_type);
      outer_bindings.push_back(converter.Convert(division[i]->outer));
      outer_block_vars.push_back(outer_var);
      // generate a new iter var for outer block
      PrimExpr base = division[i]->IsInner() ? 0 : outer_var * division[i]->inner_extent;
      if (const auto* op = division[i]->inner.as<arith::IterSumExprNode>()) {
        base = base + op->base;
        inner_bindings.push_back(base + converter.Convert(arith::IterSumExpr(op->args, 0)));
      } else {
        inner_bindings.push_back(base + converter.Convert(division[i]->inner));
      }
      inner_block_vars.push_back(iter_var);
      bv_iters[iter_var->var] = Range::FromMinExtent(base, division[i]->inner_extent);
    }
    block_var_no[iter_var->var] = i;
  }
  Block inner_block = block;
  inner_block.CopyOnWrite()->iter_vars = inner_block_vars;
  BlockRealize inner_br = block_realize;
  inner_br.CopyOnWrite()->binding_values = inner_bindings;
  inner_br.CopyOnWrite()->predicate = division.back()->inner_extent;
  inner_br.CopyOnWrite()->block = inner_block;
  // Regenerate inner_loops
  Stmt body = inner_br;
  for (const auto& inner_loop : inner_loops) {
    auto loop_node = make_object<LoopNode>(*inner_loop);
    loop_node->body = body;
    body = Loop(loop_node);
  }
  // Calculate outer block's read/write region
  auto rewrite_range = [&](const Range& range) -> Range {
    auto res = arith::DetectIter(range->min, bv_iters, &analyzer);
    CHECK(bool(res));
    auto normalized_expr = res.value();
    PrimExpr extent = 1;
    if (normalized_expr->args.size() == 1) {
      CHECK(analyzer.CanProve(normalized_expr->args[0]->scale - range->extent == 0));
      extent = normalized_expr->args[0]->extent;
    }
    return Range::FromMinExtent(normalized_expr->base, extent * range->extent);
  };
  std::vector<TensorRegion> reads, writes;
  auto rewrite_region = [&](std::vector<TensorRegion>* regions, Array<TensorRegion> old_regions) {
    for (auto tensor_region : old_regions) {
      std::vector<Range> region;
      for (const auto& range : tensor_region->region) {
        region.push_back(rewrite_range(range));
      }
      (*regions).emplace_back(tensor_region->buffer, region);
    }
  };
  rewrite_region(&reads, block->reads);
  rewrite_region(&writes, block->writes);
  // Generate a new outer block
  auto outer_block = Block(outer_block_vars, reads, writes, body, Array<BufferAllocate>(),
                           Array<Annotation>(), "blockized_" + block->tag);
  auto outer_realize =
      BlockRealize(outer_bindings, division.back()->outer_extent, outer_block, exec_scope);

  this->Replace(loop_sref, outer_realize, {{inner_block, block}});
  UpdateScope(GetParentBlockSRef(this->stmt2ref.at(outer_block.get()))->stmt, this->stmt2ref,
              &this->scopes);
  UpdateScope(outer_block.get(), this->stmt2ref, &this->scopes);

  // Check loop binding
  this->ValidateLoops();
  return this->stmt2ref.at(outer_block.get());
}
}  // namespace tir
}  // namespace tvm
