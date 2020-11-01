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
  CHECK(CheckOneLine(GetRef<Stmt>(loop_sref->stmt)))
      << "ValueError: Only one line subtree can be blockize";
  // get the inner block
  Array<StmtSRef> child_blocks = GetChildBlocks(loop_sref);
  CHECK_EQ(child_blocks.size(), 1) << "ValueError: Only one line subtree can be blockize";
  StmtSRef block_sref = child_blocks[0];
  BlockRealize block_realize = GetBlockRealize(block_sref);
  Block block = block_realize->block;
  // collect loops inside/outside loop_sref
  std::vector<const LoopNode*> outer_loops, inner_loops;
  std::vector<Var> inner_iters;
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
    }
    iters[current_loop->loop_var] = Range::FromMinExtent(current_loop->min, current_loop->extent);
    if (current_sref == loop_sref) inner = false;
  }
  arith::Analyzer analyzer;
  auto division = arith::SubspaceDivision(block_realize->binding_values, iters, inner_iters,
                                          block_realize->predicate, &analyzer);
  LOG(INFO) << "Division" << division;
  CHECK(!division.empty())
      << "ValueError: The bindings of the block below can not be blockized by loops under "
      << loop->loop_var;
  // Generate inner block
  arith::IterVarMapConverter converter(&analyzer);
  Array<IterVar> outer_block_vars;
  Array<PrimExpr> inner_bindings, outer_bindings;
  std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> block_var_no;
  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    const IterVar iter_var = block->iter_vars[i];
    const IterVar outer_var(Range::FromMinExtent(0, division[i]->outer_extent),
                            iter_var->var.copy_with_suffix("o"), iter_var->iter_type);
    outer_bindings.push_back(converter.Convert(division[i]->outer));
    outer_block_vars.push_back(outer_var);

    PrimExpr base = is_one(division[i]->outer_extent) ? 0 : outer_var * division[i]->inner_extent;
    if (const auto* op = division[i]->inner.as<arith::IterSumExprNode>()) {
      base = base + op->base;
      inner_bindings.push_back(base + converter.Convert(arith::IterSumExpr(op->args, 0)));
    } else {
      inner_bindings.push_back(base + converter.Convert(division[i]->inner));
    }
    block_var_no[iter_var->var] = i;
    LOG(INFO) << "outer_var " << outer_var << " = " << outer_bindings.back();
    LOG(INFO) << "inner_var " << iter_var << " = " << inner_bindings.back();
  }
  BlockRealize inner_br = block_realize;
  inner_br.CopyOnWrite()->binding_values = inner_bindings;
  inner_br.CopyOnWrite()->predicate = division.back()->inner_extent;
  // Regenerate inner_loops
  Stmt body = inner_br;
  for (const auto& inner_loop : inner_loops) {
    auto loop_node = make_object<LoopNode>(*inner_loop);
    loop_node->body = body;
    body = Loop(loop_node);
  }
  // Calculate outer block's IO region
  auto rewrite_range = [&](const Range& range) -> Range {
    // Detect that the range is under valid pattern
    arith::PVar<Var> v;
    arith::PVar<PrimExpr> d, c;
    // LOG(INFO) << range;
    if (c.Match(range->extent) && (d + v * c).Match(range->min)) {
      // [d + v*c: d + v*c + c]
      auto it = block_var_no.find(v.Eval());
      CHECK(it != block_var_no.end())
          << "ValueError: The TensorRegion of the block below can not be blockized";
      PrimExpr base = analyzer.Simplify(d.Eval() * c.Eval());
      PrimExpr extent = analyzer.Simplify(division[it->second]->inner_extent * c.Eval());
      // LOG(INFO) << base << " " << outer_block_vars[it->second] << " " << extent;
      return Range::FromMinExtent(base + outer_block_vars[it->second] * extent, extent);
    } else if (c.Match(range->extent) && (v * c).Match(range->min)) {
      // [v*c : v*c + c]
      auto it = block_var_no.find(v.Eval());
      CHECK(it != block_var_no.end())
          << "ValueError: The TensorRegion of the block below can not be blockized";
      PrimExpr extent = analyzer.Simplify(division[it->second]->inner_extent * c.Eval());
      // LOG(INFO) << outer_block_vars[it->second] << " " << extent;
      return Range::FromMinExtent(outer_block_vars[it->second] * extent, extent);
    } else if (is_one(range->extent) && v.Match_(range->min)) {
      // [v : v + 1]
      auto it = block_var_no.find(v.Eval());
      CHECK(it != block_var_no.end())
          << "ValueError: The TensorRegion of the block below can not be blockized";
      PrimExpr extent = analyzer.Simplify(division[it->second]->inner_extent);
      // LOG(INFO) << outer_block_vars[it->second] << " " << extent;
      return Range::FromMinExtent(outer_block_vars[it->second] * extent, extent);
    } else {
      LOG(FATAL) << "ValueError: The TensorRegion of the block below can not be blockized";
      return Range(0, 0);
    }
  };

  std::vector<TensorRegion> reads, writes;
  auto rewrite_region = [&](std::vector<TensorRegion>* regions, Array<TensorRegion> old_regions) {
    for (size_t i = 0; i < old_regions.size(); ++i) {
      auto tensor_region = old_regions[i];
      std::vector<Range> region;
      for (const auto& range : tensor_region->region) {
        region.push_back(rewrite_range(range));
      }
      (*regions).push_back(TensorRegion(tensor_region->buffer, region));
    }
  };
  rewrite_region(&reads, block->reads);
  rewrite_region(&writes, block->writes);

  auto outer_block = Block(outer_block_vars, reads, writes, body, Array<BufferAllocate>(),
                           Array<Annotation>(), "blockized_" + block->tag);
  auto outer_realize =
      BlockRealize(outer_bindings, division.back()->outer_extent, outer_block, exec_scope);

  this->Replace(loop_sref, outer_realize);
  LOG(INFO) << this->func;
  // Check loop binding
  this->ValidateLoops();

  return this->stmt2ref.at(outer_block.get());
}
}  // namespace tir
}  // namespace tvm
