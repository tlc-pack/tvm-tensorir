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

bool ListContains(const Array<StmtSRef>& list, const StmtSRef& element) {
  for (const StmtSRef& ele : list) {
    if (ele.same_as(element)) {
      return true;
    }
  }
  return false;
}

StmtSRef ScheduleNode::decompose_reduction(const StmtSRef& block_sref, const StmtSRef& loop_sref) {
  /*!
   *  Check
   *    - block is reduction
   *    - loop is higher than all the loops related to reduce block var
   *  Mutate
   *    - generate loops related to data par block vars
   *    - generate corresponding init block and update block
   */
  // A bunch of type checkings
  const auto* block = block_sref->GetStmt<BlockNode>();
  const auto* loop = loop_sref->GetStmt<LoopNode>();
  CHECK(block != nullptr)
      << "TypeError: 'decompose_reduction' expect a block as first argument, but get type: "
      << block_sref->stmt->GetTypeKey();
  CHECK(loop != nullptr)
      << "TypeError: 'decompose_reduction' expect a loop as second argument, but get type: "
      << loop_sref->stmt->GetTypeKey();
  const auto* reduce_step = block->body.as<ReduceStepNode>();
  CHECK(reduce_step != nullptr) << "TypeError: 'decompose_reduction' expects the body of the block "
                                   "is ReduceStep, but get type: "
                                << block->body->GetTypeKey();
  const auto* buffer_load = reduce_step->lhs.as<BufferLoadNode>();
  CHECK(buffer_load != nullptr)
      << "TypeError: 'decompose_reduction' expects the body of the reduce step "
         "is BufferLoad, but get type: "
      << reduce_step->lhs->GetTypeKey();
  Array<StmtSRef> loops = GetLoopsInScope(block_sref);
  const BlockRealizeNode* realize = GetBlockRealize(block_sref).get();
  // Cond 0. Check loop_sref is block_sref's ancestor
  CHECK(ListContains(loops, loop_sref))
      << "ValueError: 'decompose_reduction' expect the loop to be an ancestor of block";
  // Cond 1. Check block is reduction
  CHECK(GetParentScope(block_sref).IsReduction(block_sref))
      << "decompose_reduction expect the block to be a reduction block";
  // Cond 2. Check 'loop' is higher than all the loops related to block var of type reduction
  for (int i = 0, n = block->iter_vars.size(); i < n; ++i) {
    // For each block var of type kCommReduce, check its binding
    const IterVar& iter_var = block->iter_vars[i];
    const PrimExpr& binding = realize->binding_values[i];
    if (iter_var->iter_type != IterVarType::kCommReduce) {
      continue;
    }
    for (const StmtSRef& higher_loop : loops) {
      // Only check loops higher than the target loop
      if (higher_loop.same_as(loop_sref)) {
        break;
      }
      // loop_var of a higher loop shouldn't contain loop var
      const Var& loop_var = higher_loop->GetStmt<LoopNode>()->loop_var;
      CHECK(!ExprContainsVar(binding, loop_var))
          << "ValueError: 'decompose_reduction' expect the loop to be higher "
             "than all the loops related to reduce block var";
    }
  }
  // Mutate
  ObjectPtr<BlockNode> init_block = make_object<BlockNode>();
  ObjectPtr<BlockRealizeNode> init_realize = make_object<BlockRealizeNode>();
  init_block->tag = block->tag + "_init";
  init_realize->binding_values = {};
  init_realize->predicate = realize->predicate;
  init_realize->block = Block(init_block);
  // Step 1. Create new block vars and their bindings
  // Maps an old block var to the new corresponding block var
  std::unordered_map<const VarNode*, const VarNode*> block_var_map;
  for (int i = 0, n = block->iter_vars.size(); i < n; ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    const PrimExpr& binding = realize->binding_values[i];
    // Only process data parallel block vars
    if (iter_var->iter_type != IterVarType::kDataPar) {
      continue;
    }
    // Create a new block var
    IterVar new_iter_var(/*dom=*/iter_var->dom,
                         /*var=*/iter_var->var.copy_with_suffix("_init"),
                         /*iter_type=*/iter_var->iter_type,
                         /*thread_tag=*/iter_var->thread_tag);
    // Add a block var and its binding
    init_block->iter_vars.push_back(new_iter_var);
    init_realize->binding_values.push_back(binding);
    // Add a mapping from old block vars to new block vars
    block_var_map[iter_var->var.get()] = new_iter_var->var.get();
  }
  // Step 2. After copying block vars, substitute them in init block
  init_block->body = SubstituteInScope(
      BufferStore(buffer_load->buffer, reduce_step->comm_reducer->identity_element[0],
                  buffer_load->indices),
      block_var_map);
  for (const TensorRegion& write : block->writes) {
    init_block->writes.push_back(SubstituteTensorRegion(write, block_var_map));
  }
  // Step 3. Create loops above the init block
  Stmt body = BlockRealize(init_realize);
  for (int i = static_cast<int>(loops.size()) - 1; i >= 0; --i) {
    const auto* higher_loop = loops[i]->GetStmt<LoopNode>();
    for (const auto& expr : init_realize->binding_values) {
      // Skip irrelavent loops
      if (!ExprContainsVar(expr, higher_loop->loop_var)) {
        continue;
      }
      // Create a new equivalent to the loop
      Var old_loop_var = higher_loop->loop_var;
      Var new_loop_var = old_loop_var.copy_with_suffix("_init");
      std::unordered_map<const VarNode*, const VarNode*> var_map = {
          {old_loop_var.get(), new_loop_var.get()}};
      body = Loop(/*loop_var=*/new_loop_var,
                  /*min=*/higher_loop->min,
                  /*extent=*/higher_loop->extent,
                  /*annotations=*/higher_loop->annotations,
                  /*body=body*/ SubstituteInScope(body, var_map));
    }
    // Only consider loops higher than the given loop
    if (loops[i].same_as(loop_sref)) {
      break;
    }
  }
  // Step 4. Create the parent of the new loop
  if (const auto* parent = loop_sref->parent->GetStmt<LoopNode>()) {
    this->Replace(GetRef<StmtSRef>(loop_sref->parent),
                  Loop(/*loop_var=*/parent->loop_var,
                       /*min=*/parent->min,
                       /*extent=*/parent->extent,
                       /*annotations=*/parent->annotations,
                       /*body=*/SeqStmt::Flatten(Array<Stmt>{body, parent->body})));
  } else if (const auto* parent = loop_sref->parent->GetStmt<BlockNode>()) {
    this->Replace(GetRef<StmtSRef>(loop_sref->parent),
                  Block(/*iter_vars=*/parent->iter_vars,
                        /*reads=*/parent->reads,
                        /*writes=*/parent->writes,
                        /*body=*/SeqStmt::Flatten(Array<Stmt>{body, parent->body}),
                        /*allocations=*/parent->allocations,
                        /*annotations=*/parent->annotations,
                        /*tag=*/parent->tag));
  } else {
    LOG(FATAL) << "TyepError: 'decompose_reduction' is applied to loop whose parent's type is not "
                  "unsupported: "
               << loop_sref->parent->stmt->GetTypeKey();
  }
  // Step 5. Change the reduction block to update block
  Block update_block(
      /*iter_vars=*/block->iter_vars,
      /*reads=*/block->reads,
      /*writes=*/block->writes,
      /*body=*/BufferStore(buffer_load->buffer, reduce_step->ApplyCombiner(), buffer_load->indices),
      /*allocations=*/block->allocations,
      /*annotations=*/block->annotations,
      /*tag=*/block->tag + "_update");
  this->Replace(block_sref, update_block, {{update_block, GetRef<Block>(block)}});
  // Update scope information
  UpdateScope(GetParentBlockSRef(block_sref)->stmt, this->stmt2ref, &this->scopes);
  return stmt2ref.at(init_block.get());
}

void ScheduleNode::merge_reduction(const StmtSRef& init_sref, const StmtSRef& update_sref) {
  /*!
   * Check
   *   - init_block is under the same scope with update_sref
   *   - LCA is higher than all the loops related to update_block's reduce block var
   *   - init_block's write region is the same as update_block's write region under LCA
   *   - the merged block is decomposable (i.e satisfying the check's of decompose_reduction)
   * Mutate
   *   - delete init_block
   *   - generate reduction block
   */

  // Check
  const auto* init = init_sref->GetStmt<BlockNode>();
  const auto* update = update_sref->GetStmt<BlockNode>();
  CHECK(init != nullptr) << "merge_reduction expect a block as first argument";
  CHECK(update != nullptr) << "merge_reduction expect a block as second argument";

  // Check init_block is under the same scope with update_sref
  CHECK_EQ(GetParentBlockSRef(init_sref), GetParentBlockSRef(update_sref))
      << "merge_reduction expect the init_block and update_block to be under the same scope";
  const auto& scope_root = GetParentBlockSRef(init_sref);

  // Check init_block's write region is the same as update_block's write region under LCA
  CHECK_EQ(init->writes.size(), 1);
  CHECK_EQ(update->writes.size(), 1);
  const auto& lca = LowestCommonAncestor({init_sref, update_sref}, scope_root);
  const auto& init_region = RelaxRegion(init_sref, lca, init->writes[0]);
  const auto& update_region = RelaxRegion(update_sref, lca, update->writes[0]);
  CHECK_EQ(init_region->region.size(), update_region->region.size());
  ExprDeepEqual equal;
  for (size_t i = 0; i < init_region->region.size(); ++i) {
    CHECK(equal(init_region->region[i]->min, update_region->region[i]->min));
    CHECK(equal(init_region->region[i]->extent, update_region->region[i]->extent));
  }

  // Check the merged block is decomposable
  CHECK(this->scopes.at(scope_root).CanMergeReduction(init_sref, update_sref));
  const auto* init_body = init->body.as<BufferStoreNode>();
  const auto* update_body = update->body.as<BufferStoreNode>();

  // Check LCA is higher than all the loops related to update_block's reduce block var
  Array<StmtSRef> loops = GetLoopsInScope(update_sref);
  const auto* br = GetBlockRealize(update_sref).operator->();
  if (!scope_root.same_as(lca)) {
    for (const auto& loop : loops) {
      if (loop.same_as(lca)) break;
      const auto* loop_ptr = loop->GetStmt<LoopNode>();
      for (size_t i = 0; i < update->iter_vars.size(); ++i) {
        if (update->iter_vars[i]->iter_type == IterVarType::kCommReduce) {
          CHECK(!ExprContainsVar(br->binding_values[i], loop_ptr->loop_var))
              << "merge_reduction expect lca to be higher than all the loops related to "
                 "update_block's reduce block var";
        }
      }
    }
  }

  // Mutate
  // Delete init block and its single-branched ancestors
  const auto& removed = RemoveLeaf(init_sref, scope_root);
  this->Replace(lca, removed.second);

  // Change the update block to reduction block
  auto merged_block = make_object<BlockNode>(*update);
  merged_block->body = ReduceStep::FromInitUpdate(this->reducers_, init_body->value,
                                                  GetRef<BufferStore>(update_body));
  Map<Block, Block> block_map;
  block_map.Set(Block(merged_block), GetRef<Block>(update));
  this->Replace(update_sref, Block(merged_block), block_map);

  // update scope information
  UpdateScope(GetParentBlockSRef(update_sref)->stmt, this->stmt2ref, &this->scopes);
}

}  // namespace tir
}  // namespace tvm
