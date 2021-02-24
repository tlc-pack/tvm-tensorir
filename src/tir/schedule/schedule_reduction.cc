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

bool ListContainsElement(const Array<StmtSRef>& list, const StmtSRef& element) {
  for (const StmtSRef& ele : list) {
    if (ele.same_as(element)) {
      return true;
    }
  }
  return false;
}

StmtSRef ScheduleNode::decompose_reduction(const StmtSRef& block_sref,
                                           const Optional<StmtSRef>& loop_sref_opt) {
  /*!
   *  Check
   *    - block is reduction
   *    - loop is higher than all the loops related to reduce block var, or loop is None
   *  Mutate
   *    - If loop is not None:
   *      - generate loops related to data par block vars
   *      - generate corresponding init block and update block
   *    - If loop is None:
   *      - substitute `tir.init()` with IfThenElse statement
   */
  // A bunch of type checking
  ICHECK(block_sref.defined())
      << "ValueError: 'decompose_reduction' expect a block as first argument, but get value 'None'";
  const auto* block = block_sref->GetStmt<BlockNode>();
  if (loop_sref_opt) {
    // 'loop' is not 'None'.
    StmtSRef loop_sref = loop_sref_opt.value();
    const auto* loop = loop_sref->GetStmt<ForNode>();
    CHECK(block != nullptr)
        << "TypeError: 'decompose_reduction' expect a block as first argument, but get type: "
        << block_sref->stmt->GetTypeKey();
    CHECK(loop != nullptr)
        << "TypeError: 'decompose_reduction' expect a loop as second argument, but get type: "
        << loop_sref->stmt->GetTypeKey();
    CHECK(block->init.defined()) << "ValueError: 'decompose_reduction' expect a reduction block, "
                                    "but the block has no init block";
    Array<StmtSRef> loops = GetAxes(block_sref);
    const BlockRealizeNode* realize = GetBlockRealize(block_sref).get();
    // Cond 0. Check loop_sref is an ancestor of block_sref
    CHECK(ListContainsElement(loops, loop_sref))
        << "ValueError: 'decompose_reduction' expect the loop to be an ancestor of block";
    // Cond 1. Check block is reduction
    CHECK(GetParentScope(block_sref)->IsReduction(block_sref))
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
        const Var& loop_var = higher_loop->GetStmt<ForNode>()->loop_var;
        CHECK(!StmtExprContainsVar(binding, loop_var))
            << "ValueError: 'decompose_reduction' expect the loop to be higher "
               "than all the loops related to reduce block var";
      }
    }
    // Mutate
    ObjectPtr<BlockNode> init_block = make_object<BlockNode>();
    ObjectPtr<BlockRealizeNode> init_realize = make_object<BlockRealizeNode>();
    init_block->name_hint = block->name_hint + "_init";
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
    init_block->body = SubstituteInScope(block->init.value(), block_var_map);
    for (const BufferRegion& write : block->writes) {
      init_block->writes.push_back(SubstituteBufferRegion(write, block_var_map));
    }
    // Step 3. Create loops above the init block
    Stmt body = BlockRealize(init_realize);
    for (int i = static_cast<int>(loops.size()) - 1; i >= 0; --i) {
      const auto* higher_loop = loops[i]->GetStmt<ForNode>();
      for (const PrimExpr& expr : init_realize->binding_values) {
        // Skip irrelevant loops
        if (!StmtExprContainsVar(expr, higher_loop->loop_var)) {
          continue;
        }
        // Create a new equivalent to the loop
        Var old_loop_var = higher_loop->loop_var;
        Var new_loop_var = old_loop_var.copy_with_suffix("_init");
        std::unordered_map<const VarNode*, const VarNode*> var_map = {
            {old_loop_var.get(), new_loop_var.get()}};
        body = For(/*loop_var=*/new_loop_var,
                   /*min=*/higher_loop->min,
                   /*extent=*/higher_loop->extent,
                   /*kind=*/ForKind::kSerial,
                   /*body=body*/ SubstituteInScope(body, var_map));
      }
      // Only consider loops higher than the given loop
      if (loops[i].same_as(loop_sref)) {
        break;
      }
    }
    // Step 4. Create the parent of the new loop
    if (const auto* parent = loop_sref->parent->GetStmt<ForNode>()) {
      this->Replace(GetRef<StmtSRef>(loop_sref->parent),
                    For(/*loop_var=*/parent->loop_var,
                        /*min=*/parent->min,
                        /*extent=*/parent->extent,
                        /*kind=*/parent->kind,
                        /*body=*/SeqStmt::Flatten(Array<Stmt>{body, parent->body}),
                        /*thread_binding*/parent->thread_binding,
                        /*annotations*/parent->annotations),
                    {});
    } else if (const auto* parent = loop_sref->parent->GetStmt<BlockNode>()) {
      auto block_node = make_object<BlockNode>(*parent);
      block_node->body = SeqStmt::Flatten(Array<Stmt>{body, parent->body});
      block_node->init = NullOpt;
      Block new_block = Block(block_node);
      this->Replace(GetRef<StmtSRef>(loop_sref->parent), new_block,
                    {{new_block, GetRef<Block>(parent)}});
    } else {
      LOG(FATAL)
          << "TypeError: 'decompose_reduction' is applied to loop whose parent's type is not "
             "unsupported: "
          << loop_sref->parent->stmt->GetTypeKey();
    }
    // Step 5. Change the reduction block to update block
    auto update_block_node = make_object<BlockNode>(*block);
    update_block_node->name_hint = block->name_hint + "_update";
    update_block_node->init = NullOpt;
    Block update_block(update_block_node);
    this->Replace(block_sref, update_block, {{update_block, GetRef<Block>(block)}});
    // Update scope information
    UpdateScope(GetParentBlockSRef(block_sref)->stmt, this->stmt2ref, &this->scopes);
    return stmt2ref.at(init_block.get());
  } else {
    // 'loop' is 'None'. Convert `tir.init()` to a conjunction of conditions.
    CHECK(block->init.defined()) << "ValueError: 'decompose_reduction' expect a reduction block, "
                                    "but the block has no init block";
    PrimExpr condition = const_true();
    for (const IterVar& var : block->iter_vars) {
      if (var->iter_type == IterVarType::kCommReduce) {
        condition = And(condition, EQ(var, var->dom->min));
      }
    }
    condition = arith::Analyzer().Simplify(condition);
    Stmt body;
    if (is_one(condition)) {
      body = block->body;
    } else {
      body = SeqStmt({IfThenElse(condition, block->init.value()), block->body});
    }
    auto block_node = make_object<BlockNode>(*block);
    block_node->name_hint = block->name_hint + "_update";
    block_node->init = NullOpt;
    block_node->body = body;
    Block new_block(block_node);

    this->Replace(block_sref, new_block, {{new_block, GetRef<Block>(block)}});
    // Update scope information
    UpdateScope(GetParentBlockSRef(block_sref)->stmt, this->stmt2ref, &this->scopes);
    return stmt2ref.at(new_block.get());
  }
}

void ScheduleNode::merge_reduction(const StmtSRef& init_sref, const StmtSRef& update_sref) {
  /*!
   * Check
   *   - init_sref is under the same scope with update_sref
   *   - LCA is higher than all the loops related to update_block's reduce block var
   *   - init_block's write region is the same as update_block's write region under LCA
   *   - the merged block is decomposable (i.e satisfying the check's of decompose_reduction)
   * Mutate
   *   - delete init_block
   *   - generate reduction block
   */
  // Type checks
  const auto* init = init_sref->GetStmt<BlockNode>();
  const auto* update = update_sref->GetStmt<BlockNode>();
  CHECK(init != nullptr) << "TypeError: 'merge_reduction' expects 'init' of type Block, but get: "
                         << init_sref->stmt->GetTypeKey();
  CHECK(update != nullptr)
      << "TypeError: 'merge_reduction' expects 'update' of type Block, but get: "
      << update_sref->stmt->GetTypeKey();
  const auto* init_body = init->body.as<BufferStoreNode>();
  const auto* update_body = update->body.as<BufferStoreNode>();
  CHECK(init_body != nullptr && update_body != nullptr)
      << "ValueError: 'merge_reduction' expects the body of init and update block to be "
         "BufferStore";
  Optional<CommReducer> reducer;
  Optional<PrimExpr> reducer_lhs, reducer_rhs;
  CommReducer::FromInitUpdate(init_body->value, GetRef<BufferStore>(update_body), reducer,
                              reducer_lhs, reducer_rhs, Span());
  CHECK(reducer.defined())
      << "ValueError: 'merge_reduction' pattern detect failed. No reducer pattern matched for "
      << init_body->value << " and " << GetRef<BufferStore>(update_body);
  const BlockRealizeNode* init_realize = GetBlockRealize(init_sref).get();
  const BlockRealizeNode* update_realize = GetBlockRealize(update_sref).get();
  ExprDeepEqual equal;
  CHECK(equal(init_realize->predicate, update_realize->predicate))
      << "ValueError: 'merge_reduction' expects the predicate of init and update to be the same";
  const StmtSRef& scope = GetParentBlockSRef(init_sref);
  StmtSRef lca = LowestCommonAncestor({init_sref, update_sref}, scope);
  // Cond 1. Check init_block is under the same scope with update_sref
  CHECK_EQ(scope.get(), GetParentBlockSRef(update_sref).get())
      << "TypeError: 'merge_reduction' expects the 'init' and 'update' to be under the same scope";
  // Cond 3. Write region of 'init' is the same as that of 'update' under LCA
  {
    CHECK_EQ(init->writes.size(), 1)
        << "ValueError: 'merge_reduction' expects 'init' with only one write region";
    CHECK_EQ(update->writes.size(), 1)
        << "ValueError: 'merge_reduction' expects 'update' with only one write region";
    BufferRegion init_region = RelaxRegion(init_sref, lca, init->writes[0]);
    BufferRegion update_region = RelaxRegion(update_sref, lca, update->writes[0]);
    CHECK_EQ(init_region->region.size(), update_region->region.size())
        << "ValueError: 'merge_reduction' has inconsistent ranks between the write region of "
           "'init' and that of 'update'";
    for (size_t i = 0; i < init_region->region.size(); ++i) {
      CHECK(equal(init_region->region[i]->min, update_region->region[i]->min) &&
            equal(init_region->region[i]->extent, update_region->region[i]->extent))
          << "ValueError: 'merge_reduction' has inconsistent write domain on axis " << i;
    }
  }
  // Cond 4. Check the merged block is decomposable
  CHECK(this->scopes.at(scope)->CanMergeReduction(init_sref, update_sref));
  // Cond 2. Check LCA is higher than all the loops related to update_block's reduce block var
  if (!scope.same_as(lca)) {
    for (const StmtSRef& higher_loop : GetAxes(update_sref)) {
      if (higher_loop.same_as(lca)) {
        break;
      }
      const Var& loop_var = higher_loop->GetStmt<ForNode>()->loop_var;
      for (int i = 0, n = update->iter_vars.size(); i < n; ++i) {
        const IterVar& iter_var = update->iter_vars[i];
        const PrimExpr& binding = update_realize->binding_values[i];
        if (iter_var->iter_type != IterVarType::kCommReduce) {
          continue;
        }
        CHECK(!StmtExprContainsVar(binding, loop_var))
            << "ValueError: 'merge_reduction' expects LCA "
               "to be higher than all the loops related to "
               "update_block's reduce block var";
      }
    }
  }
  // Mutate
  // Step 1. Delete init block and its single-branched ancestors
  std::pair<Stmt, Stmt> removed = RemoveLeaf(init_sref, scope);
  this->Replace(lca, removed.second, {});
  // Step 2. Change the update block to reduction block
  BufferStore new_init = GetRef<BufferStore>(update_body);
  new_init.CopyOnWrite()->value = init_body->value;
  auto merged_node = make_object<BlockNode>(*update);
  merged_node->init = new_init;
  Block merged(merged_node);
  this->Replace(update_sref, merged, {{merged, GetRef<Block>(update)}});
  // Update scope information
  UpdateScope(GetParentBlockSRef(update_sref)->stmt, this->stmt2ref, &this->scopes);
}

StmtSRef ScheduleNode::rfactor(const StmtSRef& loop_sref, int factor_axis) {
  const auto* loop = loop_sref->GetStmt<ForNode>();
  CHECK(loop) << "TypeError: Only support rfactor a loop for now, but get type: "
              << loop_sref->stmt->GetTypeKey();
  CHECK(CheckOneLine(GetRef<Stmt>(loop_sref->stmt)))
      << "ValueError: Only one line subtree can be rfactor";
  // get the inner block
  Array<StmtSRef> child_blocks = GetChildBlocks(loop_sref);
  CHECK_EQ(child_blocks.size(), 1) << "ValueError: Only one line subtree can be rfactor";
  StmtSRef block_sref = child_blocks[0];
  BlockRealize block_realize = GetBlockRealize(block_sref);
  Block block = block_realize->block;
  // Check the block is reduction block
  BlockScope scope = GetParentScope(block_sref);
  CHECK(scope->IsReduction(block_sref)) << "ValueError: can only rfactor a reduction block";
  // Collect the info of loop&block iter relation
  std::unordered_set<const VarNode*> data_par_loops, reduce_loops;
  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    std::unordered_set<const VarNode*>* set = nullptr;
    if (block->iter_vars[i]->iter_type == IterVarType::kDataPar) {
      set = &data_par_loops;
    } else if (block->iter_vars[i]->iter_type == IterVarType::kCommReduce) {
      set = &reduce_loops;
    }
    if (set != nullptr) {
      PreOrderVisit(block_realize->binding_values[i], [set] (const ObjectRef& node) {
        if (const auto* var = node.as<VarNode>()) {
          set->insert(var);
          return false;
        }
        return true;
      });
    }
  }
  // Get the init BufferStore and update buffer store
  const auto* init = block->init.as<BufferStoreNode>();
  const auto* update = block->body.as<BufferStoreNode>();
  CHECK(init) << "ValueError: the init of the block ought to be a BufferStore stmt";
  CHECK(update) << "ValueError: the body of the block ought to be a BufferStore stmt";
  Optional<CommReducer> reducer;
  Optional<PrimExpr> reducer_lhs, reducer_rhs;
  CommReducer::FromInitUpdate(init->value, GetRef<BufferStore>(update), reducer, reducer_lhs,
                              reducer_rhs, Span());
  CHECK(reducer.defined()) << "ValueError: 'merge_reduction' pattern detect failed. "
                           << "No reducer pattern matched for " << init->value << " and "
                           << GetRef<BufferStore>(update);
  ICHECK(reducer_lhs.defined() && reducer_rhs.defined());
  PrimExpr lhs = reducer_lhs.value();
  PrimExpr rhs = reducer_rhs.value();
  // Get the loops outside the block
  std::unordered_map<Var, Range, ObjectPtrHash, ObjectPtrEqual> iters;
  Array<StmtSRef> loops = GetAxes(block_sref);
  for (auto it = loops.rbegin(); it != loops.rend(); ++it) {
    const auto* l = (*it)->GetStmt<ForNode>();
    ICHECK(l) << "InternalError: GetAxes returns a block sref";
    CHECK(!data_par_loops.count(l->loop_var.get()) || !reduce_loops.count(l->loop_var.get()))
        << "ValueError: loop " << l->loop_var << " is related with both data_par and reduce iters ";
    iters[l->loop_var] = Range::FromMinExtent(l->min, l->extent);
  }
  // Do subspace division with subspace {loop}
  arith::Analyzer analyzer;
  Array<arith::DivisionForm> division =
      arith::SubspaceDivision(block_realize->block->iter_vars, block_realize->binding_values, iters,
                              {loop->loop_var}, block_realize->predicate, &analyzer);
  arith::IterVarMapConverter converter(&analyzer);
  CHECK(is_one(division.back()->inner_extent))
      << "ValueError: can not rfactor a loop related with predicate";
  // create rf block
  IterVar rf_iter(Range::FromMinExtent(loop->min, loop->extent),
                  Var("v" + loop->loop_var->name_hint), IterVarType::kDataPar);
  BlockRealize rf_block_realize = block_realize;
  Block rf_block = block;
  std::vector<PrimExpr> rf_bindings;
  std::vector<IterVar> rf_iters;
  std::unordered_map<const VarNode*, PrimExpr> var_map;
  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    if (block->iter_vars[i]->iter_type == IterVarType::kDataPar) {
      CHECK(division[i]->IsOuter())
          << "ValueError: can not rfactor a loop that touches data par block vars";
      if (!division[i]->IsInner()) {
        rf_bindings.push_back(converter.Convert(division[i]->outer));
        IterVar new_iter = block->iter_vars[i];
        new_iter.CopyOnWrite()->dom = Range::FromMinExtent(0, division[i]->outer_extent);
        rf_iters.push_back(new_iter);
      } else {
        var_map[block->iter_vars[i]->var.get()] = 0;
      }
    } else {
      if (!division[i]->IsOuter()) {
        if (!division[i]->IsInner()) {
          var_map[block->iter_vars[i]->var.get()] =
              block->iter_vars[i] * division[i]->inner_extent +
              Substitute(converter.Convert(division[i]->inner), {{loop->loop_var, rf_iter->var}});

          rf_bindings.push_back(converter.Convert(division[i]->outer));
          IterVar new_iter = block->iter_vars[i];
          new_iter.CopyOnWrite()->dom = Range::FromMinExtent(0, division[i]->outer_extent);
          rf_iters.push_back(new_iter);
        } else {
          var_map[block->iter_vars[i]->var.get()] =
              Substitute(converter.Convert(division[i]->inner), {{loop->loop_var, rf_iter->var}});
        }
      } else {
        if (!division[i]->IsInner()) {
          rf_bindings.push_back(converter.Convert(division[i]->outer));
          IterVar new_iter = block->iter_vars[i];
          new_iter.CopyOnWrite()->dom = Range::FromMinExtent(0, division[i]->outer_extent);
          rf_iters.push_back(new_iter);
        } else {
          var_map[block->iter_vars[i]->var.get()] = 0;
        }
      }
    }
  }
  rf_bindings.push_back(loop->loop_var);
  rf_iters.push_back(rf_iter);
  CHECK(0 <= factor_axis && factor_axis <= static_cast<int>(update->buffer->shape.size()))
      << "ValueError: factor_axis should be in range [0, " << update->buffer->shape.size() << "]";
  Array<PrimExpr> rf_shape = update->buffer->shape;
  Array<PrimExpr> rf_indices = update->indices;
  rf_shape.insert(rf_shape.begin() + factor_axis, loop->extent);
  rf_indices.insert(rf_indices.begin() + factor_axis, rf_iter->var);
  Buffer rf_buf = update->buffer;
  rf_buf.CopyOnWrite()->shape = rf_shape;
  rf_buf.CopyOnWrite()->name = rf_buf->name + "_rf";
  rf_buf.CopyOnWrite()->data = rf_buf->data.copy_with_suffix("_rf");
  BufferStore rf_update = GetRef<BufferStore>(update);
  rf_update.CopyOnWrite()->buffer = rf_buf;
  rf_update.CopyOnWrite()->indices = rf_indices;
  rf_update.CopyOnWrite()->value =
      reducer.value().get()->operator()({BufferLoad(rf_buf, rf_indices)}, {rhs})[0];
  std::vector<BufferRegion> rf_reads, rf_writes;
  auto rf_region = [&](Array<BufferRegion> regions, std::vector<BufferRegion>& rf_regions) {
    for (const auto& t_region : regions) {
      if (t_region->buffer.same_as(update->buffer)) {
        Region region = t_region->region;
        region.insert(region.begin() + factor_axis, Range::FromMinExtent(rf_iter->var, 1));
        rf_regions.emplace_back(rf_buf, region);
      } else {
        rf_regions.push_back(SubstituteBufferRegion(t_region, var_map));
      }
    }
  };
  rf_region(rf_block->reads, rf_reads);
  rf_region(rf_block->writes, rf_writes);
  rf_block.CopyOnWrite()->body = Substitute((Stmt)rf_update, var_map);
  rf_block.CopyOnWrite()->iter_vars = rf_iters;
  rf_block.CopyOnWrite()->reads = rf_reads;
  rf_block.CopyOnWrite()->writes = rf_writes;
  rf_block.CopyOnWrite()->init = BufferStore(rf_buf, init->value, rf_indices);
  rf_block_realize.CopyOnWrite()->block = rf_block;
  rf_block_realize.CopyOnWrite()->binding_values = rf_bindings;
  // create write back block
  BlockRealize wb_block_realize = block_realize;
  Block wb_block = block;
  std::vector<PrimExpr> wb_bindings;
  std::vector<IterVar> wb_iters;
  var_map.clear();
  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    if (block->iter_vars[i]->iter_type == IterVarType::kDataPar) {
      wb_iters.emplace_back(block->iter_vars[i]->dom, block->iter_vars[i]->var.copy_with_suffix(""),
                            block->iter_vars[i]->iter_type);
      wb_bindings.push_back(block_realize->binding_values[i]);
      var_map[block->iter_vars[i]->var.get()] = wb_iters.back();
    }
  }
  wb_iters.emplace_back(Range::FromMinExtent(loop->min, loop->extent),
                        Var("v" + loop->loop_var->name_hint), IterVarType::kCommReduce);
  wb_bindings.push_back(loop->loop_var);
  var_map[rf_iter->var.get()] = wb_iters.back();

  auto wb_region = [&](const BufferLoad& load) {
    std::vector<Range> region;
    for (const auto& index : load->indices) region.push_back(Range::FromMinExtent(index, 1));
    return BufferRegion(load->buffer, region);
  };
  BufferStore wb_update = GetRef<BufferStore>(update);
  BufferLoad wb_lhs = Downcast<BufferLoad>(Substitute((PrimExpr)lhs, var_map));
  BufferLoad wb_rhs = Downcast<BufferLoad>(
      Substitute((PrimExpr)BufferLoad(rf_update->buffer, rf_update->indices), var_map));
  wb_update.CopyOnWrite()->value = reducer.value().get()->operator()({wb_lhs}, {wb_rhs})[0];
  wb_update = Downcast<BufferStore>(Substitute((Stmt)wb_update, var_map));
  wb_block.CopyOnWrite()->body = wb_update;
  wb_block.CopyOnWrite()->reads = {wb_region(wb_lhs), wb_region(wb_rhs)};
  wb_block.CopyOnWrite()->writes = {wb_region(wb_lhs)};
  wb_block.CopyOnWrite()->iter_vars = wb_iters;
  wb_block.CopyOnWrite()->init = BufferStore(wb_update->buffer, init->value, wb_update->indices);
  wb_block_realize.CopyOnWrite()->block = wb_block;
  wb_block_realize.CopyOnWrite()->binding_values = wb_bindings;
  // create loops outside write back block and rfactor block
  Stmt rf_body = rf_block_realize, wb_body = wb_block_realize;
  Var wb_loop_var = loop->loop_var.copy_with_suffix("");
  wb_body = For(wb_loop_var, loop->min, loop->extent, ForKind::kSerial,
                SubstituteInScope(wb_body, {{loop->loop_var.get(), wb_loop_var.get()}}));
  Optional<StmtSRef> top;
  for (int i = loops.size() - 1; i >= 0; --i) {
    const auto* l = loops[i]->GetStmt<ForNode>();
    ICHECK(l) << "InternalError: GetAxes returns a block sref";
    if (l->body->IsInstance<SeqStmtNode>()) {
      CHECK(i != (int)loops.size() - 1) << "ValueError: can not rfactor";
      top = loops[i + 1];
      break;
    }
    if (l != loop) {
      // copy this loop outside rfactor block
      For rf_loop = GetRef<For>(l);
      rf_loop.CopyOnWrite()->body = rf_body;
      rf_body = rf_loop;
    }
    if (data_par_loops.count(l->loop_var.get())) {
      // copy this loop outside write back block
      wb_loop_var = l->loop_var.copy_with_suffix("");
      wb_body = For(wb_loop_var, l->min, l->extent, ForKind::kSerial,
                    SubstituteInScope(wb_body, {{l->loop_var.get(), wb_loop_var.get()}}));
    }
  }
  For rf_loop = GetRef<For>(loop);
  rf_loop.CopyOnWrite()->body = rf_body;
  rf_body = rf_loop;
  if (!top) top = loops[0];

  // insert rf block and wb block under top
  auto insert = [](Stmt body, int64_t pos, std::vector<Stmt> input) -> SeqStmt {
    if (pos == -1) return SeqStmt(input);
    std::vector<Stmt> res;
    if (const auto* op = body.as<SeqStmtNode>()) {
      for (const auto& stmt : op->seq) res.push_back(stmt);
    } else {
      LOG(FATAL);
    }
    res.insert(res.begin() + pos, input.begin(), input.end());
    return SeqStmt(res);
  };
  if (const auto* parent = top.value()->parent->GetStmt<ForNode>()) {
    SeqStmt parent_body = insert(parent->body, top.value()->seq_index, {rf_body, wb_body});
    this->Replace(GetRef<StmtSRef>(top.value()->parent),
                  For(parent->loop_var, parent->min, parent->extent, ForKind::kSerial, parent_body),
                  {{wb_block, block}});
  } else if (const auto* parent = top.value()->parent->GetStmt<BlockNode>()) {
    SeqStmt parent_body = insert(parent->body, top.value()->seq_index, {rf_body, wb_body});
    auto block_node = make_object<BlockNode>(*parent);
    block_node->body = parent_body;
    block_node->init = NullOpt;
    Block new_block = Block(block_node);
    this->Replace(GetRef<StmtSRef>(top.value()->parent), new_block,
                  {{new_block, GetRef<Block>(parent)}, {wb_block, block}});
  }
  // insert rf buffer into scope block's allocation
  StmtSRef scope_sref = GetParentBlockSRef(block_sref);
  Block scope_block = GetRef<Block>(scope_sref->GetStmt<BlockNode>()),
        new_scope_block = scope_block;
  new_scope_block.CopyOnWrite()->alloc_buffers.push_back(rf_buf);
  this->Replace(scope_sref, new_scope_block, {{new_scope_block, scope_block}});
  // Update scope information
  UpdateScope(scope_sref->stmt, this->stmt2ref, &this->scopes);

  return stmt2ref.at(rf_block.get());
}

struct Internal {
  static StmtSRef DecomposeReduction(Schedule self, StmtSRef block_sref,
                                     Optional<StmtSRef> loop_sref) {
    return self->decompose_reduction(block_sref, loop_sref);
  }
  static void MergeReduction(Schedule self, StmtSRef init_block_sref, StmtSRef update_block_sref) {
    self->merge_reduction(init_block_sref, update_block_sref);
  }
  static StmtSRef RFactor(Schedule self, StmtSRef loop_sref, int factor_axis) {
    return self->rfactor(loop_sref, factor_axis);
  }
};

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleDecomposeReduction")
    .set_body_typed(Internal::DecomposeReduction);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleMergeReduction").set_body_typed(Internal::MergeReduction);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleRfactor").set_body_typed(Internal::RFactor);

}  // namespace tir
}  // namespace tvm
