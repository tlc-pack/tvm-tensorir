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
#include "../utils.h"

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

StmtSRef DecomposeReduction(ScheduleState self, const StmtSRef& block_sref,
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
  const auto* block = block_sref->StmtAs<BlockNode>();
  if (loop_sref_opt) {
    // 'loop' is not 'None'.
    StmtSRef loop_sref = loop_sref_opt.value();
    const auto* loop = loop_sref->StmtAs<ForNode>();
    CHECK(block != nullptr)
        << "TypeError: 'decompose_reduction' expect a block as first argument, but get type: "
        << block_sref->stmt->GetTypeKey();
    CHECK(loop != nullptr)
        << "TypeError: 'decompose_reduction' expect a loop as second argument, but get type: "
        << loop_sref->stmt->GetTypeKey();
    CHECK(block->init.defined()) << "ValueError: 'decompose_reduction' expect a reduction block, "
                                    "but the block has no init block";
    Array<StmtSRef> loops = GetLoops(block_sref);
    const BlockRealizeNode* realize = GetBlockRealize(block_sref).get();
    // Cond 0. Check loop_sref is an ancestor of block_sref
    CHECK(ListContainsElement(loops, loop_sref))
        << "ValueError: 'decompose_reduction' expect the loop to be an ancestor of block";
    // Cond 1. Check block is reduction
    CHECK(ReductionBlock(self, block_sref, GetScopeRoot(block_sref).value()))
        << "decompose_reduction expect the block to be a reduction block";
    // Cond 2. Check 'loop' is higher than all the loops related to block var of type reduction
    for (int i = 0, n = block->iter_vars.size(); i < n; ++i) {
      // For each block var of type kCommReduce, check its binding
      const IterVar& iter_var = block->iter_vars[i];
      const PrimExpr& binding = realize->iter_values[i];
      if (iter_var->iter_type != IterVarType::kCommReduce) {
        continue;
      }
      for (const StmtSRef& higher_loop : loops) {
        // Only check loops higher than the target loop
        if (higher_loop.same_as(loop_sref)) {
          break;
        }
        // loop_var of a higher loop shouldn't contain loop var
        const Var& loop_var = higher_loop->StmtAs<ForNode>()->loop_var;
        CHECK(!StmtExprContainsVar(binding, loop_var))
            << "ValueError: 'decompose_reduction' expect the loop to be higher "
               "than all the loops related to reduce block var";
      }
    }
    // Mutate
    ObjectPtr<BlockNode> init_block = make_object<BlockNode>();
    ObjectPtr<BlockRealizeNode> init_realize = make_object<BlockRealizeNode>();
    init_block->name_hint = block->name_hint + "_init";
    init_realize->iter_values = {};
    init_realize->predicate = realize->predicate;
    init_realize->block = Block(init_block);
    // Step 1. Create new block vars and their bindings
    // Maps an old block var to the new corresponding block var
    std::unordered_map<const VarNode*, const VarNode*> block_var_map;
    for (int i = 0, n = block->iter_vars.size(); i < n; ++i) {
      const IterVar& iter_var = block->iter_vars[i];
      const PrimExpr& binding = realize->iter_values[i];
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
      init_realize->iter_values.push_back(binding);
      // Add a mapping from old block vars to new block vars
      block_var_map[iter_var->var.get()] = new_iter_var->var.get();
    }
    // Step 2. After copying block vars, substitute them in init block
    init_block->body =
        Substitute(block->init.value(), [&block_var_map](const Var& var) -> Optional<PrimExpr> {
          auto it = block_var_map.find(var.get());
          if (it != block_var_map.end()) {
            return GetRef<PrimExpr>(it->second);
          } else {
            return NullOpt;
          }
        });
    for (const BufferRegion& write : block->writes) {
      init_block->writes.push_back(SubstituteBufferRegion(write, block_var_map));
    }
    // Step 3. Create loops above the init block
    Stmt body = BlockRealize(init_realize);
    for (int i = static_cast<int>(loops.size()) - 1; i >= 0; --i) {
      const auto* higher_loop = loops[i]->StmtAs<ForNode>();
      for (const PrimExpr& expr : init_realize->iter_values) {
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
    if (const auto* parent = loop_sref->parent->StmtAs<ForNode>()) {
      self->Replace(GetRef<StmtSRef>(loop_sref->parent),
                    For(/*loop_var=*/parent->loop_var,
                        /*min=*/parent->min,
                        /*extent=*/parent->extent,
                        /*kind=*/parent->kind,
                        /*body=*/SeqStmt::Flatten(Array<Stmt>{body, parent->body}),
                        /*thread_binding*/ parent->thread_binding,
                        /*annotations*/ parent->annotations),
                    {});
    } else if (const auto* parent = loop_sref->parent->StmtAs<BlockNode>()) {
      auto block_node = make_object<BlockNode>(*parent);
      block_node->body = SeqStmt::Flatten(Array<Stmt>{body, parent->body});
      block_node->init = NullOpt;
      Block new_block = Block(block_node);
      self->Replace(GetRef<StmtSRef>(loop_sref->parent), new_block,
                    {{GetRef<Block>(parent), new_block}});
      UpdateAffineFlag(self, GetRef<StmtSRef>(loop_sref->parent));
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
    self->Replace(block_sref, update_block, {{GetRef<Block>(block), update_block}});
    // Update scope information
    UpdateScope(self, block_sref);
    UpdateAffineFlag(self, block_sref);
    StmtSRef init_block_sref = self->stmt2ref.at(init_block.get());
    UpdateAffineFlag(self, init_block_sref);
    return init_block_sref;
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

    self->Replace(block_sref, new_block, {{GetRef<Block>(block), new_block}});
    // Update scope information
    UpdateScope(self, block_sref);
    UpdateAffineFlag(self, block_sref);
    return self->stmt2ref.at(new_block.get());
  }
}

void MergeReduction(ScheduleState self, const StmtSRef& init_sref, const StmtSRef& update_sref) {
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
  const auto* init = init_sref->StmtAs<BlockNode>();
  const auto* update = update_sref->StmtAs<BlockNode>();
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
  StmtSRef scope = GetScopeRoot(init_sref).value();
  StmtSRef lca = LowestCommonAncestor({init_sref, update_sref}, scope);
  // Cond 1. Check init_block is under the same scope with update_sref
  CHECK_EQ(scope.get(), GetScopeRoot(update_sref).get())
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
  CHECK(CanMergeReduction(self, init_sref, update_sref, scope));
  // Cond 2. Check LCA is higher than all the loops related to update_block's reduce block var
  if (!scope.same_as(lca)) {
    for (const StmtSRef& higher_loop : GetLoops(update_sref)) {
      if (higher_loop.same_as(lca)) {
        break;
      }
      const Var& loop_var = higher_loop->StmtAs<ForNode>()->loop_var;
      for (int i = 0, n = update->iter_vars.size(); i < n; ++i) {
        const IterVar& iter_var = update->iter_vars[i];
        const PrimExpr& binding = update_realize->iter_values[i];
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
  self->Replace(lca, removed.second, {});
  // Step 2. Change the update block to reduction block
  BufferStore new_init = GetRef<BufferStore>(update_body);
  new_init.CopyOnWrite()->value = init_body->value;
  auto merged_node = make_object<BlockNode>(*update);
  merged_node->init = new_init;
  Block merged(merged_node);
  self->Replace(update_sref, merged, {{GetRef<Block>(update), merged}});
  // Update scope information
  UpdateScope(self, update_sref);
  UpdateAffineFlag(self, update_sref);
}

StmtSRef RFactor(ScheduleState self, const StmtSRef& loop_sref, int factor_axis) {
  const auto* loop = loop_sref->StmtAs<ForNode>();

  // *****************************************************
  // *    Condition Checks and Information Collection    *
  // *****************************************************

  // Check some conditions for rfactor. Get the block and block-realize.
  CHECK(loop) << "TypeError: Only support loop rfactor. But the input type is: "
              << loop_sref->stmt->GetTypeKey();
  Array<StmtSRef> child_blocks = GetChildBlocks(self, loop_sref);
  CHECK_EQ(child_blocks.size(), 1) << "ValueError: The loop should have exactly one child block";
  StmtSRef block_sref = child_blocks[0];
  BlockRealize block_realize = GetBlockRealize(block_sref);
  Block block = block_realize->block;
  StmtSRef scope_root = GetScopeRoot(block_sref).value();
  CHECK(ReductionBlock(self, block_sref, scope_root))
      << "ValueError: We can only do rfactor for loops of a reduction block";

  // Collect the information of the reduction.
  // Get the `init` identity and the `update` combiner of the reduction.
  const auto* init = block->init.as<BufferStoreNode>();
  const auto* update = block->body.as<BufferStoreNode>();
  CHECK(init && update) << "ValueError: Currently rfactor only supports the cases that the init "
                           "and body of the reduction block are BufferStores";
  CHECK(0 <= factor_axis && factor_axis <= static_cast<int>(update->buffer->shape.size()))
      << "ValueError: factor_axis should be in range [0, " << update->buffer->shape.size() << "]";
  // Extract the commutative reducer, combiner lhs and combiner rhs from the reduction identity and
  // the reduction combiner. The lhs will be used when constructing the write-back block, and the
  // rhs will be used when constructing the rfactor block.
  Optional<CommReducer> reducer;
  Optional<PrimExpr> update_lhs, update_rhs;
  CommReducer::FromInitUpdate(init->value, GetRef<BufferStore>(update), reducer, update_lhs,
                              update_rhs, Span());
  CHECK(reducer.defined()) << "ValueError: No matched reducer for identity \"" << init->value
                           << "\" and combiner \"" << GetRef<BufferStore>(update) << "\"";
  ICHECK(update_lhs.defined() && update_rhs.defined());

  // Collect the information of loops and blocks.
  /*! \brief The loop vars that are touched by at least one data parallel block var */
  std::unordered_set<const VarNode*> data_par_iters;
  /*! \brief The loop vars that are touched by at least one reduction block var */
  std::unordered_set<const VarNode*> reduce_iters;
  /*! \brief The block vars which touch the rfactor loop */
  std::unordered_set<const IterVarNode*> touch_iters;

  // Collect:
  //  1. the vars that are touched by data parallel block vars (stored in `data_par_iters`),
  //  2. the vars that are touched by reduction block vars (stored in `reduce_iters`), and
  //  3. the block vars which touch the rfactor loop (stored in `touch_iters`).
  for (int i = 0; i < static_cast<int>(block->iter_vars.size()); ++i) {
    std::unordered_set<const VarNode*>* set = nullptr;
    if (block->iter_vars[i]->iter_type == IterVarType::kDataPar) {
      set = &data_par_iters;
    } else if (block->iter_vars[i]->iter_type == IterVarType::kCommReduce) {
      set = &reduce_iters;
    } else {
      LOG(FATAL)
          << "ValueError: rfactor only supports data parallel block vars and reduction block vars";
    }
    PreOrderVisit(block_realize->iter_values[i], [&](const ObjectRef& node) {
      if (const auto* var = node.as<VarNode>()) {
        set->insert(var);
        if (var == loop->loop_var.get()) {
          // The i-th block var touches the rfactor loop.
          touch_iters.insert(block->iter_vars[i].get());
        }
        return false;
      }
      return true;
    });
  }

  // Collect the loops of the reduction block.
  /*!
   * \brief A mapping which maps a loop var to its corresponding For loop for all the reduction
   *         block's outer loops
   */
  std::unordered_map<const VarNode*, For> loop_vars;
  Array<StmtSRef> loops = GetLoops(block_sref);
  for (const StmtSRef& l_sref : loops) {
    const ForNode* l = TVM_SREF_TO_FOR(l, l_sref);
    if (l == loop) {
      CHECK(!data_par_iters.count(l->loop_var.get()))
          << "ValueError: The rfactor loop cannot be touched by data parallel block vars";
    } else {
      CHECK(!(data_par_iters.count(l->loop_var.get()) && reduce_iters.count(l->loop_var.get())))
          << "ValueError: It is not supported that loop \"" << l->loop_var
          << "\" is touched by both data parallel block vars and reduction block vars";
    }
    loop_vars[l->loop_var.get()] = GetRef<For>(l);
  }

  // *****************************************************
  // *   RFactor Block & Write-Back Block Construction   *
  // *****************************************************

  /*
   * Start constructing the rfactor block. The main difficulty to construct the rfactor block is to
   * create its block vars. So here we briefly introduce the algorithm to create the block vars.
   *   1. Create a block var for the rfactor loop. The binding of this block var is the loop var,
   *      and the block var is data parallel.
   *   2. For all the old block's block vars, there are two cases:
   *     (a) If it is data parallel block var, or a reduction block var which doesn't touch the
   *         rfactor loop, we keep it and its binding in the rfactor block.
   *     (b) Otherwise it is a reduction block var which touches the rfactor loop. In this case, we
   *         "split" the block var into one or more new block vars and do not keep the old block
   *         var. More specifically, we create a new reduction block var for each loop var that
   *         appears in the reduction block var's binding (except for the rfactor loop), and the
   *         binding of the new block var is exactly the loop var. (Note that for each loop var, we
   *         create at most one block var, even if there are multiple old block vars which touch
   *         both this loop and the rfactor loop).
   *         Then we substitute the appearances of the old block var with the new created block vars
   *         by recording two mappings: one maps loops vars to new created block vars which is used
   *         for binding substitution, and another maps old block vars to new expressions which is
   *         used for substitutions of the old block vars.
   */
  /*!
   * \brief A mapping which maps old block vars to new expressions. The old vars will be replaced by
   * the expressions in future substitution.
   */
  std::unordered_map<const VarNode*, PrimExpr> var_map;
  /*!
   * \brief A mapping which maps loop vars to new created block vars. This map is used to
   * substitute the loop vars which appear in the bindings of some old block vars with the new
   * created block vars.
   */
  std::unordered_map<const VarNode*, PrimExpr> iter_map;

  // The new block vars and the bindings of the rfactor block.
  std::vector<IterVar> rf_block_iters;
  std::vector<PrimExpr> rf_bindings;

  // Create a new data parallel block var for the rfactor loop.
  IterVar rf_block_var(Range::FromMinExtent(loop->min, loop->extent),
                       Var("v" + loop->loop_var->name_hint), IterVarType::kDataPar);
  iter_map[loop->loop_var.get()] = rf_block_var;
  rf_block_iters.push_back(rf_block_var);
  rf_bindings.push_back(loop->loop_var);

  // Create other block vars for the rfactor block.
  for (int i = 0; i < static_cast<int>(block->iter_vars.size()); ++i) {
    IterVar old_block_var = block->iter_vars[i];
    PrimExpr old_binding = block_realize->iter_values[i];
    if (old_block_var->iter_type == IterVarType::kDataPar) {
      if (is_one(old_block_var->dom->extent)) {
        // If the extent of this block var is 1, we can substitute the appearances of this block var
        // with its minimum value.
        var_map[old_block_var->var.get()] = old_block_var->dom->min;
      } else {
        // Otherwise, we reuse the old data parallel block var and its corresponding binding.
        rf_block_iters.push_back(old_block_var);
        rf_bindings.push_back(old_binding);
      }
    } else {
      ICHECK(old_block_var->iter_type == kCommReduce);
      if (touch_iters.count(old_block_var.get())) {
        // This block var touches the rfactor loop. So next we try to create a new block var for all
        // loop vars that appear in the old binding.
        PreOrderVisit(old_binding, [&](const ObjectRef& node) {
          if (const auto* var = node.as<VarNode>()) {
            auto it = loop_vars.find(var);
            if (it == loop_vars.end()) {
              // `var` is not a loop var. So we go back.
              return false;
            }
            const For& l = it->second;
            if (iter_map.find(var) == iter_map.end()) {
              // We haven't created the new block var for `var`. So here we create it, append it
              // and its binding to `rf_block_iters` and `rf_bindings` respectively.
              IterVar new_iter_var(Range::FromMinExtent(l->min, l->extent),
                                   Var("v" + l->loop_var->name_hint), IterVarType::kCommReduce);
              iter_map[var] = new_iter_var;
              rf_block_iters.push_back(new_iter_var);
              rf_bindings.push_back(GetRef<Var>(var));
            }
            return false;
          }
          return true;
        });
        // Substitute the original binding with new block vars. Store the result expression
        // in `var_map` for future substitution.
        var_map[old_block_var->var.get()] = Substitute(old_binding, iter_map);
      } else {
        // This block var doesn't touch the rfactor loop. So we reuse the old reduction block var
        // and its corresponding binding.
        rf_block_iters.push_back(old_block_var);
        rf_bindings.push_back(old_binding);
      }
    }
  }
  ICHECK_EQ(rf_block_iters.size(), rf_bindings.size());

  // Construct other parts of the rfactor block and block_realize.
  Array<PrimExpr> rf_shape = update->buffer->shape;
  Array<PrimExpr> rf_indices = update->indices;
  rf_shape.insert(rf_shape.begin() + factor_axis, loop->extent);
  rf_indices.insert(rf_indices.begin() + factor_axis, rf_block_var->var);
  Buffer rf_buf = update->buffer;
  rf_buf.CopyOnWrite()->shape = rf_shape;
  rf_buf.CopyOnWrite()->name = rf_buf->name + ".rf";
  rf_buf.CopyOnWrite()->data = rf_buf->data.copy_with_suffix(".rf");
  BufferStore rf_update(
      rf_buf,
      reducer.value().get()->operator()({BufferLoad(rf_buf, rf_indices)}, {update_rhs.value()})[0],
      rf_indices);
  auto f_rf_create_rw_region = [&](const Array<BufferRegion>& regions) {
    Array<BufferRegion> new_regions;
    for (const BufferRegion& buf_region : regions) {
      if (buf_region->buffer.same_as(update->buffer)) {
        Region region = buf_region->region;
        region.insert(region.begin() + factor_axis, Range::FromMinExtent(rf_block_var->var, 1));
        new_regions.push_back(SubstituteBufferRegion(BufferRegion(rf_buf, region), var_map));
      } else {
        new_regions.push_back(SubstituteBufferRegion(buf_region, var_map));
      }
    }
    return std::move(new_regions);
  };
  Block rf_block(
      /*iter_vars=*/rf_block_iters,
      /*reads=*/f_rf_create_rw_region(block->reads),
      /*writes=*/f_rf_create_rw_region(block->writes),
      /*name_hint=*/block->name_hint + "_rf",
      /*body=*/Substitute(static_cast<Stmt>(rf_update), var_map),
      /*init=*/
      Substitute(static_cast<Stmt>(BufferStore(rf_buf, init->value, rf_indices)), var_map));
  BlockRealize rf_block_realize(rf_bindings, block_realize->predicate, rf_block);
  // Finish constructing the rfactor block.

  // Start constructing the write-back block.
  var_map.clear();
  // The new block vars and their bindings of the write-back block.
  std::vector<IterVar> wb_block_iters;
  std::vector<PrimExpr> wb_bindings;

  // Create new block vars.
  for (int i = 0; i < static_cast<int>(block->iter_vars.size()); ++i) {
    IterVar old_block_var = block->iter_vars[i];
    if (old_block_var->iter_type == IterVarType::kDataPar) {
      wb_block_iters.emplace_back(old_block_var->dom, old_block_var->var.copy_with_suffix(""),
                                  kDataPar);
      wb_bindings.push_back(block_realize->iter_values[i]);
      var_map[old_block_var->var.get()] = wb_block_iters.back();
    }
  }
  wb_block_iters.emplace_back(Range::FromMinExtent(loop->min, loop->extent),
                              Var("v" + loop->loop_var->name_hint), IterVarType::kCommReduce);
  wb_bindings.push_back(loop->loop_var);
  var_map[rf_block_var->var.get()] = wb_block_iters.back();

  // Create other parts of the write-back block and block_realize.
  auto f_wb_create_rw_region = [&](const BufferLoad& load) {
    std::vector<Range> region;
    for (const PrimExpr& index : load->indices) {
      region.push_back(Range::FromMinExtent(index, 1));
    }
    return BufferRegion(load->buffer, region);
  };
  BufferLoad wb_lhs =
      Downcast<BufferLoad>(Substitute(static_cast<PrimExpr>(update_lhs.value()), var_map));
  BufferLoad wb_rhs = Downcast<BufferLoad>(Substitute(
      static_cast<PrimExpr>(BufferLoad(rf_update->buffer, rf_update->indices)), var_map));
  BufferStore wb_update(update->buffer, reducer.value().get()->operator()({wb_lhs}, {wb_rhs})[0],
                        update->indices);
  wb_update = Downcast<BufferStore>(Substitute(static_cast<Stmt>(wb_update), var_map));
  Block wb_block(/*iter_vars=*/wb_block_iters,
                 /*reads=*/{f_wb_create_rw_region(wb_lhs), f_wb_create_rw_region(wb_rhs)},
                 /*writes=*/{f_wb_create_rw_region(wb_lhs)},
                 /*name_hint=*/block->name_hint,
                 /*body=*/wb_update,
                 /*init=*/BufferStore(wb_update->buffer, init->value, wb_update->indices));
  BlockRealize wb_block_realize(wb_bindings, block_realize->predicate, wb_block);
  // Finish constructing the write-back block.

  // *****************************************************
  // *                Loop Construction                  *
  // *****************************************************

  // Construct the loops outside the rfactor block and the write-back block.
  Stmt rf_body = std::move(rf_block_realize);
  Stmt wb_body = std::move(wb_block_realize);
  Var wb_loop_var = loop->loop_var.copy_with_suffix("");
  wb_body = For(wb_loop_var, loop->min, loop->extent, ForKind::kSerial,
                SubstituteInScope(wb_body, {{loop->loop_var.get(), wb_loop_var.get()}}));
  // `replace_top` is the deepest loop whose parent in TIR might be a SeqStmt. It will be used for
  // IR replacement later.
  Optional<StmtSRef> replace_top = NullOpt;
  for (int i = static_cast<int>(loops.size()) - 1; i >= 0; --i) {
    const ForNode* l = TVM_SREF_TO_FOR(l, loops[i]);
    if (l->body->IsInstance<SeqStmtNode>()) {
      ICHECK_NE(i, static_cast<int>(loops.size()) - 1)
          << "ValueError: The body of the innermost loop must not be a SeqStmt";
      replace_top = loops[i + 1];
      break;
    }
    // Wrap the rfactor block with this loop.
    For rf_loop = GetRef<For>(l);
    rf_loop.CopyOnWrite()->body = rf_body;
    rf_body = rf_loop;

    if (data_par_iters.count(l->loop_var.get())) {
      // Wrap the write-back block with this loop if the loop is a data parallel loop.
      wb_loop_var = l->loop_var.copy_with_suffix("");
      wb_body = For(wb_loop_var, l->min, l->extent, ForKind::kSerial,
                    SubstituteInScope(wb_body, {{l->loop_var.get(), wb_loop_var.get()}}));
    }
  }
  if (!replace_top.defined()) {
    replace_top = loops[0];
  }

  // *****************************************************
  // *           Schedule Replacement & Update           *
  // *****************************************************

  // Insert the rfactor block and the write-back block under the `replace_top` loop.
  auto f_insert_loop = [](const Stmt& body, int64_t pos, std::vector<Stmt> input) -> SeqStmt {
    if (pos == -1) {
      return SeqStmt(input);
    }
    if (const auto* seq_stmt = body.as<SeqStmtNode>()) {
      std::vector<Stmt> res;
      for (const Stmt& stmt : seq_stmt->seq) {
        res.push_back(stmt);
      }
      res.erase(res.begin() + pos);
      res.insert(res.begin() + pos, input.begin(), input.end());
      return SeqStmt(res);
    } else {
      LOG(FATAL) << "TypeError: `body` must be a SeqStmt in this case. But its type is: "
                 << body->GetTypeKey();
      throw;
    }
  };
  if (const auto* loop_parent = replace_top.value()->parent->StmtAs<ForNode>()) {
    ObjectPtr<ForNode> p_new_loop = make_object<ForNode>(*loop_parent);
    p_new_loop->body =
        f_insert_loop(loop_parent->body, replace_top.value()->seq_index, {rf_body, wb_body});
    self->Replace(GetRef<StmtSRef>(replace_top.value()->parent), For(p_new_loop),
                  {{block, wb_block}});
  } else {
    const auto* block_parent = replace_top.value()->parent->StmtAs<BlockNode>();
    ICHECK(block_parent);
    ObjectPtr<BlockNode> block_node = make_object<BlockNode>(*block_parent);
    block_node->body =
        f_insert_loop(block_parent->body, replace_top.value()->seq_index, {rf_body, wb_body});
    Block new_block(block_node);
    self->Replace(GetRef<StmtSRef>(replace_top.value()->parent), new_block,
                  {{GetRef<Block>(block_parent), new_block}, {block, wb_block}});
  }

  // Append the new rfactor buffer to the scope root block's buffer allocation.
  Block scope_block = GetRef<Block>(scope_root->StmtAs<BlockNode>());
  Block new_scope_block = scope_block;
  new_scope_block.CopyOnWrite()->alloc_buffers.push_back(rf_buf);
  self->Replace(scope_root, new_scope_block, {{scope_block, new_scope_block}});
  // Update scope information.
  StmtSRef rf_block_sref = self->stmt2ref.at(rf_block.get());
  UpdateScope(self, scope_root);
  UpdateAffineFlag(self, scope_root);
  UpdateAffineFlag(self, rf_block_sref);
  return rf_block_sref;
}

struct RFactorTraits : public UnpackedInstTraits<RFactorTraits> {
  static constexpr const char* kName = "RFactor";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv, Integer factor_axis) {
    return sch->RFactor(loop_rv, factor_axis->value);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv, Integer factor_axis) {
    PythonAPICall py("rfactor");
    py.Input("loop", loop_rv);
    py.Input("factor_axis", factor_axis->value);
    py.SingleOutput(outputs);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

struct DecomposeReductionTraits : public UnpackedInstTraits<DecomposeReductionTraits> {
  static constexpr const char* kName = "DecomposeReduction";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, Optional<LoopRV> loop_rv) {
    return sch->DecomposeReduction(block_rv, loop_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, String loop_rv) {
    PythonAPICall py("decompose_reduction");
    py.Input("block", block_rv);
    py.Input("loop", loop_rv);
    py.SingleOutput(outputs);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

struct MergeReductionTraits : public UnpackedInstTraits<MergeReductionTraits> {
  static constexpr const char* kName = "MergeReduction";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV init_block_rv,
                                      BlockRV update_block_rv) {
    return sch->MergeReduction(init_block_rv, update_block_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String init_block_rv,
                                 String update_block_rv) {
    PythonAPICall py("merge_reduction");
    py.Input("init_block", init_block_rv);
    py.Input("update_block", update_block_rv);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND(RFactorTraits);
TVM_REGISTER_INST_KIND(DecomposeReductionTraits);
TVM_REGISTER_INST_KIND(MergeReductionTraits);

}  // namespace tir
}  // namespace tvm
