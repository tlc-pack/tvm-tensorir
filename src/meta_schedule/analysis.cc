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
#include "./analysis.h"  // NOLINT(build/include)

#include <tvm/arith/analyzer.h>
#include <tvm/tir/stmt_functor.h>

#include "../arith/pattern_match.h"
#include "../tir/schedule/schedule_common.h"  // TODO(@junrushao1994): replace it
#include "./utils.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Checks if the specific expr is an integer constant
 * \param x The expr to be checked
 * \return A boolean flag indicating if it is a constant integer, or broadcast of constant integer
 */
static bool IsConstInt(const PrimExpr& x) {
  if (x->IsInstance<tir::IntImmNode>()) {
    return true;
  }
  if (const auto* op = x.as<tir::BroadcastNode>()) {
    return op->value->IsInstance<tir::IntImmNode>();
  }
  return false;
}

/*!
 * \brief Check if an expression consists of a single variable, or a variable +/i an constant
 * \param expr The expression to be checked
 * \param result Output, the var inside if it satisfies the condition
 * \return A boolean indicating if it satisfies the condition
 */
static bool IsVarPlusMinusConst(const PrimExpr& expr, tir::Var* result) {
  // match: "var"
  if (const auto* var = expr.as<tir::VarNode>()) {
    *result = GetRef<tir::Var>(var);
    return true;
  }
  arith::PVar<tir::Var> var;
  arith::PVar<IntImm> shift;
  // match: "var +/- shift"
  if ((var + shift).Match(expr) || (var - shift).Match(expr) || (shift + var).Match(expr)) {
    *result = var.Eval();
    return true;
  }
  return false;
}

bool IsTrivialBinding(Schedule sch, BlockRV block_rv) {
  tir::StmtSRef block_sref = sch->Eval(block_rv);
  const auto* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block) << "TypeError: Expects Block, but gets: " << block_sref->stmt->GetTypeKey();
  tir::BlockRealize realize = tir::GetBlockRealize(block_sref);
  Array<tir::StmtSRef> loops = sch->sch->GetLoopsInScope(block_sref);
  const Array<PrimExpr>& bindings = realize->binding_values;
  if (loops.size() != bindings.size()) {
    return false;
  }
  int n = loops.size();
  arith::Analyzer analyzer;
  for (int i = 0; i < n; ++i) {
    const PrimExpr& bind = bindings[i];
    const auto* loop = loops[i]->GetStmt<tir::LoopNode>();
    CHECK(loop) << "TypeError: Expects Loop, but gets: " << loops[i]->stmt->GetTypeKey();
    if (!analyzer.CanProve(bind == loop->loop_var)) {
      return false;
    }
  }
  return true;
}

Array<Integer> GetBlockVarTypes(Schedule sch, BlockRV block_rv) {
  tir::StmtSRef block_sref = sch->Eval(block_rv);
  const auto* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block) << "TypeError: Expects Block, but gets: " << block_sref->stmt->GetTypeKey();
  Array<Integer> result;
  for (const tir::IterVar& iter_var : block->iter_vars) {
    int iter_type = iter_var->iter_type;
    result.push_back(iter_type);
  }
  return result;
}

bool IsLeafBlock(Schedule sch, BlockRV block_rv) {
  tir::StmtSRef block_sref = sch->Eval(block_rv);
  const auto* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block) << "TypeError: Expects Block, but gets: " << block_sref->stmt->GetTypeKey();
  bool is_leaf = true;
  tir::PreOrderVisit(block->body, [&is_leaf](const ObjectRef& obj) -> bool {
    if (is_leaf == false) {
      return false;
    }
    if (obj->IsInstance<tir::BlockNode>()) {
      is_leaf = false;
      return false;
    }
    return true;
  });
  return is_leaf;
}

bool IsLeafBlockWithSingleStmt(Schedule sch, BlockRV block_rv) {
  tir::StmtSRef block_sref = sch->Eval(block_rv);
  const auto* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block) << "TypeError: Expects Block, but gets: " << block_sref->stmt->GetTypeKey();
  const tir::Stmt& body = block->body;
  if (body->IsInstance<tir::BufferStoreNode>()) {
    return true;
  }
  if (body->IsInstance<tir::ReduceStepNode>()) {
    return true;
  }
  return false;
}

tir::BufferLoad GetBufferStore(Schedule sch, BlockRV block_rv) {
  tir::StmtSRef block_sref = sch->Eval(block_rv);
  const auto* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block) << "TypeError: Expects Block, but gets: " << block_sref->stmt->GetTypeKey();
  if (const auto* body = block->body.as<tir::BufferStoreNode>()) {
    return tir::BufferLoad(body->buffer, body->indices);
  }
  if (const auto* body = block->body.as<tir::ReduceStepNode>()) {
    const auto* buffer_update = body->lhs.as<tir::BufferLoadNode>();
    CHECK(buffer_update) << "TypeError: LHS of ReduceStep is expected to be BufferLoad, but gets: "
                         << body->lhs->GetTypeKey();
    return GetRef<tir::BufferLoad>(buffer_update);
  }
  LOG(FATAL) << "ValueError: `GetBufferStore` only applies to a leaf block whose body is single "
                "statement, but get: "
             << GetRef<tir::Block>(block);
  throw;
}

Array<tir::BufferLoad> GetBufferLoad(Schedule sch, BlockRV block_rv) {
  tir::StmtSRef block_sref = sch->Eval(block_rv);
  const auto* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block) << "TypeError: Expects Block, but gets: " << block_sref->stmt->GetTypeKey();
  Array<tir::BufferLoad> reads;
  auto f_visit = [&reads](const ObjectRef& obj) {
    if (const auto* load = obj.as<tir::BufferLoadNode>()) {
      reads.push_back(GetRef<tir::BufferLoad>(load));
    }
  };
  if (const auto* body = block->body.as<tir::BufferStoreNode>()) {
    tir::PostOrderVisit(body->value, f_visit);
    return reads;
  }
  if (const auto* body = block->body.as<tir::ReduceStepNode>()) {
    tir::PostOrderVisit(body->rhs, f_visit);
    return reads;
  }
  LOG(FATAL) << "ValueError: `GetBufferLoad` only applies to a leaf block whose body is single "
                "statement, but get: "
             << GetRef<tir::Block>(block);
  throw;
}

int CountOp(Schedule sch, BlockRV block_rv, Op op) {
  tir::StmtSRef block_sref = sch->Eval(block_rv);
  const auto* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block) << "TypeError: Expects Block, but gets: " << block_sref->stmt->GetTypeKey();
  int count = 0;
  tir::PostOrderVisit(block->body, [&count, &op](const ObjectRef& obj) {
    if (const auto* call = obj.as<tir::CallNode>()) {
      if (call->op.same_as(op)) {
        ++count;
      }
    }
  });
  return count;
}

bool HasBranch(Schedule sch, BlockRV block_rv) {
  tir::StmtSRef block_sref = sch->Eval(block_rv);
  const auto* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block) << "TypeError: Expects Block, but gets: " << block_sref->stmt->GetTypeKey();
  bool has_branch = false;
  arith::Analyzer analyzer;
  const Op& op_if_then_else = Op::Get("tir.if_then_else");
  auto f_visit = [&has_branch, &analyzer, &op_if_then_else](const ObjectRef& obj) -> bool {
    if (has_branch) {
      // stop visiting
      return false;
    }
    if (const auto* realize = obj.as<tir::BlockRealizeNode>()) {
      // Case 1: BlockRealize
      if (!analyzer.CanProve(realize->predicate == 1)) {
        has_branch = true;
      }
    } else if (obj->IsInstance<tir::IfThenElseNode>() || obj->IsInstance<tir::SelectNode>()) {
      // Case 2: IfThenElse / Select
      has_branch = true;
    } else if (const auto* call = obj.as<tir::CallNode>()) {
      // Case 3: Call
      if (call->op.same_as(op_if_then_else)) {
        has_branch = true;
      }
    }
    return !has_branch;
  };
  tir::PreOrderVisit(tir::GetBlockRealize(block_sref), f_visit);
  return has_branch;
}

Optional<Array<tir::Var>> BlockVarsUsedInStore(Schedule sch, BlockRV block_rv) {
  tir::StmtSRef block_sref = sch->Eval(block_rv);
  const auto* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block) << "TypeError: Expects Block, but gets: " << block_sref->stmt->GetTypeKey();
  if (!IsLeafBlockWithSingleStmt(sch, block_rv)) {
    return NullOpt;
  }
  // Collect block vars
  std::unordered_set<const tir::VarNode*> block_vars;
  for (const tir::IterVar& iter_var : block->iter_vars) {
    block_vars.insert(iter_var->var.get());
  }
  Array<tir::Var> result;
  tir::BufferLoad store = GetBufferStore(sch, block_rv);
  for (const PrimExpr& idx : store->indices) {
    if (IsConstInt(idx)) {
      continue;
    }
    tir::Var var;
    if (IsVarPlusMinusConst(idx, &var)) {
      if (block_vars.count(var.get())) {
        result.push_back(var);
        continue;
      }
    }
    return NullOpt;
  }
  return result;
}

int CountMissingBlockVars(tir::BufferLoad load, Array<tir::Var> block_vars) {
  int n_missing = 0;
  // Collect vars that are used in indices of BufferLoad
  std::unordered_set<const tir::VarNode*> vars_in_load;
  for (const PrimExpr& idx : load->indices) {
    tir::PostOrderVisit(idx, [&vars_in_load](const ObjectRef& obj) {
      if (const auto* var = obj.as<tir::VarNode>()) {
        vars_in_load.insert(var);
      }
    });
  }
  // Enumerate and count missing ones
  for (const tir::Var& var : block_vars) {
    if (!vars_in_load.count(var.get())) {
      ++n_missing;
    }
  }
  return n_missing;
}

Optional<Array<Bool>> InspectLoadIndices(Schedule sch, BlockRV block_rv) {
  // Filter out block vars that corresponding to indices in BufferStore
  Optional<Array<tir::Var>> store = BlockVarsUsedInStore(sch, block_rv);
  if (!store.defined()) {
    return NullOpt;
  }
  // Index those BufferStore indices
  std::unordered_map<const tir::VarNode*, int> store_indices;
  {
    int index = 0;
    for (const tir::Var& var : store.value()) {
      store_indices[var.get()] = index++;
    }
  }
  bool surjective = true;
  bool injective = true;
  bool ordered = true;
  Array<tir::BufferLoad> loads = GetBufferLoad(sch, block_rv);
  for (const tir::BufferLoad& load : loads) {
    // load -> store mapping result
    std::vector<int> load_mapped_to_store_index;
    // Number of times that a store axis is mapped to
    std::vector<int> store_be_mapped_times(store_indices.size(), 0);
    // Enumerate each index, collect the load -> store mapping info
    for (const PrimExpr& idx : load->indices) {
      if (IsConstInt(idx)) {
        continue;
      }
      tir::Var var;
      // Check if it matches a block var
      if (IsVarPlusMinusConst(idx, &var)) {
        if (store_indices.count(var.get())) {
          int index = store_indices.at(var.get());
          load_mapped_to_store_index.push_back(index);
          store_be_mapped_times[index] += 1;
          continue;
        }
      }
      // If not, the load-store mapping does not exist
      return NullOpt;
    }
    // Check `store_be_mapped_times` to determine if the mapping is injective and surjective
    for (int times : store_be_mapped_times) {
      // If there is a store axis that doesn't have corresponding any load axis
      if (times == 0) {
        surjective = false;
      }
      // If there is a store axis that has more than 2 corresponding load axes
      if (times >= 2) {
        injective = false;
      }
    }
    // Check `load_mapped_to_store_index` to determine if the mapping is in order
    for (size_t i = 1; i < load_mapped_to_store_index.size(); ++i) {
      if (load_mapped_to_store_index[i - 1] > load_mapped_to_store_index[i]) {
        ordered = false;
        break;
      }
    }
  }
  return Array<Bool>{Bool(surjective), Bool(injective), Bool(ordered)};
}

bool HasReduceBlockVar(Schedule sch, BlockRV block_rv) {
  Array<Integer> iter_types = GetBlockVarTypes(sch, block_rv);
  for (const Integer& iter_type : iter_types) {
    if (iter_type == tir::IterVarType::kCommReduce) {
      return true;
    }
  }
  return false;
}

bool NeedsMultiLevelTiling(Schedule sch, BlockRV block_rv) {
  // Right now it only works with a leaf block with a single statement
  if (!IsTrivialBinding(sch, block_rv)) {
    return false;
  }
  // Get block vars used in BufferStore
  Optional<Array<tir::Var>> block_vars = BlockVarsUsedInStore(sch, block_rv);
  if (!block_vars.defined()) {
    return false;
  }
  // Check reuse
  Array<tir::BufferLoad> loads = GetBufferLoad(sch, block_rv);
  int n_missing = 0;
  for (const tir::BufferLoad& load : loads) {
    n_missing += CountMissingBlockVars(load, block_vars.value());
  }
  if (n_missing >= 2) {
    return true;
  }
  if (n_missing == 0) {
    return false;
  }
  // n_missing == 1, check reduction axes
  return HasReduceBlockVar(sch, block_rv);
}

void DoMultiLevelTiling(Schedule sch, BlockRV block_rv, String tiling_structure) {
  // Do the multi-level tiling
  std::vector<int> s_idx = FindCharPos(tiling_structure, 'S');
  std::vector<int> r_idx = FindCharPos(tiling_structure, 'R');
  std::vector<std::vector<LoopRV>> order(tiling_structure.size());
  Array<LoopRV> axes = sch->GetAxes(block_rv);
  Array<Integer> iter_types = GetBlockVarTypes(sch, block_rv);
  CHECK_EQ(axes.size(), iter_types.size());
  int n = axes.size();
  for (int i = 0; i < n; ++i) {
    std::vector<int>* idx = nullptr;
    if (iter_types[i] == tir::IterVarType::kDataPar) {
      idx = &s_idx;
    } else if (iter_types[i] == tir::IterVarType::kCommReduce) {
      idx = &r_idx;
    } else {
      continue;
    }
    int n_tiles = idx->size();
    Array<tir::Var> factors = sch->SamplePerfectTile(/*n=*/n_tiles, /*loop=*/axes[i]);
    Array<LoopRV> splits =
        sch->Split(/*loop=*/axes[i], /*factors=*/{factors.begin(), factors.end()});
    for (int j = 0; j < n_tiles; ++j) {
      order[idx->at(j)].push_back(splits[j]);
    }
  }
  sch->Reorder(ConcatArray(order));
}

TVM_DLL bool IsElementWiseMatch(Schedule sch, BlockRV producer_rv, BlockRV consumer_rv) {
  const auto* producer_block = sch->Eval(producer_rv)->GetStmt<tir::BlockNode>();
  const auto* consumer_block = sch->Eval(consumer_rv)->GetStmt<tir::BlockNode>();
  CHECK(producer_block);
  CHECK(consumer_block);
  CHECK_GE(producer_block->writes.size(), 1U);
  // Check condition 1: They have the same output size
  const tir::TensorRegion& write_region = producer_block->writes[0];
  for (const tir::TensorRegion& region : producer_block->writes) {
    if (!write_region->buffer.same_as(region->buffer)) {
      continue;
    }
    if (!DomainEqual(write_region->region, region->region)) {
      return false;
    }
  }
  for (const tir::TensorRegion& region : consumer_block->writes) {
    if (!write_region->buffer.same_as(region->buffer)) {
      continue;
    }
    if (!DomainEqual(write_region->region, region->region)) {
      return false;
    }
  }
  // Check condition 2: The read is elementwise
  Optional<Array<tir::Var>> block_vars = BlockVarsUsedInStore(sch, consumer_rv);
  if (!block_vars.defined()) {
    return false;
  }
  if (Optional<Array<Bool>> access = InspectLoadIndices(sch, consumer_rv)) {
    CHECK_EQ(access.value().size(), 3);
    bool surjective = access.value()[0];
    bool injective = access.value()[1];
    bool order = access.value()[2];
    return surjective && injective && order;
  }
  return false;
}

bool IsOutputBlock(Schedule sch, BlockRV block_rv) {
  tir::StmtSRef block_sref = sch->Eval(block_rv);
  const auto* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block);
  tir::StmtSRef parent_sref = sch->sch->GetParentBlockSRef(block_sref);
  const auto* parent = parent_sref->GetStmt<tir::BlockNode>();
  CHECK(parent);
  for (const tir::TensorRegion& write : block->writes) {
    for (const tir::TensorRegion& parent_write : parent->writes) {
      if (write->buffer.same_as(parent_write->buffer)) {
        return true;
      }
    }
  }
  return false;
}

bool AutoTensorizeComparator::CompareBuffer(const tir::Buffer& lhs, const tir::Buffer& rhs) {
  if (lhs.same_as(rhs)) return true;
  // Remap both buffer itself and buffer data
  // Skip buffer shape
  bool equal = DefEqual(lhs, rhs) && DefEqual(lhs->data, rhs->data) &&
               lhs->buffer_type == rhs->buffer_type && CompareType(lhs->dtype, rhs->dtype);
  if (equal) rhs_buffer_map_[rhs] = lhs;
  return equal;
}

bool CanTensorizeRewrite(Schedule sch, BlockRV block_rv, tir::PrimFunc desc_func) {
  const tir::StmtSRef& block_sref = sch->Eval(block_rv);
  const tir::BlockRealize& block = tir::GetBlockRealize(block_sref);
  // Get desc_block, collect the loops outside desc_block at the same time
  std::set<const tir::VarNode*> desc_loop_vars;
  std::vector<const tir::LoopNode*> desc_loops;
  const tir::BlockRealizeNode* desc_block = nullptr;
  const auto* desc_body = desc_func->body.as<tir::BlockRealizeNode>();
  CHECK(desc_body);
  tir::PostOrderVisit(desc_body->block->body, [&](const ObjectRef& obj) {
    if (obj->IsInstance<tir::BlockRealizeNode>()) {
      desc_block = obj.as<tir::BlockRealizeNode>();
      return false;
    } else if (obj->IsInstance<tir::LoopNode>()) {
      const auto* desc_loop_ptr = obj.as<tir::LoopNode>();
      desc_loops.push_back(desc_loop_ptr);
      desc_loop_vars.insert(desc_loop_ptr->loop_var.get());
      if (!tir::is_zero(desc_loop_ptr->min)) return false;
    }
    return true;
  });
  std::reverse(desc_loops.begin(), desc_loops.end());
  CHECK(desc_block);
  // compare the block with desc_block
  // Ignore the scope of buffers when comparing, since we can do cache_read/write
  AutoTensorizeComparator comparator;
  if (!comparator.VisitStmt(block, GetRef<tir::BlockRealize>(desc_block))) return false;
  // matched, try to do tiling automatically if possible
  // Now the heuristic is that if block's block var binding is constant + loop var,
  // in other words, with tir.block(..., vi=Ci+i, vj=Cj+j, vk=Ck+k), then we split and reorder
  // i, j, k according to the loops outside desc_block
  // Collect the loops outside block
  std::set<const tir::VarNode*> loop_vars;
  std::vector<const tir::LoopNode*> loops;
  for (const tir::StmtSRefNode* sref = block_sref->parent;; sref = sref->parent) {
    const auto* loop_ptr = sref->GetStmt<tir::LoopNode>();
    if (loop_ptr == nullptr || loop_ptr->body->IsInstance<tir::SeqStmtNode>()) break;
    loops.push_back(loop_ptr);
    loop_vars.insert(loop_ptr->loop_var.get());
    CHECK(tir::is_zero(loop_ptr->min));
  }
  std::reverse(loops.begin(), loops.end());
  // map from block loops to desc block loops
  arith::Analyzer analyzer;
  std::map<const tir::StmtSRefNode*, int> loop_map;
  size_t offset = block->binding_values.size() - desc_block->binding_values.size();
  for (size_t i = 0; i < desc_block->binding_values.size(); ++i) {
    // find the corresponding loop of the ith block var of block
    const tir::LoopNode* loop_i = nullptr;
    for (const auto& loop : loops) {
      const PrimExpr& r = analyzer.Simplify(block->binding_values[i + offset] - loop->loop_var);
      if (!tir::ExprContainsVar(r, loop_vars)) {
        loop_i = loop;
        break;
      }
    }
    // find the corresponding loop of the ith block var of desc block
    int desc_loop_i = -1;
    for (size_t j = 0; j < desc_loops.size(); ++j) {
      const PrimExpr& r =
          analyzer.Simplify(desc_block->binding_values[i] - desc_loops[j]->loop_var);
      if (!tir::ExprContainsVar(r, desc_loop_vars)) {
        desc_loop_i = j;
        break;
      }
    }
    // check mapping
    const tir::StmtSRefNode* loop_sref = sch->sch->stmt2ref[loop_i].operator->();
    auto it = loop_map.find(loop_sref);
    if (it == loop_map.end()) {
      loop_map[loop_sref] = desc_loop_i;
    } else {
      if (it->second != desc_loop_i) return false;
    }
    for (auto it : loop_map) {
      const IntImm& loop_extent =
          Downcast<IntImm>(analyzer.Simplify(it.first->GetStmt<tir::LoopNode>()->extent));
      const IntImm& desc_loop_extent =
          Downcast<IntImm>(analyzer.Simplify(desc_loops[it.second]->extent));
      if (loop_extent->value % desc_loop_extent->value != 0) return false;
    }
  }
  return true;
}

void DoTensorizeRewrite(Schedule sch, BlockRV block_rv, tir::PrimFunc desc_func) {
  const tir::StmtSRef& block_sref = sch->Eval(block_rv);
  const tir::BlockRealize& block = tir::GetBlockRealize(block_sref);
  // Get desc_block, collect the loops outside desc_block at the same time
  std::set<const tir::VarNode*> desc_loop_vars;
  std::vector<const tir::LoopNode*> desc_loops;
  const tir::BlockRealizeNode* desc_block = nullptr;
  const auto* desc_body = desc_func->body.as<tir::BlockRealizeNode>();
  CHECK(desc_body);
  tir::PostOrderVisit(desc_body->block->body, [&](const ObjectRef& obj) {
    if (obj->IsInstance<tir::BlockRealizeNode>()) {
      desc_block = obj.as<tir::BlockRealizeNode>();
      return false;
    } else if (obj->IsInstance<tir::LoopNode>()) {
      const auto* desc_loop_ptr = obj.as<tir::LoopNode>();
      desc_loops.push_back(desc_loop_ptr);
      desc_loop_vars.insert(desc_loop_ptr->loop_var.get());
      CHECK(tir::is_zero(desc_loop_ptr->min));
    }
    return true;
  });
  std::reverse(desc_loops.begin(), desc_loops.end());
  // compare the block with desc_block
  // Ignore the scope of buffers when comparing, since we can do cache_read/write
  AutoTensorizeComparator comparator;
  CHECK(comparator.VisitStmt(block, GetRef<tir::BlockRealize>(desc_block)));
  // matched, try to do auto tiling if possible
  // Now the heuristic is that if block's block var binding is constant + loop var,
  // in other words, with tir.block(..., vi=Ci+i, vj=Cj+j, vk=Ck+k), then we split and reorder
  // i, j, k according to the loops outside desc_block
  // Collect the loops outside block
  std::set<const tir::VarNode*> loop_vars;
  std::vector<LoopRV> loops;
  const auto& all_loops = sch->GetAxes(block_rv);
  for (auto it = all_loops.rbegin(); it != all_loops.rend(); ++it) {
    const auto* loop_ptr = sch->Eval(*it)->GetStmt<tir::LoopNode>();
    if (loop_ptr->body->IsInstance<tir::SeqStmtNode>()) break;
    loops.push_back(*it);
    loop_vars.insert(loop_ptr->loop_var.get());
    CHECK(tir::is_zero(loop_ptr->min));
  }
  // build map from block loops to desc block loops
  arith::Analyzer analyzer;
  std::unordered_map<LoopRV, int, ObjectHash, ObjectEqual> loop_map;
  size_t offset = block->binding_values.size() - desc_block->binding_values.size();
  for (size_t i = 0; i < desc_block->binding_values.size(); ++i) {
    // find the corresponding loop of the ith block var of block
    LoopRV loop_i;
    for (const auto& loop : loops) {
      const PrimExpr& r = analyzer.Simplify(block->binding_values[i + offset] -
                                            sch->Eval(loop)->GetStmt<tir::LoopNode>()->loop_var);
      if (!tir::ExprContainsVar(r, loop_vars)) {
        loop_i = loop;
        break;
      }
    }
    // find the corresponding loop of the ith block var of desc block
    int desc_loop_i = -1;
    for (size_t j = 0; j < desc_loops.size(); ++j) {
      const PrimExpr& r =
          analyzer.Simplify(desc_block->binding_values[i] - desc_loops[j]->loop_var);
      if (!tir::ExprContainsVar(r, desc_loop_vars)) {
        desc_loop_i = j;
        break;
      }
    }
    // check mapping is consistent
    auto it = loop_map.find(loop_i);
    if (it == loop_map.end()) {
      loop_map[loop_i] = desc_loop_i;
    } else {
      CHECK_EQ(it->second, desc_loop_i);
    }
  }
  // split the loops
  std::unordered_set<const tir::StmtSRefNode*> subloops_set;
  std::vector<LoopRV> reorder_suffix;
  reorder_suffix.resize(loop_map.size());
  for (const auto& it : loop_map) {
    const IntImm& loop_extent =
        Downcast<IntImm>(analyzer.Simplify(sch->Eval(it.first)->GetStmt<tir::LoopNode>()->extent));
    const IntImm& desc_loop_extent =
        Downcast<IntImm>(analyzer.Simplify(desc_loops[it.second]->extent));
    CHECK_EQ(loop_extent->value % desc_loop_extent->value, 0);
    const auto& res = sch->Split(it.first, {div(loop_extent, desc_loop_extent), desc_loop_extent});
    reorder_suffix[it.second] = res[1];
    subloops_set.insert(sch->Eval(res[1]).operator->());
  }
  // reorder the loops
  std::vector<LoopRV> reorder_list;
  const auto& all_loops_after_split = sch->GetAxes(block_rv);
  bool meet = false;
  for (const auto& loop : all_loops_after_split) {
    if (subloops_set.count(sch->Eval(loop).operator->())) {
      meet = true;
    } else {
      if (meet) reorder_list.push_back(loop);
    }
  }
  reorder_list.insert(reorder_list.end(), reorder_suffix.begin(), reorder_suffix.end());
  sch->Reorder(reorder_list);
}

TVM_REGISTER_GLOBAL("meta_schedule.analysis.IsTrivialBinding").set_body_typed(IsTrivialBinding);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.GetBlockVarTypes").set_body_typed(GetBlockVarTypes);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.IsLeafBlock").set_body_typed(IsLeafBlock);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.IsLeafBlockWithSingleStmt")
    .set_body_typed(IsLeafBlockWithSingleStmt);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.GetBufferStore").set_body_typed(GetBufferStore);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.GetBufferLoad").set_body_typed(GetBufferLoad);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.CountOp").set_body_typed(CountOp);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.HasBranch").set_body_typed(HasBranch);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.BlockVarsUsedInStore")
    .set_body_typed(BlockVarsUsedInStore);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.CountMissingBlockVars")
    .set_body_typed(CountMissingBlockVars);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.InspectLoadIndices").set_body_typed(InspectLoadIndices);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.HasReduceBlockVar").set_body_typed(HasReduceBlockVar);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.NeedsMultiLevelTiling")
    .set_body_typed(NeedsMultiLevelTiling);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.DoMultiLevelTiling").set_body_typed(DoMultiLevelTiling);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.IsElementWiseMatch").set_body_typed(IsElementWiseMatch);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.IsOutputBlock").set_body_typed(IsOutputBlock);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.CanTensorizeRewrite")
    .set_body_typed(CanTensorizeRewrite);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.DoTensorizeRewrite").set_body_typed(DoTensorizeRewrite);

}  // namespace meta_schedule
}  // namespace tvm
