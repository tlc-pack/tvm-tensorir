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
#include "./access_analysis.h"

#include "../arith/pattern_match.h"
#include "./auto_scheduler_utils.h"
#include "./loop_tree.h"

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_NODE_TYPE(BaseAccessPatternNode);
TVM_REGISTER_NODE_TYPE(DummyAccessPatternNode);
TVM_REGISTER_NODE_TYPE(LeafAccessPatternNode);

DummyAccessPattern::DummyAccessPattern() { data_ = make_object<DummyAccessPatternNode>(); }

/*!
 * \brief Get TIR statement from LeafStmt
 * \param obj The leaf statement
 * \return The TIR statement
 */
tir::Stmt StmtFromLeaf(const ObjectRef& obj) {
  const auto* leaf = obj.as<LeafStmtNode>();
  CHECK(leaf) << "TypeError: Expect LeafStmtNode, but gets: " << obj->GetTypeKey();
  return leaf->stmt;
}

/*!
 * \brief Check if an expression consists of a single variable, or a variable +/i an constant
 * \param expr The expression to be checked
 * \param result Output, the var inside if it satisfies the condition
 * \return A boolean indicating if it satisfies the condition
 */
bool IsVarPlusMinusConst(const PrimExpr& expr, tir::Var* result) {
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

/*!
 * \brief On a leaf block, checks if all buffer store and reduction update are in form of
 *     A[v_0 +/- const][v_i +/- const]
 * in which v_i are block variables, and const are constant integers.
 * This means the indices trivally correlates with block variables,
 * and it directly corresponds to TE-style buffer store.
 * TODO(@junrushao1994): we might need to check each block var is correlated at most once.
 * For now, it checks the following condition:
 * 1) The block has only one statement, either ReduceStep and BufferStore
 * 2) The buffer store/update trivially correlates to block variables
 * \param node The node in the loop tree to be checked
 * \param all_trivial_store Output, a flag indicating if it satisfies the condition
 * \returns The block vars in the order they are used
 */
std::vector<tir::Var> CheckAllTrivialStore(const LoopTree& node, bool* all_trivial_store) {
  // A block contains only statement
  if (node->children.size() != 1) {
    *all_trivial_store = false;
    return {};
  }
  // Collect block variables to a lookup table
  CHECK(node->block_realize.defined())
      << "InternalError: Cannot collect block vars on a null block";
  std::unordered_set<const tir::VarNode*> block_vars;
  for (const tir::IterVar& iter_var : node->block_realize.value()->block->iter_vars) {
    block_vars.insert(iter_var->var.get());
  }
  // Collect the store indices
  const auto* leaf = node->children[0].as<LeafStmtNode>();
  CHECK(leaf != nullptr) << "TypeError: Expect LeafStmtNode, but gets: "
                         << node->children[0]->GetTypeKey();
  // For each index in the store indices, check if they are
  // 1) constant
  // 2) trivially derived from a block var
  std::vector<tir::Var> result;
  for (const PrimExpr& idx : leaf->write->indices) {
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
    *all_trivial_store = false;
    return {};
  }
  *all_trivial_store = true;
  return result;
}

/*!
 * \brief Checks if the leaf block has branching statement/intrinsic inside
 * \param node The block to be checked
 * \return A boolean flag indicating the check result
 */
bool CheckHasBranch(const LoopTree& node) {
  if (node->block_realize.defined()) {
    arith::Analyzer analyzer;
    const tir::BlockRealizeNode* realize = node->block_realize.value().get();
    // if the predicate is not always 1, then there is a branch
    if (!analyzer.CanProve(realize->predicate == 1)) {
      return true;
    }
  }
  for (const ObjectRef& child : node->children) {
    if (const auto* leaf = child.as<LeafStmtNode>()) {
      bool has_branch = false;
      tir::PostOrderVisit(leaf->stmt, [&has_branch](const ObjectRef& obj) {
        if (obj->IsInstance<tir::IfThenElseNode>()) {
          has_branch = true;
        } else if (obj->IsInstance<tir::SelectNode>()) {
          has_branch = true;
        } else if (const auto* call = obj.as<tir::CallNode>()) {
          if (call->op.same_as(tir::builtin::if_then_else())) {
            has_branch = true;
          }
        }
      });
      if (has_branch) {
        return true;
      }
    }
  }
  return false;
}

/*!
 * \brief Checks if the leaf block has branching statement/intrinsic inside
 * \param node The block to be checked
 * \return A boolean flag indicating the check result
 */
bool CheckHasExpensiveOp(const LoopTree& node) {
  const tvm::Op& exp = tvm::Op::Get("tir.exp");
  for (const ObjectRef& child : node->children) {
    // For each child Stmt
    if (const auto* leaf = child.as<LeafStmtNode>()) {
      bool has_expensive_op = false;
      tir::PostOrderVisit(leaf->stmt, [&has_expensive_op, &exp](const ObjectRef& obj) {
        // Checks a TIR function call
        if (const auto* call = obj.as<tir::CallNode>()) {
          // If it is tir.exp
          if (call->op.same_as(exp)) {
            has_expensive_op = true;
          }
        }
      });
      if (has_expensive_op) {
        return true;
      }
    }
  }
  return false;
}

/*!
 * \brief Checks the axes on each buffer load can be mapped to block vars used in the buffer store
 * \param node The block to be checked
 * \param stores The block vars used in order in buffer store
 * \param exists If the mapping exists
 * \param surjective If each block var is used at least once
 * \param injective If each block var is used at most once
 * \param ordered If the mapping maintains order
 */
void AnalyzeLoadStoreMapping(const LoopTree& node, const Array<tir::Var>& stores, bool* exists,
                             bool* surjective, bool* injective, bool* ordered) {
  // Only consider block with one statement
  if (node->children.size() != 1 || !node->block_realize.defined()) {
    *exists = false;
    *surjective = false;
    *injective = false;
    *ordered = false;
    return;
  }
  // Collect indices of the block vars
  std::unordered_map<const tir::VarNode*, int> store_vars;
  {
    int index = 0;
    for (const tir::Var& var : stores) {
      store_vars[var.get()] = index++;
    }
  }
  // Those flags can be changed from true to false, but not false to true
  *exists = true;
  *surjective = true;
  *injective = true;
  *ordered = true;
  // Visit each buffer loads
  tir::PostOrderVisit(StmtFromLeaf(node->children[0]), [&](const ObjectRef& obj) {
    // 1) If we have determined the mapping does not exist, return
    // 2) If the current object is not BufferLoad that we are interested in, skip
    if (*exists == false || !obj->IsInstance<tir::BufferLoadNode>()) {
      return;
    }
    const auto* buffer_load = static_cast<const tir::BufferLoadNode*>(obj.get());
    std::vector<int> store_be_mapped_times(store_vars.size(), 0);
    std::vector<int> load_mapped_to_store_index;
    // For each non-constant index
    for (const PrimExpr& idx : buffer_load->indices) {
      if (IsConstInt(idx)) {
        continue;
      }
      tir::Var var;
      // Check if it matches a block var
      if (IsVarPlusMinusConst(idx, &var)) {
        if (store_vars.count(var.get())) {
          int index = store_vars.at(var.get());
          load_mapped_to_store_index.push_back(index);
          store_be_mapped_times[index] += 1;
          continue;
        }
      }
      // If not, the load-store mapping does not exist
      *exists = false;
      *surjective = false;
      *injective = false;
      *ordered = false;
      return;
    }
    // Check `store_be_mapped_times` to determine if the mapping is injective and surjective
    for (int times : store_be_mapped_times) {
      // If there is a store axis that doesn't have corresponding any load axis
      if (times == 0) {
        *surjective = false;
      }
      // If there is a store axis that has more than 2 corresponding load axes
      if (times >= 2) {
        *injective = false;
      }
    }
    // Check `load_mapped_to_store_index` to determine if the mapping is in order
    int last = -1;
    for (int store_axis : load_mapped_to_store_index) {
      if (last > store_axis) {
        *ordered = false;
        break;
      }
      last = store_axis;
    }
  });
}

/*!
 * \brief Anaylze data reuse pattern. For each buffer load, we check how many store axes that do not
 * appear in load axes, in which we could see re-use opportunities
 * \param node The leaf block
 * \param stores The block vars used in buffer store
 * \return The number of axes missing
 */
int AnalyzeAxesReuse(const LoopTree& node, const Array<tir::Var>& stores) {
  if (!node->block_realize.defined() || node->children.size() != 1) {
    return false;
  }
  int n_missing = 0;
  tir::PostOrderVisit(StmtFromLeaf(node->children[0]), [&n_missing, &stores](const ObjectRef& obj) {
    if (const auto* load = obj.as<tir::BufferLoadNode>()) {
      // Collect all variables used in the buffer loaed
      std::unordered_set<const tir::VarNode*> vars_in_load;
      for (const PrimExpr& idx : load->indices) {
        tir::PostOrderVisit(idx, [&vars_in_load](const ObjectRef& obj) {
          if (const auto* var = obj.as<tir::VarNode>()) {
            vars_in_load.insert(var);
          }
        });
      }
      // Count the number of axes that do not appear
      for (const tir::Var& store_axes : stores) {
        if (!vars_in_load.count(store_axes.get())) {
          ++n_missing;
        }
      }
    }
  });
  return n_missing;
}

LeafAccessPattern::LeafAccessPattern(const LoopTree& node) {
  ObjectPtr<LeafAccessPatternNode> n = make_object<LeafAccessPatternNode>();
  n->num_stmts = node->children.size();
  n->has_branch = CheckHasBranch(node);
  n->has_expensive_op = CheckHasExpensiveOp(node);
  n->block_vars_in_trivial_store = CheckAllTrivialStore(node, &n->all_trivial_store);
  if (n->all_trivial_store) {
    AnalyzeLoadStoreMapping(node, n->block_vars_in_trivial_store, &n->lsmap_exists,
                            &n->lsmap_surjective, &n->lsmap_injective, &n->lsmap_ordered);
  } else {
    n->lsmap_exists = false;
    n->lsmap_surjective = false;
    n->lsmap_injective = false;
    n->lsmap_ordered = false;
  }
  if (n->all_trivial_store) {
    n->num_axes_reuse = AnalyzeAxesReuse(node, n->block_vars_in_trivial_store);
  } else {
    n->num_axes_reuse = 0;
  }
  data_ = std::move(n);
}

Map<LoopTree, BaseAccessPattern> AnalyzeAccess(const LoopTree& root) {
  std::unordered_map<LoopTree, BaseAccessPattern, ObjectPtrHash, ObjectPtrEqual> result;
  std::function<void(const LoopTree&)> fvisit = [&result, &fvisit](const LoopTree& node) -> void {
    int num_children = 0;
    for (const ObjectRef& obj : node->children) {
      if (const auto* child = obj.as<LoopTreeNode>()) {
        fvisit(GetRef<LoopTree>(child));
        ++num_children;
      }
    }
    if (num_children == 0) {
      result[node] = LeafAccessPattern(node);
    } else {
      result[node] = DummyAccessPattern();
    }
  };
  fvisit(root);
  return result;
}

TVM_REGISTER_GLOBAL("auto_scheduler.access_analysis.AnalyzeAccess").set_body_typed(AnalyzeAccess);

}  // namespace auto_scheduler
}  // namespace tvm
