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

#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt_functor.h>

#include "schedule_common.h"

namespace tvm {
namespace tir {

Array<Var> GatherVars(const ObjectRef& stmt_or_expr) {
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> result;
  PreOrderVisit(stmt_or_expr, [&result](const ObjectRef& node) -> bool {
    if (const auto* var = node.as<VarNode>()) {
      result.insert(GetRef<Var>(var));
    }
    return true;
  });
  return std::vector<Var>(result.begin(), result.end());
}

class StatementInliner : public StmtExprMutator {
 public:
  explicit StatementInliner(const BlockRealize& realize, Map<Block, Block>* block_sref_map,
                            const std::unordered_map<const StmtNode*, const StmtNode*>& replace_map)
      : realize_(realize), block_sref_map_(block_sref_map), replace_map_(replace_map) {
    const Block& block = realize->block;
    const auto store = block->body.as<BufferStoreNode>();
    value_ = store->value;
    CHECK_EQ(block->writes.size(), 1);
    for (const auto& index : store->indices) {
      const auto* variable = index.as<VarNode>();
      CHECK(variable) << "Only support inline direct access block";
      Var var = GetRef<Var>(variable);
      vars_.push_back(var);
    }
    Array<Var> value_vars = GatherVars(value_);
    for (const auto& x : value_vars) {
      CHECK(std::find_if(vars_.begin(), vars_.end(),
                         [=](const Var& var) -> bool { return var.same_as(x); }) != vars_.end())
        << "Not All variable in value can be replaced by index vars";
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    // Find the original block before leaf removing
    bool is_scope_block = is_scope_block_;
    is_scope_block_ = false;
    const StmtNode* node = op;
    for (const auto& pair : replace_map_) {
      if (pair.second == op) {
        node = pair.first;
        break;
      }
    }
    Block origin_block = Downcast<Block>(GetRef<Stmt>(node));
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BlockNode>();
    CHECK(op != nullptr);

    // Update Allocation
    const Buffer& buffer = realize_->block->writes[0]->buffer;
    Array<BufferAllocate> allocations;
    for (const auto allocate : op->allocations) {
      if (allocate->buffer != buffer) allocations.push_back(allocate);
    }

    // Update read region only for none-scope block
    Array<TensorRegion> reads(nullptr);
    if (is_scope_block) reads = op->reads;

    auto block = Block(op->iter_vars, reads, op->writes, op->body, allocations,
                       op->annotations, op->tag);
    block_sref_map_->Set(block, origin_block);
    return std::move(block);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    const Buffer& buffer = realize_->block->writes[0]->buffer;
    if (op->buffer == buffer) {
      std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> vmap;
      for (size_t i = 0; i < op->indices.size(); ++i) {
        vmap[vars_[i]] = op->indices[i];
      }
      return Substitute(value_, vmap);
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

 private:
  /*! The block realize of the block to be inlined*/
  const BlockRealize& realize_;
  /*! The block vars of the block to be inlined*/
  Array<Var> vars_;
  /*! The buffer store value*/
  PrimExpr value_;
  /*! The block sref map using in Repalce*/
  Map<Block, Block>* block_sref_map_;
  const std::unordered_map<const StmtNode*, const StmtNode*>& replace_map_;
  /*! Whether this block is the scope block (first visited block)*/
  bool is_scope_block_ = true;
};

void ScheduleNode::compute_inline(const StmtSRef& block_sref) {
  // conditions:
  // 1. only write to one element
  // 2. is terminal block
  // -> The inner stmt is a BufferStore
  const auto* block = block_sref->GetStmt<BlockNode>();
  const BlockRealize& realize = GetBlockRealize(block_sref);
  const StmtSRef& parent_block_sref = GetParentBlockSRef(block_sref);
  const auto* scope_block = parent_block_sref->GetStmt<BlockNode>();
  const Scope& scope = scopes.at(parent_block_sref);
  CHECK(block->body.as<BufferStoreNode>())
    << "Can only inline single assignment statement";
  CHECK_EQ(block->writes.size(), 1)
    << "Can only inline statement with one output";
  CHECK(scope.IsComplete(block_sref))
    << "Can only inline a complete block";

  // Remove leaf
  std::pair<Stmt, Stmt> removed = RemoveLeaf(block_sref, parent_block_sref);
  std::unordered_map<const StmtNode*, const StmtNode*> replace_map = {
      {removed.first.get(), removed.second.get()}
  };
  Stmt replaced = StmtReplacer(replace_map)(GetRef<Stmt>(scope_block));

  // Inline
  Map<Block, Block> block_sref_map;
  StatementInliner inliner(realize, &block_sref_map, replace_map);
  Stmt inlined_stmt = inliner(replaced);

  this->Replace(parent_block_sref, inlined_stmt, block_sref_map);
}

}  // namespace tir
}  // namespace tvm
