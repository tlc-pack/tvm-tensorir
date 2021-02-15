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
  explicit StatementInliner(const BlockNode* block, Map<Block, Block>* block_sref_map,
                            const std::unordered_map<const StmtNode*, const StmtNode*>& replace_map)
      : block_(block), block_sref_map_(block_sref_map), replace_map_(replace_map) {
    const auto store = block_->body.as<BufferStoreNode>();
    value_ = store->value;
    CHECK_EQ(block_->writes.size(), 1);
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
    const Buffer& buffer = block_->writes[0]->buffer;
    Array<BufferAllocate> allocations;
    for (const auto allocate : op->allocations) {
      if (allocate->buffer != buffer) allocations.push_back(allocate);
    }

    Array<TensorRegion> reads(nullptr);
    if (is_scope_block) {
      reads = op->reads;
    } else {
      // Update read region only for none-scope block
      BlockReadWriteCollector block_read_write_collector(allocations);
      block_read_write_collector(op->body);
      reads = block_read_write_collector.reads();
    }

    Block block(op->iter_vars, reads, op->writes, op->body, allocations, op->annotations, op->tag,
                op->init);

    block_sref_map_->Set(block, origin_block);
    return std::move(block);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    const Buffer& buffer = block_->writes[0]->buffer;
    if (buffer.same_as(op->buffer)) {
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
  const BlockNode* block_;
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
  /*!
   * Check:
   *    1. The inner stmt of block_sref if a BufferStore
   *    2. block_sref if a complete Block
   */
  const auto* block = block_sref->GetStmt<BlockNode>();
  const StmtSRef& scope_block_sref = GetParentBlockSRef(block_sref);
  const auto* scope_block = scope_block_sref->GetStmt<BlockNode>();
  const Scope& scope = scopes.at(scope_block_sref);
  CHECK(block->body.as<BufferStoreNode>())
      << "ValueError: 'compute_inline' can only inline single assignment statement";
  CHECK_EQ(block->writes.size(), 1)
      << "ValueError: 'compute_inline' can only inline statement with one output";
  CHECK(scope->IsComplete(block_sref))
      << "ValueError: 'compute_inline' can only inline a complete block";

  // Remove leaf
  std::pair<Stmt, Stmt> removed = RemoveLeaf(block_sref, scope_block_sref);
  std::unordered_map<const StmtNode*, const StmtNode*> replace_map = {
      {removed.first.get(), removed.second.get()}};
  Stmt replaced = StmtReplacer(replace_map)(GetRef<Stmt>(scope_block));

  // Inline
  Map<Block, Block> block_sref_map;
  StatementInliner inliner(block, &block_sref_map, replace_map);
  Stmt inlined_stmt = inliner(replaced);

  this->Replace(scope_block_sref, inlined_stmt, block_sref_map);
}

class ReverseStatementInliner : public StmtExprMutator {
 public:
  explicit ReverseStatementInliner(const BlockNode* block, const BlockNode* producer,
                                   Map<Block, Block>* block_sref_map)
      : block_(block), producer_(producer), block_sref_map_(block_sref_map) {
    // Check BufferStore of producer is like Buffer[v0, v1, ...]
    const auto* store = producer_->body.as<BufferStoreNode>();
    value_ = store->value;
    CHECK_EQ(producer_->writes.size(), 1);
    for (const auto& index : store->indices) {
      const auto* variable = index.as<VarNode>();
      CHECK(variable)
          << "ValueError: 'reverse_compute_inline' only supports inline direct access block";
      Var var = GetRef<Var>(variable);
      new_vars_.push_back(var);
      old_vars_.push_back(NullValue<Var>());
    }
    Array<Var> value_vars = GatherVars(store->value);
    for (const auto& x : value_vars) {
      CHECK(std::find_if(new_vars_.begin(), new_vars_.end(),
                         [=](const Var& var) -> bool { return var.same_as(x); }) != new_vars_.end())
          << "ValueError: Not all variable in value can be replaced by index vars";
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    bool is_producer = op == producer_;
    Block origin_producer = Downcast<Block>(GetRef<Stmt>(producer_));
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BlockNode>();
    CHECK(op != nullptr);
    // update allocation
    const Buffer& buffer = producer_->writes[0]->buffer;
    Array<BufferAllocate> allocations;
    for (const auto allocate : op->allocations) {
      if (allocate->buffer != buffer) allocations.push_back(allocate);
    }
    // update read/write region
    BlockReadWriteCollector block_read_write_collector(allocations);
    block_read_write_collector(op->body);
    Block block(op->iter_vars, block_read_write_collector.reads(),
                block_read_write_collector.writes(), op->body, allocations, op->annotations,
                op->tag, op->init);
    if (is_producer) block_sref_map_->Set(block, origin_producer);
    return std::move(Block(block));
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    const Buffer& buffer = producer_->writes[0]->buffer;
    if (buffer.same_as(op->buffer)) {
      // find the BufferStore of producer, now check the BufferLoad inside the old store
      const auto* old_store = block_->body.as<BufferStoreNode>();
      PrimExpr value = VisitExpr(old_store->value);
      // check BufferStore of block is substitutable
      auto vmap = [&](const Var& var) -> Optional<PrimExpr> {
        for (size_t i = 0; i < old_vars_.size(); ++i) {
          if (old_vars_[i].same_as(var) || new_vars_[i].same_as(var)) return new_vars_[i];
        }
        LOG(FATAL) << "ValueError: indices not match";
        return NullOpt;
      };
      std::vector<PrimExpr> new_indices;
      for (const auto& index : old_store->indices) new_indices.push_back(Substitute(index, vmap));
      return BufferStore(old_store->buffer, Substitute(value, vmap), new_indices);
    }
    return GetRef<Stmt>(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    const Buffer& buffer = producer_->writes[0]->buffer;
    if (buffer.same_as(op->buffer)) {
      for (size_t i = 0; i < op->indices.size(); ++i) {
        const auto* var = op->indices[i].as<VarNode>();
        CHECK(var) << "ValueError: indices not match";
        if (!old_vars_[i].defined()) {
          old_vars_[i] = GetRef<Var>(var);
        } else {
          CHECK(old_vars_[i].same_as(GetRef<Var>(var))) << "ValueError: indices not match";
        }
      }
      return value_;
    }
    return GetRef<PrimExpr>(op);
  }

 private:
  /*! The block to be reverse inlined*/
  const BlockNode* block_;
  /*! The producer of the block to be reverse inlined*/
  const BlockNode* producer_;
  /*! The block vars in block_*/
  std::vector<Var> old_vars_;
  /*! The block vars in producer_*/
  std::vector<Var> new_vars_;
  /*! The buffer store value*/
  PrimExpr value_;
  /*! The block sref map using in Repalce*/
  Map<Block, Block>* block_sref_map_;
};

void ScheduleNode::reverse_compute_inline(const StmtSRef& block_sref) {
  /*!
   * Check:
   *    1. block_sref is complete
   *    2. The inner stmt of block_sref is a BufferStore
   *    3. block_sref has only one producer
   *    4. The producer is complete
   *    5. The inner stmt of producer is a BufferStore
   *    6. The producer has only one consumer(which is block_sref)
   */
  const auto* block = block_sref->GetStmt<BlockNode>();
  CHECK(block != nullptr)
      << "TypeError: 'reverse_compute_at' expects 'block' to be a block, but get type: "
      << block_sref->stmt->GetTypeKey();
  const StmtSRef& scope_block_sref = GetParentBlockSRef(block_sref);
  const auto* scope_block = scope_block_sref->GetStmt<BlockNode>();
  const Scope& scope = scopes.at(scope_block_sref);
  // Cond 1. Check block_sref is complete
  CHECK(scope->IsComplete(block_sref))
      << "ValueError: 'reverse_compute_inline' expects the 'block' to be a complete block";
  // Cond 2. The inner stmt of block_sref if a BufferStore
  CHECK(block->body.as<BufferStoreNode>())
      << "ValueError: 'reverse_compute_inline' expects the 'block' contains a single BufferStore";
  // Cond 3. block_sref has only one RAW producer
  const auto& producers = scope->GetPredecessors(block_sref);
  CHECK_EQ(producers.size(), 1)
      << "ValueError: 'reverse_compute_inline' expects the 'block' has only one producer";
  CHECK(producers[0]->type == DepType::kRAW)
      << "ValueError: 'reverse_compute_inline' expects the 'block' has only one producer";
  const StmtSRef& producer_sref = producers[0]->dst;
  // Cond 4. The producer is complete
  CHECK(scope->IsComplete(producer_sref))
      << "ValueError: 'reverse_compute_inline' expects the producer of 'block' to be complete";
  // Cond 5. The inner stmt of producer is a BufferStore
  const auto* producer = producer_sref->GetStmt<BlockNode>();
  CHECK(producer->body.as<BufferStoreNode>())
      << "ValueError: 'reverse_compute_inline' expects the producer of 'block' to contain a single "
         "BufferStore";
  // Cond 6. The producer has only one consumer(which is block_sref)
  const auto& consumers = scope->GetSuccessors(producer_sref);
  CHECK_EQ(consumers.size(), 1) << "ValueError: 'reverse_compute_inline' expects 'block' is the "
                                   "only consumer of its producer";
  CHECK_EQ(consumers[0]->dst, block_sref) << "ValueError: 'reverse_compute_inline' expects 'block' "
                                             "is the only consumer of its producer";

  // Remove leaf
  std::pair<Stmt, Stmt> removed = RemoveLeaf(block_sref, scope_block_sref);
  std::unordered_map<const StmtNode*, const StmtNode*> replace_map = {
      {removed.first.get(), removed.second.get()}};
  Stmt replaced = StmtReplacer(replace_map)(GetRef<Stmt>(scope_block));
  // Inline
  Map<Block, Block> block_sref_map;
  ReverseStatementInliner inliner(block, producer, &block_sref_map);
  Stmt inlined_stmt = inliner(replaced);

  this->Replace(scope_block_sref, inlined_stmt, block_sref_map);
}

}  // namespace tir
}  // namespace tvm
