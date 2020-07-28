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

#include <tvm/tir/scope.h>

#include "./schedule_common.h"

namespace tvm {
namespace tir {

TVM_REGISTER_NODE_TYPE(ScopeNode);
TVM_REGISTER_NODE_TYPE(DepEdgeNode);

DepEdge::DepEdge(StmtSRef dst, DepType type) {
  ObjectPtr<DepEdgeNode> node = make_object<DepEdgeNode>();
  node->dst = std::move(dst);
  node->type = type;
  data_ = std::move(node);
}

Scope::Scope() { data_ = make_object<ScopeNode>(); }

void Scope::AddEdge(const StmtSRef& from, const StmtSRef& to, DepType type) {
  if (!from.same_as(to)) {
    ScopeNode* self = operator->();
    self->forward_edges[from].push_back(DepEdge(to, type));
    self->backward_edges[to].push_back(DepEdge(from, type));
  }
}

Array<DepEdge> Scope::GetSuccessors(const StmtSRef& block_sref) const {
  const auto& edges = (*this)->forward_edges;
  auto iter = edges.find(block_sref);
  if (iter != edges.end()) {
    return iter->second;
  } else {
    return Array<DepEdge>();
  }
}

Array<DepEdge> Scope::GetPredecessors(const StmtSRef& block_sref) const {
  const auto& edges = (*this)->backward_edges;
  auto iter = edges.find(block_sref);
  if (iter != edges.end()) {
    return iter->second;
  } else {
    return Array<DepEdge>();
  }
}

bool Scope::IsDominate(const StmtSRef& block_sref) const {
  const BlockNode* block = block_sref->GetStmt<BlockNode>();
  CHECK(block != nullptr) << "InternalError: Scope::IsDominate only works on tir::Block";
  // Condition: Block is the only writer to its outputs
  const auto& buffer_writters = (*this)->buffer_writers;
  for (const TensorRegion& write_region : block->writes) {
    CHECK(buffer_writters.count(write_region->buffer))
        << "InternalError: buffer \"" << write_region->buffer->name
        << "\" does not exist in the current scope, when querying block:\n"
        << GetRef<Block>(block);
    // Check if the buffer is only written once (by the given block)
    if (buffer_writters.at(write_region->buffer).size() != 1) {
      return false;
    }
  }
  return true;
}

bool Scope::IsComplete(const StmtSRef& block_sref) const {
  const BlockNode* block = block_sref->GetStmt<BlockNode>();
  CHECK(block != nullptr)
      << "InternalError: Scope::IsComplete only accepts tir::Block, but get type: "
      << block_sref->stmt->GetTypeKey();
  // Cond 1. A complete block must be dominate
  if (!IsDominate(block_sref)) {
    return false;
  }
  // Cond 2. Check if all the block vars are data parallel
  for (const auto& iter_var : block->iter_vars) {
    if (iter_var->iter_type != kDataPar) {
      return false;
    }
  }
  // Cond 3. Check if there is no overlap between buffers read and buffers written
  for (const auto& write : block->writes) {
    const Buffer& buffer = write->buffer;
    for (const auto& read : block->reads) {
      if (buffer.same_as(read->buffer)) {
        return false;
      }
    }
  }
  return true;
}

/*!
 * \brief Check if each reduction instance is valid. Particularly, check:
 * 1) Each iteration variable is either data parallel or reduction
 * 2) Indices used to access the output buffer are not related to or affected by reduction iteration
 * variables.
 * \param iter_vars Iteration variables of the reduction
 * \param output_buffer_indices Indices used to access the output buffer
 * \return A boolean indicating if the reduction instance is valid
 */
bool CheckReductionInstance(const Array<IterVar>& iter_vars,
                            const Array<PrimExpr>& output_buffer_indices) {
  for (const auto& iter_var : iter_vars) {
    IterVarType kind = iter_var->iter_type;
    // Check 1. Ecah iter_var can only be data parallel or reduction
    if (kind != kDataPar && kind != kCommReduce) {
      return false;
    }
    // Check 2. Each reduction iter_var should not be used to index output buffer
    if (kind == kCommReduce) {
      for (const PrimExpr& idx : output_buffer_indices) {
        if (ExprContainsVar(idx, iter_var->var)) {
          return false;
        }
      }
    }
  }
  return true;
}

bool Scope::IsReduction(const StmtSRef& block_sref) const {
  const BlockNode* block = block_sref->GetStmt<BlockNode>();
  CHECK(block != nullptr)
      << "InternalError: Scope::IsReduction only accepts tir::Block, but get type: "
      << block_sref->stmt->GetTypeKey();
  // Cond 0. Block binding is valid
  if (!block_sref->binding_valid) {
    return false;
  }
  // Cond 1. Dominate block
  if (!IsDominate(block_sref)) {
    return false;
  }
  // Cond 2. Check the block body is reduction
  const auto* reduce = block->body.as<ReduceStepNode>();
  if (reduce == nullptr) {
    return false;
  }
  // Cond 3. All block vars are either data parallel or reduction, and reduction vars should not
  // affect indexing the output buffer
  const auto* buffer_load = reduce->lhs.as<BufferLoadNode>();
  CHECK(buffer_load != nullptr) << "InternalError: ReduceStepNode::lhs should be BufferLoadNode";
  return CheckReductionInstance(block->iter_vars, buffer_load->indices);
}

bool Scope::IsCompactDataFlow(const StmtSRef& subtree_sref, const ScheduleNode* schedule) const {
  for (const auto& block : schedule->GetChildBlocks(subtree_sref)) {
    if (!IsComplete(block) && !IsReduction(block)) {
      return false;
    }
  }
  return true;
}

bool Scope::CanMergeReduction(const StmtSRef& init_sref, const StmtSRef& update_sref) const {
  const auto* init = init_sref->GetStmt<BlockNode>();
  const auto* update = update_sref->GetStmt<BlockNode>();
  CHECK(init != nullptr) << "InternalError: Scope::CanMergeReduction only accepts tir::Block as "
                            "init_block, but get type:"
                         << init_sref->stmt->GetTypeKey();
  CHECK(update != nullptr) << "InternalError: Scope::CanMergeReduction only accepts tir::Block as "
                              "update_block, but get type:"
                           << update_sref->stmt->GetTypeKey();
  // Cond 1. Check the binding of update block is valid
  if (!update_sref->binding_valid) {
    return false;
  }
  // Cond 2. Check init_block and update_block are the only two producers for their output buffer
  for (const auto& write_region : update->writes) {
    const Array<StmtSRef>& writers = (*this)->buffer_writers.at(write_region->buffer);
    if (writers.size() != 2) {
      return false;
    }
    if (!writers[0].same_as(init_sref) && !writers[0].same_as(update_sref)) {
      return false;
    }
    if (!writers[1].same_as(init_sref) && !writers[1].same_as(update_sref)) {
      return false;
    }
  }
  // Cond 3. init and update share the same buffer
  const auto* init_body = init->body.as<BufferStoreNode>();
  const auto* update_body = update->body.as<BufferStoreNode>();
  CHECK(init_body != nullptr)
      << "InternalError: init_block should contain only a BufferStore as its lhs, but get type: "
      << init->body->GetTypeKey();
  CHECK(update_body != nullptr)
      << "InternalError: update_block should contain only a BufferStore as its lhs, but get type:"
      << update->body->GetTypeKey();
  if (!init_body->buffer.same_as(update_body->buffer)) {
    return false;
  }
  // Access must be the same dimensional
  CHECK_EQ(init_body->indices.size(), update_body->indices.size())
      << "InternalError: indexing to the same buffer with different dimensions";
  // Cond 4. All block vars of update_block are either data parallel or reduction,
  // and reduction vars of update_block should not affect indexing the output buffer
  return CheckReductionInstance(update->iter_vars, update_body->indices);
}

void Scope::AddChildBlock(
    const StmtSRef& child_sref,
    std::unordered_map<Buffer, Array<StmtSRef>, ObjectPtrHash, ObjectPtrEqual>* _buffer_readers) {
  const BlockNode* block = child_sref->GetStmt<BlockNode>();
  CHECK(block) << "InternalError: Scope::AddChildBlock only accepts a Block as child_sref";
  auto& buffer_readers = *_buffer_readers;
  auto& buffer_writers = (*this)->buffer_writers;
  // Step 1. Update `buffer_readers` and `buffer_writer` for each buffer
  for (const TensorRegion& region : block->writes) {
    buffer_writers[region->buffer].push_back(child_sref);
  }
  for (const TensorRegion& region : block->reads) {
    buffer_readers[region->buffer].push_back(child_sref);
  }
  // Check and update block dependencies: RAW, WAW, WAR.
  // Note: AddEdge is effectively NOP on self-loops
  // Step 2. Update RAW dependency
  for (const TensorRegion& region : block->reads) {
    if (buffer_writers.count(region->buffer)) {
      for (const StmtSRef& from : buffer_writers[region->buffer]) {
        this->AddEdge(from, child_sref, DepType::kRAW);
      }
    }
  }
  // Step 3. Update WAW dependency
  for (const TensorRegion& region : block->writes) {
    if (buffer_writers.count(region->buffer)) {
      for (const StmtSRef& from : buffer_writers[region->buffer]) {
        this->AddEdge(from, child_sref, DepType::kWAW);
      }
    }
  }
  // Step 4. Check WAR dependency: not allowed in the IR
  for (const TensorRegion& region : block->writes) {
    if (buffer_readers.count(region->buffer)) {
      for (const StmtSRef& from : buffer_readers[region->buffer]) {
        CHECK(from.same_as(child_sref)) << "TypeError: WAR dependency is not allowed";
      }
    }
  }
}

}  // namespace tir
}  // namespace tvm
