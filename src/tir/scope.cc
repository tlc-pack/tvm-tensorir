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
#include "schedule/schedule_common.h"

namespace tvm {
namespace tir {

DepEdge::DepEdge(StmtSRef dst, DepType type) {
  ObjectPtr<DepEdgeNode> node = make_object<DepEdgeNode>();
  node->dst = std::move(dst);
  node->type = type;
  data_ = std::move(node);
}

void Scope::AddEdge(const StmtSRef& from, const StmtSRef& to, DepType type) {
  if (!from.same_as(to)) {
    ScopeNode* node = operator->();
    node->forward_edges[from].push_back(DepEdge(to, type));
    node->backward_edges[to].push_back(DepEdge(from, type));
  }
}

Array<DepEdge> Scope::GetSuccessors(const StmtSRef& block) const {
  auto iter = operator->()->forward_edges.find(block);
  if (iter != operator->()->forward_edges.end()) {
    return iter->second;
  } else {
    return Array<DepEdge>();
  }
}

Array<DepEdge> Scope::GetPredecessors(const StmtSRef& block) const {
  auto iter = operator->()->backward_edges.find(block);
  if (iter != operator->()->backward_edges.end()) {
    return iter->second;
  } else {
    return Array<DepEdge>();
  }
}

bool Scope::IsDominate(const StmtSRef &block) const {
  const auto* n = DowncastPtr<BlockNode>(block->node);
  CHECK(n != nullptr);

  // Check the block is the only producer for every output tensors
  for (const auto& write : n->writes) {
    const Buffer& buffer = write->buffer;
    if (operator->()->write_map.at(buffer).size() != 1) {
      return false;
    }
  }
  return true;
}

bool Scope::IsComplete(const StmtSRef& block) const {
  // A complete block must be dominate
  if (!IsDominate(block)) return false;

  const auto* n = DowncastPtr<BlockNode>(block->node);
  CHECK(n != nullptr);
  // Check all the block vars are at data_par IterType
  for (const auto& iter_var : n->iter_vars) {
    if (iter_var->iter_type != kDataPar) {
      return false;
    }
  }

  // The Complete block can not read the writing tensors
  for (const auto& write : n->writes) {
    const Buffer& buffer = write->buffer;
    for (const auto& read : n->reads) {
      if (buffer.same_as(read->buffer)) {
        return false;
      }
    }
  }

  return true;
}

bool Scope::IsReduction(const StmtSRef& block) const {
  const auto* n = DowncastPtr<BlockNode>(block->node);
  CHECK(n != nullptr);

  // Check the binding of block is valid
  CHECK(block->binding_valid);

  // A complete block must be dominate
  CHECK(IsDominate(block));

  // Check all the block vars are at data_par/reduce IterType
  for (const auto& iter_var : n->iter_vars) {
    if (iter_var->iter_type != kDataPar && iter_var->iter_type != kCommReduce) {
      return false;
    }
  }

  // Check the block body is reduction
  const auto* reduction = DowncastPtr<ReduceStepNode>(n->body.operator->());
  const auto* lhs = DowncastPtr<BufferLoadNode>(reduction->lhs.operator->());
  CHECK(reduction != nullptr);

  // Check all the writing block vars are data_par
  for (const auto& iter_var : n->iter_vars)
    if (iter_var->iter_type != kDataPar)
      for (const auto index : lhs->indices)
        CHECK(!RelatedWithVar(iter_var->var, index));

  return true;
}

bool Scope::CanMergeReduction(const StmtSRef &init_block, const StmtSRef &update_block) const {
  const auto* init = DowncastPtr<BlockNode>(init_block->node);
  const auto* update = DowncastPtr<BlockNode>(update_block->node);

  // Check init_block and update_block both contains a single BufferStore
  const auto* init_body = DowncastPtr<BufferStoreNode>(init->body.operator->());
  const auto* update_body = DowncastPtr<BufferStoreNode>(update->body.operator->());
  CHECK(init_body != nullptr) << "init block should contain only a BufferStore";
  CHECK(update_body != nullptr) << "update block should contain only a BufferStore";
  CHECK_EQ(init_body->buffer, update_body->buffer);
  CHECK_EQ(init_body->indices.size(), update_body->indices.size());

  // Check init_block and update_block are the only producers for its output tensor
  for (const auto& write : update->writes) {
    const Buffer& buffer = write->buffer;
    if (operator->()->write_map.at(buffer).size() != 2) {
      return false;
    } else {
      CHECK(operator->()->write_map.at(buffer)[0].same_as(init_block)
            || operator->()->write_map.at(buffer)[0].same_as(update_block));
      CHECK(operator->()->write_map.at(buffer)[1].same_as(init_block)
            || operator->()->write_map.at(buffer)[1].same_as(update_block));
    }
  }

  // Check the binding of update_block is valid
  CHECK(update_block->binding_valid);

  // Check all the block vars of update_block are at data_par/reduce IterType
  for (const auto& iter_var : update->iter_vars) {
    if (iter_var->iter_type != kDataPar && iter_var->iter_type != kCommReduce) {
      return false;
    }
  }

  // Check all the writing block vars of update_block are data_par
  for (const auto& iter_var : update->iter_vars)
    if (iter_var->iter_type != kDataPar)
      for (const auto index : update_body->indices)
        CHECK(!RelatedWithVar(iter_var->var, index));

  return true;
}

TVM_REGISTER_NODE_TYPE(ScopeNode);
TVM_REGISTER_NODE_TYPE(DepEdgeNode);

}  // namespace tir
}  // namespace tvm
