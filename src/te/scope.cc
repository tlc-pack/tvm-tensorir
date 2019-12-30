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

#include <tvm/te/scope.h>

namespace tvm {
namespace te {

void Scope::AddEdge(const StmtSRef& from, const StmtSRef& to) {
  if (!from.same_as(to)) {
    ScopeNode* node = operator->();
    node->forward_edges[from].push_back(to);
    node->backward_edges[to].push_back(from);
  }
}

Array<StmtSRef> Scope::GetSuccessors(const StmtSRef& block) const {
  auto iter = operator->()->forward_edges.find(block);
  if (iter != operator->()->forward_edges.end()) {
    return iter->second;
  } else {
    return Array<StmtSRef>();
  }
}

Array<StmtSRef> Scope::GetPredecessors(const StmtSRef& block) const {
  auto iter = operator->()->backward_edges.find(block);
  if (iter != operator->()->backward_edges.end()) {
    return iter->second;
  } else {
    return Array<StmtSRef>();
  }
}

TVM_REGISTER_NODE_TYPE(ScopeNode);

}  // namespace te
}  // namespace tvm
