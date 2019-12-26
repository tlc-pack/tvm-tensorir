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

#include <tvm/te/dependency_graph.h>

namespace tvm {
namespace te {

DepEdge::DepEdge(StmtSRef dst, EdgeType type) {
  NodePtr<DepEdgeNode> node = make_node<DepEdgeNode>();
  node->dst = std::move(dst);
  node->type = type;
  data_ = std::move(node);
}

void DependencyGraph::AddEdge(const StmtSRef& from, const StmtSRef& to, EdgeType type) {
  if (!from.same_as(to)) {
    DependencyGraphNode* node = operator->();
    node->forward_edges[from].push_back(DepEdge(to, type));
    node->backward_edges[to].push_back(DepEdge(from, type));
  }
}

Array<StmtSRef> DependencyGraph::GetSuccessor(const StmtSRef& block) const {
  Array<StmtSRef> ret;
  auto iter = operator->()->forward_edges.find(block);
  if (iter != operator->()->forward_edges.end()) {
    for (const auto& x : iter->second) {
      ret.push_back(x->dst);
    }
  }
  return ret;
}

Array<StmtSRef> DependencyGraph::GetPredecessor(const StmtSRef& block) const {
  Array<StmtSRef> ret;
  auto iter = operator->()->backward_edges.find(block);
  if (iter != operator->()->backward_edges.end()) {
    for (const auto& x : iter->second) {
      ret.push_back(x->dst);
    }
  }
  return ret;
}

TVM_REGISTER_NODE_TYPE(DependencyGraphNode);
TVM_REGISTER_NODE_TYPE(DepEdgeNode);

}  // namespace te
}  // namespace tvm
