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
#include <tvm/ir_visitor.h>

namespace tvm {
namespace te {

class DependencyAnalyzer : public IRVisitor {
 public:
  explicit DependencyAnalyzer(Function func) : func_(func) {
    NodePtr<DependencyGraphNode> node = make_node<DependencyGraphNode>();
    graph_ = DependencyGraph(node);
  }
  void Visit_(const BlockNode* op) {
    Block block = GetRef<Block>(op);
    for (const auto& read : op->reads) {
      const auto& read_buffer = read->buffer;
      for (const auto& write_block : write_map_[read_buffer]) {
        graph_.AddEdge(block, write_block, kWAR);
      }
    }
  }

  DependencyGraph GetDependency() {
    Visit(func_->body);
    return graph_;
  }

 private:
  std::unordered_map<Buffer, std::list<Block>,
                     NodeHash, NodeEqual> write_map_;
  Function func_;
  DependencyGraph graph_;
};

DepEdge::DepEdge(Block dst, EdgeType type) {
  NodePtr<DepEdgeNode> node = make_node<DepEdgeNode>();
  node->dst = std::move(dst);
  node->type = std::move(type);
  data_ = std::move(node);
}

DependencyGraph::DependencyGraph(Function func) {
  data_ = DependencyAnalyzer(func).GetDependency().data_;
}

void DependencyGraph::AddEdge(Block from, Block to, EdgeType type) {
  if (!from.same_as(to)) {
    operator->()->forward_edges[from].push_back(DepEdge(to, type));
    operator->()->backward_edges[to].push_back(DepEdge(from, type));
  }
}

Array<Block> DependencyGraph::GetSuccessor(Block block) const {
  Array<Block> ret;
  auto iter = operator->()->forward_edges.find(block);
  if (iter != operator->()->forward_edges.end()) {
    for (const auto& x : iter->second) {
      ret.push_back(x->dst);
    }
  }
  return ret;
}

Array<Block> DependencyGraph::GetPredecessor(Block block) const {
  Array<Block> ret;
  auto iter = operator->()->backward_edges.find(block);
  if (iter != operator->()->backward_edges.end()) {
    for (const auto& x : iter->second) {
      ret.push_back(x->dst);
    }
  }
  return ret;
}

void DependencyGraph::InlineBlock(Block block) {
  auto& forward_edges = operator->()->forward_edges;
  auto& backward_edges = operator->()->backward_edges;

  std::list<DepEdge> successors = forward_edges[block];
  std::list<DepEdge> predecessors = backward_edges[block];

  // delete old edges
  forward_edges.erase(block);
  backward_edges.erase(block);

  for (const auto& src : successors) {
    auto& edges = backward_edges[src->dst];
    auto iter = edges.begin();
    while (iter != edges.end()) {
      if ((*iter)->dst == block) {
        iter = edges.erase(iter);
      } else {
        ++iter;
      }
    }
  }

  for (const auto& src : predecessors) {
    auto& edges = forward_edges[src->dst];
    auto iter = edges.begin();
    while (iter != edges.end()) {
      if ((*iter)->dst == block) {
        iter = edges.erase(iter);
      } else {
        ++iter;
      }
    }
  }

  // relink new edges
  for (const auto& src : predecessors) {
    for (const auto& dst : successors) {
      if (src->type == kRAW && dst->type == kRAW) {
        AddEdge(src->dst, dst->dst, kRAW);
      } else if (src->type == kWAW && dst->type == kWAW) {
        AddEdge(src->dst, dst->dst, kWAW);
      } else if (src->type == kWAR && dst->type == kWAW) {
        AddEdge(src->dst, dst->dst, kWAR);
      }
      // for all other cases, their relation does not change
    }
  }
}

TVM_REGISTER_NODE_TYPE(DependencyGraphNode);
TVM_REGISTER_NODE_TYPE(DepEdgeNode);

}  // namespace te
}  // namespace tvm
