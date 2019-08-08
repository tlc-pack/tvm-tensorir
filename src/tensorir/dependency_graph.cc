/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Dependency between blocks
 */

#include <vector>
#include <algorithm>
#include <string>
#include <utility>
#include "dependency_graph.h"

namespace tvm {
namespace tensorir {

// Maker
Edge EdgeNode::make(BlockTreeNode dst, EdgeType type) {
  NodePtr<EdgeNode> node = make_node<EdgeNode>();
  node->dst = dst;
  node->type = type;
  return Edge(node);
}

DependencyGraph DependencyGraphNode::make(ScheduleTreeNode node) {
  ReadWriteGraph rwgraph;
  StdNodeMap<ScheduleTreeNode, size_t> block_order;
  StdNodeSet<Tensor> tensor_list;

  // gather read/write info
  size_t ct = 0;
  std::function<void(ScheduleTreeNode)> access_visitor = [&](ScheduleTreeNode n) {
    if (const AxisTreeNodeNode* node = n.as<AxisTreeNodeNode>()) {
      for (auto x : node->children) {
        access_visitor(x);
      }
    } else if (const BlockTreeNodeNode* node = n.as<BlockTreeNodeNode>()) {
      BlockTreeNode block = GetRef<BlockTreeNode>(node);
      for (auto x : node->inputs) {
        rwgraph.InsertRead(block, x->data);
        tensor_list.insert(x->data);
      }
      for (auto x : node->outputs) {
        rwgraph.InsertWrite(block, x->data);
        tensor_list.insert(x->data);
      }
      block_order[block] = ct++;
    }
  };
  access_visitor(node);

  // build dependency for all blocks that write to/read from a same tensor
  DependencyGraph dep_graph(make_node<DependencyGraphNode>());
  for (const auto& t : tensor_list) {
    enum AccessType {
      kRead = 0,
      kWrite = 1
    };

    std::vector<std::pair<BlockTreeNode, AccessType> > blocks;
    for (auto x : rwgraph.GetReadBy(t)) {
      blocks.push_back(std::make_pair(x.first, kRead));
    }
    for (auto x : rwgraph.GetWriteBy(t)) {
      blocks.push_back(std::make_pair(x.first, kWrite));
    }

    // sort according to original order
    std::sort(blocks.begin(), blocks.end(), [block_order]
        (const std::pair<BlockTreeNode, AccessType>& a,
         const std::pair<BlockTreeNode, AccessType>& b) -> bool {
      return block_order.at(a.first) < block_order.at(b.first);
    });

    // scan blocks
    size_t last_write = std::string::npos;
    for (size_t i = 0; i < blocks.size(); ++i) {
      if (blocks[i].second == kWrite) {
        if (last_write == std::string::npos) {
          for (size_t j = 0; j < i; ++j) {
            dep_graph.AddEdge(blocks[j].first, blocks[i].first, EdgeType::kWAR);
          }
        } else {
          for (size_t j = last_write+1; j < i; ++j) {
            dep_graph.AddEdge(blocks[j].first, blocks[i].first, EdgeType::kWAR);
          }
          dep_graph.AddEdge(blocks[last_write].first, blocks[i].first, EdgeType::kWAW);
        }
        last_write = i;
      } else {   // blocks[i].second == kRead
        if (last_write == std::string::npos) {
          continue;
        } else {
          dep_graph.AddEdge(blocks[last_write].first, blocks[i].first, EdgeType::kRAW);
        }
      }
    }
  }

  return dep_graph;
}

// Graph operations
void DependencyGraph::AddNode(BlockTreeNode block) {
  // Do nothing, because std::unordered_map will init value for unseen key.
}

void DependencyGraph::AddEdge(BlockTreeNode from, BlockTreeNode to, EdgeType type) {
  if (from == to) {
    return;
  }
  operator->()->forward_edges[from].push_back(EdgeNode::make(to, type));
  operator->()->backward_edges[to].push_back(EdgeNode::make(from, type));
}

Set<BlockTreeNode> DependencyGraph::GetSuccessor(BlockTreeNode block) const {
  Set<BlockTreeNode> ret;
  auto iter = operator->()->forward_edges.find(block);
  if (iter != operator->()->forward_edges.end()) {
    for (const auto& x : iter->second) {
      ret.insert(x->dst);
    }
  }
  return ret;
}

Set<BlockTreeNode> DependencyGraph::GetPredecessor(BlockTreeNode block) const {
  Set<BlockTreeNode> ret;
  auto iter = operator->()->backward_edges.find(block);
  if (iter != operator->()->backward_edges.end()) {
    for (const auto& x : iter->second) {
      ret.insert(x->dst);
    }
  }
  return ret;
}

// Operations for schedule
void DependencyGraph::InlineNode(BlockTreeNode stmt) {
  auto& forward_edges = operator->()->forward_edges;
  auto& backward_edges = operator->()->backward_edges;

  std::list<Edge> successors = forward_edges[stmt];
  std::list<Edge> predecessors = backward_edges[stmt];

  // delete old edges
  forward_edges.erase(stmt);
  backward_edges.erase(stmt);

  for (const auto& src : successors) {
    auto& edges = backward_edges[src->dst];
    auto iter = edges.begin();
    while (iter != edges.end()) {
      if ((*iter)->dst == stmt) {
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
      if ((*iter)->dst == stmt) {
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
      // or only changes to "RAR" which is useless.
    }
  }
}

// Debug tool
std::ostream& PrintDependencyGraph(std::ostream &os,  DependencyGraph g) {
  for (const auto& x : g->forward_edges) {
    for (const auto& edge : x.second) {
      switch (edge->type) {
        case kWAR: os << "WAR: "; break;
        case kWAW: os << "WAW: "; break;
        case kRAW: os << "RAW: "; break;
        case kRAR: os << "RAR: "; break;
        default:   os << "XXX: "; break;
      }
      os << x.first->stmt << "  ->  " << edge->dst->stmt << "\n";
    }
  }

  return os;
}

}  // namespace tensorir
}  // namespace tvm
