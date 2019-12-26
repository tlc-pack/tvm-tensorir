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

/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Dependency between blocks
 */

#ifndef TVM_TE_DEPENDENCY_GRAPH_H_
#define TVM_TE_DEPENDENCY_GRAPH_H_

#include <tvm/te/ir.h>
#include <tvm/te/stmt_sref.h>
#include <vector>
#include <unordered_map>

namespace tvm {
namespace te {

class StmtSRef;

// Dependency type. NOTE: Currently only kRAW is useful
enum EdgeType : int {
  kRAW,
  kWAW,
  kWAR,
  kUnknown
};

/*!
 * \brief A edge in dependency graph
 */
class DepEdge;
class DepEdgeNode : public Node {
 public:
  StmtSRef dst;
  EdgeType type;

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "te.DefEdge";
  TVM_DECLARE_NODE_TYPE_INFO(DepEdgeNode, Node);
};

class DepEdge : public NodeRef {
 public:
  DepEdge(StmtSRef, EdgeType type);
  TVM_DEFINE_NODE_REF_METHODS(DepEdge, NodeRef, DepEdgeNode);
};

/*!
 * \brief Dependency Graph that stores read/write dependency between Blocks
 */
class DependencyGraph;
class DependencyGraphNode : public Node {
 public:
  /*! \brief The forward dependency edges of the block*/
  std::unordered_map<StmtSRef, std::vector<DepEdge>, NodeHash, NodeEqual> forward_edges;
  /*! \brief The backward dependency edges of the block*/
  std::unordered_map<StmtSRef, std::vector<DepEdge>, NodeHash, NodeEqual> backward_edges;

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "te.DependencyGraph";
  TVM_DECLARE_NODE_TYPE_INFO(DependencyGraphNode, Node);
};

class DependencyGraph : public NodeRef {
 public:
  /*!
   * \brief Add a dependency edge.
   * \param from The departure of the edge
   * \param to The destination of the edge
   * \param type The dependency type
   */
  void AddEdge(const StmtSRef& from, const StmtSRef& to, EdgeType type);
  /*!
  * \brief Get all blocks that are dependent on block.
  * \param stmt The query block
  */
  Array<StmtSRef> GetSuccessor(const StmtSRef& block) const;
  /*!
   * \brief get all blocks that this block dependent on.
   * \param stmt The query block
   * */
  Array<StmtSRef> GetPredecessor(const StmtSRef& block) const;

  TVM_DEFINE_NODE_REF_METHODS(DependencyGraph, NodeRef, DependencyGraphNode);

  DependencyGraphNode* operator->() {
    return static_cast<DependencyGraphNode*>(data_.get());
  }
};


}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_DEPENDENCY_GRAPH_H_
