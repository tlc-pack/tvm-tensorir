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

#ifndef TVM_TE_BLOCK_DEPENDENCY_H_
#define TVM_TE_BLOCK_DEPENDENCY_H_

#include <tvm/te/ir.h>
#include <tvm/te/stmt_sref.h>
#include <vector>
#include <unordered_map>

namespace tvm {
namespace te {

class StmtSRef;

/*!
 * \brief Dependency Graph that stores read/write dependency between Blocks
 * \note It is not a traditional and complete dependency graph, but only a
 *       dependency hint. If there is an edge from A to B, iff B writes at
 *       least one of the read tensors of A. That's means B must produce the
 *       necessary element (but not all the element) before A under the Lowest
 *       Common Ancestor (LCA) Loop of the A and B.
 */
class BlockDependency;
class BlockDependencyNode : public Node {
 public:
  /*! \brief The forward dependency edges of the block*/
  std::unordered_map<StmtSRef, Array<StmtSRef>, NodeHash, NodeEqual> forward_edges;
  /*! \brief The backward dependency edges of the block*/
  std::unordered_map<StmtSRef, Array<StmtSRef>, NodeHash, NodeEqual> backward_edges;

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "te.BlockDependency";
  TVM_DECLARE_NODE_TYPE_INFO(BlockDependencyNode, Node);
};

class BlockDependency : public NodeRef {
 public:
  /*!
   * \brief Add a dependency edge.
   * \param from The departure of the edge
   * \param to The destination of the edge
   */
  void AddEdge(const StmtSRef& from, const StmtSRef& to);
  /*!
  * \brief Get all blocks that are dependent on block.
  * \param stmt The query block
  */
  Array<StmtSRef> GetSuccessors(const StmtSRef& block) const;
  /*!
   * \brief get all blocks that this block dependent on.
   * \param stmt The query block
   * */
  Array<StmtSRef> GetPredecessors(const StmtSRef& block) const;

  TVM_DEFINE_NODE_REF_METHODS(BlockDependency, NodeRef, BlockDependencyNode);

  BlockDependencyNode* operator->() {
    return static_cast<BlockDependencyNode*>(data_.get());
  }
};


}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_BLOCK_DEPENDENCY_H_
