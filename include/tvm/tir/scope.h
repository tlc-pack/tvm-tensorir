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
 *  \brief Scope information using in TIR
 */

#ifndef TVM_TIR_SCOPE_H_
#define TVM_TIR_SCOPE_H_

#include <tvm/tir/stmt.h>
#include <tvm/tir/ir.h>
#include <tvm/tir/stmt_sref.h>
#include <vector>
#include <unordered_map>

namespace tvm {
namespace tir {

class StmtSRef;

/*!
 * \brief Dependency Graph that stores read/write dependency between Blocks
 * \note It is not a traditional and complete dependency graph, but only a
 *       dependency hint. If there is an edge from A to B, iff B writes at
 *       least one of the read tensors of A. That's means B must produce the
 *       necessary element (but not all the element) before A under the Lowest
 *       Common Ancestor (LCA) Loop of the A and B.
 */
class Scope;
class ScopeNode : public Object {
 public:
  /*! \brief The forward dependency edges of the block */
  std::unordered_map<StmtSRef, Array<StmtSRef>, ObjectHash, ObjectEqual> forward_edges;
  /*! \brief The backward dependency edges of the block */
  std::unordered_map<StmtSRef, Array<StmtSRef>, ObjectHash, ObjectEqual> backward_edges;
  /*! \brief The mapping from the buffer to the blocks who write it */
  std::unordered_map<Buffer, Array<StmtSRef>, ObjectHash, ObjectEqual> write_map;

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "te.Scope";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScopeNode, Object);
};

class Scope : public ObjectRef {
 public:
  /*!
   * \brief Add a dependency edge.
   * \param from The departure of the edge
   * \param to The destination of the edge
   */
  void AddEdge(const StmtSRef& from, const StmtSRef& to);
  /*!
  * \brief get all blocks that this block dependent on.
  * \param stmt The query block
  */
  Array<StmtSRef> GetSuccessors(const StmtSRef& block) const;
  /*!
   * \brief Get all blocks that are dependent on block.
   * \param stmt The query block
   * */
  Array<StmtSRef> GetPredecessors(const StmtSRef& block) const;

  TVM_DEFINE_OBJECT_REF_METHODS(Scope, ObjectRef, ScopeNode);

  ScopeNode* operator->() {
    return static_cast<ScopeNode*>(data_.get());
  }
};


}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCOPE_H_
