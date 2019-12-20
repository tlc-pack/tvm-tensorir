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
 * \file tvm/te/schedule.h
 * \brief The schedule of TE IR
 */

#ifndef TVM_TE_SCHEDULE_H_
#define TVM_TE_SCHEDULE_H_

#include <tvm/te/ir.h>
#include <tvm/te/dependency_graph.h>
#include <tvm/te/schedule_tree.h>
#include <string>
#include <unordered_map>

namespace tvm {
namespace te {

class Schedule;
class ScheduleNode : public Node {
 public:
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("func", &func);
  }
  /*! \brief The function to be scheduled */
  Function func;
  /*! \brief The root of auxiliary structure */
  ScheduleTreeNodeRef root;
  /*! \brief The dependency graph used in some primitives */
  Map<BlockTreeNodeRef, DependencyGraph> dependency_graph;
  /*! \brief The mapping from output buffer to its producer blocks */
  Map<Buffer, Array<BlockTreeNodeRef>> write_map;
  /*! \brief The mapping from stmt to its auxiliary structure */
  std::unordered_map<const StmtNode*, ScheduleTreeNodeRef> stmt_map;

  static constexpr const char* _type_key = "te.Schedule";
  TVM_DECLARE_NODE_TYPE_INFO(ScheduleNode, Node);
 private:
  friend class Schedule;
};

class Schedule : public NodeRef {
 public:
  explicit Schedule(Function func,
                    ScheduleTreeNodeRef root,
                    Map<BlockTreeNodeRef, DependencyGraph> dependency_graph,
                    Map<Buffer, Array<BlockTreeNodeRef>> write_map,
                    std::unordered_map<const StmtNode*, ScheduleTreeNodeRef> stmt_map);
  TVM_DEFINE_NODE_REF_METHODS(Schedule, NodeRef, ScheduleNode);

  /*!
   * \brief Get block from its tag
   * \param tag The query tag
   * \return the block list
   * */
  Array<BlockTreeNodeRef> GetBlock(std::string tag) const;

  /*!
   * \brief Get block from its output tensor
   * \param tag The query buffer
   * \return the block list
   * */
  Array<BlockTreeNodeRef> GetBlock(Buffer buffer) const;

  /*!
   * \brief Get all blocks in the schedule
   * \return the block list
   * */
  Array<BlockTreeNodeRef> Blocks() const;

  /*!
   * \brief Get axes of the block
   * \param block The query block
   * \return the axis list
   * */
  Array<AxisTreeNodeRef> GetAxes(BlockTreeNodeRef block) const;

  /*!
   * \brief fuse two consecutive axises of one computation.
   * \param outer The outer loop
   * \param inner The inner loop
   * \return the fused loop
   * */
  AxisTreeNodeRef fuse(AxisTreeNodeRef outer, AxisTreeNodeRef inner);

  /*!
   * \brief split a specified axis into two axises by factor.
   * \param loop The loop to be split
   * \param factor The split factor
   * \return the loops after splitting
   * */
  Array<AxisTreeNodeRef> split(AxisTreeNodeRef loop, Expr factor);

  /*!
   * \brief make one block inline, then the body of computation
   *  will be expanded and inserted at the address where the tensor
   *  is required.
   * \param block the inline block
   */
  void compute_inline(BlockTreeNodeRef block);

 private:
  void Replace(ScheduleTreeNodeRef old_node, Stmt new_stmt);

  static BlockTreeNodeRef GetFatherBlock(ScheduleTreeNodeRef node);

  static Array<Stmt> GetChildren(const Stmt& stmt);

  bool IsCompleteBlock(BlockTreeNodeRef block);

  void UpdateChildren(const Stmt& stmt, const ScheduleTreeNodeRef& father);

  inline ScheduleNode* operator->() {
    return static_cast<ScheduleNode*>(data_.get());
  }
};

}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_SCHEDULE_H_
