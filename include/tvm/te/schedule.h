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

namespace tvm {
namespace te {

class Schedule;
class ScheduleNode : public Node {
 public:
  void VisitAttrs(AttrVisitor* v) {}
  static constexpr const char* _type_key = "te.Schedule";
  TVM_DECLARE_NODE_TYPE_INFO(ScheduleNode, Node);
 private:
  friend class Schedule;
  /*! \brief The function to be scheduled */
  Function func_;
  /*! \brief The mapping from AST node to its father */
  Map<Stmt, Stmt> father_map_;
  /*! \brief The dependency graph used in some primitives */
  DependencyGraph dependency_graph_;
  /*! \brief The maping from output buffer to its producer blocks */
  Map<Buffer, Array<Block>> write_map_;
};

class Schedule : public NodeRef {
 public:
  explicit Schedule(Function func,
                    DependencyGraph dependency_graph,
                    Map<Buffer, Array<Block>> write_map);
  TVM_DEFINE_MUTABLE_NODE_REF_METHODS(Schedule, NodeRef, ScheduleNode);

  /*!
   * \brief Get block from its tag
   * \param tag The query tag
   * \return the block list
   * */
  Array<Block> GetBlock(std::string tag) const;

  /*!
   * \brief Get block from its output tensor
   * \param tag The query buffer
   * \return the block list
   * */
  Array<Block> GetBlock(Buffer buffer) const;

  /*!
   * \brief Get all blocks in the schedule
   * \return the block list
   * */
  Array<Block> Blocks() const;
  /*!
   * \brief Get axes of the block
   * \param block The query block
   * \return the axis list
   * */
  Array<Loop> GetAxes(Block block) const;

 private:
  /*!
   * \brief Update the father of AST node
   * \param father_stmt the node whose children need update.
   * \param recursive whether recursively update whole sub AST.
   */
  void UpdateFather(Stmt father_stmt, bool recursive = false);

};

}  // namespace te
}  // namespace tvm

#endif  //TVM_TE_SCHEDULE_H_
