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
 * "AS IS" BASIS, WITHOUT WAStmtStmtANTIES OStmt CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) by Contributors
 * \file schedule_creator.h
 */

#ifndef TVM_TE_SCHEDULE_SCHEDULE_CREATOR_H_
#define TVM_TE_SCHEDULE_SCHEDULE_CREATOR_H_
#include <tvm/ir_mutator.h>
#include <tvm/te/schedule.h>
#include <tvm/te/schedule_tree.h>

namespace tvm {
namespace te {
/*! \brief Create a schedule from a function */
class ScheduleCreator : public IRMutator {
 public:
  /*!
   * \brief Constructor
   * \param func The target function
   */
  explicit ScheduleCreator(Function func);

  Stmt Mutate_(const te::BlockNode* op, const Stmt& s);
  Stmt Mutate_(const te::LoopNode* op, const Stmt& s);
  Stmt Mutate_(const te::SeqStmtNode* op, const Stmt& s);
  Stmt Mutate_(const ir::Block* op, const Stmt& s);
  /*!
   * \brief Create a schedule
   * \return the schedule
   */
  Schedule Create();
 private:
  Function func_;
  Map<Buffer, Array<BlockTreeNodeRef>> write_map_;
  Map<BlockTreeNodeRef, DependencyGraph> dependency_graph_;
  std::unordered_map<const StmtNode*, ScheduleTreeNodeRef> stmt_map;
  ScheduleTreeNodeRef father_node_;
  BlockTreeNodeRef current_block_;
};

}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_SCHEDULE_SCHEDULE_CREATOR_H_
