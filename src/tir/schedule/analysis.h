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
#ifndef TVM_TIR_SCHEDULE_ANALYSIS_H_
#define TVM_TIR_SCHEDULE_ANALYSIS_H_

#include <tvm/tir/schedule/state.h>

namespace tvm {
namespace tir {

/******** Verification ********/
/*!
 * \brief Verify the sref tree state is consistent with the IR
 * \param self The schedule state containing the sref to be verified
 * \throw An exception will be thrown if the sref tree is not valid
 */
void VerifySRefTree(const ScheduleState& self);

/******** Block-loop relation ********/
/*!
 * \brief Retrieve blocks in a specific function with its name
 * \param self The schedule state
 * \param name The name of the blocks to be retrieved
 * \param func_name The name of the function
 * \return A list of blocks with the specific name
 */
Array<StmtSRef> GetBlocks(const ScheduleState& self, const String& name, const String& func_name);
/*!
 * \brief Get the parent loops of the block in its scope, from outer to inner
 * \param self The schedule state
 * \param block_sref The query block
 * \return A list of loops above the given block in its scope, from outer to inner
 */
Array<StmtSRef> GetLoops(const StmtSRef& block_sref);
/*!
 * \brief Get the leaf blocks of a scope where a specific block/loop is in
 * \param self The schedule state
 * \param parent_sref The StmtSRef that points to the parent block/loop
 * \return A list of leaf blocks
 */
Array<StmtSRef> GetChildBlocks(const ScheduleState& self, const StmtSRef& parent_sref);

/*!
 * \brief Get the sref to the scope root block, exclusive
 * \param sref The block or loop sref to be retrieved
 * \return The sref to the scope root block
 */
TVM_DLL StmtSRef GetScopeSRef(const StmtSRef& sref);

/*!
 * \brief Get block from its tag
 * \param tag The query tag
 * \return the block schedulable reference list
 */
TVM_DLL Array<StmtSRef> GetBlocks(const ScheduleState& self, const String& name);

/*!
 * \brief Get loops of the block
 * \param block The query block
 * \return the loop sref list
 */
TVM_DLL Array<StmtSRef> GetAxes(const ScheduleState& self, const StmtSRef& block_sref);

/*!
 * \brief Get the child blocks of a specific parent block/loop
 * \param parent_sref The StmtSRef that points to the parent block/loop
 * \param inclusive If true and parent_sref is a block, return a single-element list containing
 * parent_sref
 * \return A list of child blocks
 */
TVM_DLL Array<StmtSRef> GetChildBlocks(const ScheduleState& self, const StmtSRef& parent_sref,
                                       bool inclusive);

/*!
 * \brief Get the producer of a specific block
 * \return The producers
 */
TVM_DLL Array<StmtSRef> GetProducers(const ScheduleState& self, const StmtSRef& block_sref);

/*!
 * \brief Get the consumers of a specific block
 * \return The consumers
 */
TVM_DLL Array<StmtSRef> GetConsumers(const ScheduleState& self, const StmtSRef& block_sref);

TVM_DLL bool HasSingleChild(const StmtSRef& loop_or_block_sref);

TVM_DLL IterVarType GetLoopIterType(const ScheduleState& self, const StmtSRef& loop_sref);

TVM_DLL Array<StmtSRef> CollectComputeLocation(const ScheduleState& self,
                                               const StmtSRef& block_sref);

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_ANALYSIS_H_
