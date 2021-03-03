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

#include "./utils.h"

namespace tvm {
namespace tir {

TVM_DLL bool ValidateBlockBinding(const BlockRealize& realize,
                                  const Map<Var, Range>& loop_var_ranges);

/*! \brief Check the region cover for the single consumer block */
TVM_DLL void VerifyRegionCover(const ScheduleState& self, const StmtSRef& consumer_block_sref);

/*! \brief Verify the correctness of the sref tree */
TVM_DLL void VerifySRefTree(const ScheduleState& self);

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
