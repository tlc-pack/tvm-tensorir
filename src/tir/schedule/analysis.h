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

class PrimFuncNode;

/******** Verification ********/
/*!
 * \brief Verify the sref tree state is consistent with the IR
 * \param self The schedule state containing the sref to be verified
 * \throw An exception will be thrown if the sref tree is not valid
 */
void VerifySRefTree(const ScheduleState& self);
/*!
 * \brief Verify the correctness of the flags cached in BlockInfo
 * \param self The schedule state to be verified
 * \note An exception will be thrown out if the info is not valid
 */
void VerifyBlockInfo(const ScheduleState& self);

/******** Binding ********/

bool ValidateBlockBinding(const BlockRealize& realize, const Map<Var, Range>& loop_var_ranges);

IterVarType GetLoopIterType(const ScheduleState& self, const StmtSRef& loop_sref);

/******** Scope ********/

/*!
 * \brief Get the sref to the scope root block, exclusive
 * \param sref The block or loop sref to be retrieved
 * \return The sref to the scope root block
 */
StmtSRef GetScopeRoot(const StmtSRef& sref);

/*!
 * \brief Check whether a subtree satisfies the one-way fine-grained data flow check
 * \details Suppose a loop tree has several blocks on the leaves.
 * We can sort them by DFS order as B1, B2, ...., Bn.
 * The subtree satisfies compact data flow if
 * - All the blocks are complete/reduction
 * - Bi doesn't read the buffers that Bi+1, Bi+2, ... Bn will write
 * - Suppose Bi reads Bj's output buffer(j < i) and Loop k is the LCA of Bi and
 * Bj, Bj's output region covers Bi's input under Loop k
 * \param self The schedule state
 * \param child_blocks The schedule that the scope is in
 * \return A boolean indicating if the subtree satisfies the one-way fine-grained data flow check
 * \note Condition 2 and 3 are global condition of a schedulable IR,
 * so it is omitted in the check.
 */
bool IsCompactDataFlow(const ScheduleState& self, const StmtSRef& scope_root,
                       const Array<StmtSRef>& child_blocks);

/*!
 * \brief Check whether the block is a complete block under the scope
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \param scope_root The sref to the root block of the scope that `block_sref` is in
 * \return A boolean indicating if the block is a complete block
 * \note Definition of a complete block:
 * 1) dominant: the block is the only writer of its output, which dominates the reader of
 * its output buffers
 * 2) all block vars are data parallel
 * 3) no overlap between the buffers it reads and writes
 */
bool CompleteBlock(const ScheduleState& self, const StmtSRef& block_sref,
                   const StmtSRef& scope_root);

/*!
 * \brief Check whether the block is a reduction block under the scope
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \param scope_root The sref to the root block of the scope that `block_sref` is in
 * \return A boolean indicating if the block is a reduction block
 * \note Definition of a reduction block:
 * 1) dominant: the block is the only writer of its output, which dominates the reader of
 * its output buffers
 * 2) all block vars are data parallel or reduction
 * 3) block bindings are quasi-affine expressions
 * 4) has the init statement
 * 5) reduction block vars are not used to index output buffers
 */
bool ReductionBlock(const ScheduleState& self, const StmtSRef& block_sref,
                    const StmtSRef& scope_root);

/*!
 * \brief Check the merged block of init_block and update_block is a reduction block
 * \param self The schedule state
 * \param init_block_sref the query init block
 * \param update_block_sref the query update block
 * \param scope_root The sref to the scope root where `init_sref` and `update_sref` are in
 * \return Whether the merged block of init_block and update_block is a reduction block
 */
bool CanMergeReduction(const ScheduleState& self, const StmtSRef& init_block_sref,
                       const StmtSRef& update_block_sref, const StmtSRef& scope_root);

/*!
 * \brief Check the region cover for a consumer block if each region it reads are fully covered by
 * its producers.
 * \param self The schedule state
 * \param consumer_block_sref The consumer block to be checked
 * \param scope_root The sref to the scope root where `consumer_block_sref` is in
 * \return A boolean flag indicating if the read regions of the specific
 * consumer are fully covered by its predecessors
 */
bool RegionCoveredConsumer(const ScheduleState& self, const StmtSRef& consumer_block_sref,
                           const StmtSRef& scope_root);

/******** Block-loop relation ********/

StmtSRef GetSRefTreeRoot(const StmtSRef& sref);

/******** Misc ********/

bool HasSingleChild(const StmtSRef& loop_or_block_sref);

Array<StmtSRef> CollectComputeLocation(const ScheduleState& self, const StmtSRef& block_sref);

/*!
 * \brief Get the pointer to the PrimFunc that the statement pointed by sref belongs to
 * \param self The state of scheduling
 * \param sref The sref to the statement in the query
 * \return A pointer to the PrimFunc the statement belongs to
 */
const PrimFuncNode* GetRootPrimFunc(const ScheduleState& self, const StmtSRef& sref);

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_ANALYSIS_H_
