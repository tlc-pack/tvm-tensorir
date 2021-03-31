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
 * \brief Verifies the sref tree state is consistent with the IR
 * \param self The schedule state containing the sref to be verified
 * \throw An exception will be thrown if the sref tree is not valid
 */
void VerifySRefTree(const ScheduleState& self);
/*!
 * \brief Verifies the cached flags in the schedule state, including:
 * - affine_binding
 * - region_cover
 * - stage_pipeline
 * \param self The schedule state to be verified
 * \throw An exception will be thrown if some srefs are not valid
 */
void VerifyCachedFlags(const ScheduleState& self);

/******** Scope ********/
/*!
 * \brief Gets the sref to the scope root block, exclusive
 * \param sref The block or loop sref to be retrieved
 * \return The sref to the scope root block. NullOpt if `sref` is the root block of the IR
 */
Optional<StmtSRef> GetScopeRoot(const StmtSRef& sref);

/*!
 * \brief Checks if scope the specified sref is in is a stage-pipeline and return it
 * \param prim The name of the schedule primitive
 * \param self The schedule state
 * \param sref The sref whose scope is to be checked
 * \throw ScheduleError if the sref has been the root of the AST (so it has no scope root), or its
 * scope root is not a stage pipeline
 * \return The block sref to the scope root
 */
StmtSRef GetScopeRootAndCheckStagePipeline(const ScheduleState& self, const StmtSRef& sref);

/*!
 * \brief Checks whether the block is a complete block under the scope
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \param scope_root The sref to the root block of the scope that `block_sref` is in
 * \return A boolean indicating if the block is a complete block
 * \note Definition of a complete block:
 * 1) All block vars are data parallel
 * 2) Dominant: the block is the only writer of its output,
 * dominating the reader of its output buffers
 * 3) No overlap between the buffers the block reads and writes
 */
bool IsCompleteBlock(const ScheduleState& self, const StmtSRef& block_sref,
                     const StmtSRef& scope_root);

/*!
 * \brief Checks if the block is a complete block
 * \param self The schedule state
 * \param block_sref The sref to the block whose completeness is to be checked
 * \param scope_root_sref The scope root of the block
 * \throw ScheduleError If the block is not a complete block
 */
void CheckCompleteBlock(const ScheduleState& self, const StmtSRef& block_sref,
                        const StmtSRef& scope_root_sref);

/******** Binding ********/
/*!
 * \brief Verifies if the block binding in a specific BlockRealize is an affine binding.
 * The binding can be represented as an injective affine map from the loop iterators.
 * \param realize The BlockRealize to be analyzed
 * \param loop_var_ranges The ranges of the loop variables
 * \param analyzer The analyzer
 * \return A boolean flag indicating if the binding is affine
 */
bool IsAffineBinding(const BlockRealize& realize, const Map<Var, Range>& loop_var_ranges,
                     arith::Analyzer* analyzer);

/*!
 * \brief Extracts the ranges of loop variables in a path of the sref tree
 * \param low_inclusive The lowest node in the path
 * \param high_exclusive The highest node in the path, defaults to the scope root if not specified
 * \param extra_relax_scope If the scope is not global, the method will look beyond the limit and
 * retrieve extra domains. For example,
 * - if the storage scope is warp, it will look upwards for threadIdx.x
 * - if the storage scope is shared, it will look for threadIdx.x/y/z
 * \return The loop domain
 */
Map<Var, Range> LoopDomainOfSRefTreePath(const StmtSRef& low_inclusive,
                                         const Optional<StmtSRef>& high_exclusive = NullOpt,
                                         const runtime::StorageScope& extra_relax_scope =  //
                                         runtime::StorageScope{runtime::StorageRank::kGlobal, ""});

/*!
 * \brief Returns the block var binding
 * \param realize The BlockRealize to be analyzed
 * \return The block var binding
 */
Map<Var, PrimExpr> GetBindings(const BlockRealize& realize);

/******** Block-loop relation ********/
/*!
 * \brief Retrieves blocks in a specific function with its name
 * \param self The schedule state
 * \param name The name of the blocks to be retrieved
 * \param func_name The name of the function
 * \return A list of blocks with the specific name
 */
Array<StmtSRef> GetBlocks(const ScheduleState& self, const String& name, const String& func_name);
/*!
 * \brief Gets the parent loops of the block in its scope, from outer to inner
 * \param self The schedule state
 * \param block_sref The query block
 * \return A list of loops above the given block in its scope, from outer to inner
 */
Array<StmtSRef> GetLoops(const StmtSRef& block_sref);
/*!
 * \brief Gets the leaf blocks of a scope where a specific block/loop is in
 * \param self The schedule state
 * \param parent_sref The StmtSRef that points to the parent block/loop
 * \return A list of leaf blocks
 */
Array<StmtSRef> GetChildBlocks(const ScheduleState& self, const StmtSRef& parent_sref);
/*!
 * \brief Verify the correctness of the sref tree
 * \param self The schedule state containing the sref to be verified
 * \note An exception will be thrown out if the sref tree is not valid
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
/*!
 * \brief Get block from its tag
 * \param tag The query tag
 * \return the block schedulable reference list
 */
Array<StmtSRef> GetBlocks(const ScheduleState& self, const String& name);

/*!
 * \brief Get loops of the block
 * \param block The query block
 * \return the loop sref list
 */
Array<StmtSRef> GetAxes(const ScheduleState& self, const StmtSRef& block_sref);

/*!
 * \brief Get the child blocks of a specific parent block/loop
 * \param parent_sref The StmtSRef that points to the parent block/loop
 * \param inclusive If true and parent_sref is a block, return a single-element list containing
 * parent_sref
 * \return A list of child blocks
 */
Array<StmtSRef> GetChildBlocks(const ScheduleState& self, const StmtSRef& parent_sref,
                               bool inclusive = false);

/*!
 * \brief Get the producer of a specific block
 * \return The producers
 */
Array<StmtSRef> GetProducers(const ScheduleState& self, const StmtSRef& block_sref);

/*!
 * \brief Get the consumers of a specific block
 * \return The consumers
 */
Array<StmtSRef> GetConsumers(const ScheduleState& self, const StmtSRef& block_sref);

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
