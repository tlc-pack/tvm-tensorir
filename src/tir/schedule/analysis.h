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

#include <tvm/arith/analyzer.h>
#include <tvm/tir/schedule/state.h>

#include <unordered_set>
#include <vector>

#include "../../runtime/thread_storage_scope.h"

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

/******** IR Module ********/

/*!
 * \brief Get the root block of the PrimFunc that the input StmtSRef is in
 * \param sref The queried StmtSRef
 * \return The root block
 */
Block GetRootBlock(StmtSRef sref);

/*!
 * \brief Get PrimFunc and GlobalVar that the root block belongs to
 * \param mod The IRModule
 * \param root_block The root block of the PrimFunc
 * \param result_g_var The result GlobalVar
 * \return The result PrimFunc where the root block belongs to
 * \note This function returns the pointer instead of ObjectRef to avoid later copy-on-write
 */
const PrimFuncNode* GetRootPrimFunc(const IRModule& mod, const StmtNode* root_block,
                                    GlobalVar* result_g_var);

/******** Scope ********/
/*!
 * \brief Checks if scope the specified sref is in is a stage-pipeline and return it
 * \param self The schedule state
 * \param sref The sref whose scope is to be checked
 * \param require_stage_pipeline A boolean indicating whether to check stage pipeline
 * \throw ScheduleError if the sref has been the root of the AST (so it has no scope root), or its
 * scope root is not a stage pipeline
 * \return The block sref to the scope root
 */
StmtSRef GetScopeRoot(const ScheduleState& self, const StmtSRef& sref, bool require_stage_pipeline);

/*!
 * \brief Checks whether the block is a complete block under the scope
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \param scope_root_sref The sref to the root block of the scope that `block_sref` is in
 * \return A boolean indicating if the block is a complete block
 * \note Definition of a complete block:
 * 1) All block vars are data parallel
 * 2) Dominant: the block is the only writer of its output,
 * dominating the reader of its output buffers
 * 3) No overlap between the buffers the block reads and writes
 */
bool IsCompleteBlock(const ScheduleState& self, const StmtSRef& block_sref,
                     const StmtSRef& scope_root_sref);

/*!
 * \brief Check if the block is a complete block under the scope
 * \param self The schedule state
 * \param block_sref The sref to the block whose completeness is to be checked
 * \param scope_root_sref The scope root of the block
 * \throw ScheduleError If the block is not a complete block
 */
void CheckCompleteBlock(const ScheduleState& self, const StmtSRef& block_sref,
                        const StmtSRef& scope_root_sref);

/*!
 * \brief Check whether the block is a reduction block under the scope
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \param scope_root_sref The sref to the root block of the scope that `block_sref` is in
 * \return A boolean indicating if the block is a reduction block
 * \note Definition of a reduction block:
 * 1) The block has the `init` statement
 * 2) All the block bindings are quasi-affine expressions
 * 3) All block vars are either data parallel block vars or reduction block vars
 * 4) Dominant: the block is the only writer of its output, dominating the reader of its output
 * buffers
 * 5) The reduction block vars are not used to index the output buffers
 */
bool IsReductionBlock(const ScheduleState& self, const StmtSRef& block_sref,
                      const StmtSRef& scope_root_sref);

/*!
 * \brief Check if the block is a reduction block under the scope
 * \param self The schedule state
 * \param block_sref The sref of the block to be checked
 * \param scope_root_sref The scope root of the block
 * \throw ScheduleError If the block is not a reduction block
 */
void CheckReductionBlock(const ScheduleState& self, const StmtSRef& block_sref,
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

/*!
 * \brief Get the vars involved in the bindings of data parallel block vars and reduction block
 * vars, respectively
 * \param block_realize The BlockRealize to be analyzed
 * \param data_par_vars The vars that appear in the binding of any data parallel block iter
 * \param reduce_vars The vars that appear in the binding of any reduction block iter
 * \return A boolean indicating whether the block has block iters that is neither a data parallel
 * block iter nor a reduction block iter
 */
bool GetVarsTouchedByBlockIters(const BlockRealize& block_realize,
                                std::unordered_set<const VarNode*>* data_par_vars,
                                std::unordered_set<const VarNode*>* reduce_vars);

/******** Block-loop relation ********/

/*!
 * \brief Gets StmtSRefs of leaf blocks of a scope where a specific block/loop is in
 * \param self The schedule state
 * \param parent_sref The StmtSRef that points to the parent block/loop
 * \return A list of StmtSRefs of leaf block
 */
Array<StmtSRef> GetChildBlockSRefOnSRefTree(const ScheduleState& self, const StmtSRef& parent_sref);

/*!
 * \brief Gets the BlockRealize of the leaf blocks of a scope where a specific block/loop is in
 * \param parent_sref The StmtSRef that points to the parent block/loop
 * \return A list of leaf BlockRealize
 */
Array<BlockRealize> GetChildBlockRealizeOnSRefTree(const StmtSRef& parent_sref);

/*!
 * \brief Get the BlockRealize of the single child block of the block or loop specified by
 * `parent_sref` on SRef tree, or throw an exception if there is 0 or multiple child blocks
 * \param self The schedule state
 * \param parent_sref The StmtSRef that points to the parent block/loop
 * \return The BlockRealize of the single child block
 * \throw ScheduleError If there is 0 or multiple child blocks
 */
BlockRealize CheckGetSingleChildBlockRealizeOnSRefTree(const ScheduleState& self,
                                                       const StmtSRef& parent_sref);
/*!
 * \brief Get the BlockRealize of the input block
 * \param self The schedule state
 * \param block_sref The StmtSRef of the queried block
 * \return The BlockRealize of the input block
 */
BlockRealize GetBlockRealize(const ScheduleState& self, const StmtSRef& block_sref);

/******** Commutative Reducer ********/

/*!
 * \brief Get the list of the registered reducer-getter functions
 * \return The list of the registered reducer-getter functions
 * \sa ReducerRegistry
 */
std::vector<runtime::TypedPackedFunc<CommReducer(DataType)>> GetReducerGetters();

/*!
 * \brief Given the input identity and the combiner BufferStore of a reduction, extract the
 * corresponding commutative reducer and its lhs, rhs if possible.
 * \param identity The identity of the reduction
 * \param combiner The combiner of the reduction
 * \param result_reducer The extracted CommReducer
 * \param lhs The extracted lhs of the reducer
 * \param rhs The extracted rhs of the reducer
 * \return A boolean indicating whether a corresponding commutative reducer is found
 */
bool FromIdentityCombiner(const PrimExpr& identity, const BufferStore& combiner,
                          CommReducer* result_reducer, PrimExpr* lhs, PrimExpr* rhs);

/******** Misc: not upstream-ed yet ********/

/*!
 * \brief Verify the correctness of the flags cached in BlockInfo
 * \param self The schedule state to be verified
 * \note An exception will be thrown out if the info is not valid
 */
void VerifyBlockInfo(const ScheduleState& self);

bool ValidateBlockBinding(const BlockRealize& realize, const Map<Var, Range>& loop_var_ranges);

IterVarType GetLoopIterType(const ScheduleState& self, const StmtSRef& loop_sref);

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

StmtSRef GetSRefTreeRoot(const StmtSRef& sref);

bool HasSingleChild(const StmtSRef& loop_or_block_sref);

Array<StmtSRef> CollectComputeLocation(const ScheduleState& self, const StmtSRef& block_sref);

class PrimFuncNode;

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_ANALYSIS_H_
