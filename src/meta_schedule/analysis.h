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
#ifndef SRC_META_SCHEDULE_ANALYSIS_H_
#define SRC_META_SCHEDULE_ANALYSIS_H_

#include <utility>

#include "../tir/schedule/schedule_common.h"
#include "./schedule.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Checks if
 * 1) number of blocks vars equals to number of loop vars
 * 2) each block var is bound to a loop var directly
 * 3) the order is preserved, i.e. the i-th block var is the i-th loop var
 * \param sch The TIR schedule class
 * \param block_sref The block to be analyzed
 * \return A boolean indicating if the block binding is trivial
 */
TVM_DLL bool IsTrivialBinding(const tir::Schedule& sch, const tir::StmtSRef& block_sref);

/*!
 * \brief Returns the IterVarType of each block var
 * \param sch The TIR schedule class
 * \param block_sref The block to be analyzed
 * \return An array of integers, the IterVarTypes corresponding to each block var in order
 * \sa tir::IterVarType
 */
TVM_DLL Array<Integer> GetBlockVarTypes(const tir::Schedule& sch, const tir::StmtSRef& block_sref);

/*!
 * \brief Check if the iter types of all block vars are data parallel
 * \param sch The TIR schedule class
 * \param block_sref The block to be analyzed
 * \return A boolean indicating if the block is spatial
 */
TVM_DLL bool IsSpatial(const tir::Schedule& sch, const tir::StmtSRef& block_sref);

/*!
 * \brief Checks if the specific block is a leaf block and its body is a single statement
 * \param sch The TIR schedule class
 * \param block_sref The block to be analyzed
 * \return A boolean indiciating if the block is a leaf block and its body is a single statement
 */
TVM_DLL bool IsSingleStmtLeaf(const tir::Schedule& sch, const tir::StmtSRef& block_sref);

/*!
 * \brief Checks if a block is output block
 * \param sch The TIR schedule class
 * \param block_sref The block to be analyzed
 * \return A boolean flag indicating if it is an output block
 */
TVM_DLL bool IsOutputBlock(const tir::Schedule& sch, const tir::StmtSRef& block_sref);

/*!
 * \brief Count the number of occurrence of an operator, i.e. tir.exp
 * \param sch The TIR schedule class
 * \param block_sref The block to be analyzed
 * \param op The operator to be counted
 * \return An integer indicating the number of its occurrence
 */
TVM_DLL int CountOp(const tir::Schedule& sch, const tir::StmtSRef& block_sref, const Op& op);

/*!
 * \brief Check if there is any branch in the given block, which includes
 * 1) block predicate
 * 2) if-then-else statement
 * 3) select expression
 * 4) if-then-else operator
 * \param sch The TIR schedule class
 * \param block_sref The block to be analyzed
 * \return A boolean indicating there is at least a branch in the given block
 */
TVM_DLL bool HasBranch(const tir::Schedule& sch, const tir::StmtSRef& block_sref);

/*!
 * \brief Checks whether the producer and consumer matches in elementwise way.
 * Assuming consumer_sref is the only consumer of producer_sref.
 * \param sch The meta schedule class
 * \param producer_sref The producer block
 * \param consumer_sref The consumer block
 * \return A boolean flag indicating if they match
 */
TVM_DLL bool IsElementWiseMatch(const tir::Schedule& sch, const tir::StmtSRef& producer_sref,
                                const tir::StmtSRef& consumer_sref);

/*!
 * \brief Checks if a block needs multi-level tiling
 * \param sch The TIR schedule class
 * \param block_sref The block to be analyzed
 * \return A boolean flag indicating if the block needs multi-level tiling
 */
TVM_DLL bool NeedsMultiLevelTiling(const tir::Schedule& sch, const tir::StmtSRef& block_sref);

/*!
 * \brief Checks if a block can be inlined
 * \param sch The TIR schedule class
 * \param block_sref The block to be analyzed
 * \return A boolean flag indicating if the block needs multi-level tiling
 */
TVM_DLL bool IsStrictlyInlineable(const tir::Schedule& sch, const tir::StmtSRef& block_sref);

/*!
 * \brief Checks if a block is potential to rewrite and do tensorize
 * \param sch The meta schedule class
 * \param block_rv The block random variable to be analyzed
 * \param desc_func The description function of TensorIntrin we want to match
 * \return A boolean flag indicating if is able to rewrite and do tensorize
 */
TVM_DLL bool CanTensorizeRewrite(const tir::Schedule& sch, const tir::StmtSRef& block_sref,
                                 const tir::PrimFunc& desc_func);

/*!
 * \brief Rewrite a block to do tensorize in the future
 * \param sch The meta schedule class
 * \param block_rv The block random variable to be analyzed
 * \param desc_func The description function of TensorIntrin we want to match
 */
TVM_DLL void DoTensorizeRewrite(Schedule sch, BlockRV block_rv, tir::PrimFunc desc_func);

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_ANALYSIS_H_
