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

#include "./schedule.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Checks if
 * 1) number of blocks vars equals to number of loop vars
 * 2) each block var is bound to a loop var directly
 * 3) the order is preserved, i.e. the i-th block var is the i-th loop var
 * \param sch The meta schedule class
 * \param block The block random variable to be analyzed
 * \return A boolean indicating if the block binding is trivial
 */
TVM_DLL bool IsTrivialBinding(Schedule sch, BlockRV block);

/*!
 * \brief Returns the IterVarType of each block var
 * \param sch The meta schedule class
 * \param block The block random variable to be analyzed
 * \return An array of integers, the IterVarTypes corresponding to each block var in order
 * \sa tir::IterVarType
 */
TVM_DLL Array<Integer> GetBlockVarTypes(Schedule sch, BlockRV block);

/*!
 * \brief Checks if the specific block is a leaf block
 * \param sch The meta schedule class
 * \param block The block random variable to be analyzed
 * \return A boolean indiciating if the block is a leaf block
 */
TVM_DLL bool IsLeafBlock(Schedule sch, BlockRV block);

/*!
 * \brief Checks if the specific block is a leaf block and its body is a single statement
 * \param sch The meta schedule class
 * \param block The block random variable to be analyzed
 * \return A boolean indiciating if the block is a leaf block and its body is a single statement
 */
TVM_DLL bool IsLeafBlockWithSingleStmt(Schedule sch, BlockRV block);

/*!
 * \brief Get the buffer written in the single statement of a leaf statement
 * \param sch The meta schedule class
 * \param block The block random variable to be analyzed
 * \return A BufferLoad indicating the buffer and its indices to be written
 * \note It is intended to return type BufferLoad, because it has included all necessary info
 */
TVM_DLL tir::BufferLoad GetBufferStore(Schedule sch, BlockRV block);

/*!
 * \brief Get all the buffers read in the single statement of a leaf statement
 * \param sch The meta schedule class
 * \param block The block random variable to be analyzed
 * \return An array of BufferLoad indicating the buffers and their indices to be read
 */
TVM_DLL Array<tir::BufferLoad> GetBufferLoad(Schedule sch, BlockRV block);

/*!
 * \brief Count the number of occurrence of an operator, i.e. tir.exp
 * \param sch The meta schedule class
 * \param block The block random variable to be analyzed
 * \param op The operator to be counted
 * \return An integer indicating the number of its occurrence
 */
TVM_DLL int CountOp(Schedule sch, BlockRV block, Op op);

/*!
 * \brief Check if there is any branch in the given block, which includes
 * 1) block predicate
 * 2) if-then-else statement
 * 3) select expression
 * 4) if-then-else operator
 * \param sch The meta schedule class
 * \param block The block random variable to be analyzed
 * \return A boolean indicating there is at least a branch in the given block
 */
TVM_DLL bool HasBranch(Schedule sch, BlockRV block);

/*!
 * \brief Check if the specifc block satisfies
 * 1) it is a leaf block with a single statement as its body
 * 2) indices in BufferStore are either constants, or block vars +/- constants
 * If condition is satisfied, return an array of block vars
 * that are used in BufferStore indices in the same order as they appears in indices
 * \param sch The meta schedule class
 * \param block The block random variable to be analyzed
 * \return An array of block vars, in the same order as they appears in indices,
 * if the condition is satisfied; NullOpt otherwise
 */
TVM_DLL Optional<Array<tir::Var>> BlockVarsUsedInStore(Schedule sch, BlockRV block);

/*!
 * \brief Count the number of block vars that are not used in the BufferLoad
 * \param load The BufferLoad to be examined
 * \param block_vars The list of block vars
 * \return An integer indicating number of block vars that are not used
 */
TVM_DLL int CountMissingBlockVars(tir::BufferLoad load, Array<tir::Var> block_vars);

/*!
 * \brief Inspect the mapping between indices in all BufferLoads and block vars used in BufferStore.
 * First, call `BlockVarsUsedInStore` to get block vars.
 * Second, for each BufferLoad and its indices, check
 * 1) exists: the mapping from load -> block vars exists
 * 2) surjective: every block var is mapped to at least once
 * 3) injective: every block var is mapped to at most once
 * 4) order: the mapping is kept in order
 * If the mapping doesn't exist, then return NullOpt;
 * Otherwise, return (surjective, injective, order)
 * \param sch The meta schedule class
 * \param block The block random variable to be analyzed
 * \return NullOpt if the mapping doesn't exist, otherwise (surjective, injective, order)
 * \sa BlockVarsUsedInStore
 */
TVM_DLL Optional<Array<Bool>> InspectLoadIndices(Schedule sch, BlockRV block);

TVM_DLL bool HasReduceBlockVar(Schedule sch, BlockRV block);

TVM_DLL bool NeedsMultiLevelTiling(Schedule sch, BlockRV block);

TVM_DLL void DoMultiLevelTiling(Schedule sch, BlockRV block, String tiling_structure);

TVM_DLL bool IsElementWiseMatch(Schedule sch, BlockRV producer, BlockRV consumer);

TVM_DLL bool IsOutputBlock(Schedule sch, BlockRV block);

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_ANALYSIS_H_
