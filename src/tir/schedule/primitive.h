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
#ifndef TVM_TIR_SCHEDULE_PRIMITIVES_PRIMITIVES_H_
#define TVM_TIR_SCHEDULE_PRIMITIVES_PRIMITIVES_H_

#include <tvm/tir/schedule/schedule.h>

#include <vector>

namespace tvm {
namespace tir {

class Sampler;

/******** Sampling ********/

TVM_DLL std::vector<int64_t> SamplePerfectTile(tir::ScheduleState self, Sampler* sampler,
                                               const tir::StmtSRef& loop_sref,  //
                                               int n,                           //
                                               int max_innermost_factor,        //
                                               Optional<Array<Integer>>* decision);

TVM_DLL int64_t SampleCategorical(tir::ScheduleState self, Sampler* sampler,  //
                                  const Array<Integer>& candidates,           //
                                  const Array<FloatImm>& probs,               //
                                  Optional<Integer>* decision);

TVM_DLL tir::StmtSRef SampleComputeLocation(tir::ScheduleState self, Sampler* sampler,  //
                                            const tir::StmtSRef& block_sref,            //
                                            Optional<Integer>* decision);

/******** Schedule: Block/Loop Relationship ********/

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
Array<StmtSRef> GetLoops(const StmtSRef& block_sref);

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

/******** Schedule: loops ********/

/*!
 * \brief Fuse two consecutive loops of one computation.
 * \param outer_sref The outer loop
 * \param inner_sref The inner loop
 * \return The fused loop
 */
TVM_DLL StmtSRef Fuse(ScheduleState self, const StmtSRef& outer_sref, const StmtSRef& inner_sref);

/*!
 * \brief Split a specified loop into two loops by factor.
 * \param loop_sref The loop to be split
 * \param nparts The extent of the new outer loop
 * \param factor The extent of the new inner loop
 * \return The loops after splitting
 */
TVM_DLL Array<StmtSRef> Split(ScheduleState self, const StmtSRef& loop_sref, const PrimExpr& nparts,
                              const PrimExpr& factor);

/*!
 * \brief reorder a list of loops
 * \param order the order of loops
 */
TVM_DLL void Reorder(ScheduleState self, const Array<StmtSRef>& order);

/******** Schedule: compute location ********/

/*!
 * \brief Move the block under the loop and regenerate the loops to cover the producing region.
 * \param block_sref The block to be moved
 * \param loop_sref The target loop
 * \param preserve_unit_loop Keep the trivial loops whose extent is 1
 */
TVM_DLL void ComputeAt(ScheduleState self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
                       bool preserve_unit_loop);

/*!
 * \brief Move the block under the loop and regenerate the loops to cover the producing region.
 * \param block_sref The block to be moved
 * \param loop_sref The target loop
 * \param preserve_unit_loop Keep the trivial loops whose extent is 1
 */
TVM_DLL void ReverseComputeAt(ScheduleState self, const StmtSRef& block_sref,
                              const StmtSRef& loop_sref, bool preserve_unit_loop);

/*!
 * \brief Make the block inline
 * \param block_sref The sref of the block
 */
TVM_DLL void ComputeInline(ScheduleState self, const StmtSRef& block_sref);

/*!
 * \brief Make the block inline
 * \param block_sref The sref of block
 */
TVM_DLL void ReverseComputeInline(ScheduleState self, const StmtSRef& block_sref);

/******** Schedule: parallelize / annotate ********/

/*!
 * \brief vectorize a loop
 * \param loop_sref the loop to be vectorized
 */
TVM_DLL void Vectorize(ScheduleState self, const StmtSRef& loop_sref);

/*!
 * \brief parallelize a loop
 * \param loop_sref the loop to be paralleled
 */
TVM_DLL void Parallel(ScheduleState self, const StmtSRef& loop_sref);

/*!
 * \brief unroll a loop
 * \param loop_sref the loop to be unrolled
 */
TVM_DLL void Unroll(ScheduleState self, const StmtSRef& loop_sref);

/*!
 * \brief parallel a loop
 * \param loop_sref the loop to be paralleled
 */
TVM_DLL void Bind(ScheduleState self, const StmtSRef& loop_sref, const IterVar& thread);

/*!
 * \brief add double_buffer annotation to a complete block
 * \param block_sref the block of interest
 */
TVM_DLL void DoubleBuffer(ScheduleState self, const StmtSRef& block_sref);

/*!
 * \brief Set the storage scope of a buffer, which is the i-th write buffer of the given block
 * \param block_sref The producer of the buffer
 * \param i The index of the buffer in block's write region
 * \param storage_scope The storage scope to be set
 */
TVM_DLL void SetScope(ScheduleState self, const StmtSRef& block_sref, int i,
                      const String& storage_scope);

/*!
 * \brief Set alignment requirement for specific dimension such that
 *        stride[axis] == k * factor + offset for some k
 * \param block_sref The producer block of the buffer
 * \param buffer_index The index of the buffer in block's write region
 * \param axis The dimension to be specified for alignment
 * \param factor The factor multiple of alignment
 * \param offset The required offset factor
 */
TVM_DLL void StorageAlign(ScheduleState self, const StmtSRef& block_sref, int buffer_index,
                          int axis, int factor, int offset);

/******** Schedule: annotate ********/

/*!
 * \brief add annotation to a loop
 * \param loop_sref the loop of interest
 * \param pragma_type the attribute key
 * \param pragma_value the attribute value
 */
TVM_DLL void Pragma(ScheduleState self, const StmtSRef& loop_sref, const String& pragma_type,
                    const PrimExpr& pragma_value);

TVM_DLL void MarkLoop(ScheduleState self, const StmtSRef& loop_sref, const String& ann_key,
                      const PrimExpr& ann_val);

TVM_DLL void MarkBlock(ScheduleState self, const StmtSRef& block_sref, const String& ann_key,
                       const PrimExpr& ann_val);

/******** Schedule: cache read/write ********/

/*!
 * \brief Create a cache read of original tensor for readers.
 * \param block_sref The consumer of the buffer
 * \param i The index of the buffer in block's read region
 * \param storage_scope The storage scope
 */
TVM_DLL StmtSRef CacheRead(ScheduleState self, const StmtSRef& block_sref, int i,
                           const String& storage_scope);

/*!
 * \brief Create a cache write of original tensor, before storing into tensor.
 * \param block_sref The producer of the buffer
 * \param i The index of the buffer in block's write region
 * \param storage_scope The storage scope
 */
TVM_DLL StmtSRef CacheWrite(ScheduleState self, const StmtSRef& block_sref, int i,
                            const String& storage_scope);

/******** Schedule: reduction ********/

/*!
 * \brief Factor a reduction block by the specified loop
 * \param loop_sref The loop outside block we want to do rfactor
 * \param factor_axis The position where the new dimension is placed in the new generated rfactor
 *                    buffer
 * \return The sref of the rfactor block
 */
TVM_DLL StmtSRef RFactor(ScheduleState self, const StmtSRef& loop_sref, int factor_axis);

/*!
 * \brief Decompose reduction block_sref into init&update blocks
 * \param block_sref the reduction block_sref
 * \param loop_sref the position where init block_sref will be
 * \return the sref of init block
 */
TVM_DLL StmtSRef DecomposeReduction(ScheduleState self, const StmtSRef& block_sref,
                                    const Optional<StmtSRef>& loop_sref);

/*!
 * \brief Merge init and reduction block into reduction block
 * \param init_sref the init block
 * \param update_sref the update block
 */
TVM_DLL void MergeReduction(ScheduleState self, const StmtSRef& init_sref,
                            const StmtSRef& update_sref);

/******** Blockize / Tensorize ********/

/*!
 * \brief make subtree rooted by loop_sref into a block
 * \param loop_sref the subtree root
 * \return the loop_sref of new block
 */
TVM_DLL StmtSRef Blockize(ScheduleState self, const StmtSRef& loop_sref);

/*!
 * \brief Tensorize the computation enclosed by loop with tensor_intrin
 * \param loop_sref the loop/block to be tensorized
 * \param intrinsic the tensor intrinsic
 */
TVM_DLL void Tensorize(ScheduleState self, const StmtSRef& loop_sref,
                       const TensorIntrin& intrinsic);

/******** Misc ********/

TVM_DLL void InlineArgument(ScheduleState self, int i, const String& func_name);

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_PRIMITIVES_PRIMITIVES_H_
