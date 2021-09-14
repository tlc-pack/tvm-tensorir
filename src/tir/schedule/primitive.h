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

#include <tvm/tir/schedule/state.h>

#include <vector>

namespace tvm {
namespace tir {

class Sampler;

/******** Schedule: Sampling ********/

TVM_DLL std::vector<int64_t> SamplePerfectTile(tir::ScheduleState self, Sampler* sampler,
                                               const tir::StmtSRef& loop_sref, int n,
                                               int max_innermost_factor,
                                               Optional<Array<Integer>>* decision);
TVM_DLL int64_t SampleCategorical(tir::ScheduleState self, Sampler* sampler,
                                  const Array<Integer>& candidates, const Array<FloatImm>& probs,
                                  Optional<Integer>* decision);
TVM_DLL tir::StmtSRef SampleComputeLocation(tir::ScheduleState self, Sampler* sampler,
                                            const tir::StmtSRef& block_sref,
                                            Optional<Integer>* decision);

/******** Schedule: Get blocks & loops ********/

TVM_DLL Array<StmtSRef> GetBlocks(const ScheduleState& self, const String& name,
                                  const String& func_name = "main");
TVM_DLL Array<StmtSRef> GetLoops(const StmtSRef& block_sref);
TVM_DLL Array<StmtSRef> GetChildBlocks(const ScheduleState& self, const StmtSRef& parent_sref,
                                       bool inclusive = false);
TVM_DLL Array<StmtSRef> GetProducers(const ScheduleState& self, const StmtSRef& block_sref);
TVM_DLL Array<StmtSRef> GetConsumers(const ScheduleState& self, const StmtSRef& block_sref);

/******** Schedule: Transform loops ********/

/*!
 * Split a loop into a list of consecutive loops. It requires:
 * 1) The loop can't have annotation or thread binding.
 * 2) The loop must start with 0.
 * \param self The state of the schedule
 * \param loop_sref The sref to the loop being split
 * \param factors The splitting factors
 * \return An array of srefs to the loops after splitting
 */
TVM_DLL Array<StmtSRef> Split(ScheduleState self, const StmtSRef& loop_sref,
                              const Array<PrimExpr>& factors);
/*!
 * \brief Fuse a list of consecutive loops into one. It requires:
 * 1) The loops can't have annotations or thread bindings.
 * 2) The inner loop must be the only child of the outer loop.
 * 3) All loops must start with 0.
 * \param self The state of the schedule
 * \param loop_srefs An array of srefs to the loops to be fused
 * \return The sref to the fused loop
 */
TVM_DLL StmtSRef Fuse(ScheduleState self, const Array<StmtSRef>& loop_srefs);
TVM_DLL void Reorder(ScheduleState self, const Array<StmtSRef>& order);

/******** Schedule: Manipulate ForKind ********/

TVM_DLL void Parallel(ScheduleState self, const StmtSRef& loop_sref);
TVM_DLL void Vectorize(ScheduleState self, const StmtSRef& loop_sref);
TVM_DLL void Unroll(ScheduleState self, const StmtSRef& loop_sref);
TVM_DLL void Bind(ScheduleState self, const StmtSRef& loop_sref, const IterVar& thread);

/******** Schedule: Insert cache stages ********/

TVM_DLL StmtSRef CacheRead(ScheduleState self, const StmtSRef& block_sref, int i,
                           const String& storage_scope);
TVM_DLL StmtSRef CacheWrite(ScheduleState self, const StmtSRef& block_sref, int i,
                            const String& storage_scope);
/******** Schedule: Compute location ********/

TVM_DLL void ComputeAt(ScheduleState self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
                       bool preserve_unit_loop);
TVM_DLL void ReverseComputeAt(ScheduleState self, const StmtSRef& block_sref,
                              const StmtSRef& loop_sref, bool preserve_unit_loop);

/*!
 * \brief Inline a block into its consumer(s). It requires:
 * 1) The block is a complete non-root block, which only produces one buffer
 * 2) The block must not be the only leaf in the scope.
 * 3) The body of the block must be a BufferStore statement in the form of,
 *    A[i, j, k, ...] = ...
 * where the indices of the LHS are all distinct atomic variables,
 * and no variables other than those indexing variables are allowed in the statement.
 * \param self The state of the schedule
 * \param block_sref The sref to the block to be inlined to its consumer(s)
 */
TVM_DLL void ComputeInline(ScheduleState self, const StmtSRef& block_sref);
/*!
 * \brief Inline a block into its only producer. It requires:
 * 1) The block is a complete non-root block, which only produces and consumers one buffer
 * 2) The block must not be the only leaf in the scope.
 * 3) The only producer of the block is a read-after-write producer and a complete non-root block
 * 4) The body of the block must be a BufferStore statement in the form of,
 *    B[f(i, j, k, ...)] = g(i, j, k, A[i, j, k, ...] ...)
 * where the indices of each `BufferLoad` on the RHS are all distinct atomic variables,
 * and no variables other than those indexing variables are allowed in the statement.
 * \param self The state of the schedule
 * \param block_sref The sref to the block to be inlined to its producer
 */
TVM_DLL void ReverseComputeInline(ScheduleState self, const StmtSRef& block_sref);

/******** Schedule: Reduction ********/

TVM_DLL StmtSRef RFactor(ScheduleState self, const StmtSRef& loop_sref, int factor_axis);
TVM_DLL StmtSRef DecomposeReduction(ScheduleState self, const StmtSRef& block_sref,
                                    const Optional<StmtSRef>& loop_sref);
TVM_DLL void MergeReduction(ScheduleState self, const StmtSRef& init_sref,
                            const StmtSRef& update_sref);

/******** Schedule: Blockize & Tensorize ********/

TVM_DLL StmtSRef Blockize(ScheduleState self, const StmtSRef& loop_sref);
TVM_DLL void Tensorize(ScheduleState self, const StmtSRef& loop_sref,
                       const TensorIntrin& intrinsic);

/******** Schedule: Annotation ********/

TVM_DLL void MarkLoop(ScheduleState self, const StmtSRef& loop_sref, const String& ann_key,
                      const PrimExpr& ann_val);
TVM_DLL void MarkBlock(ScheduleState self, const StmtSRef& block_sref, const String& ann_key,
                       const PrimExpr& ann_val);
TVM_DLL void Pragma(ScheduleState self, const StmtSRef& loop_sref, const String& pragma_type,
                    const PrimExpr& pragma_value);

/******** Schedule: Misc ********/

TVM_DLL void DoubleBuffer(ScheduleState self, const StmtSRef& block_sref);
TVM_DLL void SetScope(ScheduleState self, const StmtSRef& block_sref, int i,
                      const String& storage_scope);
TVM_DLL void StorageAlign(ScheduleState self, const StmtSRef& block_sref, int buffer_index,
                          int axis, int factor, int offset);
TVM_DLL void InlineArgument(ScheduleState self, int i, const String& func_name);
TVM_DLL void SoftwarePipeline(ScheduleState self, const StmtSRef& loop_sref, int num_stages);

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_PRIMITIVES_PRIMITIVES_H_
