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
#include "./traced_schedule.h"

namespace tvm {
namespace tir {

Schedule Schedule::Traced(IRModule mod, int64_t seed, int debug_mode,
                          ScheduleErrorRenderLevel error_render_level) {
  ObjectPtr<TracedScheduleNode> n = make_object<TracedScheduleNode>();
  n->state_ = ScheduleState(mod, debug_mode);
  n->error_render_level_ = error_render_level;
  n->sampler_.Seed(seed);
  n->symbol_table_ = {};
  n->analyzer_ = std::make_unique<arith::Analyzer>();
  return Schedule(std::move(n));
}

Schedule TracedScheduleNode::Copy(int64_t new_seed) const {
  throw;  //
}

/******** Schedule: Sampling ********/

Array<ExprRV> TracedScheduleNode::SamplePerfectTile(const LoopRV& loop_rv, int n,
                                                    int max_innermost_factor,
                                                    Optional<Array<Integer>> decision) {
  throw;  //
}

ExprRV TracedScheduleNode::SampleCategorical(const Array<Integer>& candidates,
                                             const Array<FloatImm>& probs,
                                             Optional<Integer> decision) {
  throw;  //
}

LoopRV TracedScheduleNode::SampleComputeLocation(const BlockRV& block_rv,
                                                 Optional<Integer> decision) {
  throw;  //
}

/******** Schedule: Get blocks & loops ********/

BlockRV TracedScheduleNode::GetBlock(const String& name, const String& func_name) {
  throw;  //
}

Array<LoopRV> TracedScheduleNode::GetLoops(const BlockRV& block_rv) {
  throw;  //
}

Array<BlockRV> TracedScheduleNode::GetChildBlocks(const BlockRV& block_rv) {
  throw;  //
}

Array<BlockRV> TracedScheduleNode::GetChildBlocks(const LoopRV& loop_rv) {
  throw;  //
}

Array<BlockRV> TracedScheduleNode::GetProducers(const BlockRV& block_rv) {
  throw;  //
}

Array<BlockRV> TracedScheduleNode::GetConsumers(const BlockRV& block_rv) {
  throw;  //
}

/******** Schedule: Transform loops ********/

LoopRV TracedScheduleNode::Fuse(const Array<LoopRV>& loop_rvs) {
  throw;  //
}

Array<LoopRV> TracedScheduleNode::Split(const LoopRV& loop_rv,
                                        const Array<Optional<ExprRV>>& factor_rvs) {
  throw;  //
}

void TracedScheduleNode::Reorder(const Array<LoopRV>& order) {
  throw;  //
}

/******** Schedule: Manipulate ForKind ********/

void TracedScheduleNode::Parallel(const LoopRV& loop_rv) {
  throw;  //
}

void TracedScheduleNode::Vectorize(const LoopRV& loop_rv) {
  throw;  //
}

void TracedScheduleNode::Unroll(const LoopRV& loop_rv) {
  throw;  //
}

void TracedScheduleNode::Bind(const LoopRV& loop_rv, const String& thread) {
  throw;  //
}

/******** Schedule: Insert cache stages ********/

BlockRV TracedScheduleNode::CacheRead(const BlockRV& block_rv, int i, const String& storage_scope) {
  throw;  //
}

BlockRV TracedScheduleNode::CacheWrite(const BlockRV& block_rv, int i,
                                       const String& storage_scope) {
  throw;  //
}

/******** Schedule: Compute location ********/

void TracedScheduleNode::ComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                                   bool preserve_unit_loop) {
  throw;  //
}

void TracedScheduleNode::ReverseComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                                          bool preserve_unit_loop) {
  throw;  //
}

void TracedScheduleNode::ComputeInline(const BlockRV& block_rv) {
  throw;  //
}

void TracedScheduleNode::ReverseComputeInline(const BlockRV& block_rv) {
  throw;  //
}

/******** Schedule: Reduction ********/

BlockRV TracedScheduleNode::RFactor(const LoopRV& loop_rv, int factor_axis) {
  throw;  //
}

BlockRV TracedScheduleNode::DecomposeReduction(const BlockRV& block_rv,
                                               const Optional<LoopRV>& loop_rv) {
  throw;  //
}

void TracedScheduleNode::MergeReduction(const BlockRV& init_block_rv,
                                        const BlockRV& update_block_rv) {
  throw;  //
}

/******** Schedule: Blockize & Tensorize ********/

BlockRV TracedScheduleNode::Blockize(const LoopRV& loop_rv) {
  throw;  //
}

void TracedScheduleNode::Tensorize(const LoopRV& loop_rv, const String& intrin_name) {
  throw;  //
}

/******** Schedule: Annotation ********/

void TracedScheduleNode::MarkLoop(const LoopRV& loop_rv, const String& ann_key,
                                  const PrimExpr& ann_val) {
  throw;  //
}

void TracedScheduleNode::MarkBlock(const BlockRV& block_rv, const String& ann_key,
                                   const PrimExpr& ann_val) {
  throw;  //
}

void TracedScheduleNode::Pragma(const LoopRV& loop_rv, const String& pragma_type,
                                const ExprRV& pragma_value) {
  throw;  //
}

/******** Schedule: Misc ********/

void TracedScheduleNode::EnterPostProc() {
  throw;  //
}

void TracedScheduleNode::DoubleBuffer(const BlockRV& block_rv) {
  throw;  //
}

void TracedScheduleNode::SetScope(const BlockRV& block_rv, int i, const String& storage_scope) {
  throw;  //
}

void TracedScheduleNode::StorageAlign(const BlockRV& block_rv, int buffer_index, int axis,
                                      int factor, int offset) {
  throw;  //
}

void TracedScheduleNode::InlineArgument(int i, const String& func_name) {
  throw;  //
}

}  // namespace tir
}  // namespace tvm
