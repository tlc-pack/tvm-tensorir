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
#ifndef TVM_TIR_SCHEDULE_TRACED_SCHEDULE_H_
#define TVM_TIR_SCHEDULE_TRACED_SCHEDULE_H_

#include "./concrete_schedule.h"

namespace tvm {
namespace tir {

class TracedScheduleNode : public ConcreteScheduleNode {
  friend class Schedule;

 protected:
  Trace trace_;

 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
    // `state_` is not visited
    // `error_render_level_` is not visited
    // `rand_state_` is not visited
    // `symbol_table_` is not visited
    // `analyzer_` is not visitied
    // `trace_` is not visited
  }

  ~TracedScheduleNode() = default;

  static constexpr const char* _type_key = "tir.TracedSchedule";
  TVM_DECLARE_FINAL_OBJECT_INFO(TracedScheduleNode, ScheduleNode);

 public:
  Optional<Trace> trace() const final { return trace_; }
  Schedule Copy(tir::TRandState new_seed = -1) const final;

 public:
  /******** Schedule: Sampling ********/
  Array<ExprRV> SamplePerfectTile(const LoopRV& loop_rv, int n, int max_innermost_factor,
                                  Optional<Array<Integer>> decision = NullOpt) final;
  Array<Array<ExprRV>> SampleShapeGenericTiles(
      const Array<LoopRV>& loop_rvs, const std::vector<int>& ns, const Target& target,
      int max_innermost_factor, Optional<Array<Array<Integer>>> decision = NullOpt) final;
  ExprRV SampleCategorical(const Array<Integer>& candidates, const Array<FloatImm>& probs,
                           Optional<Integer> decision = NullOpt) final;
  LoopRV SampleComputeLocation(const BlockRV& block_rv, Optional<Integer> decision = NullOpt) final;

  /******** Schedule: Get blocks & loops ********/

  BlockRV GetBlock(const String& name, const String& func_name = "main") final;
  Array<LoopRV> GetLoops(const BlockRV& block_rv) final;
  Array<BlockRV> GetChildBlocks(const BlockRV& block_rv) final;
  Array<BlockRV> GetChildBlocks(const LoopRV& loop_rv) final;
  Array<BlockRV> GetProducers(const BlockRV& block_rv) final;
  Array<BlockRV> GetConsumers(const BlockRV& block_rv) final;

  /******** Schedule: Transform loops ********/

  LoopRV Fuse(const Array<LoopRV>& loop_rvs) final;
  Array<LoopRV> Split(const LoopRV& loop_rv, const Array<Optional<ExprRV>>& factor_rvs) final;
  void Reorder(const Array<LoopRV>& order) final;

  /******** Schedule: Manipulate ForKind ********/

  void Parallel(const LoopRV& loop_rv) final;
  void Vectorize(const LoopRV& loop_rv) final;
  void Unroll(const LoopRV& loop_rv) final;
  void Bind(const LoopRV& loop_rv, const IterVar& thread) final {
    LOG(FATAL) << "NotImplementedError: Bind with an IterVar is not supported";
    throw;
  }
  void Bind(const LoopRV& loop_rv, const String& thread) final;

  /******** Schedule: Insert cache stages ********/

  BlockRV CacheRead(const BlockRV& block_rv, int i, const String& storage_scope) final;
  BlockRV CacheWrite(const BlockRV& block_rv, int i, const String& storage_scope) final;

  /******** Schedule: Compute location ********/

  void ComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv, bool preserve_unit_loop) final;
  void ReverseComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                        bool preserve_unit_loop) final;
  void ComputeInline(const BlockRV& block_rv) final;
  void ReverseComputeInline(const BlockRV& block_rv) final;

  /******** Schedule: Reduction ********/

  BlockRV RFactor(const LoopRV& loop_rv, int factor_axis) final;
  BlockRV DecomposeReduction(const BlockRV& block_rv, const Optional<LoopRV>& loop_rv) final;
  void MergeReduction(const BlockRV& init_block_rv, const BlockRV& update_block_rv) final;

  /******** Schedule: Blockize & Tensorize ********/

  BlockRV Blockize(const LoopRV& loop_rv) final;
  void Tensorize(const LoopRV& loop_rv, const TensorIntrin& intrin) final {
    LOG(FATAL) << "NotImplementedError: Tensorize with a tensor intrinsic is not supported, please "
                  "use register the intrinsic and use its name instead";
    throw;
  }
  void Tensorize(const LoopRV& loop_rv, const String& intrin_name) final;

  /******** Schedule: Annotation ********/

  void MarkLoop(const LoopRV& loop_rv, const String& ann_key, const ObjectRef& ann_val) final;
  void MarkBlock(const BlockRV& block_rv, const String& ann_key, const ObjectRef& ann_val) final;
  void Pragma(const LoopRV& loop_rv, const String& pragma_type, const ExprRV& pragma_value) final;

  /******** Schedule: Misc ********/

  void EnterPostproc() final;
  void DoubleBuffer(const BlockRV& block_rv) final;
  void SetScope(const BlockRV& block_rv, int i, const String& storage_scope) final;
  void StorageAlign(const BlockRV& block_rv, int buffer_index, int axis, int factor,
                    int offset) final;
  void InlineArgument(int i, const String& func_name) final;
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_TRACED_SCHEDULE_H_
