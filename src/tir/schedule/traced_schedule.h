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
    // `symbol_table_` is not visited
    // `analyzer_` is not visitied
    // `trace_` is not visited
  }

  ~TracedScheduleNode() = default;

 public:
  Optional<Trace> trace() const final { return trace_; }
  Schedule Copy() const final;

 public:
  /******** Schedule: Sampling ********/
  /*!
   * \brief Sample an integer given the probability distribution
   * \param candidates The candidates
   * \param probs The probability distribution of the candidates
   * \param decision The sampling decision
   * \return The random variable sampled from candidates
   */
  ExprRV SampleCategorical(const Array<Integer>& candidates, const Array<FloatImm>& probs,
                           Optional<Integer> decision = NullOpt) final;

  /******** Schedule: Get blocks & loops ********/
  BlockRV GetBlock(const String& name, const String& func_name = "main") final;
  Array<LoopRV> GetLoops(const BlockRV& block_rv) final;
  /******** Schedule: Transform loops ********/
  LoopRV Fuse(const Array<LoopRV>& loop_rvs) final;
  Array<LoopRV> Split(const LoopRV& loop_rv, const Array<Optional<ExprRV>>& factor_rvs) final;
  /******** Schedule: Manipulate ForKind ********/
  void Parallel(const LoopRV& loop_rv) final;
  void Vectorize(const LoopRV& loop_rv) final;
  void Bind(const LoopRV& loop_rv, const String& thread_axis) final;
  void Unroll(const LoopRV& loop_rv) final;
  /******** Schedule: Insert cache stages ********/
  BlockRV CacheRead(const BlockRV& block_rv, int read_buffer_index,
                    const String& storage_scope) final;
  BlockRV CacheWrite(const BlockRV& block_rv, int write_buffer_index,
                     const String& storage_scope) final;
  /******** Schedule: Compute location ********/
  void ComputeInline(const BlockRV& block_rv) final;
  void ReverseComputeInline(const BlockRV& block_rv) final;
  /******** Schedule: Reduction ********/
  BlockRV RFactor(const LoopRV& loop_rv, int factor_axis) final;
  /******** Schedule: Block annotation ********/
  void StorageAlign(const BlockRV& block_rv, int buffer_index, int axis, int factor,
                    int offset) final;
  /******** Schedule: Blockize & Tensorize ********/
  /******** Schedule: Annotation ********/
  /******** Schedule: Misc ********/
  void EnterPostproc() final;
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_TRACED_SCHEDULE_H_
