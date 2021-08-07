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
#ifndef SRC_META_SCHEDULE_SCHEDULE_H_
#define SRC_META_SCHEDULE_SCHEDULE_H_

#include <tvm/tir/schedule/schedule.h>

#include "../tir/schedule/concrete_schedule.h"
#include "./instruction.h"
#include "./sampler.h"
#include "./trace.h"

namespace tvm {
namespace meta_schedule {

class Schedule;

/*! \brief The meta schedule class */
class ScheduleNode : public tir::ConcreteScheduleNode {
 private:
  friend class tir::Schedule;
  using tir::ConcreteScheduleNode::Copy;

 protected:
  friend class Schedule;
  using TSymbolTable = tir::ConcreteScheduleNode::TSymbolTable;

  /*! \brief The schedule state */
  using tir::ConcreteScheduleNode::state_;
  /*! \brief The symbol table */
  using tir::ConcreteScheduleNode::symbol_table_;

 public:
  /*! \brief The trace of the program execution */
  Trace trace;
  /*! \brief The random number sampler state */
  int64_t random_state;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `state_` is not visited
    // `symbol_table_` is not visited
    v->Visit("trace", &trace);
    // `sampler` is not visited
  }

  static constexpr const char* _type_key = "meta_schedule.Schedule";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleNode, tir::ConcreteScheduleNode);

 public:
  /**************** Utility ****************/
  /*!
   * \brief Copy the schedule into a new one. Operation on the new schedule won't affect the
   * original schedule, and vice versa.
   * \return A new schedule.
   */
  Schedule Copy(int64_t new_seed) const;

  void Seed(int64_t seed = -1) final;

  /**************** Sampling ****************/
  /*!
   * \brief Apply the instruction SamplePerfectTile
   * \param n The number of loops after tiling
   * \param loop_rv The loop to be tiled
   * \param max_innermost_factor The maximum factor in the innermost loop, -1 if disabled
   * \return An array of random variables, the result of sampling
   */
  Array<tir::Var> SamplePerfectTile(const LoopRV& loop_rv,     //
                                    int n,                     //
                                    int max_innermost_factor,  //
                                    Optional<Array<Integer>> decision = NullOpt) final;
  /*!
   * \brief Sample an integer given the probability distribution
   * \param candidates The candidates
   * \param probs The probability distribution of the candidates
   * \return The random variable
   */
  tir::Var SampleCategorical(const Array<Integer>& candidates,  //
                             const Array<FloatImm>& probs,      //
                             Optional<Integer> decision = NullOpt) final;
  /*!
   * \brief Sample a compute-at location from a block
   * \param block_rv A block to be computed at
   * \return The loop to be computed at
   */
  LoopRV SampleComputeLocation(const BlockRV& block_rv, Optional<Integer> decision = NullOpt) final;
  /**************** Block/Loop Relationship ****************/
  /*!
   * \brief Apply the instruction GetBlock
   * \param name The name of the block to get retrieved
   * \return A block random variable, the return value of the instruction
   */
  BlockRV GetBlock(const String& name) final;
  /*!
   * \brief Apply the instruction GetAxes
   * \param block_rv The block used to retrieve the axes
   * \return An array of loop random variables
   */
  Array<LoopRV> GetAxes(const BlockRV& block_rv) final;
  /*!
   * \brief Get the child blocks of a specific parent block/loop
   * \param block_rv The random variable that points to the parent block
   * \return A list of child blocks
   */
  Array<BlockRV> GetChildBlocks(const BlockRV& block_rv) final;
  /*!
   * \brief Get the child blocks of a specific parent block/loop
   * \param loop_rv The random variable that points to the parent loop
   * \return A list of child blocks
   */
  Array<BlockRV> GetChildBlocks(const LoopRV& loop_rv) final;
  /*!
   * \brief Get the producer of a specific block
   * \return The producers
   */
  Array<BlockRV> GetProducers(const BlockRV& block_rv) final;
  /*!
   * \brief Get the consumers of a specific block
   * \return The consumers
   */
  Array<BlockRV> GetConsumers(const BlockRV& block_rv) final;

  /******** Schedule: loops ********/
  /*!
   * \brief Fuse two consecutive loops of one computation.
   * \param outer_loop_rv The outer loop
   * \param inner_loop_rv The inner loop
   * \return The fused loop
   */
  LoopRV Fuse(const Array<LoopRV>& loop_rvs) final;

  /*!
   * \brief Split a specified loop into two loops by factor.
   * \param loop_rv The loop to be split
   * \param nparts The extent of the new outer loop
   * \param factor The extent of the new inner loop
   * \return The loops after splitting
   */
  Array<LoopRV> Split(const LoopRV& loop_rv, const Array<Optional<PrimExpr>>& factors) final;

  /*!
   * \brief reorder a list of loops
   * \param order the order of loops
   */
  void Reorder(const Array<LoopRV>& order) final;

  /******** Schedule: compute location ********/
  /*!
   * \brief Move the block under the loop and regenerate the loops to cover the producing region.
   * \param block_rv The block to be moved
   * \param loop_rv The target loop
   * \param preserve_unit_loop Keep the trivial loops whose extent is 1
   */
  void ComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv, bool preserve_unit_loop) final;
  /*!
   * \brief Move the block under the loop and regenerate the loops to cover the producing region.
   * \param block_rv The block to be moved
   * \param loop_rv The target loop
   * \param preserve_unit_loop Keep the trivial loops whose extent is 1
   */
  void ReverseComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                        bool preserve_unit_loop) final;
  /*!
   * \brief Make the block inline
   * \param block_rv The block
   */
  void ComputeInline(const BlockRV& block_rv) final;
  /*!
   * \brief Make the block inline
   * \param block_rv The block
   */
  void ReverseComputeInline(const BlockRV& block_rv) final;

  /******** Schedule: parallelize / annotate ********/
  /*!
   * \brief vectorize a loop
   * \param loop_rv the loop to be vectorized
   */
  void Vectorize(const LoopRV& loop_rv) final;
  /*!
   * \brief parallelize a loop
   * \param loop_rv the loop to be paralleled
   */
  void Parallel(const LoopRV& loop_rv) final;
  /*!
   * \brief unroll a loop
   * \param loop_rv the loop to be unrolled
   */
  void Unroll(const LoopRV& loop_rv) final;
  /*!
   * \brief bind a loop to a thread axis
   * \param loop_rv the loop to be paralleled
   * \param thread The thread axis
   */
  void Bind(const LoopRV& loop_rv, const tir::IterVar& thread) final;
  /*!
   * \brief parallel a loop
   * \param loop_rv the loop to be paralleled
   */
  void Bind(const LoopRV& loop_rv, const String& thread) final;
  /*!
   * \brief add double_buffer annotation to a complete block
   * \param block_rv the block of interest
   */
  void DoubleBuffer(const BlockRV& block_rv) final;
  /*!
   * \brief Set the storage scope of a buffer, where the buffer is given as the i-th write buffer
   *        of the input block
   * \param block_rv The producer of the buffer
   * \param i The index of the buffer in block's write region
   * \param storage_scope The storage scope to be set
   */
  void SetScope(const BlockRV& block_rv, int i, const String& storage_scope) final;
  /*!
   * \brief add annotation to a loop
   * \param loop_rv the loop of interest
   * \param pragma_type the attribute key
   * \param pragma_value the attribute value
   */
  void Pragma(const LoopRV& loop_rv, const String& pragma_type, const ExprRV& pragma_value) final;
  /*!
   * \brief Set alignment requirement for specific dimension such that
   *        stride[axis] == k * factor + offset for some k
   * \param block_rv The producer block of the buffer
   * \param buffer_index The index of the buffer in block's write region
   * \param axis The dimension to be specified for alignment
   * \param factor The factor multiple of alignment
   * \param offset The required offset factor
   */
  void StorageAlign(const BlockRV& block_rv, int buffer_index, int axis, int factor,
                    int offset) final;

  /******** Schedule: cache read/write ********/
  /*!
   * \brief Create a cache read of original tensor for readers.
   * \param block_rv The consumer of the buffer
   * \param i The index of the buffer in block's read region
   * \param storage_scope The storage scope
   */
  BlockRV CacheRead(const BlockRV& block_rv, int i, const String& storage_scope) final;
  /*!
   * \brief Create a cache write of original tensor, before storing into tensor.
   * \param block_rv The producer of the buffer
   * \param i The index of the buffer in block's write region
   * \param storage_scope The storage scope
   */
  BlockRV CacheWrite(const BlockRV& block_rv, int i, const String& storage_scope) final;

  /******** Schedule: reduction ********/
  /*!
   * \brief Factor a reduction block by the specified loop
   * \param loop_rv The loop outside block we want to do rfactor
   * \param factor_axis The position where the new dimension is placed in the new generated rfactor
   *                    buffer
   * \return The rfactor block
   */
  BlockRV RFactor(const LoopRV& loop_rv, int factor_axis) final;
  /*!
   * \brief Decompose reduction block_rv into init&update blocks
   * \param block_rv the reduction block_rv
   * \param loop_rv the position where init block_rv will be
   * \return The init block
   */
  BlockRV DecomposeReduction(const BlockRV& block_rv, const Optional<LoopRV>& loop_rv) final;
  /*!
   * \brief Merge init and reduction block into reduction block
   * \param init_block_rv the init block
   * \param update_block_rv the update block
   */
  void MergeReduction(const BlockRV& init_block_rv, const BlockRV& update_block_rv) final;

  /******** Schedule: Blockize / Tensorize ********/
  /*!
   * \brief make subtree rooted by loop_rv into a block
   * \param loop_rv the subtree root
   * \return the loop_rv of new block
   */
  BlockRV Blockize(const LoopRV& loop_rv) final;
  /*!
   * \brief Tensorize the computation enclosed by loop with tensor_intrin
   * \param loop_rv the loop/block to be tensorized
   * \param intrin the tensor intrinsic
   */
  void Tensorize(const LoopRV& loop_rv, const tir::TensorIntrin& intrin) final;
  /*!
   * \brief Tensorize the computation enclosed by loop with tensor_intrin
   * \param loop_rv The loop/block to be tensorized
   * \param intrin_name Name of the tensor intrinsic
   */
  void Tensorize(const LoopRV& loop_rv, const String& intrin_name) final;

  /******** Schedule: Marks and NO-OPs ********/
  /*!
   * \brief Mark a loop
   * \param loop The loop to be marked
   * \param ann_key The annotation key
   * \param ann_val The annotation value
   */
  void MarkLoop(const LoopRV& loop_rv, const String& ann_key, const PrimExpr& ann_val);
  /*!
   * \brief Mark a block
   * \param block The block to be marked
   * \param ann_key The annotation key
   * \param ann_val The annotation value
   */
  void MarkBlock(const BlockRV& block_rv, const String& ann_key, const PrimExpr& ann_val);
  /*! \brief An NOP indicating entrance of post processing */
  void EnterPostProc();

  /******** Schedule: Misc ********/
  void InlineArgument(int i, const String& func_name) override;
};

class Schedule : public tir::Schedule {
 public:
  using TSymbolTable = ScheduleNode::TSymbolTable;
  explicit Schedule(tir::PrimFunc func, int64_t seed = -1, int debug_mode = false);
  explicit Schedule(IRModule mod, int64_t seed = -1, int debug_mode = false);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Schedule, tir::Schedule, ScheduleNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SCHEDULE_H_
