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
#ifndef TVM_TIR_SCHEDULE_SCHEDULE_H_
#define TVM_TIR_SCHEDULE_SCHEDULE_H_

#include <tvm/ir/module.h>
#include <tvm/tir/schedule/state.h>

namespace tvm {
namespace tir {

/**************** Random variable: ExprRV ****************/

using ExprRV = PrimExpr;

/**************** Random variable: BlockRV ****************/

/*! \brief A random variable that evaluates to a TIR block */
class BlockRVNode : public runtime::Object {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}
  static constexpr const char* _type_key = "tir.BlockRV";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockRVNode, Object);
};

/*!
 * \brief Managed reference to BlockRVNode
 * \sa BlockRVNode
 */
class BlockRV : public runtime::ObjectRef {
 public:
  /*! \brief Constructor */
  TVM_DLL BlockRV();
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BlockRV, ObjectRef, BlockRVNode);
};

/**************** Random variable: LoopRV ****************/

/*! \brief A random variable that evaluates to a TIR loop axis */
class LoopRVNode : public runtime::Object {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "tir.LoopRV";
  TVM_DECLARE_FINAL_OBJECT_INFO(LoopRVNode, Object);
};

/*!
 * \brief Managed reference to LoopRVNode
 * \sa LoopRVNode
 */
class LoopRV : public runtime::ObjectRef {
 public:
  /*! \brief Constructor */
  TVM_DLL LoopRV();

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(LoopRV, ObjectRef, LoopRVNode);
};

/**************** The schedule class ****************/

class Schedule;

class ScheduleNode : public runtime::Object {
 public:
  /*! \brief The internal state of scheduling */
  ScheduleState state;

  virtual ~ScheduleNode() = default;

  static constexpr const char* _type_key = "tir.Schedule";
  TVM_DECLARE_BASE_OBJECT_INFO(ScheduleNode, Object);

 public:
  /*! \brief Get the actual concrete name of the class */
  virtual String GetClassName() const = 0;

  /*!
   * \brief Copy the schedule and guarantee that
   * 1) SRef tree is completely reconstructed
   * 2) Schedule function is untouched
   * 3) For all the random variables, they are valid in both original copy and the new copy, but
   * points to different StmtSRefs, because the SRef tree is reconstructed
   */
  virtual Schedule Copy() const = 0;
  /*!
   * \brief Seed the randomness
   * \param seed The new random seed
   */
  virtual void Seed(int64_t seed = -1) = 0;
  /*!
   * \brief Take the PrimFunc out of the schedule
   */
  virtual IRModule Module() const = 0;

 public:
  /******** Lookup random variables ********/
  virtual Block Get(const BlockRV& block_rv) const = 0;

  virtual For Get(const LoopRV& loop_rv) const = 0;

  virtual int64_t Get(const Var& var_rv) const = 0;

  virtual PrimExpr Get(const ExprRV& expr_rv) const = 0;

  virtual StmtSRef GetSRef(const BlockRV& block_rv) const = 0;

  virtual StmtSRef GetSRef(const LoopRV& loop_rv) const = 0;

  virtual StmtSRef GetSRef(const Stmt& stmt) const = 0;

  virtual StmtSRef GetSRef(const StmtNode* stmt) const = 0;

 public:
  /******** Sampling ********/
  /*!
   * \brief Apply the instruction SamplePerfectTile
   * \param n The number of loops after tiling
   * \param loop_rv The loop to be tiled
   * \param max_innermost_factor The maximum factor in the innermost loop, -1 if disabled
   * \return An array of random variables, the result of sampling
   */
  virtual Array<Var> SamplePerfectTile(const LoopRV& loop_rv,     //
                                       int n,                     //
                                       int max_innermost_factor,  //
                                       Optional<Array<ObjectRef>> decision = NullOpt) = 0;
  /*!
   * \brief Sample an integer given the probability distribution
   * \param candidates The candidates
   * \param probs The probability distribution of the candidates
   * \return The random variable
   */
  virtual Var SampleCategorical(const Array<Integer>& candidates,  //
                                const Array<FloatImm>& probs,      //
                                Optional<ObjectRef> decision = NullOpt) = 0;
  /*!
   * \brief Sample a compute-at location from a block
   * \param block A block to be computed at
   * \return The loop to be computed at
   */
  virtual LoopRV SampleComputeLocation(const BlockRV& block_rv,
                                       Optional<ObjectRef> decision = NullOpt) = 0;

 public:
  /******** Block/Loop relation ********/
  /*!
   * \brief Get block from its tag
   * \param tag The query tag
   * \return the block schedulable reference list
   */
  virtual BlockRV GetBlock(const String& name) = 0;
  /*!
   * \brief Get loops of the block
   * \param block The query block
   * \return A list of loops, from outer to inner
   */
  virtual Array<LoopRV> GetAxes(const BlockRV& block_rv) = 0;
  /*!
   * \brief Get the child blocks of a specific parent block/loop
   * \param block_rv The random variable that points to the parent block
   * \return A list of child blocks
   * TODO(@junrushao1994): revisit
   */
  virtual Array<BlockRV> GetChildBlocks(const BlockRV& block_rv) = 0;
  /*!
   * \brief Get the child blocks of a specific parent block/loop
   * \param loop_rv The random variable that points to the parent loop
   * \return A list of child blocks
   */
  virtual Array<BlockRV> GetChildBlocks(const LoopRV& loop_rv) = 0;
  /*!
   * \brief Get the producer of a specific block
   * \return The producers
   */
  virtual Array<BlockRV> GetProducers(const BlockRV& block_rv) = 0;
  /*!
   * \brief Get the consumers of a specific block
   * \return The consumers
   */
  virtual Array<BlockRV> GetConsumers(const BlockRV& block_rv) = 0;

  /******** Schedule: loops ********/
  /*!
   * \brief Fuse two consecutive loops of one computation.
   * \param loop_rvs The loop random variables to be fused
   * \return The fused loop
   */
  virtual LoopRV Fuse(const Array<LoopRV>& loop_rvs) = 0;
  /*!
   * \brief Split a specified loop into two loops by factor.
   * \param loop_rv The loop to be split
   * \param factors The tiling factors, and exactly one of which is NullOpt or -1
   * \return The loops after splitting
   */
  virtual Array<LoopRV> Split(const LoopRV& loop_rv, const Array<Optional<ExprRV>>& factors) = 0;
  /*!
   * \brief reorder a list of loops
   * \param order the order of loops
   */
  virtual void Reorder(const Array<LoopRV>& order) = 0;

  /******** Schedule: compute location ********/
  /*!
   * \brief Move the block under the loop and regenerate the loops to cover the producing region.
   * \param block_rv The block to be moved
   * \param loop_rv The target loop
   * \param preserve_unit_loop Keep the trivial loops whose extent is 1
   */
  virtual void ComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                         bool preserve_unit_loop) = 0;
  /*!
   * \brief Move the block under the loop and regenerate the loops to cover the producing region.
   * \param block_rv The block to be moved
   * \param loop_rv The target loop
   * \param preserve_unit_loop Keep the trivial loops whose extent is 1
   */
  virtual void ReverseComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                                bool preserve_unit_loop) = 0;
  /*!
   * \brief Make the block inline
   * \param block_rv The block
   */
  virtual void ComputeInline(const BlockRV& block_rv) = 0;
  /*!
   * \brief Make the block inline
   * \param block_rv The block
   */
  virtual void ReverseComputeInline(const BlockRV& block_rv) = 0;

  /******** Schedule: parallelize / annotate ********/
  /*!
   * \brief vectorize a loop
   * \param loop_rv the loop to be vectorized
   */
  virtual void Vectorize(const LoopRV& loop_rv) = 0;
  /*!
   * \brief parallelize a loop
   * \param loop_rv the loop to be paralleled
   */
  virtual void Parallel(const LoopRV& loop_rv) = 0;
  /*!
   * \brief unroll a loop
   * \param loop_rv the loop to be unrolled
   */
  virtual void Unroll(const LoopRV& loop_rv) = 0;
  /*!
   * \brief bind a loop to a thread axis
   * \param loop_rv the loop to be paralleled
   * \param thread The thread axis
   */
  virtual void Bind(const LoopRV& loop_rv, const IterVar& thread) = 0;
  /*!
   * \brief parallel a loop
   * \param loop_rv the loop to be paralleled
   * \param thread The thread axis
   */
  virtual void Bind(const LoopRV& loop_rv, const String& thread) = 0;
  /*!
   * \brief add double_buffer annotation to a complete block
   * \param block_rv the block of interest
   */
  virtual void DoubleBuffer(const BlockRV& block_rv) = 0;
  /*!
   * \brief add annotation to a loop
   * \param loop_rv the loop of interest
   * \param pragma_type the attribute key
   * \param pragma_value the attribute value
   */
  virtual void Pragma(const LoopRV& loop_rv, const String& pragma_type,
                      const ExprRV& pragma_value) = 0;

  /******** Schedule: cache read/write ********/
  /*!
   * \brief Create a cache read of original tensor for readers.
   * \param block_rv The consumer of the buffer
   * \param i The index of the buffer in block's read region
   * \param storage_scope The storage scope
   */
  virtual BlockRV CacheRead(const BlockRV& block_rv, int i, const String& storage_scope) = 0;
  /*!
   * \brief Create a cache write of original tensor, before storing into tensor.
   * \param block_rv The producer of the buffer
   * \param i The index of the buffer in block's write region
   * \param storage_scope The storage scope
   */
  virtual BlockRV CacheWrite(const BlockRV& block_rv, int i, const String& storage_scope) = 0;

  /******** Schedule: reduction ********/
  /*!
   * \brief rfactor a reduction block using loop
   * \param loop_rv the loop outside block we want to do rfactor
   * \param factor_axis the position where the new axis is placed
   * \return The new block
   * TODO(@junrushao1994): do we need a concrete integer here?
   */
  virtual BlockRV RFactor(const LoopRV& loop_rv, int factor_axis) = 0;
  /*!
   * \brief Decompose reduction block_rv into init&update blocks
   * \param block_rv the reduction block_rv
   * \param loop_rv the position where init block_rv will be
   * \return The init block
   */
  virtual BlockRV DecomposeReduction(const BlockRV& block_rv, const Optional<LoopRV>& loop_rv) = 0;
  /*!
   * \brief Merge init and reduction block into reduction block
   * \param init_block_rv the init block
   * \param update_block_rv the update block
   */
  virtual void MergeReduction(const BlockRV& init_block_rv, const BlockRV& update_block_rv) = 0;

  /******** Schedule: blockize / tensorize ********/
  /*!
   * \brief make subtree rooted by loop_rv into a block
   * \param loop_rv the subtree root
   * \return the loop_rv of new block
   */
  virtual BlockRV Blockize(const LoopRV& loop_rv, const String& exec_scope) = 0;
  /*!
   * \brief Tensorize the computation enclosed by loop with tensor_intrin
   * \param loop_rv the loop/block to be tensorized
   * \param intrin the tensor intrinsic
   */
  virtual void Tensorize(const LoopRV& loop_rv, const TensorIntrin& intrin) = 0;
  /*!
   * \brief Tensorize the computation enclosed by loop with tensor_intrin
   * \param loop_rv The loop/block to be tensorized
   * \param intrin_name Name of the tensor intrinsic
   */
  virtual void Tensorize(const LoopRV& loop_rv, const String& intrin_name) = 0;
};

class Schedule : public runtime::ObjectRef {
 public:
  TVM_DLL static Schedule Concrete(PrimFunc func, int64_t seed, bool debug_mode);
  TVM_DLL static Schedule Meta(PrimFunc func, int64_t seed, bool debug_mode);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Schedule, runtime::ObjectRef, ScheduleNode);
};

TVM_DLL String Repr(const PrimFunc& func);
TVM_DLL String Repr(const IRModule& mod);
TVM_DLL String Repr(const Schedule& self);

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_SCHEDULE_H_
