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

#include <tvm/tir/function.h>
#include <tvm/tir/schedule/state.h>

namespace tvm {
namespace tir {

/**************** Random variable: BlockRV ****************/

/*! \brief A random variable that evaluates to a TensorIR block */
class BlockRVNode : public runtime::Object {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}
  static constexpr const char* _type_key = "tir.BlockRV";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockRVNode, runtime::Object);
};

/*!
 * \brief Managed reference to BlockRVNode
 * \sa BlockRVNode
 */
class BlockRV : public runtime::ObjectRef {
 public:
  /*! \brief Constructor */
  TVM_DLL BlockRV();
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BlockRV, runtime::ObjectRef, BlockRVNode);
};

/**************** Random variable: LoopRV ****************/

/*! \brief A random variable that evaluates to a TensorIR for loop */
class LoopRVNode : public runtime::Object {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}
  static constexpr const char* _type_key = "tir.LoopRV";
  TVM_DECLARE_FINAL_OBJECT_INFO(LoopRVNode, runtime::Object);
};

/*!
 * \brief Managed reference to LoopRVNode
 * \sa LoopRVNode
 */
class LoopRV : public runtime::ObjectRef {
 public:
  /*! \brief Constructor */
  TVM_DLL LoopRV();
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(LoopRV, runtime::ObjectRef, LoopRVNode);
};

/**************** Random variable: ExprRV ****************/

/*! \brief An expr random variable */
using ExprRV = PrimExpr;

using ExprRVNode = PrimExprNode;

/**************** The Schedule class ****************/

class Schedule;

/*! \brief The user-facing schedule class */
class ScheduleNode : public runtime::Object {
  friend class Schedule;

 public:
  virtual ~ScheduleNode() = default;

  static constexpr const char* _type_key = "tir.Schedule";
  TVM_DECLARE_BASE_OBJECT_INFO(ScheduleNode, runtime::Object);

 public:
  /******** Sampling ********/
  /*!
   * \brief Sample the factors to perfect tiling a specific LoopRV
   * \param loop_rv The loop to be tiled
   * \param n The number of loops after tiling
   * \param max_innermost_factor The maximum factor in the innermost loop, -1 if disabled
   * \param decision The sampling decision
   * \return An array of n random variables, the result of sampling
   */
  virtual Array<ExprRV> SamplePerfectTile(const LoopRV& loop_rv,     //
                                          int n,                     //
                                          int max_innermost_factor,  //
                                          Optional<Array<Integer>> decision = NullOpt) = 0;
  /*!
   * \brief Sample an integer given the probability distribution
   * \param candidates The candidates
   * \param probs The probability distribution of the candidates
   * \param decision The sampling decision
   * \return The random variable sampled from candidates
   */
  virtual ExprRV SampleCategorical(const Array<Integer>& candidates,  //
                                   const Array<FloatImm>& probs,      //
                                   Optional<Integer> decision = NullOpt) = 0;
  /*!
   * \brief Sample a compute-at location on a BlockRV so that its producer can compute at that loop
   * \param block_rv The consumer block to be computed at
   * \return The sampled loop to be computed at
   */
  virtual LoopRV SampleComputeLocation(const BlockRV& block_rv,
                                       Optional<Integer> decision = NullOpt) = 0;

 public:
  /*! \brief Get the IRModule associated with this schedule. */
  virtual IRModule mod() const { return state()->mod; }
  /*! \return The internal state of scheduling */
  virtual ScheduleState state() const = 0;
  /*!
   * \brief Returns a copy of the schedule, including both its state and its symbol table,
   * guaranteeing that
   * 1) SRef tree is completely reconstructed;
   * 2) The IRModule being scheduled is not modified;
   * 3) All the random variables are valid in the copy, pointing to the correpsonding sref
   * reconstructed
   */
  virtual Schedule Copy() const = 0;
  /*!
   * \brief Seed the randomness
   * \param seed The new random seed, -1 if use device random, otherwise non-negative
   */
  virtual void Seed(int64_t seed = -1) {
    LOG(FATAL) << "ValueError: The schedule cannot be seeded because no randomness is allowed";
  }

 public:
  /******** Lookup/Remove random variables ********/
  /*!
   * \brief Get the block corresponding to the specific BlockRV
   * \param block_rv The BlockRV to be looked up
   * \return The corresponding block
   */
  virtual Block Get(const BlockRV& block_rv) const = 0;
  /*!
   * \brief Get the for loop corresponding to the specific LoopRV
   * \param loop_rv The LoopRV to be looked up
   * \return The corresponding for loop
   */
  virtual For Get(const LoopRV& loop_rv) const = 0;
  /*!
   * \brief Get the expr corresponding to the specific random variable
   * \param expr_rv The random variable to be looked up
   * \return The corresponding expr
   */
  virtual PrimExpr Get(const ExprRV& expr_rv) const = 0;
  /*!
   * \brief Get the block sref corresponding to the specific BlockRV
   * \param block_rv The BlockRV to be looked up
   * \return The corresponding block sref
   */
  virtual StmtSRef GetSRef(const BlockRV& block_rv) const = 0;
  /*!
   * \brief Get the loop sref corresponding to the specific LoopRV
   * \param loop_rv The LoopRV to be looked up
   * \return The corresponding loop sref
   */
  virtual StmtSRef GetSRef(const LoopRV& loop_rv) const = 0;
  /*!
   * \brief Get the block/loop sref corresponding to the specific statement
   * \param stmt The statement to be looked up
   * \return The corresponding block/loop sref
   */
  virtual StmtSRef GetSRef(const StmtNode* stmt) const;
  /*!
   * \brief Get the block/loop sref corresponding to the specific statement
   * \param stmt The statement to be looked up
   * \return The corresponding block/loop sref
   */
  StmtSRef GetSRef(const Stmt& stmt) const { return this->GetSRef(stmt.get()); }
  /*!
   * \brief Remove a block random variable from the symbol table
   * \param block_rv The random variable to be removed
   */
  virtual void RemoveRV(const BlockRV& block_rv) = 0;
  /*!
   * \brief Remove a loop random variable from the symbol table
   * \param loop_rv The random variable to be removed
   */
  virtual void RemoveRV(const LoopRV& loop_rv) = 0;
  /*!
   * \brief Remove an integer random variable from the symbol table
   * \param expr_rv The random variable to be removed
   */
  virtual void RemoveRV(const ExprRV& expr_rv) = 0;

 public:
  /******** Block/Loop relation ********/
  /*!
   * \brief Retrieve a block in a specific function with its name
   * \param name The name of the block to be retrieved
   * \param func_name The name of the function
   * \return The block retrieved
   * \note Indexing error is raised if 0 or multiple blocks exist with the specific name
   */
  virtual BlockRV GetBlock(const String& name, const String& func_name = "main") = 0;
  /*!
   * \brief Get the parent loops of the block in its scope, from outer to inner
   * \param block_rv The query block
   * \return A list of loops above the given block in its scope, from outer to inner
   */
  virtual Array<LoopRV> GetLoops(const BlockRV& block_rv) = 0;
  /*!
   * \brief Get the leaf blocks of a specific scope
   * \param block_rv The block where the scope is rooted
   * \return A list of child blocks
   */
  virtual Array<BlockRV> GetChildBlocks(const BlockRV& block_rv) = 0;
  /*!
   * \brief Get the leaf blocks of under a specific loop
   * \param loop_rv The loop under which collecting is conducted
   * \return A list of child blocks
   */
  virtual Array<BlockRV> GetChildBlocks(const LoopRV& loop_rv) = 0;
  /*!
   * \brief Get the producer of a specific block
   * \param block_rv The block to be queried
   * \return The producers
   */
  virtual Array<BlockRV> GetProducers(const BlockRV& block_rv) = 0;
  /*!
   * \brief Get the consumers of a specific block
   * \param block_rv The block to be queried
   * \return The consumers
   */
  virtual Array<BlockRV> GetConsumers(const BlockRV& block_rv) = 0;

  /******** Schedule: loops ********/
  /*!
   * \brief Fuse consecutive loops into one.
   * \param loop_rvs The loop random variables to be fused
   * \return The fused loop
   */
  virtual LoopRV Fuse(const Array<LoopRV>& loop_rvs) = 0;
  /*!
   * \brief Split a specified loop into two or more with the specific factor.
   * \param loop_rv The loop to be split
   * \param factors The tiling factors, and at most one of which is NullOpt or -1, which means that
   * factor is inferred.
   * \return The loops after splitting
   */
  virtual Array<LoopRV> Split(const LoopRV& loop_rv, const Array<Optional<ExprRV>>& factors) = 0;
  /*!
   * \brief Reorder a list of loops
   * \param order The order after reordering
   */
  virtual void Reorder(const Array<LoopRV>& order) = 0;

  /******** Schedule: compute location ********/
  /*!
   * \brief Compute the producer block under its consumer's specific loop,
   * and regenerate the loops to cover the region needed by the consumer.
   * \param block_rv The block to be moved
   * \param loop_rv The target loop
   * \param preserve_unit_loop Keep the trivial loops whose extent is 1
   */
  virtual void ComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                         bool preserve_unit_loop) = 0;
  /*!
   * \brief Compute the consumer block under its producer's specific loop,
   * and regenerate the loops to cover the region needed by the producer.
   * \param block_rv The block to be moved
   * \param loop_rv The target loop
   * \param preserve_unit_loop Keep the trivial loops whose extent is 1
   */
  virtual void ReverseComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                                bool preserve_unit_loop) = 0;
  /*!
   * \brief Remove the block, and produce the value needed inplace
   * where it is used on the consumer.
   * \param block_rv The block to be inlined
   */
  virtual void ComputeInline(const BlockRV& block_rv) = 0;
  /*!
   * \brief Remove the block, and produce the value needed inplace
   * where it is used on the producer.
   * \param block_rv The block to be reverse-inlined
   */
  virtual void ReverseComputeInline(const BlockRV& block_rv) = 0;

  /******** Schedule: parallelize / annotate ********/
  /*!
   * \brief Vectorize a loop
   * \param loop_rv The loop to be vectorized
   */
  virtual void Vectorize(const LoopRV& loop_rv) = 0;
  /*!
   * \brief Parallelize a loop
   * \param loop_rv The loop to be paralleled
   */
  virtual void Parallel(const LoopRV& loop_rv) = 0;
  /*!
   * \brief Unroll a loop
   * \param loop_rv The loop to be unrolled
   */
  virtual void Unroll(const LoopRV& loop_rv) = 0;
  /*!
   * \brief Bind a loop to a thread axis
   * \param loop_rv The loop to be bound
   * \param thread The thread axis
   */
  virtual void Bind(const LoopRV& loop_rv, const IterVar& thread) = 0;
  /*!
   * \brief Bind a loop to a thread axis
   * \param loop_rv The loop to be bound
   * \param thread The thread axis
   */
  virtual void Bind(const LoopRV& loop_rv, const String& thread) = 0;
  /*!
   * \brief Add `double_buffer` annotation to a block
   * \param block_rv The block to be annotated
   */
  virtual void DoubleBuffer(const BlockRV& block_rv) = 0;
  /*!
   * \brief Set the storage scope of a buffer, where the buffer is given as the i-th write buffer
   *        of the input block
   * \param block_rv The producer of the buffer
   * \param i The index of the buffer in block's write region
   * \param storage_scope The storage scope to be set
   */
  virtual void SetScope(const BlockRV& block_rv, int i, const String& storage_scope) = 0;
  /*!
   * \brief Add a pragma annotation to a specific loop
   * \param loop_rv The loop to be annotated
   * \param pragma_type The attribute key
   * \param pragma_value The attribute value
   */
  virtual void Pragma(const LoopRV& loop_rv, const String& pragma_type,
                      const ExprRV& pragma_value) = 0;
  /*!
   * \brief Set alignment requirement for specific dimension such that
   *        stride[axis] == k * factor + offset for some k.
   * \param block_rv The producer block of the buffer
   * \param buffer_index The index of the buffer in block's write region
   * \param axis The dimension to be specified for alignment
   * \param factor The factor multiple of alignment
   * \param offset The required offset factor
   */
  virtual void StorageAlign(const BlockRV& block_rv, int buffer_index, int axis, int factor,
                            int offset) = 0;

  /******** Schedule: cache read/write ********/
  /*!
   * \brief Create a block that reads a buffer region into a read cache
   * \param block_rv The consumer of the buffer
   * \param i The index of the buffer in block's read region
   * \param storage_scope The storage scope
   */
  virtual BlockRV CacheRead(const BlockRV& block_rv, int i, const String& storage_scope) = 0;
  /*!
   * \brief Create a block that writes a buffer region into a write cache
   * \param block_rv The producer of the buffer
   * \param i The index of the buffer in block's write region
   * \param storage_scope The storage scope
   */
  virtual BlockRV CacheWrite(const BlockRV& block_rv, int i, const String& storage_scope) = 0;

  /******** Schedule: reduction ********/
  /*!
   * \brief Factor a reduction block by the specified loop
   * \param loop_rv The loop outside block we want to do rfactor
   * \param factor_axis The position where the new dimension is placed in the new generated rfactor
   *                      buffer
   * \return The rfactor block
   */
  virtual BlockRV RFactor(const LoopRV& loop_rv, int factor_axis) = 0;
  /*!
   * \brief Decompose a reduction block into init block and update block
   * \param block_rv The reduction block
   * \param loop_rv The position where init block is inserted
   * \return The init block
   */
  virtual BlockRV DecomposeReduction(const BlockRV& block_rv, const Optional<LoopRV>& loop_rv) = 0;
  /*!
   * \brief Construct a reduction block by merging the init and update block
   * \param init_block_rv The init block
   * \param update_block_rv The update block
   */
  virtual void MergeReduction(const BlockRV& init_block_rv, const BlockRV& update_block_rv) = 0;

  /******** Schedule: blockize / tensorize ********/
  /*!
   * \brief Make subtree rooted by a specific loop into a block
   * \param loop_rv The root of the subtree
   * \return The new block
   */
  virtual BlockRV Blockize(const LoopRV& loop_rv) = 0;
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

  /******** Schedule: Misc ********/
  virtual void InlineArgument(int i, const String& func_name) = 0;
};

/*!
 * \brief Managed reference to ScheduleNode
 *
 * A schedule is a set of transformations that change the order of computation but
 * preserve the semantics of computation. Some example of schedules:
 * 1) Split a loop into two;
 * 2) Reorder two loops;
 * 3) Inline the computation of a specific buffer into its consumer
 *
 * The schedule class stores auxiliary information to schedule correctly and efficiently.
 *
 * Link to tutorial: https://tvm.apache.org/docs/tutorials/language/schedule_primitives.html
 *
 * \sa ScheduleNode
 */
class Schedule : public runtime::ObjectRef {
 public:
  /*!
   * \brief Construct a concrete TensorIR schedule from an IRModule
   * \param mod The IRModule to be scheduled
   * \param debug_mode Do extra correctness checking after the class creation
   * and each time after calling the Replace method.
   * \return The concrete schedule created
   * \sa ScheduleDebugMask
   * \note The checks performed includes:
   * 1) VerifySRefTree
   * 2) VerifyAffineBinding
   * 3) VerifyRegionCover
   * 4) VerifyStagePipeline
   */
  TVM_DLL static Schedule Concrete(IRModule mod, int debug_mode);
  TVM_DLL static Schedule Meta(PrimFunc func, int64_t seed, int debug_mode);
  TVM_DLL static Schedule Meta(IRModule mod, int64_t seed, int debug_mode);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Schedule, runtime::ObjectRef, ScheduleNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_SCHEDULE_H_
