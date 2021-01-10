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

#include <tvm/tir/schedule.h>

#include <unordered_map>

#include "./instruction.h"
#include "./sampler.h"
#include "./trace.h"

namespace tvm {
namespace meta_schedule {

class Schedule;

/*! \brief The meta schedule class */
class ScheduleNode : public Object {
 public:
  /*! \brief Type of the symbol table, which maps a random variable to its value */
  using TSymbolTable = Map<ObjectRef, Optional<ObjectRef>>;

 public:
  /*! \brief The original TIR PrimFunc to be scheduled */
  tir::PrimFunc orig_func;
  /*! \brief The TIR schedule in the current stage */
  tir::Schedule sch{nullptr};
  /*! \brief The trace of the program execution */
  Trace trace;
  /*! \brief The symbol table with information of all defined variables in the meta schedule */
  TSymbolTable sym_tab;
  /*! \brief The random number generator */
  Sampler sampler;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("orig_func", &orig_func);
    v->Visit("sch", &sch);
    v->Visit("trace", &trace);
    v->Visit("sym_tab", &sym_tab);
    // `sampler` is not visited
  }
  static constexpr const char* _type_key = "meta_schedule.Schedule";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleNode, Object);
  /**************** Utility ****************/
  /*!
   * \brief Seed the randomness
   * \param seed The new random seed
   */
  void Seed(int seed);
  /*!
   * \brief Copy the schedule into a new one. Operation on the new schedule won't affect the
   * original schedule, and vice versa.
   * \return A new schedule.
   */
  Schedule Copy(int new_seed) const;
  /**************** Evaluation of random variables ****************/
  /*!
   * \brief Evaluate the value of a random variable of type Block
   * \param block The block random variable to be evaluated
   * \return The TIR SRef to the block evaluated
   */
  tir::StmtSRef Eval(const BlockRV& block);
  /*!
   * \brief Evaluate the value of a random variable of type Loop
   * \param loop The loop random variable to be evaluated
   * \return The TIR SRef to the block evaluated
   */
  tir::StmtSRef Eval(const LoopRV& loop);
  /*!
   * \brief Evaluate the value of a random variable of type Buffer
   * \param buffer The buffer random variable to be evaluated
   * \return A TIR buffer, the result of evaluation
   */
  tir::Buffer Eval(const BufferRV& buffer);
  /*!
   * \brief Evaluate the value of a PrimExpr, containing random variable of type tir::Var
   * \param expr The expression containing random variables to be evaluated
   * \return The result of the evaluation
   */
  int Eval(const PrimExpr& expr);
  /*!
   * \brief Evaluate the value of a random variable of type Loop, or inline_rv, or root_rv
   * \param loop The loop random variable to be evaluated
   * \return The TIR SRef to the block evaluated, or inline_rv, or root_rv
   */
  ObjectRef EvalLoopExtended(const LoopRV& loop);
  /**************** Sampling ****************/
  enum class Order : int {
    outer_to_inner = 0,
    inner_to_order = 1,
  };
  enum class Mode : int {
    max = 0,
    rand = 1,
  };
  /*!
   * \brief Apply the instruction SamplePerfectTile
   * \param n_splits The number of loops after tiling
   * \param loop The loop to be tiled
   * \param max_innermost_factor The maximum factor in the innermost loop
   * \return An array of random variables, the result of sampling
   */
  Array<tir::Var> SamplePerfectTile(int n_splits, const LoopRV& loop, int max_innermost_factor = 16,
                                    const Optional<Array<ObjectRef>>& decision = NullOpt);
  /*!
   * \brief Apply the instruction SampleTileFactor
   * \param n_splits The number of loops after tiling
   * \param loop The loop to be tiled
   * \param where The distribution of tile size to be sampled
   * \return An array of random variables, the result of sampling
   */
  Array<tir::Var> SampleTileFactor(int n_splits, const LoopRV& loop, const Array<Integer>& where,
                                   const Optional<Array<ObjectRef>>& decision = NullOpt);
  /*!
   * \brief Sample an integer in [min_inclusive, max_exclusive)
   * \param min_inclusive The left boundary, inclusive
   * \param max_exclusive The right boundary, exclusive
   * \return The integer sampled
   */
  tir::Var SampleInt(const PrimExpr& min_inclusive, const PrimExpr& max_exclusive,
                     const Optional<ObjectRef>& decision = NullOpt);
  /*!
   * \brief Sample an integer given the probability distribution
   * \param candidates The candidates
   * \param probs The probability distribution of the candidates
   * \return The random variable
   */
  tir::Var SampleCategorical(const Array<Integer>& candidates, const Array<FloatImm>& probs,
                             const Optional<ObjectRef>& decision = NullOpt);
  /*!
   * \brief Sample a compute-at location from a block
   * \param block A block to be computed at
   * \return The loop to be computed at
   */
  LoopRV SampleComputeLocation(const BlockRV& block, const Optional<ObjectRef>& decision = NullOpt);
  /**************** Block/Loop Relationship ****************/
  /*!
   * \brief Get the producer of a specific block
   * \return The producers
   */
  Array<BlockRV> GetProducers(const BlockRV& block);
  /*!
   * \brief Get the consumers of a specific block
   * \return The consumers
   */
  Array<BlockRV> GetConsumers(const BlockRV& block);
  /*!
   * \brief Apply the instruction GetBlock
   * \param name The name of the block to get retrieved
   * \return A block random variable, the return value of the instruction
   */
  BlockRV GetBlock(const String& name);
  /*!
   * \brief Apply the instruction GetAxes
   * \param block The block used to retrieve the axes
   * \return An array of loop random variables
   */
  Array<LoopRV> GetAxes(const BlockRV& block);
  /*!
   * \brief Get the buffers the block reads
   * \param block The block
   * \return A list of buffers the block reads
   */
  Array<BufferRV> GetReadBuffers(const BlockRV& block);
  /*!
   * \brief Get the buffers the block writes
   * \param block The block
   * \return A list of buffers the block writes
   */
  Array<BufferRV> GetWriteBuffers(const BlockRV& block);
  /*!
   * \brief Get the root blocks which are direct children of the root node
   * \return An array of block random variables, the sub-root blocks
   * \note It is not useful, should remove
   */
  Array<BlockRV> GetRootBlocks();
  /*!
   * \brief Get the leaf blocks who do not have any child block
   * \return An array of block random variables, the leaf blocks
   * \note It is not useful, should remove
   */
  Array<BlockRV> GetLeafBlocks();
  /**************** Scheduling Primitives ****************/
  /*!
   * \brief Mark a loop
   * \param loop The loop to be marked
   * \param ann_key The annotation key
   * \param ann_val The annotation value
   */
  void MarkLoop(const LoopRV& loop, const String& ann_key, const PrimExpr& ann_val);
  /*!
   * \brief Mark a block
   * \param block The block to be marked
   * \param ann_key The annotation key
   * \param ann_val The annotation value
   */
  void MarkBlock(const BlockRV& block, const String& ann_key, const PrimExpr& ann_val);
  /*!
   * \brief Fuse the loops
   * \param loops The loops to be fused
   * \return The fused loop
   */
  LoopRV Fuse(const Array<LoopRV>& loops);
  /*!
   * \brief Apply the instruction Split
   * \param loop The loop to be split
   * \param factors The split factors
   * \return An array of loop random variables
   * \note If there is no NullOpt in factors, will split from inner to outer, and factors[0] is not
   * used
   */
  Array<LoopRV> Split(const LoopRV& loop, const Array<Optional<PrimExpr>>& factors);
  /*!
   * \brief Apply the instruction Reorder
   * \param after_axes The axes to be reordered
   */
  void Reorder(const Array<LoopRV>& after_axes);
  /*!
   * \brief Move the block under the loop and regenerate the loops to cover the producing region.
   * \param block The block to be moved
   * \param loop The loop to be moved to
   */
  void ComputeAt(const BlockRV& block, const LoopRV& loop);
  /*!
   * \brief Move the block under the loop and regenerate the loops to cover the producing region.
   * \param block The block to be moved
   * \param loop The loop to be moved to
   */
  void ReverseComputeAt(const BlockRV& block, const LoopRV& loop);
  /*!
   * \brief Apply the instruction compute_inline
   * \param block The block to be computed inline
   */
  void ComputeInline(const BlockRV& block);
  /*!
   * \brief Apply the instruction reverse+compute_inline
   * \param block The block to be reverse computed inline
   */
  void ReverseComputeInline(const BlockRV& block);
  /*!
   * \brief Apply the instruction cache_read
   * \param block The read block of the buffer to be cached
   * \param i The index of the buffer in block's read region
   * \param storage_scope The storage scope
   * \return The cache write stage
   */
  BlockRV CacheRead(const BlockRV& block, int i, const String& storage_scope);
  /*!
   * \brief Apply the instruction cache_write
   * \param block The write block of the buffer to be cached
   * \param i The index of the buffer in block's write region
   * \param storage_scope The storage scope
   * \return The cache write stage
   */
  BlockRV CacheWrite(const BlockRV& block, int i, const String& storage_scope);
  /*!
   * \brief Apply blockize to the schedule
   * \param loop The loop to be blockized
   * \param exec_scope The execution scope
   * \return A block random variable pointing to the new block
   */
  BlockRV Blockize(const LoopRV& loop, const String& exe_scope);
  /*!
   * \brief Apply the instruction DecomposeReduction
   * \param block The block to be decomposed
   * \param loop The loop to be decomposed at
   * \return The block random variable indicating the decomposition result
   */
  BlockRV DecomposeReduction(const BlockRV& block, const LoopRV& loop);
  /*!
   * \brief Parallelize a specific loop
   * \param loop The loop to be parallelized
   */
  void Parallel(const LoopRV& loop);
  /*!
   * \brief Vectorize a specific loop
   * \param loop The loop to be vectorized
   */
  void Vectorize(const LoopRV& loop);
  /*!
   * \brief Unroll a specific loop
   * \param loop The loop to be unrolled
   */
  void Unroll(const LoopRV& loop);
  /*!
   * \brief Bind a thread_axis to a specific loop
   * \param loop The loop to be unrolled
   * \param thread_axis The thread axis to be bound to the loop
   */
  void Bind(const LoopRV& loop, const String& thread_axis);
  /*! \brief An NOP indicating entrance of post processing*/
  void EnterPostProc();
};

class Schedule : public ObjectRef {
 public:
  using TSymbolTable = ScheduleNode::TSymbolTable;
  /*!
   * \brief Constructor
   * \param orig_func The original TIR PrimFunc to be scheduled
   * \param sch The TIR schedule in the current stage
   * \param trace The trace of the program execution
   * \param sym_tab The symbol table with information of all defined variables in the meta schedule
   * \param seed The random seed
   */
  explicit Schedule(tir::PrimFunc orig_func, tir::Schedule sch, Trace trace, TSymbolTable sym_tab,
                    Optional<Integer> seed);
  /*!
   * \brief Constructor: other fields are created with default value
   * \param orig_func The original TIR PrimFunc to be scheduled
   * \param seed The random seed
   */
  explicit Schedule(tir::PrimFunc orig_func, Optional<Integer> seed);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Schedule, ObjectRef, ScheduleNode);
};

/*!
 * \brief Get the string representation of a schedule
 * \param sch The schedule to be stringified
 * \return The string representation of a schedule
 */
inline String Repr(const Schedule& sch) {
  const auto* f = runtime::Registry::Get("script.AsTVMScript");
  CHECK(f) << "IndexError: global function \"script.AsTVMScript\" not found";
  String s = (*f)(sch->sch->func, false);
  return s;
}

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SCHEDULE_H_
