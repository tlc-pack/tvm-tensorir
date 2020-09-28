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
#include <utility>

#include "./instruction.h"
#include "./random_variable.h"
#include "./sampler.h"

namespace tvm {
namespace meta_schedule {

class Schedule;

/*! \brief An entry in the symbol table in meta schedule */
class SymbolTableEntry {
 public:
  /*! \brief The index of the instruction that generates the current random variable */
  int source;
  /*! \brief The value of the current random variable */
  Optional<ObjectRef> value;
  /*!
   * \brief Constructor
   * \param source The index of the instruction that generates the current random variable
   * \param value The value of the current random variable
   */
  explicit SymbolTableEntry(int source, Optional<ObjectRef> value)
      : source(source), value(std::move(value)) {}
};

/*!
 * \brief The symbol table, which maps a random variable to a SymbolTableEntry
 * \sa SymbolTableEntry
 */
using SymbolTable = std::unordered_map<ObjectRef, SymbolTableEntry, ObjectPtrHash, ObjectPtrEqual>;

/*! \brief The meta schedule class */
class ScheduleNode : public Object {
 public:
  /*! \brief The original TIR PrimFunc to be scheduled */
  tir::PrimFunc orig_func;
  /*! \brief The TIR schedule in the current stage */
  tir::Schedule sch{nullptr};
  /*! \brief The trace of instructions used */
  Array<Instruction> trace;
  /*! \brief The symbol table with information of all defined variables in the meta schedule */
  SymbolTable sym_tab;
  /*! \brief The random number generator */
  Sampler sampler;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("orig_func", &orig_func);
    v->Visit("sch", &sch);
    v->Visit("trace", &trace);
    // `sym_tab` is not visited
    // `sampler` is not visited
  }
  static constexpr const char* _type_key = "meta_schedule.Schedule";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleNode, Object);
  /**************** Utility ****************/
  /*!
   * \brief Copy the schedule into a new one. Operation on the new schedule won't affect the
   * original schedule, and vice versa.
   * \return A new schedule.
   */
  Schedule copy() const;
  /**************** Evaluation of random variables ****************/
  /*!
   * \brief Evaluate the value of a random variable of type Block
   * \param block The block random variable to be evaluated
   * \return The TIR SRef to the block evaluated
   */
  tir::StmtSRef Eval(const BlockRV& block);
  /*!
   * \brief Evaluate the value of a random variable of type LoopAxis
   * \param loop The loop random variable to be evaluated
   * \return The TIR SRef to the block evaluated
   */
  tir::StmtSRef Eval(const LoopRV& loop);
  /*!
   * \brief Evaluate the value of a PrimExpr, containing random variable of type tir::Var
   * \param expr The expression containing random variables to be evaluated
   * \return The result of the evaluation
   */
  int Eval(const PrimExpr& expr);
  /**************** Sampling ****************/
  /*!
   * \brief Apply the instruction SampleTileFactor
   * \param n The number of loops after tiling
   * \param loop The loop to be tiled
   * \param where The distribution of tile size to be sampled
   * \return An array of random variables, the result of sampling
   */
  Array<tir::Var> SampleTileFactor(int n, LoopRV loop, Array<Integer> where);
  /**************** Block Relationship ****************/
  /*!
   * \brief Get the only consumer of a specific block
   * \param block The block to be queried
   * \return A block, its only consumer; or NullOpt if it does not exist
   */
  Optional<BlockRV> GetOnlyConsumer(const BlockRV& block);
  /**************** Scheduling Primitives ****************/
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
   * \brief Apply the instruction Split
   * \param loop The loop to be split
   * \param factors The split factors
   * \return An array of loop random variables
   * \note Splitting from inner to outer, and factors[0] is not used
   */
  Array<LoopRV> Split(const LoopRV& loop, const Array<PrimExpr>& factors);
  /*!
   * \brief Apply the instruction Reorder
   * \param after_axes The axes to be reordered
   */
  void Reorder(const Array<LoopRV>& after_axes);
  /*!
   * \brief Apply the instruction compute_inline
   * \param block The block to be computed inline
   */
  void ComputeInline(const BlockRV& block);
  /*!
   * \brief Apply the instruction cache_write
   * \param block The block to be buffered
   * \param storage_scope The storage scope
   * \return The cache write stage
   */
  BlockRV CacheWrite(const BlockRV& block, const String& storage_scope);
  /*!
   * \brief Apply the instruction DecomposeReduction
   * \param block The block to be decomposed
   * \param loop The loop to be decomposed at
   * \return The block random variable indicating the decomposition result
   */
  BlockRV DecomposeReduction(const BlockRV& block, const LoopRV& loop);
  /**************** Replay ****************/
  /*!
   * \brief Replay the trace to generate a new state of scheduling
   */
  void ReplayOnce();
};

class Schedule : public ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param orig_func The original TIR PrimFunc to be scheduled
   * \param sch The TIR schedule in the current stage
   * \param trace The trace of instructions used
   * \param sym_tab The symbol table with information of all defined variables in the meta schedule
   */
  explicit Schedule(tir::PrimFunc orig_func, tir::Schedule sch, Array<Instruction> trace,
                    SymbolTable sym_tab, Sampler sampler);
  /*!
   * \brief Constructor: other fields are created with default value
   * \param orig_func The original TIR PrimFunc to be scheduled
   */
  explicit Schedule(tir::PrimFunc orig_func);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Schedule, ObjectRef, ScheduleNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SCHEDULE_H_
