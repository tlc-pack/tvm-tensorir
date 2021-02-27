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
  using TSymbolTable = Map<ObjectRef, ObjectRef>;

 public:
  ScheduleState state;
  TSymbolTable symbol_table;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("state", &state);
    v->Visit("symbol_table", &symbol_table);
  }

  static constexpr const char* _type_key = "tir.Schedule";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleNode, Object);

 public:
  /*!
   * \brief Copy the schedule and guarantee that
   * 1) SRef tree is completely reconstructed
   * 2) Schedule function is untouched
   * 3) For all the random variables, they are valid in both original copy and the new copy, but
   * points to different StmtSRefs, because the SRef tree is reconstructed
   */
  virtual Schedule Copy() const;

 public:
  /******** Lookup random variables ********/
  Block Get(const BlockRV& block_rv) const;

  For Get(const LoopRV& loop_rv) const;

  int64_t Get(const Var& var_rv) const;

  int64_t Get(const ExprRV& expr_rv) const;

  StmtSRef GetSRef(const BlockRV& block_rv) const;

  StmtSRef GetSRef(const LoopRV& loop_rv) const;

 public:
  /******** Block/Loop relation ********/
  /*!
   * \brief Get block from its tag
   * \param tag The query tag
   * \return the block schedulable reference list
   */
  virtual Array<StmtSRef> GetBlock(const String& name) const;
  /*!
   * \brief Get loops of the block
   * \param block The query block
   * \return the loop sref list
   */
  virtual Array<StmtSRef> GetAxes(const StmtSRef& block) const;
  /*!
   * \brief Get the child blocks of a specific parent block/loop
   * \param parent_sref The StmtSRef that points to the parent block/loop
   * \return A list of child blocks
   */
  virtual Array<StmtSRef> GetChildBlocks(const StmtSRef& parent_sref) const;

  /******** Schedule: loops ********/
  /*!
   * \brief Fuse two consecutive loops of one computation.
   * \param outer_sref The outer loop
   * \param inner_sref The inner loop
   * \return The fused loop
   */
  virtual StmtSRef Fuse(const StmtSRef& outer_sref, const StmtSRef& inner_sref);
  /*!
   * \brief Split a specified loop into two loops by factor.
   * \param loop_sref The loop to be split
   * \param nparts The extent of the new outer loop
   * \param factor The extent of the new inner loop
   * \return The loops after splitting
   */
  virtual Array<StmtSRef> Split(const StmtSRef& loop_sref, const PrimExpr& nparts,
                                const PrimExpr& factor);
  /*!
   * \brief reorder a list of loops
   * \param order the order of loops
   */
  virtual void Reorder(const Array<StmtSRef>& order);

  /******** Schedule: compute location ********/
  /*!
   * \brief Move the block under the loop and regenerate the loops to cover the producing region.
   * \param block_sref The block to be moved
   * \param loop_sref The target loop
   * \param preserve_unit_loop Keep the trivial loops whose extent is 1
   */
  virtual void ComputeAt(const StmtSRef& block_sref, const StmtSRef& loop_sref,
                         bool preserve_unit_loop);
  /*!
   * \brief Move the block under the loop and regenerate the loops to cover the producing region.
   * \param block_sref The block to be moved
   * \param loop_sref The target loop
   * \param preserve_unit_loop Keep the trivial loops whose extent is 1
   */
  virtual void ReverseComputeAt(const StmtSRef& block_sref, const StmtSRef& loop_sref,
                                bool preserve_unit_loop);
  /*!
   * \brief Make the block inline
   * \param block_sref The sref of the block
   */
  virtual void ComputeInline(const StmtSRef& block_sref);
  /*!
   * \brief Make the block inline
   * \param block_sref The sref of block
   */
  virtual void ReverseComputeInline(const StmtSRef& block_sref);

  /******** Schedule: parallelize / annotate ********/
  /*!
   * \brief vectorize a loop
   * \param loop_sref the loop to be vectorized
   */
  virtual void Vectorize(const StmtSRef& loop_sref);
  /*!
   * \brief parallelize a loop
   * \param loop_sref the loop to be paralleled
   */
  virtual void Parallel(const StmtSRef& loop_sref);
  /*!
   * \brief unroll a loop
   * \param loop_sref the loop to be unrolled
   */
  virtual void Unroll(const StmtSRef& loop_sref);
  /*!
   * \brief parallel a loop
   * \param loop_sref the loop to be paralleled
   */
  virtual void Bind(const StmtSRef& loop_sref, const IterVar& thread);
  /*!
   * \brief add double_buffer annotation to a complete block
   * \param block_sref the block of interest
   */
  virtual void DoubleBuffer(const StmtSRef& block_sref);
  /*!
   * \brief add annotation to a loop
   * \param loop_sref the loop of interest
   * \param pragma_type the attribute key
   * \param pragma_value the attribute value
   */
  virtual void Pragma(const StmtSRef& loop_sref, const String& pragma_type,
                      const PrimExpr& pragma_value);

  /******** Schedule: cache read/write ********/
  /*!
   * \brief Create a cache read of original tensor for readers.
   * \param block_sref The consumer of the buffer
   * \param i The index of the buffer in block's read region
   * \param storage_scope The storage scope
   */
  virtual StmtSRef CacheRead(const StmtSRef& block_sref, int i, const String& storage_scope);
  /*!
   * \brief Create a cache write of original tensor, before storing into tensor.
   * \param block_sref The producer of the buffer
   * \param i The index of the buffer in block's write region
   * \param storage_scope The storage scope
   */
  virtual StmtSRef CacheWrite(const StmtSRef& block_sref, int i, const String& storage_scope);

  /******** Schedule: reduction ********/
  /*!
   * \brief rfactor a reduction block using loop
   * \param loop_sref the loop outside block we want to do rfactor
   * \param factor_axis the position where the new axis is placed
   * \return the sref of new block
   * TODO(@junrushao1994): do we need a concrete integer here?
   */
  virtual StmtSRef RFactor(const StmtSRef& loop_sref, int factor_axis);
  /*!
   * \brief Decompose reduction block_sref into init&update blocks
   * \param block_sref the reduction block_sref
   * \param loop_sref the position where init block_sref will be
   * \return the sref of init block
   */
  virtual StmtSRef DecomposeReduction(const StmtSRef& block_sref,
                                      const Optional<StmtSRef>& loop_sref);
  /*!
   * \brief Merge init and reduction block into reduction block
   * \param init_sref the init block
   * \param update_sref the update block
   */
  virtual void MergeReduction(const StmtSRef& init_sref, const StmtSRef& update_sref);

  /******** Blockize / Tensorize ********/
  /*!
   * \brief make subtree rooted by loop_sref into a block
   * \param loop_sref the subtree root
   * \return the loop_sref of new block
   */
  virtual StmtSRef Blockize(const StmtSRef& loop_sref, const String& exec_scope);
  /*!
   * \brief Tensorize the computation enclosed by loop with tensor_intrin
   * \param loop_sref the loop/block to be tensorized
   * \param intrinsic the tensor intrinsic
   */
  virtual void Tensorize(const StmtSRef& loop_sref, const TensorIntrin& intrinsic);
};

class Schedule : public runtime::ObjectRef {
  using TSymbolTable = ScheduleNode::TSymbolTable;

 public:
  /******** Constructors ********/
  /*!
   * \brief Constructor
   * \param state The schedule state
   * \param symbol_table The symbol table
   */
  TVM_DLL explicit Schedule(ScheduleState state, TSymbolTable symbol_table = {});
  /*!
   * \brief Constructor
   * \param func The func to be scheduled
   * \param debug_mode Use debug mode in scheduling
   */
  TVM_DLL explicit Schedule(PrimFunc func, bool debug_mode = false);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Schedule, runtime::ObjectRef, ScheduleNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_SCHEDULE_H_
