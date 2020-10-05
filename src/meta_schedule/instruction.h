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
#ifndef SRC_META_SCHEDULE_INSTRUCTION_H_
#define SRC_META_SCHEDULE_INSTRUCTION_H_

#include <tvm/ir/attrs.h>
#include <tvm/tir/var.h>

namespace tvm {
namespace meta_schedule {

class ScheduleNode;

/**************** Random variables ****************/

/*! \brief A random variable that evaluates to a TIR block */
class BlockRVNode : public runtime::Object {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}
  static constexpr const char* _type_key = "meta_schedule.BlockRV";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockRVNode, Object);
};

/*!
 * \brief Managed reference to BlockRVNode
 * \sa BlockRVNode
 */
class BlockRV : public runtime::ObjectRef {
 public:
  /*! \brief Constructor */
  BlockRV();
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BlockRV, ObjectRef, BlockRVNode);
};

/*! \brief A random variable that evaluates to a TIR loop axis */
class LoopRVNode : public runtime::Object {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}
  static constexpr const char* _type_key = "meta_schedule.LoopRV";
  TVM_DECLARE_FINAL_OBJECT_INFO(LoopRVNode, Object);
};

/*!
 * \brief Managed reference to LoopRVNode
 * \sa LoopRVNode
 */
class LoopRV : public runtime::ObjectRef {
 public:
  /*! \brief Constructor */
  LoopRV();
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(LoopRV, ObjectRef, LoopRVNode);
};

/**************** Instruction ****************/

/*! \brief Base class for all meta scheduling instrructions */
class InstructionNode : public Object {
 public:
  /*! \brief The input random variables it consumers */
  Array<ObjectRef> inputs;
  /*! \brief The output random variables it produces */
  Array<ObjectRef> outputs;
  /*! \brief The attributes of the instruction */
  Attrs inst_attrs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("inputs", &inputs);
    v->Visit("outputs", &outputs);
    v->Visit("inst_attrs", &inst_attrs);
  }

  static constexpr const char* _type_key = "meta_schedule.Instruction";
  TVM_DECLARE_FINAL_OBJECT_INFO(InstructionNode, Object);
};

/*!
 * \brief Managed reference to InstructionNode
 * \sa InstructionNode
 */
class Instruction : public ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param inputs The input random variables it consumers
   * \param outputs The output random variables it produces
   * \param attrs The attributes of the instruction
   */
  explicit Instruction(Array<ObjectRef> inputs, Array<ObjectRef> outputs, Attrs attrs);

  /*!
   * \brief Apply an instruction to the specific schedule, and return the outputs
   * \param sch The schedule to be applied
   * \param inst_attrs Attributes of the instruction
   * \param inputs The inputs to the instruction
   * \return The outputs of the instruction applied
   */
  static Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Attrs& inst_attrs,
                                          const Array<ObjectRef>& inputs);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Instruction, ObjectRef, InstructionNode);

 protected:
  /*! \brief Constructor. The node should never be constructed directly. */
  Instruction() = default;
};

/**************** Sampling ****************/

/*! \brief Attrs of the instruction to sample perfect tile factors */
struct SamplePerfectTileAttrs : public tvm::AttrsNode<SamplePerfectTileAttrs> {
  /*! \brief The number of loops after tiling */
  int n_splits;
  /*! \brief The maximum factor in the innermost loop */
  int max_innermost_factor;

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param n_splits The number of loops after tiling
   * \param loop The loop to be tiled
   * \param max_innermost_factor The maximum factor in the innermost loop
   * \param outputs Outputs of the instruction
   * \return The instruction created
   */
  static Instruction MakeInst(int n_splits, const LoopRV& loop, int max_innermost_factor,
                              const Array<tir::Var>& outputs);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;

  TVM_DECLARE_ATTRS(SamplePerfectTileAttrs, "meta_schedule.attrs.SamplePerfectTileAttrs") {
    TVM_ATTR_FIELD(n_splits);
    TVM_ATTR_FIELD(max_innermost_factor);
  }
};

/*! \brief Attrs of the instruction to sample tiling factors */
struct SampleTileFactorAttrs : public tvm::AttrsNode<SampleTileFactorAttrs> {
  /*! \brief The number of loops after tiling */
  int n_splits;
  /*! \brief The distribution to be sampled from */
  Array<Integer> where;

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param n_splits The number of loops after tiling
   * \param loop The loop to be tiled
   * \param where The distribution to be sampled from
   * \param outputs Outputs of the instruction
   * \return The instruction created
   */
  static Instruction MakeInst(int n_splits, const LoopRV& loop, const Array<Integer>& where,
                              const Array<tir::Var>& outputs);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;

  TVM_DECLARE_ATTRS(SampleTileFactorAttrs, "meta_schedule.attrs.SampleTileFactorAttrs") {
    TVM_ATTR_FIELD(n_splits);
    TVM_ATTR_FIELD(where);
  }
};

/**************** Block/Loop Relationship ****************/

/*! \brief Attrs of the instruction that gets the only consumer of a specific block */
struct GetOnlyConsumerAttrs : public tvm::AttrsNode<GetOnlyConsumerAttrs> {
  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The block to be queried
   * \param output The output of the query
   * \return The instruction created
   */
  static Instruction MakeInst(const BlockRV& block, const BlockRV& output);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;

  TVM_DECLARE_ATTRS(GetOnlyConsumerAttrs, "meta_schedule.attrs.GetOnlyConsumerAttrs") {}
};

/*! \brief Attrs of the instruction that gets a specific block by its name */
struct GetBlockAttrs : public tvm::AttrsNode<GetBlockAttrs> {
  /*! \brief The name of the block */
  String name;

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param name The name of the block
   * \param output The output of the query
   * \return The instruction created
   */
  static Instruction MakeInst(const String& name, const BlockRV& output);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;

  TVM_DECLARE_ATTRS(GetBlockAttrs, "meta_schedule.attrs.GetBlockAttrs") { TVM_ATTR_FIELD(name); }
};

/*! \brief Attrs of the instruction that gets loop axes on top of a specifc block */
struct GetAxesAttrs : public tvm::AttrsNode<GetAxesAttrs> {
  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The name of the block
   * \param outputs The outputs of the query
   * \return The instruction created
   */
  static Instruction MakeInst(const BlockRV& block, const Array<LoopRV>& outputs);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;

  TVM_DECLARE_ATTRS(GetAxesAttrs, "meta_schedule.attrs.GetAxesAttrs") {}
};

/**************** Scheduling Primitives ****************/

/*! \brief Attrs of the instruction that applies loop splitting */
struct SplitAttrs : public tvm::AttrsNode<SplitAttrs> {
  /*!
   * \brief Create instruction given the inputs and outputs
   * \param loop The loop to be split
   * \param factors Thee splitting factors
   * \param outputs The outputs of the query
   * \return The instruction created
   */
  static Instruction MakeInst(const LoopRV& loop, const Array<PrimExpr>& factors,
                              const Array<LoopRV>& outputs);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;

  TVM_DECLARE_ATTRS(SplitAttrs, "meta_schedule.attrs.SplitAttrs") {}
};

/*! \brief Attrs of the instruction that applies loop reordering */
struct ReorderAttrs : public tvm::AttrsNode<ReorderAttrs> {
  /*!
   * \brief Create instruction given the inputs and outputs
   * \param after_axes The axes to be reordered
   * \return The instruction created
   */
  static Instruction MakeInst(const Array<LoopRV>& after_axes);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;

  TVM_DECLARE_ATTRS(ReorderAttrs, "meta_schedule.attrs.ReorderAttrs") {}
};

/*! \brief Attrs of the instruction that applies compute_inline */
struct ComputeInlineAttrs : public tvm::AttrsNode<ComputeInlineAttrs> {
  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The block to be computed inline
   * \return The instruction created
   */
  static Instruction MakeInst(const BlockRV& block);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;

  TVM_DECLARE_ATTRS(ComputeInlineAttrs, "meta_schedule.attrs.ComputeInlineAttrs") {}
};

/*! \brief Attrs of the instruction that applies cache_write */
struct CacheWriteAttrs : public tvm::AttrsNode<CacheWriteAttrs> {
  /*! \brief The storage scope of the instruction cache_write */
  String storage_scope;

  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The block to be cache written
   * \param storage_scope The storage scope of the instruction
   * \param output The output of the instruction
   * \return The instruction created
   */
  static Instruction MakeInst(const BlockRV& block, const String& storage_scope,
                              const BlockRV& output);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;

  TVM_DECLARE_ATTRS(CacheWriteAttrs, "meta_schedule.attrs.CacheWriteAttrs") {
    TVM_ATTR_FIELD(storage_scope);
  }
};

/*! \brief Attrs of the instruction that applies decompose_reduction */
struct DecomposeReductionAttrs : public tvm::AttrsNode<DecomposeReductionAttrs> {
  /*!
   * \brief Create instruction given the inputs and outputs
   * \param block The reduction block to be decomposed
   * \param loop The loop to be decomposed at
   * \param output The output of the instruction
   * \return The instruction created
   */
  static Instruction MakeInst(const BlockRV& block, const LoopRV& loop, const BlockRV& output);

  /*!
   * \brief Apply the instruction to the schedule with given inputs
   * \param sch The schedule to be applied
   * \param inputs The input of the instruction
   * \return Outputs of the instruction
   */
  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;

  TVM_DECLARE_ATTRS(DecomposeReductionAttrs, "meta_schedule.attrs.DecomposeReductionAttrs") {}
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_INSTRUCTION_H_
