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

#include "./random_variable.h"

namespace tvm {
namespace meta_schedule {

class ScheduleNode;

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

  static Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Attrs& inst_attrs,
                                          const Array<ObjectRef>& inputs);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Instruction, ObjectRef, InstructionNode);

 protected:
  /*! \brief Constructor. The node should never be constructed directly. */
  Instruction() = default;
};

/**************** Sampling ****************/

struct SamplePerfectTileAttrs : public tvm::AttrsNode<SamplePerfectTileAttrs> {
  int n_splits;
  int max_innermost_factor;

  TVM_DECLARE_ATTRS(SamplePerfectTileAttrs, "meta_schedule.attrs.SamplePerfectTileAttrs") {
    TVM_ATTR_FIELD(n_splits);
    TVM_ATTR_FIELD(max_innermost_factor);
  }

  static Instruction MakeInst(int n_splits, const LoopRV& loop, int max_innermost_factor,
                              const Array<tir::Var>& outputs);

  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;
};

struct SampleTileFactorAttrs : public tvm::AttrsNode<SampleTileFactorAttrs> {
  int n_splits;
  Array<Integer> where;

  TVM_DECLARE_ATTRS(SampleTileFactorAttrs, "meta_schedule.attrs.SampleTileFactorAttrs") {
    TVM_ATTR_FIELD(n_splits);
    TVM_ATTR_FIELD(where);
  }

  static Instruction MakeInst(int n_splits, const LoopRV& loop, const Array<Integer>& where,
                              const Array<tir::Var>& outputs);

  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;
};

/**************** Block/Loop Relationship ****************/

struct GetOnlyConsumerAttrs : public tvm::AttrsNode<GetOnlyConsumerAttrs> {
  TVM_DECLARE_ATTRS(GetOnlyConsumerAttrs, "meta_schedule.attrs.GetOnlyConsumerAttrs") {}

  static Instruction MakeInst(const BlockRV& block, const BlockRV& output);

  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;
};

struct GetBlockAttrs : public tvm::AttrsNode<GetBlockAttrs> {
  String name;
  TVM_DECLARE_ATTRS(GetBlockAttrs, "meta_schedule.attrs.GetBlockAttrs") { TVM_ATTR_FIELD(name); }

  static Instruction MakeInst(const String& name, const BlockRV& output);

  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;
};

struct GetAxesAttrs : public tvm::AttrsNode<GetAxesAttrs> {
  TVM_DECLARE_ATTRS(GetAxesAttrs, "meta_schedule.attrs.GetAxesAttrs") {}

  static Instruction MakeInst(const BlockRV& block, const Array<LoopRV>& outputs);

  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;
};

/**************** Scheduling Primitives ****************/

struct SplitAttrs : public tvm::AttrsNode<SplitAttrs> {
  TVM_DECLARE_ATTRS(SplitAttrs, "meta_schedule.attrs.SplitAttrs") {}

  static Instruction MakeInst(const LoopRV& loop, const Array<PrimExpr>& factors,
                              const Array<LoopRV>& outputs);

  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;
};

struct ReorderAttrs : public tvm::AttrsNode<ReorderAttrs> {
  TVM_DECLARE_ATTRS(ReorderAttrs, "meta_schedule.attrs.ReorderAttrs") {}

  static Instruction MakeInst(const Array<LoopRV>& after_axes);

  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;
};

struct ComputeInlineAttrs : public tvm::AttrsNode<ComputeInlineAttrs> {
  TVM_DECLARE_ATTRS(ComputeInlineAttrs, "meta_schedule.attrs.ComputeInlineAttrs") {}

  static Instruction MakeInst(const BlockRV& block);

  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;
};

struct CacheWriteAttrs : public tvm::AttrsNode<CacheWriteAttrs> {
  String storage_scope;
  TVM_DECLARE_ATTRS(CacheWriteAttrs, "meta_schedule.attrs.CacheWriteAttrs") {
    TVM_ATTR_FIELD(storage_scope);
  }

  static Instruction MakeInst(const BlockRV& block, const String& storage_scope,
                              const BlockRV& output);

  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;
};

struct DecomposeReductionAttrs : public tvm::AttrsNode<DecomposeReductionAttrs> {
  TVM_DECLARE_ATTRS(DecomposeReductionAttrs, "meta_schedule.attrs.DecomposeReductionAttrs") {}

  static Instruction MakeInst(const BlockRV& block, const LoopRV& loop, const BlockRV& output);

  Array<ObjectRef> ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs) const;
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_INSTRUCTION_H_
