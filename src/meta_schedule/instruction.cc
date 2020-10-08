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
#include "./instruction.h"  // NOLINT(build/include)

#include "./schedule.h"

namespace tvm {
namespace meta_schedule {

/**************** Constructors ****************/

BlockRV::BlockRV() { data_ = make_object<BlockRVNode>(); }

LoopRV::LoopRV() { data_ = make_object<LoopRVNode>(); }

Instruction::Instruction(Array<ObjectRef> inputs, Array<ObjectRef> outputs, Attrs inst_attrs) {
  ObjectPtr<InstructionNode> n = make_object<InstructionNode>();
  n->inputs = std::move(inputs);
  n->outputs = std::move(outputs);
  n->inst_attrs = std::move(inst_attrs);
  data_ = std::move(n);
}

/**************** Utilities ****************/

#define TVM_META_SCHEDULE_CAST_INPUT(CastType, VarName, Input)                   \
  CHECK(Input->IsInstance<CastType::ContainerType>())                            \
      << "TypeError: Cannot downcast to '" << CastType::ContainerType::_type_key \
      << "' from: " << Input->GetTypeKey();                                      \
  CastType VarName = Downcast<CastType>(Input);

#define TVM_META_SCHEDULE_APPLY_INST(Schedule, Attrs, Inputs, AttrType) \
  if (const auto* n = Attrs.as<AttrType>()) {                           \
    return n->ApplyToSchedule(Schedule, Inputs);                        \
  }

template <class T>
Array<ObjectRef> AdaptOutputs(const Array<T>& outputs) {
  return {outputs.begin(), outputs.end()};
}

/**************** Instruction  ****************/

Array<ObjectRef> Instruction::ApplyToSchedule(ScheduleNode* sch, const Attrs& inst_attrs,
                                              const Array<ObjectRef>& inputs) {
  // TODO(@junrushao1994): dispatch using a vtable
  CHECK(inst_attrs.defined()) << "ValueError: `inst_attrs` is undefined";
  TVM_META_SCHEDULE_APPLY_INST(sch, inst_attrs, inputs, SamplePerfectTileAttrs);
  TVM_META_SCHEDULE_APPLY_INST(sch, inst_attrs, inputs, SampleTileFactorAttrs);
  TVM_META_SCHEDULE_APPLY_INST(sch, inst_attrs, inputs, GetOnlyConsumerAttrs);
  TVM_META_SCHEDULE_APPLY_INST(sch, inst_attrs, inputs, GetBlockAttrs);
  TVM_META_SCHEDULE_APPLY_INST(sch, inst_attrs, inputs, GetAxesAttrs);
  TVM_META_SCHEDULE_APPLY_INST(sch, inst_attrs, inputs, SplitAttrs);
  TVM_META_SCHEDULE_APPLY_INST(sch, inst_attrs, inputs, ReorderAttrs);
  TVM_META_SCHEDULE_APPLY_INST(sch, inst_attrs, inputs, ReverseComputeAtAttrs);
  TVM_META_SCHEDULE_APPLY_INST(sch, inst_attrs, inputs, ComputeInlineAttrs);
  TVM_META_SCHEDULE_APPLY_INST(sch, inst_attrs, inputs, CacheWriteAttrs);
  TVM_META_SCHEDULE_APPLY_INST(sch, inst_attrs, inputs, DecomposeReductionAttrs);
  LOG(FATAL) << "TypeError: Cannot recognize instruction attribute: " << inst_attrs->GetTypeKey();
  throw;
}

/**************** MakeInst/ApplyToSchedule: Sampling  ****************/

Instruction SamplePerfectTileAttrs::MakeInst(int n_splits, const LoopRV& loop,
                                             int max_innermost_factor,
                                             const Array<tir::Var>& outputs) {
  ObjectPtr<SamplePerfectTileAttrs> n = make_object<SamplePerfectTileAttrs>();
  n->n_splits = n_splits;
  n->max_innermost_factor = max_innermost_factor;
  return Instruction(/*inputs=*/{loop},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/Attrs(std::move(n)));
}

Array<ObjectRef> SamplePerfectTileAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                         const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_CAST_INPUT(LoopRV, loop, inputs[0]);
  return AdaptOutputs(sch->SamplePerfectTile(n_splits, loop, max_innermost_factor));
}

Instruction SampleTileFactorAttrs::MakeInst(int n_splits, const LoopRV& loop,
                                            const Array<Integer>& where,
                                            const Array<tir::Var>& outputs) {
  ObjectPtr<SampleTileFactorAttrs> n = make_object<SampleTileFactorAttrs>();
  n->n_splits = n_splits;
  n->where = where;
  return Instruction(/*inputs=*/{loop},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/Attrs(std::move(n)));
}

Array<ObjectRef> SampleTileFactorAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                        const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_CAST_INPUT(LoopRV, loop, inputs[0]);
  return AdaptOutputs(sch->SampleTileFactor(n_splits, loop, where));
}

/**************** MakeInst/ApplyToSchedule: Block/Loop Relationship  ****************/

Instruction GetOnlyConsumerAttrs::MakeInst(const BlockRV& block, const BlockRV& output) {
  ObjectPtr<GetOnlyConsumerAttrs> n = make_object<GetOnlyConsumerAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{output},
                     /*attrs=*/Attrs(std::move(n)));
}

Array<ObjectRef> GetOnlyConsumerAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                       const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_CAST_INPUT(BlockRV, block, inputs[0]);
  return {sch->GetOnlyConsumer(block)};
}

Instruction GetBlockAttrs::MakeInst(const String& name, const BlockRV& output) {
  ObjectPtr<GetBlockAttrs> n = make_object<GetBlockAttrs>();
  n->name = name;
  return Instruction(/*inputs=*/{},
                     /*outputs=*/{output},
                     /*attrs=*/Attrs(std::move(n)));
}

Array<ObjectRef> GetBlockAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 0);
  return {sch->GetBlock(name)};
}

Instruction GetAxesAttrs::MakeInst(const BlockRV& block, const Array<LoopRV>& outputs) {
  ObjectPtr<GetAxesAttrs> n = make_object<GetAxesAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/Attrs(std::move(n)));
}

Array<ObjectRef> GetAxesAttrs::ApplyToSchedule(ScheduleNode* sch,
                                               const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_CAST_INPUT(BlockRV, block, inputs[0]);
  return AdaptOutputs(sch->GetAxes(block));
}

/**************** MakeInst/ApplyToSchedule: Scheduling Primitives  ****************/

Instruction SplitAttrs::MakeInst(const LoopRV& loop, const Array<Optional<PrimExpr>>& factors,
                                 const Array<LoopRV>& outputs) {
  ObjectPtr<SplitAttrs> n = make_object<SplitAttrs>();
  Array<ObjectRef> inputs;
  inputs.reserve(1 + factors.size());
  inputs.push_back(loop);
  inputs.insert(inputs.end(), factors.begin(), factors.end());
  return Instruction(/*inputs=*/inputs,
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/Attrs(std::move(n)));
}

Array<ObjectRef> SplitAttrs::ApplyToSchedule(ScheduleNode* sch,
                                             const Array<ObjectRef>& inputs) const {
  CHECK_GE(inputs.size(), 3);
  TVM_META_SCHEDULE_CAST_INPUT(LoopRV, loop, inputs[0]);
  Array<Optional<PrimExpr>> factors;
  for (int i = 1, n = inputs.size(); i < n; ++i) {
    TVM_META_SCHEDULE_CAST_INPUT(PrimExpr, factor, inputs[i]);
    factors.push_back(factor);
  }
  return AdaptOutputs(sch->Split(loop, factors));
}

Instruction ReorderAttrs::MakeInst(const Array<LoopRV>& after_axes) {
  ObjectPtr<ReorderAttrs> n = make_object<ReorderAttrs>();
  return Instruction(/*inputs=*/{after_axes.begin(), after_axes.end()},
                     /*outputs=*/{},
                     /*attrs=*/Attrs(std::move(n)));
}

Array<ObjectRef> ReorderAttrs::ApplyToSchedule(ScheduleNode* sch,
                                               const Array<ObjectRef>& inputs) const {
  Array<LoopRV> after_axes;
  for (const ObjectRef& obj : inputs) {
    if (const auto* loop = obj.as<LoopRVNode>()) {
      after_axes.push_back(GetRef<LoopRV>(loop));
    } else {
      LOG(FATAL) << "TypeError: Expects LoopRV, but gets: " << obj->GetTypeKey();
    }
  }
  sch->Reorder(after_axes);
  return {};
}

Instruction ReverseComputeAtAttrs::MakeInst(const BlockRV& block, const LoopRV& loop) {
  ObjectPtr<ReverseComputeAtAttrs> n = make_object<ReverseComputeAtAttrs>();
  return Instruction(/*inputs=*/{block, loop},
                     /*outputs=*/{},
                     /*attrs=*/Attrs(std::move(n)));
}

Array<ObjectRef> ReverseComputeAtAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                        const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 2);
  TVM_META_SCHEDULE_CAST_INPUT(BlockRV, block, inputs[0]);
  TVM_META_SCHEDULE_CAST_INPUT(LoopRV, loop, inputs[1]);
  sch->ReverseComputeAt(block, loop);
  return {};
}

Instruction ComputeInlineAttrs::MakeInst(const BlockRV& block) {
  ObjectPtr<ComputeInlineAttrs> n = make_object<ComputeInlineAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{},
                     /*attrs=*/Attrs(std::move(n)));
}

Array<ObjectRef> ComputeInlineAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                     const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_CAST_INPUT(BlockRV, block, inputs[0]);
  sch->ComputeInline(block);
  return {};
}

Instruction CacheWriteAttrs::MakeInst(const BlockRV& block, const String& storage_scope,
                                      const BlockRV& output) {
  ObjectPtr<CacheWriteAttrs> n = make_object<CacheWriteAttrs>();
  n->storage_scope = storage_scope;
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{output},
                     /*attrs=*/Attrs(std::move(n)));
}

Array<ObjectRef> CacheWriteAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                  const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_CAST_INPUT(BlockRV, block, inputs[0]);
  return {sch->CacheWrite(block, storage_scope)};
}

Instruction DecomposeReductionAttrs::MakeInst(const BlockRV& block, const LoopRV& loop,
                                              const BlockRV& output) {
  ObjectPtr<DecomposeReductionAttrs> n = make_object<DecomposeReductionAttrs>();
  return Instruction(/*inputs=*/{block, loop},
                     /*outputs=*/{output},
                     /*attrs=*/Attrs(std::move(n)));
}

Array<ObjectRef> DecomposeReductionAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                          const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 2);
  TVM_META_SCHEDULE_CAST_INPUT(BlockRV, block, inputs[0]);
  TVM_META_SCHEDULE_CAST_INPUT(LoopRV, loop, inputs[1]);
  return {sch->DecomposeReduction(block, loop)};
}

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(BlockRVNode);
TVM_REGISTER_NODE_TYPE(LoopRVNode);
TVM_REGISTER_NODE_TYPE(InstructionNode);
TVM_REGISTER_NODE_TYPE(SamplePerfectTileAttrs);
TVM_REGISTER_NODE_TYPE(SampleTileFactorAttrs);
TVM_REGISTER_NODE_TYPE(GetBlockAttrs);
TVM_REGISTER_NODE_TYPE(GetAxesAttrs);
TVM_REGISTER_NODE_TYPE(SplitAttrs);
TVM_REGISTER_NODE_TYPE(ReorderAttrs);
TVM_REGISTER_NODE_TYPE(ReverseComputeAtAttrs);
TVM_REGISTER_NODE_TYPE(ComputeInlineAttrs);
TVM_REGISTER_NODE_TYPE(CacheWriteAttrs);
TVM_REGISTER_NODE_TYPE(DecomposeReductionAttrs);
TVM_REGISTER_NODE_TYPE(GetOnlyConsumerAttrs);

#undef TVM_META_SCHEDULE_CAST_INPUT
#undef TVM_META_SCHEDULE_APPLY_INST

}  // namespace meta_schedule
}  // namespace tvm
