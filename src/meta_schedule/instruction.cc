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

BufferRV::BufferRV() { data_ = make_object<BufferRVNode>(); }

Instruction::Instruction(Array<ObjectRef> inputs, Array<ObjectRef> outputs, InstAttrs inst_attrs) {
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

template <class T>
Array<ObjectRef> AdaptOutputs(const Array<T>& outputs) {
  return {outputs.begin(), outputs.end()};
}

/**************** Instruction  ****************/

Array<ObjectRef> Instruction::ApplyToSchedule(ScheduleNode* sch, const InstAttrs& inst_attrs,
                                              const Array<ObjectRef>& inputs) {
  CHECK(inst_attrs.defined()) << "ValueError: `inst_attrs` is undefined";
  return inst_attrs->ApplyToSchedule(sch, inputs);
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
                     /*attrs=*/InstAttrs(std::move(n)));
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
                     /*attrs=*/InstAttrs(std::move(n)));
}

Array<ObjectRef> SampleTileFactorAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                        const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_CAST_INPUT(LoopRV, loop, inputs[0]);
  return AdaptOutputs(sch->SampleTileFactor(n_splits, loop, where));
}

Instruction SampleFusibleLoopsAttrs::MakeInst(const Array<LoopRV>& loops,
                                              const Array<Integer>& loop_types, int max_extent,
                                              bool include_overflow_loop, int order, int mode,
                                              const tir::Var& output) {
  ObjectPtr<SampleFusibleLoopsAttrs> n = make_object<SampleFusibleLoopsAttrs>();
  n->loop_types = loop_types;
  n->max_extent = max_extent;
  n->include_overflow_loop = include_overflow_loop;
  n->order = order;
  n->mode = mode;
  return Instruction(/*inputs=*/{loops.begin(), loops.end()},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Array<ObjectRef> SampleFusibleLoopsAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                          const Array<ObjectRef>& inputs) const {
  Array<LoopRV> loops;
  loops.reserve(inputs.size());
  for (int i = 0, n = inputs.size(); i < n; ++i) {
    TVM_META_SCHEDULE_CAST_INPUT(LoopRV, loop, inputs[i]);
    loops.push_back(loop);
  }
  ScheduleNode::Order the_order = static_cast<ScheduleNode::Order>(this->order);
  ScheduleNode::Mode the_mode = static_cast<ScheduleNode::Mode>(this->mode);
  return {sch->SampleFusibleLoops(loops, loop_types, max_extent, include_overflow_loop, the_order,
                                  the_mode)};
}

/**************** MakeInst/ApplyToSchedule: Block/Loop Relationship  ****************/

Instruction GetOnlyConsumerAttrs::MakeInst(const BlockRV& block, const BlockRV& output) {
  ObjectPtr<GetOnlyConsumerAttrs> n = make_object<GetOnlyConsumerAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
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
                     /*attrs=*/InstAttrs(std::move(n)));
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
                     /*attrs=*/InstAttrs(std::move(n)));
}

Array<ObjectRef> GetAxesAttrs::ApplyToSchedule(ScheduleNode* sch,
                                               const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_CAST_INPUT(BlockRV, block, inputs[0]);
  return AdaptOutputs(sch->GetAxes(block));
}

Instruction GetReadBuffersAttrs::MakeInst(const BlockRV& block, const Array<BufferRV>& outputs) {
  ObjectPtr<GetReadBuffersAttrs> n = make_object<GetReadBuffersAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Array<ObjectRef> GetReadBuffersAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                      const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_CAST_INPUT(BlockRV, block, inputs[0]);
  return AdaptOutputs(sch->GetReadBuffers(block));
}

Instruction GetWriteBuffersAttrs::MakeInst(const BlockRV& block, const Array<BufferRV>& outputs) {
  ObjectPtr<GetWriteBuffersAttrs> n = make_object<GetWriteBuffersAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Array<ObjectRef> GetWriteBuffersAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                       const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_CAST_INPUT(BlockRV, block, inputs[0]);
  return AdaptOutputs(sch->GetWriteBuffers(block));
}

Instruction GetRootBlocksAttrs::MakeInst(const Array<BlockRV>& outputs) {
  ObjectPtr<GetRootBlocksAttrs> n = make_object<GetRootBlocksAttrs>();
  return Instruction(/*inputs=*/{},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Array<ObjectRef> GetRootBlocksAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                     const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 0);
  return AdaptOutputs(sch->GetRootBlocks());
}

Instruction GetLeafBlocksAttrs::MakeInst(const Array<BlockRV>& outputs) {
  ObjectPtr<GetLeafBlocksAttrs> n = make_object<GetLeafBlocksAttrs>();
  return Instruction(/*inputs=*/{},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Array<ObjectRef> GetLeafBlocksAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                     const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 0);
  return AdaptOutputs(sch->GetLeafBlocks());
}

/**************** MakeInst/ApplyToSchedule: Scheduling Primitives  ****************/

Instruction MarkLoopTypeAttrs::MakeInst(const Array<LoopRV>& loops, const Range& range,
                                        const String& mark) {
  ObjectPtr<MarkLoopTypeAttrs> n = make_object<MarkLoopTypeAttrs>();
  n->mark = mark;
  Array<ObjectRef> inputs{loops.begin(), loops.end()};
  inputs.push_back(range->min);
  inputs.push_back(range->extent);
  return Instruction(/*inputs=*/inputs,
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Array<ObjectRef> MarkLoopTypeAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                    const Array<ObjectRef>& inputs) const {
  int n_loops = static_cast<int>(inputs.size()) - 2;
  Array<LoopRV> loops;
  loops.reserve(n_loops);
  for (int i = 0; i < n_loops; ++i) {
    TVM_META_SCHEDULE_CAST_INPUT(LoopRV, loop, inputs[i]);
    loops.push_back(loop);
  }
  TVM_META_SCHEDULE_CAST_INPUT(PrimExpr, min, inputs[n_loops]);
  TVM_META_SCHEDULE_CAST_INPUT(PrimExpr, extent, inputs[n_loops + 1]);
  sch->MarkLoopType(loops, mark, Range::FromMinExtent(min, extent));
  return {};
}

Instruction MarkBlockTypeAttrs::MakeInst(const BlockRV& block, const String& mark) {
  ObjectPtr<MarkBlockTypeAttrs> n = make_object<MarkBlockTypeAttrs>();
  n->mark = mark;
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Array<ObjectRef> MarkBlockTypeAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                     const Array<ObjectRef>& inputs) const {
  TVM_META_SCHEDULE_CAST_INPUT(BlockRV, block, inputs[0]);
  sch->MarkBlockType(block, mark);
  return {};
}

Instruction FuseAttrs::MakeInst(const Array<LoopRV>& loops, const LoopRV& output) {
  ObjectPtr<FuseAttrs> n = make_object<FuseAttrs>();
  return Instruction(/*inputs=*/{loops.begin(), loops.end()},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Array<ObjectRef> FuseAttrs::ApplyToSchedule(ScheduleNode* sch,
                                            const Array<ObjectRef>& inputs) const {
  int n_loops = inputs.size();
  Array<LoopRV> loops;
  loops.reserve(n_loops);
  for (int i = 0; i < n_loops; ++i) {
    TVM_META_SCHEDULE_CAST_INPUT(LoopRV, loop, inputs[i]);
    loops.push_back(loop);
  }
  return {sch->Fuse(loops)};
}

Instruction SplitAttrs::MakeInst(const LoopRV& loop, const Array<Optional<PrimExpr>>& factors,
                                 const Array<LoopRV>& outputs) {
  ObjectPtr<SplitAttrs> n = make_object<SplitAttrs>();
  Array<ObjectRef> inputs;
  inputs.reserve(1 + factors.size());
  inputs.push_back(loop);
  inputs.insert(inputs.end(), factors.begin(), factors.end());
  return Instruction(/*inputs=*/inputs,
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n)));
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
                     /*attrs=*/InstAttrs(std::move(n)));
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
                     /*attrs=*/InstAttrs(std::move(n)));
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
                     /*attrs=*/InstAttrs(std::move(n)));
}

Array<ObjectRef> ComputeInlineAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                     const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_CAST_INPUT(BlockRV, block, inputs[0]);
  sch->ComputeInline(block);
  return {};
}

Instruction ReverseComputeInlineAttrs::MakeInst(const BlockRV& block) {
  ObjectPtr<ReverseComputeInlineAttrs> n = make_object<ReverseComputeInlineAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Array<ObjectRef> ReverseComputeInlineAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                            const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_CAST_INPUT(BlockRV, block, inputs[0]);
  sch->ReverseComputeInline(block);
  return {};
}

Instruction CacheReadAttrs::MakeInst(const BufferRV& buffer, const String& storage_scope,
                                     const BlockRV& output) {
  ObjectPtr<CacheReadAttrs> n = make_object<CacheReadAttrs>();
  n->storage_scope = storage_scope;
  return Instruction(/*inputs=*/{buffer},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Array<ObjectRef> CacheReadAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                 const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_CAST_INPUT(BufferRV, buffer, inputs[0]);
  return {sch->CacheRead(buffer, storage_scope)};
}

Instruction CacheWriteAttrs::MakeInst(const BufferRV& buffer, const String& storage_scope,
                                      const BlockRV& output) {
  ObjectPtr<CacheWriteAttrs> n = make_object<CacheWriteAttrs>();
  n->storage_scope = storage_scope;
  return Instruction(/*inputs=*/{buffer},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Array<ObjectRef> CacheWriteAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                  const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_CAST_INPUT(BufferRV, buffer, inputs[0]);
  return {sch->CacheWrite(buffer, storage_scope)};
}

Instruction BlockizeAttrs::MakeInst(const LoopRV& loop, const String& exec_scope,
                                    const BlockRV& output) {
  ObjectPtr<BlockizeAttrs> n = make_object<BlockizeAttrs>();
  n->exec_scope = exec_scope;
  return Instruction(/*inputs=*/{loop},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Array<ObjectRef> BlockizeAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_CAST_INPUT(LoopRV, loop, inputs[0]);
  return {sch->Blockize(loop, exec_scope)};
}

Instruction DecomposeReductionAttrs::MakeInst(const BlockRV& block, const LoopRV& loop,
                                              const BlockRV& output) {
  ObjectPtr<DecomposeReductionAttrs> n = make_object<DecomposeReductionAttrs>();
  return Instruction(/*inputs=*/{block, loop},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
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
TVM_REGISTER_NODE_TYPE(BufferRVNode);
TVM_REGISTER_OBJECT_TYPE(InstAttrsNode);
TVM_REGISTER_NODE_TYPE(InstructionNode);
TVM_REGISTER_NODE_TYPE(SamplePerfectTileAttrs);
TVM_REGISTER_NODE_TYPE(SampleTileFactorAttrs);
TVM_REGISTER_NODE_TYPE(SampleFusibleLoopsAttrs);
TVM_REGISTER_NODE_TYPE(GetOnlyConsumerAttrs);
TVM_REGISTER_NODE_TYPE(GetBlockAttrs);
TVM_REGISTER_NODE_TYPE(GetAxesAttrs);
TVM_REGISTER_NODE_TYPE(GetReadBuffersAttrs);
TVM_REGISTER_NODE_TYPE(GetWriteBuffersAttrs);
TVM_REGISTER_NODE_TYPE(GetRootBlocksAttrs);
TVM_REGISTER_NODE_TYPE(GetLeafBlocksAttrs);
TVM_REGISTER_NODE_TYPE(MarkLoopTypeAttrs);
TVM_REGISTER_NODE_TYPE(MarkBlockTypeAttrs);
TVM_REGISTER_NODE_TYPE(FuseAttrs);
TVM_REGISTER_NODE_TYPE(SplitAttrs);
TVM_REGISTER_NODE_TYPE(ReorderAttrs);
TVM_REGISTER_NODE_TYPE(ReverseComputeAtAttrs);
TVM_REGISTER_NODE_TYPE(ComputeInlineAttrs);
TVM_REGISTER_NODE_TYPE(ReverseComputeInlineAttrs);
TVM_REGISTER_NODE_TYPE(CacheReadAttrs);
TVM_REGISTER_NODE_TYPE(CacheWriteAttrs);
TVM_REGISTER_NODE_TYPE(BlockizeAttrs);
TVM_REGISTER_NODE_TYPE(DecomposeReductionAttrs);

#undef TVM_META_SCHEDULE_CAST_INPUT

}  // namespace meta_schedule
}  // namespace tvm
