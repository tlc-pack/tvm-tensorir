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

/**************** Utilities ****************/

#define TVM_META_SCHEDULE_INST_CAST(CastType, VarName, Input)                    \
  CHECK(Input->IsInstance<CastType::ContainerType>())                            \
      << "TypeError: Cannot downcast to '" << CastType::ContainerType::_type_key \
      << "' from: " << Input->GetTypeKey();                                      \
  CastType VarName = Downcast<CastType>(Input);

template <class T>
Array<ObjectRef> AdaptOutputs(const Array<T>& outputs) {
  return {outputs.begin(), outputs.end()};
}

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

/**************** Instruction  ****************/

Array<ObjectRef> InstructionNode::Export(const Map<ObjectRef, String>& rv_names,
                                         const Optional<Array<ObjectRef>>& decision) const {
  Array<ObjectRef> record;
  record.reserve(4);
  // record[0]: inst_attrs::_name
  record.push_back(inst_attrs->GetName());
  // record[1]: inputs
  // record[2]: outputs
  for (const Array<ObjectRef>& rvs : {this->inputs, this->outputs}) {
    Array<ObjectRef> names;
    names.reserve(rvs.size());
    for (const ObjectRef& rv : rvs) {
      if (const auto* integer = rv.as<IntImmNode>()) {
        names.push_back(GetRef<IntImm>(integer));
      } else if (rv_names.count(rv)) {
        names.push_back(rv_names.at(rv));
      } else {
        LOG(INFO) << "TypeError: Unable to handle: " << rv << ". Its type is: " << rv->GetTypeKey();
        throw;
      }
    }
    record.push_back(names);
  }
  // record[3]: (optional) inst_attrs
  // record[4]: (optional) decision
  inst_attrs->Export(&record, decision);
  return record;
}

Array<ObjectRef> Instruction::ApplyToSchedule(ScheduleNode* sch, const InstAttrs& inst_attrs,
                                              const Array<ObjectRef>& inputs) {
  return inst_attrs->ApplyToSchedule(sch, inputs);
}

Array<ObjectRef> Instruction::ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& record,
                                              Map<String, ObjectRef>* named_rvs) {
#define TVM_META_SCHEDULE_INST_VTABLE_ENTRY(AttrsTyppe) \
  { AttrsTyppe::Name(), AttrsTyppe::Import }
  static const std::unordered_map<String, std::function<InstAttrs(const Array<ObjectRef>&)>>
      vtable = {
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(SamplePerfectTileAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(SampleTileFactorAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(SampleFusibleLoopsAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(GetOnlyConsumerAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(GetBlockAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(GetAxesAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(GetReadBuffersAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(GetWriteBuffersAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(GetRootBlocksAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(GetLeafBlocksAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(MarkLoopTypeAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(MarkBlockTypeAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(FuseAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(SplitAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(ReorderAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(ComputeAtAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(ReverseComputeAtAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(ComputeInlineAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(ReverseComputeInlineAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(CacheReadAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(CacheWriteAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(BlockizeAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(DecomposeReductionAttrs),
      };
#undef TVM_META_SCHEDULE_INST_VTABLE_ENTRY
  CHECK_GE(record.size(), 3);
  CHECK_LE(record.size(), 5);
  // Extract record[0]: inst_attrs::_name
  String attrs_name = Downcast<String>(record[0]);
  // Extract record[1]: inputs
  Array<ObjectRef> inputs;
  {
    Array<ObjectRef> record_inputs = Downcast<Array<ObjectRef>>(record[1]);
    inputs.reserve(record_inputs.size());
    for (const ObjectRef& obj : record_inputs) {
      if (const auto* integer = obj.as<IntImmNode>()) {
        inputs.push_back(GetRef<Integer>(integer));
      } else if (const auto* str = obj.as<StringObj>()) {
        inputs.push_back(named_rvs->at(GetRef<String>(str)));
      } else {
        LOG(FATAL) << "TypeError: Cannot deal with type '" << obj->GetTypeKey()
                   << "' for input: " << obj;
      }
    }
  }
  // Extract record[2]: outputs
  Array<String> record_outputs = Downcast<Array<String>>(record[2]);
  // Extract record[3]: (optional) inst_attrs
  InstAttrs inst_attrs = vtable.at(attrs_name)(record);
  // Extract record[4]: (optional) decision
  Optional<Array<ObjectRef>> opt_decision = record.size() >= 5
                                                ? Downcast<Array<ObjectRef>>(record[4])
                                                : Optional<Array<ObjectRef>>(NullOpt);
  // Get the new output random variables
  Array<ObjectRef> outputs = inst_attrs->ApplyToSchedule(sch, inputs);
  {
    // link `record_outputs` and `outputs`
    CHECK_EQ(record_outputs.size(), outputs.size());
    int n = record_outputs.size();
    for (int i = 0; i < n; ++i) {
      named_rvs->Set(record_outputs[i], outputs[i]);
    }
  }
  if (opt_decision.defined()) {
    Array<ObjectRef> decision = opt_decision.value();
    CHECK_EQ(decision.size(), outputs.size());
    int n = decision.size();
    for (int i = 0; i < n; ++i) {
      sch->sym_tab.Set(outputs[i], decision[i]);
    }
    sch->decisions.Set(sch->trace.back(), decision);
  }
  return outputs;
}

/**************** MakeInst  ****************/
/**************** (MakeInst) Sampling  ****************/

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

/**************** (MakeInst) Block/Loop Relationship  ****************/

Instruction GetOnlyConsumerAttrs::MakeInst(const BlockRV& block, const BlockRV& output) {
  ObjectPtr<GetOnlyConsumerAttrs> n = make_object<GetOnlyConsumerAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction GetBlockAttrs::MakeInst(const String& name, const BlockRV& output) {
  ObjectPtr<GetBlockAttrs> n = make_object<GetBlockAttrs>();
  n->name = name;
  return Instruction(/*inputs=*/{},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction GetAxesAttrs::MakeInst(const BlockRV& block, const Array<LoopRV>& outputs) {
  ObjectPtr<GetAxesAttrs> n = make_object<GetAxesAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction GetReadBuffersAttrs::MakeInst(const BlockRV& block, const Array<BufferRV>& outputs) {
  ObjectPtr<GetReadBuffersAttrs> n = make_object<GetReadBuffersAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction GetWriteBuffersAttrs::MakeInst(const BlockRV& block, const Array<BufferRV>& outputs) {
  ObjectPtr<GetWriteBuffersAttrs> n = make_object<GetWriteBuffersAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction GetRootBlocksAttrs::MakeInst(const Array<BlockRV>& outputs) {
  ObjectPtr<GetRootBlocksAttrs> n = make_object<GetRootBlocksAttrs>();
  return Instruction(/*inputs=*/{},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction GetLeafBlocksAttrs::MakeInst(const Array<BlockRV>& outputs) {
  ObjectPtr<GetLeafBlocksAttrs> n = make_object<GetLeafBlocksAttrs>();
  return Instruction(/*inputs=*/{},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n)));
}

/**************** (MakeInst) Scheduling Primitives  ****************/

Instruction FuseAttrs::MakeInst(const Array<LoopRV>& loops, const LoopRV& output) {
  ObjectPtr<FuseAttrs> n = make_object<FuseAttrs>();
  return Instruction(/*inputs=*/{loops.begin(), loops.end()},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
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

Instruction ReorderAttrs::MakeInst(const Array<LoopRV>& after_axes) {
  ObjectPtr<ReorderAttrs> n = make_object<ReorderAttrs>();
  return Instruction(/*inputs=*/{after_axes.begin(), after_axes.end()},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction ComputeAtAttrs::MakeInst(const BlockRV& block, const LoopRV& loop) {
  ObjectPtr<ComputeAtAttrs> n = make_object<ComputeAtAttrs>();
  return Instruction(/*inputs=*/{block, loop},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction ReverseComputeAtAttrs::MakeInst(const BlockRV& block, const LoopRV& loop) {
  ObjectPtr<ReverseComputeAtAttrs> n = make_object<ReverseComputeAtAttrs>();
  return Instruction(/*inputs=*/{block, loop},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction ComputeInlineAttrs::MakeInst(const BlockRV& block) {
  ObjectPtr<ComputeInlineAttrs> n = make_object<ComputeInlineAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction ReverseComputeInlineAttrs::MakeInst(const BlockRV& block) {
  ObjectPtr<ReverseComputeInlineAttrs> n = make_object<ReverseComputeInlineAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction MarkLoopTypeAttrs::MakeInst(const Array<LoopRV>& loops, const String& mark,
                                        const PrimExpr& first_n, const PrimExpr& last_n) {
  ObjectPtr<MarkLoopTypeAttrs> n = make_object<MarkLoopTypeAttrs>();
  n->mark = mark;
  Array<ObjectRef> inputs{loops.begin(), loops.end()};
  inputs.push_back(first_n);
  inputs.push_back(last_n);
  return Instruction(/*inputs=*/inputs,
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction MarkBlockTypeAttrs::MakeInst(const BlockRV& block, const String& mark) {
  ObjectPtr<MarkBlockTypeAttrs> n = make_object<MarkBlockTypeAttrs>();
  n->mark = mark;
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction CacheReadAttrs::MakeInst(const BufferRV& buffer, const String& storage_scope,
                                     const BlockRV& output) {
  ObjectPtr<CacheReadAttrs> n = make_object<CacheReadAttrs>();
  n->storage_scope = storage_scope;
  return Instruction(/*inputs=*/{buffer},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction CacheWriteAttrs::MakeInst(const BufferRV& buffer, const String& storage_scope,
                                      const BlockRV& output) {
  ObjectPtr<CacheWriteAttrs> n = make_object<CacheWriteAttrs>();
  n->storage_scope = storage_scope;
  return Instruction(/*inputs=*/{buffer},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction BlockizeAttrs::MakeInst(const LoopRV& loop, const String& exec_scope,
                                    const BlockRV& output) {
  ObjectPtr<BlockizeAttrs> n = make_object<BlockizeAttrs>();
  n->exec_scope = exec_scope;
  return Instruction(/*inputs=*/{loop},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction DecomposeReductionAttrs::MakeInst(const BlockRV& block, const LoopRV& loop,
                                              const BlockRV& output) {
  ObjectPtr<DecomposeReductionAttrs> n = make_object<DecomposeReductionAttrs>();
  return Instruction(/*inputs=*/{block, loop},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

/**************** ApplyToSchedule  ****************/
/**************** (ApplyToSchedule) Sampling  ****************/

Array<ObjectRef> SamplePerfectTileAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                         const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  return AdaptOutputs(sch->SamplePerfectTile(n_splits, loop, max_innermost_factor));
}

Array<ObjectRef> SampleTileFactorAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                        const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  return AdaptOutputs(sch->SampleTileFactor(n_splits, loop, where));
}

Array<ObjectRef> SampleFusibleLoopsAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                          const Array<ObjectRef>& inputs) const {
  Array<LoopRV> loops;
  loops.reserve(inputs.size());
  for (int i = 0, n = inputs.size(); i < n; ++i) {
    TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[i]);
    loops.push_back(loop);
  }
  ScheduleNode::Order the_order = static_cast<ScheduleNode::Order>(this->order);
  ScheduleNode::Mode the_mode = static_cast<ScheduleNode::Mode>(this->mode);
  return {sch->SampleFusibleLoops(loops, loop_types, max_extent, include_overflow_loop, the_order,
                                  the_mode)};
}

/**************** (ApplyToSchedule) Block/Loop Relationship  ****************/

Array<ObjectRef> GetOnlyConsumerAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                       const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  return {sch->GetOnlyConsumer(block)};
}

Array<ObjectRef> GetBlockAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 0);
  return {sch->GetBlock(name)};
}

Array<ObjectRef> GetAxesAttrs::ApplyToSchedule(ScheduleNode* sch,
                                               const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  return AdaptOutputs(sch->GetAxes(block));
}

Array<ObjectRef> GetReadBuffersAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                      const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  return AdaptOutputs(sch->GetReadBuffers(block));
}

Array<ObjectRef> GetWriteBuffersAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                       const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  return AdaptOutputs(sch->GetWriteBuffers(block));
}

Array<ObjectRef> GetRootBlocksAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                     const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 0);
  return AdaptOutputs(sch->GetRootBlocks());
}

Array<ObjectRef> GetLeafBlocksAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                     const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 0);
  return AdaptOutputs(sch->GetLeafBlocks());
}

/**************** (ApplyToSchedule) Scheduling Primitives  ****************/

Array<ObjectRef> MarkLoopTypeAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                    const Array<ObjectRef>& inputs) const {
  int n_loops = static_cast<int>(inputs.size()) - 2;
  Array<LoopRV> loops;
  loops.reserve(n_loops);
  for (int i = 0; i < n_loops; ++i) {
    TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[i]);
    loops.push_back(loop);
  }
  TVM_META_SCHEDULE_INST_CAST(PrimExpr, first_n, inputs[n_loops]);
  TVM_META_SCHEDULE_INST_CAST(PrimExpr, last_n, inputs[n_loops + 1]);
  sch->MarkLoopType(loops, mark, first_n, last_n);
  return {};
}

Array<ObjectRef> MarkBlockTypeAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                     const Array<ObjectRef>& inputs) const {
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  sch->MarkBlockType(block, mark);
  return {};
}

Array<ObjectRef> FuseAttrs::ApplyToSchedule(ScheduleNode* sch,
                                            const Array<ObjectRef>& inputs) const {
  int n_loops = inputs.size();
  Array<LoopRV> loops;
  loops.reserve(n_loops);
  for (int i = 0; i < n_loops; ++i) {
    TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[i]);
    loops.push_back(loop);
  }
  return {sch->Fuse(loops)};
}

Array<ObjectRef> SplitAttrs::ApplyToSchedule(ScheduleNode* sch,
                                             const Array<ObjectRef>& inputs) const {
  CHECK_GE(inputs.size(), 3);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  Array<Optional<PrimExpr>> factors;
  for (int i = 1, n = inputs.size(); i < n; ++i) {
    TVM_META_SCHEDULE_INST_CAST(PrimExpr, factor, inputs[i]);
    factors.push_back(factor);
  }
  return AdaptOutputs(sch->Split(loop, factors));
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

Array<ObjectRef> ComputeAtAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                 const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 2);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[1]);
  sch->ComputeAt(block, loop);
  return {};
}

Array<ObjectRef> ReverseComputeAtAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                        const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 2);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[1]);
  sch->ReverseComputeAt(block, loop);
  return {};
}

Array<ObjectRef> ComputeInlineAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                     const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  sch->ComputeInline(block);
  return {};
}

Array<ObjectRef> ReverseComputeInlineAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                            const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  sch->ReverseComputeInline(block);
  return {};
}

Array<ObjectRef> CacheReadAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                 const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BufferRV, buffer, inputs[0]);
  return {sch->CacheRead(buffer, storage_scope)};
}

Array<ObjectRef> CacheWriteAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                  const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BufferRV, buffer, inputs[0]);
  return {sch->CacheWrite(buffer, storage_scope)};
}

Array<ObjectRef> BlockizeAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  return {sch->Blockize(loop, exec_scope)};
}

Array<ObjectRef> DecomposeReductionAttrs::ApplyToSchedule(ScheduleNode* sch,
                                                          const Array<ObjectRef>& inputs) const {
  CHECK_EQ(inputs.size(), 2);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[1]);
  return {sch->DecomposeReduction(block, loop)};
}

/**************** Export  ****************/
/**************** (Export) Sampling  ****************/

void SamplePerfectTileAttrs::Export(Array<ObjectRef>* record,
                                    const Optional<Array<ObjectRef>>& decision) const {
  record->push_back(Array<ObjectRef>{
      Integer(n_splits),              //
      Integer(max_innermost_factor),  //
  });
  if (decision.defined()) {
    record->push_back(decision.value());
  }
}

void SampleTileFactorAttrs::Export(Array<ObjectRef>* record,
                                   const Optional<Array<ObjectRef>>& decision) const {
  record->push_back(Array<ObjectRef>{
      Integer(n_splits),  //
      where,              //
  });
  if (decision.defined()) {
    record->push_back(decision.value());
  }
}

void SampleFusibleLoopsAttrs::Export(Array<ObjectRef>* record,
                                     const Optional<Array<ObjectRef>>& decision) const {
  record->push_back(Array<ObjectRef>{
      loop_types,                      //
      Integer(max_extent),             //
      Integer(include_overflow_loop),  //
      Integer(order),                  //
      Integer(mode),                   //
  });
  if (decision.defined()) {
    record->push_back(decision.value());
  }
}

/**************** (Export) Block/Loop Relationship  ****************/

void GetOnlyConsumerAttrs::Export(Array<ObjectRef>* record,
                                  const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
}

void GetBlockAttrs::Export(Array<ObjectRef>* record,
                           const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  record->push_back(Array<ObjectRef>{name});
}

void GetAxesAttrs::Export(Array<ObjectRef>* record,
                          const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
}

void GetReadBuffersAttrs::Export(Array<ObjectRef>* record,
                                 const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
}

void GetWriteBuffersAttrs::Export(Array<ObjectRef>* record,
                                  const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
}

void GetRootBlocksAttrs::Export(Array<ObjectRef>* record,
                                const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
}

void GetLeafBlocksAttrs::Export(Array<ObjectRef>* record,
                                const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
}

/**************** (Export) Scheduling Primitives  ****************/

void MarkLoopTypeAttrs::Export(Array<ObjectRef>* record,
                               const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  record->push_back(Array<ObjectRef>{mark});
}

void MarkBlockTypeAttrs::Export(Array<ObjectRef>* record,
                                const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  record->push_back(Array<ObjectRef>{mark});
}

void FuseAttrs::Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
}

void SplitAttrs::Export(Array<ObjectRef>* record,
                        const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
}

void ReorderAttrs::Export(Array<ObjectRef>* record,
                          const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
}

void ComputeAtAttrs::Export(Array<ObjectRef>* record,
                            const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
}

void ReverseComputeAtAttrs::Export(Array<ObjectRef>* record,
                                   const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
}
void ComputeInlineAttrs::Export(Array<ObjectRef>* record,
                                const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
}
void ReverseComputeInlineAttrs::Export(Array<ObjectRef>* record,
                                       const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
}

void CacheReadAttrs::Export(Array<ObjectRef>* record,
                            const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  record->push_back(Array<ObjectRef>{storage_scope});
}

void CacheWriteAttrs::Export(Array<ObjectRef>* record,
                             const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  record->push_back(Array<ObjectRef>{storage_scope});
}

void BlockizeAttrs::Export(Array<ObjectRef>* record,
                           const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  record->push_back(Array<ObjectRef>{exec_scope});
}

void DecomposeReductionAttrs::Export(Array<ObjectRef>* record,
                                     const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
}

/**************** Import  ****************/
/**************** (Import) Sampling  ****************/

InstAttrs SamplePerfectTileAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_GE(record.size(), 4);
  CHECK_LE(record.size(), 5);
  Array<ObjectRef> from = Downcast<Array<ObjectRef>>(record[3]);
  CHECK_EQ(from.size(), 2);
  ObjectPtr<SamplePerfectTileAttrs> n = make_object<SamplePerfectTileAttrs>();
  n->n_splits = Downcast<Integer>(from[0]);
  n->max_innermost_factor = Downcast<Integer>(from[1]);
  return InstAttrs(std::move(n));
}

InstAttrs SampleTileFactorAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_GE(record.size(), 4);
  CHECK_LE(record.size(), 5);
  Array<ObjectRef> from = Downcast<Array<ObjectRef>>(record[3]);
  CHECK_EQ(from.size(), 2);
  ObjectPtr<SampleTileFactorAttrs> n = make_object<SampleTileFactorAttrs>();
  n->n_splits = Downcast<Integer>(from[0]);
  n->where = Downcast<Array<Integer>>(from[1]);
  return InstAttrs(std::move(n));
}

InstAttrs SampleFusibleLoopsAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_GE(record.size(), 4);
  CHECK_LE(record.size(), 5);
  Array<ObjectRef> from = Downcast<Array<ObjectRef>>(record[3]);
  CHECK_EQ(from.size(), 5);
  ObjectPtr<SampleFusibleLoopsAttrs> n = make_object<SampleFusibleLoopsAttrs>();
  n->loop_types = Downcast<Array<Integer>>(from[0]);
  n->max_extent = Downcast<Integer>(from[1]);
  n->include_overflow_loop = Downcast<Integer>(from[2]);
  n->order = Downcast<Integer>(from[3]);
  n->mode = Downcast<Integer>(from[4]);
  return InstAttrs(std::move(n));
}

/**************** (Import) Block/Loop Relationship  ****************/

InstAttrs GetOnlyConsumerAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 3);
  return InstAttrs(make_object<GetOnlyConsumerAttrs>());
}

InstAttrs GetBlockAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 4);
  Array<ObjectRef> from = Downcast<Array<ObjectRef>>(record[3]);
  CHECK_EQ(from.size(), 1);
  ObjectPtr<GetBlockAttrs> n = make_object<GetBlockAttrs>();
  n->name = Downcast<String>(from[0]);
  return InstAttrs(std::move(n));
}

InstAttrs GetAxesAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 3);
  return InstAttrs(make_object<GetAxesAttrs>());
}

InstAttrs GetReadBuffersAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 3);
  return InstAttrs(make_object<GetReadBuffersAttrs>());
}

InstAttrs GetWriteBuffersAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 3);
  return InstAttrs(make_object<GetWriteBuffersAttrs>());
}

InstAttrs GetRootBlocksAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 3);
  return InstAttrs(make_object<GetRootBlocksAttrs>());
}

InstAttrs GetLeafBlocksAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 3);
  return InstAttrs(make_object<GetLeafBlocksAttrs>());
}

/**************** (Import) Scheduling Primitives  ****************/

InstAttrs MarkLoopTypeAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 4);
  Array<ObjectRef> from = Downcast<Array<ObjectRef>>(record[3]);
  CHECK_EQ(from.size(), 1);
  ObjectPtr<MarkLoopTypeAttrs> n = make_object<MarkLoopTypeAttrs>();
  n->mark = Downcast<String>(from[0]);
  return InstAttrs(std::move(n));
}

InstAttrs MarkBlockTypeAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 4);
  Array<ObjectRef> from = Downcast<Array<ObjectRef>>(record[3]);
  CHECK_EQ(from.size(), 1);
  ObjectPtr<MarkBlockTypeAttrs> n = make_object<MarkBlockTypeAttrs>();
  n->mark = Downcast<String>(from[0]);
  return InstAttrs(std::move(n));
}

InstAttrs FuseAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 3);
  return InstAttrs(make_object<FuseAttrs>());
}

InstAttrs SplitAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 3);
  return InstAttrs(make_object<SplitAttrs>());
}

InstAttrs ReorderAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 3);
  return InstAttrs(make_object<ReorderAttrs>());
}

InstAttrs ComputeAtAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 3);
  return InstAttrs(make_object<ComputeAtAttrs>());
}

InstAttrs ReverseComputeAtAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 3);
  return InstAttrs(make_object<ReverseComputeAtAttrs>());
}

InstAttrs ComputeInlineAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 3);
  return InstAttrs(make_object<ComputeInlineAttrs>());
}

InstAttrs ReverseComputeInlineAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 3);
  return InstAttrs(make_object<ReverseComputeInlineAttrs>());
}

InstAttrs CacheReadAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 4);
  Array<ObjectRef> from = Downcast<Array<ObjectRef>>(record[3]);
  CHECK_EQ(from.size(), 1);
  ObjectPtr<CacheReadAttrs> n = make_object<CacheReadAttrs>();
  n->storage_scope = Downcast<String>(from[0]);
  return InstAttrs(std::move(n));
}

InstAttrs CacheWriteAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 4);
  Array<ObjectRef> from = Downcast<Array<ObjectRef>>(record[3]);
  CHECK_EQ(from.size(), 1);
  ObjectPtr<CacheWriteAttrs> n = make_object<CacheWriteAttrs>();
  n->storage_scope = Downcast<String>(from[0]);
  return InstAttrs(std::move(n));
}

InstAttrs BlockizeAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 4);
  Array<ObjectRef> from = Downcast<Array<ObjectRef>>(record[3]);
  CHECK_EQ(from.size(), 1);
  ObjectPtr<BlockizeAttrs> n = make_object<BlockizeAttrs>();
  n->exec_scope = Downcast<String>(from[0]);
  return InstAttrs(std::move(n));
}

InstAttrs DecomposeReductionAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 3);
  return InstAttrs(make_object<DecomposeReductionAttrs>());
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
TVM_REGISTER_NODE_TYPE(ComputeAtAttrs);
TVM_REGISTER_NODE_TYPE(ReverseComputeAtAttrs);
TVM_REGISTER_NODE_TYPE(ComputeInlineAttrs);
TVM_REGISTER_NODE_TYPE(ReverseComputeInlineAttrs);
TVM_REGISTER_NODE_TYPE(CacheReadAttrs);
TVM_REGISTER_NODE_TYPE(CacheWriteAttrs);
TVM_REGISTER_NODE_TYPE(BlockizeAttrs);
TVM_REGISTER_NODE_TYPE(DecomposeReductionAttrs);

#undef TVM_META_SCHEDULE_INST_CAST

}  // namespace meta_schedule
}  // namespace tvm
