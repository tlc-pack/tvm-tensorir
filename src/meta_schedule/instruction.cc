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

LoopRV LoopRV::ComputeInlineRV() {
  static LoopRV loop_rv;
  return loop_rv;
}

LoopRV LoopRV::ComputeRootRV() {
  static LoopRV loop_rv;
  return loop_rv;
}

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
        LOG(FATAL) << "TypeError: Unable to handle: " << rv
                   << ". Its type is: " << rv->GetTypeKey();
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

Array<ObjectRef> Instruction::ImportToSchedule(ScheduleNode* sch, const Array<ObjectRef>& record,
                                               Map<String, ObjectRef>* named_rvs) {
#define TVM_META_SCHEDULE_INST_VTABLE_ENTRY(AttrsType) \
  { String(AttrsType::_name), AttrsType::Import }
  static const std::unordered_map<String, std::function<InstAttrs(const Array<ObjectRef>&)>>
      vtable = {
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(SamplePerfectTileAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(SampleTileFactorAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(SampleIntAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(SampleCategoricalAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(SampleComputeLocationAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(GetProducersAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(GetConsumersAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(GetBlockAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(GetAxesAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(GetReadBuffersAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(GetWriteBuffersAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(GetRootBlocksAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(GetLeafBlocksAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(MarkLoopAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(MarkBlockAttrs),
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
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(ParallelAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(VectorizeAttrs),
      };
#undef TVM_META_SCHEDULE_INST_VTABLE_ENTRY
  CHECK_GE(record.size(), 3);
  CHECK_LE(record.size(), 5);
  // Step 1. Extract inst_attrs::_name <= record[0]
  String attrs_name = Downcast<String>(record[0]);
  // Step 2. Extract record_inputs <= record[1], then translate record_inputs to inputs
  Array<ObjectRef> inputs;
  {
    Array<ObjectRef> record_inputs = Downcast<Array<ObjectRef>>(record[1]);
    inputs.reserve(record_inputs.size());
    for (const ObjectRef& obj : record_inputs) {
      if (const auto* integer = obj.as<IntImmNode>()) {
        inputs.push_back(GetRef<Integer>(integer));
      } else if (const auto* str_obj = obj.as<StringObj>()) {
        String str = GetRef<String>(str_obj);
        CHECK(named_rvs->count(str)) << "IndexError: Cannot find variable: " << str;
        inputs.push_back(named_rvs->at(str));
      } else {
        LOG(FATAL) << "TypeError: Cannot deal with type '" << obj->GetTypeKey()
                   << "' for input: " << obj;
      }
    }
  }
  // Step 3. Extract record_outputs <= record[2]
  Array<String> record_outputs = Downcast<Array<String>>(record[2]);
  // Step 4. Extract inst_attrs <= record[3]
  InstAttrs inst_attrs = vtable.at(attrs_name)(record);
  // Step 5. Extract decision <= record[4]
  Optional<Array<ObjectRef>> opt_decision = record.size() >= 5
                                                ? Downcast<Array<ObjectRef>>(record[4])
                                                : Optional<Array<ObjectRef>>(NullOpt);
  // Step 6. Calculate the new outputs, and translate record_outputs to outputs
  Array<ObjectRef> outputs = inst_attrs->ApplyToSchedule(sch, inputs, opt_decision);
  CHECK_EQ(record_outputs.size(), outputs.size());
  int n = record_outputs.size();
  for (int i = 0; i < n; ++i) {
    named_rvs->Set(record_outputs[i], outputs[i]);
  }
  return outputs;
}

/**************** Make  ****************/
/**************** (Make) Sampling  ****************/

Instruction SamplePerfectTileAttrs::Make(int n_splits, const LoopRV& loop, int max_innermost_factor,
                                         const Array<tir::Var>& outputs) {
  ObjectPtr<SamplePerfectTileAttrs> n = make_object<SamplePerfectTileAttrs>();
  n->n_splits = n_splits;
  n->max_innermost_factor = max_innermost_factor;
  return Instruction(/*inputs=*/{loop},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction SampleTileFactorAttrs::Make(int n_splits, const LoopRV& loop,
                                        const Array<Integer>& where,
                                        const Array<tir::Var>& outputs) {
  ObjectPtr<SampleTileFactorAttrs> n = make_object<SampleTileFactorAttrs>();
  n->n_splits = n_splits;
  n->where = where;
  return Instruction(/*inputs=*/{loop},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction SampleIntAttrs::Make(const PrimExpr& min_inclusive, const PrimExpr& max_exclusive,
                                 const tir::Var& output) {
  return Instruction(/*inputs=*/{min_inclusive, max_exclusive},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(make_object<SampleIntAttrs>()));
}

Instruction SampleCategoricalAttrs::Make(const Array<Integer>& candidates,
                                         const Array<FloatImm>& probs, const tir::Var& output) {
  ObjectPtr<SampleCategoricalAttrs> n = make_object<SampleCategoricalAttrs>();
  n->candidates = candidates;
  n->probs = probs;
  return Instruction(/*inputs=*/{},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction SampleComputeLocationAttrs::Make(const BlockRV& block, const LoopRV& output) {
  ObjectPtr<SampleComputeLocationAttrs> n = make_object<SampleComputeLocationAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

/**************** (Make) Block/Loop Relationship  ****************/

Instruction GetProducersAttrs::Make(const BlockRV& block, const Array<BlockRV>& outputs) {
  ObjectPtr<GetProducersAttrs> n = make_object<GetProducersAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction GetConsumersAttrs::Make(const BlockRV& block, const Array<BlockRV>& outputs) {
  ObjectPtr<GetConsumersAttrs> n = make_object<GetConsumersAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction GetBlockAttrs::Make(const String& name, const BlockRV& output) {
  ObjectPtr<GetBlockAttrs> n = make_object<GetBlockAttrs>();
  n->name = name;
  return Instruction(/*inputs=*/{},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction GetAxesAttrs::Make(const BlockRV& block, const Array<LoopRV>& outputs) {
  ObjectPtr<GetAxesAttrs> n = make_object<GetAxesAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction GetReadBuffersAttrs::Make(const BlockRV& block, const Array<BufferRV>& outputs) {
  ObjectPtr<GetReadBuffersAttrs> n = make_object<GetReadBuffersAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction GetWriteBuffersAttrs::Make(const BlockRV& block, const Array<BufferRV>& outputs) {
  ObjectPtr<GetWriteBuffersAttrs> n = make_object<GetWriteBuffersAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction GetRootBlocksAttrs::Make(const Array<BlockRV>& outputs) {
  ObjectPtr<GetRootBlocksAttrs> n = make_object<GetRootBlocksAttrs>();
  return Instruction(/*inputs=*/{},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction GetLeafBlocksAttrs::Make(const Array<BlockRV>& outputs) {
  ObjectPtr<GetLeafBlocksAttrs> n = make_object<GetLeafBlocksAttrs>();
  return Instruction(/*inputs=*/{},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n)));
}

/**************** (Make) Scheduling Primitives  ****************/

Instruction FuseAttrs::Make(const Array<LoopRV>& loops, const LoopRV& output) {
  ObjectPtr<FuseAttrs> n = make_object<FuseAttrs>();
  return Instruction(/*inputs=*/{loops.begin(), loops.end()},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction SplitAttrs::Make(const LoopRV& loop, const Array<Optional<PrimExpr>>& factors,
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

Instruction ReorderAttrs::Make(const Array<LoopRV>& after_axes) {
  ObjectPtr<ReorderAttrs> n = make_object<ReorderAttrs>();
  return Instruction(/*inputs=*/{after_axes.begin(), after_axes.end()},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction ComputeAtAttrs::Make(const BlockRV& block, const LoopRV& loop) {
  ObjectPtr<ComputeAtAttrs> n = make_object<ComputeAtAttrs>();
  return Instruction(/*inputs=*/{block, loop},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction ReverseComputeAtAttrs::Make(const BlockRV& block, const LoopRV& loop) {
  ObjectPtr<ReverseComputeAtAttrs> n = make_object<ReverseComputeAtAttrs>();
  return Instruction(/*inputs=*/{block, loop},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction ComputeInlineAttrs::Make(const BlockRV& block) {
  ObjectPtr<ComputeInlineAttrs> n = make_object<ComputeInlineAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction ReverseComputeInlineAttrs::Make(const BlockRV& block) {
  ObjectPtr<ReverseComputeInlineAttrs> n = make_object<ReverseComputeInlineAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction MarkLoopAttrs::Make(const LoopRV& loop, const String& ann_key,
                                const PrimExpr& ann_val) {
  Array<ObjectRef> inputs{loop};
  ObjectPtr<MarkLoopAttrs> n = make_object<MarkLoopAttrs>();
  n->ann_key = ann_key;
  if (const auto* str_imm = ann_val.as<tir::StringImmNode>()) {
    n->ann_val = str_imm->value;
  } else if (const auto* int_imm = ann_val.as<tir::IntImmNode>()) {
    n->ann_val = "";
    inputs.push_back(Integer(int_imm->value));
  }
  return Instruction(/*inputs=*/inputs,
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction MarkBlockAttrs::Make(const BlockRV& block, const String& ann_key,
                                 const PrimExpr& ann_val) {
  ObjectPtr<MarkBlockAttrs> n = make_object<MarkBlockAttrs>();
  n->ann_key = ann_key;
  return Instruction(/*inputs=*/{block, ann_val},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction CacheReadAttrs::Make(const BlockRV& block, int i, const String& storage_scope,
                                 const BlockRV& output) {
  ObjectPtr<CacheReadAttrs> n = make_object<CacheReadAttrs>();
  n->i = i;
  n->storage_scope = storage_scope;
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction CacheWriteAttrs::Make(const BlockRV& block, int i, const String& storage_scope,
                                  const BlockRV& output) {
  ObjectPtr<CacheWriteAttrs> n = make_object<CacheWriteAttrs>();
  n->i = i;
  n->storage_scope = storage_scope;
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction BlockizeAttrs::Make(const LoopRV& loop, const String& exec_scope,
                                const BlockRV& output) {
  ObjectPtr<BlockizeAttrs> n = make_object<BlockizeAttrs>();
  n->exec_scope = exec_scope;
  return Instruction(/*inputs=*/{loop},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction DecomposeReductionAttrs::Make(const BlockRV& block, const LoopRV& loop,
                                          const BlockRV& output) {
  ObjectPtr<DecomposeReductionAttrs> n = make_object<DecomposeReductionAttrs>();
  return Instruction(/*inputs=*/{block, loop},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction ParallelAttrs::Make(const LoopRV& loop) {
  return Instruction(/*inputs=*/{loop},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(make_object<ParallelAttrs>()));
}

Instruction VectorizeAttrs::Make(const LoopRV& loop) {
  return Instruction(/*inputs=*/{loop},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(make_object<VectorizeAttrs>()));
}

Instruction EnterPostProcAttrs::Make() {
  return Instruction(/*inputs=*/{},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(make_object<EnterPostProcAttrs>()));
}

/**************** ApplyToSchedule  ****************/

#define TVM_META_SCHEDULE_INST_CAST(CastType, VarName, Input)                    \
  CHECK(Input->IsInstance<CastType::ContainerType>())                            \
      << "TypeError: Cannot downcast to '" << CastType::ContainerType::_type_key \
      << "' from: " << Input->GetTypeKey();                                      \
  CastType VarName = Downcast<CastType>(Input);

template <class T>
Array<ObjectRef> AdaptOutputs(const Array<T>& outputs) {
  return {outputs.begin(), outputs.end()};
}

/**************** (ApplyToSchedule) Sampling  ****************/

Array<ObjectRef> SamplePerfectTileAttrs::ApplyToSchedule(
    ScheduleNode* sch, const Array<ObjectRef>& inputs,
    const Optional<Array<ObjectRef>>& decision) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  return AdaptOutputs(sch->SamplePerfectTile(n_splits, loop, max_innermost_factor, decision));
}

Array<ObjectRef> SampleTileFactorAttrs::ApplyToSchedule(
    ScheduleNode* sch, const Array<ObjectRef>& inputs,
    const Optional<Array<ObjectRef>>& decision) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  return AdaptOutputs(sch->SampleTileFactor(n_splits, loop, where, decision));
}

Array<ObjectRef> SampleIntAttrs::ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs,
                                                 const Optional<Array<ObjectRef>>& decision) const {
  CHECK_EQ(inputs.size(), 2);
  TVM_META_SCHEDULE_INST_CAST(PrimExpr, min_inclusive, inputs[0]);
  TVM_META_SCHEDULE_INST_CAST(PrimExpr, max_exclusive, inputs[1]);
  return {sch->SampleInt(min_inclusive, max_exclusive, decision)};
}

Array<ObjectRef> SampleCategoricalAttrs::ApplyToSchedule(
    ScheduleNode* sch, const Array<ObjectRef>& inputs,
    const Optional<Array<ObjectRef>>& decision) const {
  CHECK_EQ(inputs.size(), 0);
  return {sch->SampleCategorical(candidates, probs, decision)};
}

Array<ObjectRef> SampleComputeLocationAttrs::ApplyToSchedule(
    ScheduleNode* sch, const Array<ObjectRef>& inputs,
    const Optional<Array<ObjectRef>>& decision) const {
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  return {sch->SampleComputeLocation(block, decision)};
}

/**************** (ApplyToSchedule) Block/Loop Relationship  ****************/

Array<ObjectRef> GetProducersAttrs::ApplyToSchedule(
    ScheduleNode* sch, const Array<ObjectRef>& inputs,
    const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  return AdaptOutputs(sch->GetProducers(block));
}

Array<ObjectRef> GetConsumersAttrs::ApplyToSchedule(
    ScheduleNode* sch, const Array<ObjectRef>& inputs,
    const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  return AdaptOutputs(sch->GetConsumers(block));
}

Array<ObjectRef> GetBlockAttrs::ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs,
                                                const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 0);
  return {sch->GetBlock(name)};
}

Array<ObjectRef> GetAxesAttrs::ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs,
                                               const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  return AdaptOutputs(sch->GetAxes(block));
}

Array<ObjectRef> GetReadBuffersAttrs::ApplyToSchedule(
    ScheduleNode* sch, const Array<ObjectRef>& inputs,
    const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  return AdaptOutputs(sch->GetReadBuffers(block));
}

Array<ObjectRef> GetWriteBuffersAttrs::ApplyToSchedule(
    ScheduleNode* sch, const Array<ObjectRef>& inputs,
    const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  return AdaptOutputs(sch->GetWriteBuffers(block));
}

Array<ObjectRef> GetRootBlocksAttrs::ApplyToSchedule(
    ScheduleNode* sch, const Array<ObjectRef>& inputs,
    const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 0);
  return AdaptOutputs(sch->GetRootBlocks());
}

Array<ObjectRef> GetLeafBlocksAttrs::ApplyToSchedule(
    ScheduleNode* sch, const Array<ObjectRef>& inputs,
    const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 0);
  return AdaptOutputs(sch->GetLeafBlocks());
}

/**************** (ApplyToSchedule) Scheduling Primitives  ****************/

Array<ObjectRef> MarkLoopAttrs::ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs,
                                                const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK(inputs.size() == 1 || inputs.size() == 2);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  if (ann_val == "") {
    CHECK_EQ(inputs.size(), 2);
    TVM_META_SCHEDULE_INST_CAST(PrimExpr, val, inputs[1]);
    sch->MarkLoop(loop, ann_key, val);
  } else {
    CHECK_EQ(inputs.size(), 1);
    sch->MarkLoop(loop, ann_key, tir::StringImm(ann_val));
  }
  return {};
}

Array<ObjectRef> MarkBlockAttrs::ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs,
                                                 const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 2);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  TVM_META_SCHEDULE_INST_CAST(PrimExpr, ann_val, inputs[1]);
  sch->MarkBlock(block, ann_key, ann_val);
  return {};
}

Array<ObjectRef> FuseAttrs::ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs,
                                            const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  int n_loops = inputs.size();
  Array<LoopRV> loops;
  loops.reserve(n_loops);
  for (int i = 0; i < n_loops; ++i) {
    TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[i]);
    loops.push_back(loop);
  }
  return {sch->Fuse(loops)};
}

Array<ObjectRef> SplitAttrs::ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs,
                                             const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_GE(inputs.size(), 3);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  Array<Optional<PrimExpr>> factors;
  for (int i = 1, n = inputs.size(); i < n; ++i) {
    TVM_META_SCHEDULE_INST_CAST(PrimExpr, factor, inputs[i]);
    factors.push_back(factor);
  }
  return AdaptOutputs(sch->Split(loop, factors));
}

Array<ObjectRef> ReorderAttrs::ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs,
                                               const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
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

Array<ObjectRef> ComputeAtAttrs::ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs,
                                                 const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 2);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[1]);
  sch->ComputeAt(block, loop);
  return {};
}

Array<ObjectRef> ReverseComputeAtAttrs::ApplyToSchedule(
    ScheduleNode* sch, const Array<ObjectRef>& inputs,
    const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 2);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[1]);
  sch->ReverseComputeAt(block, loop);
  return {};
}

Array<ObjectRef> ComputeInlineAttrs::ApplyToSchedule(
    ScheduleNode* sch, const Array<ObjectRef>& inputs,
    const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  sch->ComputeInline(block);
  return {};
}

Array<ObjectRef> ReverseComputeInlineAttrs::ApplyToSchedule(
    ScheduleNode* sch, const Array<ObjectRef>& inputs,
    const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  sch->ReverseComputeInline(block);
  return {};
}

Array<ObjectRef> CacheReadAttrs::ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs,
                                                 const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  return {sch->CacheRead(block, i, storage_scope)};
}

Array<ObjectRef> CacheWriteAttrs::ApplyToSchedule(
    ScheduleNode* sch, const Array<ObjectRef>& inputs,
    const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  return {sch->CacheWrite(block, i, storage_scope)};
}

Array<ObjectRef> BlockizeAttrs::ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs,
                                                const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  return {sch->Blockize(loop, exec_scope)};
}

Array<ObjectRef> DecomposeReductionAttrs::ApplyToSchedule(
    ScheduleNode* sch, const Array<ObjectRef>& inputs,
    const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 2);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[1]);
  return {sch->DecomposeReduction(block, loop)};
}

Array<ObjectRef> ParallelAttrs::ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs,
                                                const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  sch->Parallel(loop);
  return {};
}
Array<ObjectRef> VectorizeAttrs::ApplyToSchedule(ScheduleNode* sch, const Array<ObjectRef>& inputs,
                                                 const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  sch->Vectorize(loop);
  return {};
}

Array<ObjectRef> EnterPostProcAttrs::ApplyToSchedule(
    ScheduleNode* sch, const Array<ObjectRef>& inputs,
    const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 0);
  sch->EnterPostProc();
  return {};
}

#undef TVM_META_SCHEDULE_INST_CAST

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

void SampleIntAttrs::Export(Array<ObjectRef>* record,
                            const Optional<Array<ObjectRef>>& decision) const {
  record->push_back(Array<ObjectRef>{});
  if (decision.defined()) {
    record->push_back(decision.value());
  }
}

void SampleCategoricalAttrs::Export(Array<ObjectRef>* record,
                                    const Optional<Array<ObjectRef>>& decision) const {
  record->push_back(Array<ObjectRef>{
      candidates,
      probs,
  });
  if (decision.defined()) {
    record->push_back(decision.value());
  }
}

void SampleComputeLocationAttrs::Export(Array<ObjectRef>* record,
                                        const Optional<Array<ObjectRef>>& decision) const {
  record->push_back(Array<ObjectRef>{});
  if (decision.defined()) {
    record->push_back(decision.value());
  }
}

/**************** (Export) Block/Loop Relationship  ****************/

void GetBlockAttrs::Export(Array<ObjectRef>* record,
                           const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  record->push_back(Array<ObjectRef>{name});
}

/**************** (Export) Scheduling Primitives  ****************/

void MarkLoopAttrs::Export(Array<ObjectRef>* record,
                           const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  record->push_back(Array<ObjectRef>{ann_key, ann_val});
}

void MarkBlockAttrs::Export(Array<ObjectRef>* record,
                            const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  record->push_back(Array<ObjectRef>{ann_key});
}

void CacheReadAttrs::Export(Array<ObjectRef>* record,
                            const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  record->push_back(Array<ObjectRef>{Integer(i), storage_scope});
}

void CacheWriteAttrs::Export(Array<ObjectRef>* record,
                             const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  record->push_back(Array<ObjectRef>{Integer(i), storage_scope});
}

void BlockizeAttrs::Export(Array<ObjectRef>* record,
                           const Optional<Array<ObjectRef>>& decision) const {
  CHECK(!decision.defined());
  record->push_back(Array<ObjectRef>{exec_scope});
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

InstAttrs SampleIntAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_GE(record.size(), 4);
  CHECK_LE(record.size(), 5);
  Array<ObjectRef> from = Downcast<Array<ObjectRef>>(record[3]);
  CHECK_EQ(from.size(), 0);
  return InstAttrs(make_object<SampleIntAttrs>());
}

InstAttrs SampleCategoricalAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_GE(record.size(), 4);
  CHECK_LE(record.size(), 5);
  Array<ObjectRef> from = Downcast<Array<ObjectRef>>(record[3]);
  CHECK_EQ(from.size(), 2);
  ObjectPtr<SampleCategoricalAttrs> n = make_object<SampleCategoricalAttrs>();
  n->candidates = Downcast<Array<Integer>>(from[0]);
  n->probs = Downcast<Array<FloatImm>>(from[1]);
  return InstAttrs(std::move(n));
}

InstAttrs SampleComputeLocationAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_GE(record.size(), 4);
  CHECK_LE(record.size(), 5);
  Array<ObjectRef> from = Downcast<Array<ObjectRef>>(record[3]);
  CHECK_EQ(from.size(), 0);
  return InstAttrs(make_object<SampleComputeLocationAttrs>());
}

/**************** (Import) Block/Loop Relationship  ****************/

InstAttrs GetBlockAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 4);
  Array<ObjectRef> from = Downcast<Array<ObjectRef>>(record[3]);
  CHECK_EQ(from.size(), 1);
  ObjectPtr<GetBlockAttrs> n = make_object<GetBlockAttrs>();
  n->name = Downcast<String>(from[0]);
  return InstAttrs(std::move(n));
}

/**************** (Import) Scheduling Primitives  ****************/

InstAttrs MarkLoopAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 4);
  Array<ObjectRef> from = Downcast<Array<ObjectRef>>(record[3]);
  CHECK_EQ(from.size(), 2);
  ObjectPtr<MarkLoopAttrs> n = make_object<MarkLoopAttrs>();
  n->ann_key = Downcast<String>(from[0]);
  n->ann_val = Downcast<String>(from[1]);
  return InstAttrs(std::move(n));
}

InstAttrs MarkBlockAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 4);
  Array<ObjectRef> from = Downcast<Array<ObjectRef>>(record[3]);
  CHECK_EQ(from.size(), 1);
  ObjectPtr<MarkBlockAttrs> n = make_object<MarkBlockAttrs>();
  n->ann_key = Downcast<String>(from[0]);
  return InstAttrs(std::move(n));
}

InstAttrs CacheReadAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 4);
  Array<ObjectRef> from = Downcast<Array<ObjectRef>>(record[3]);
  CHECK_EQ(from.size(), 2);
  ObjectPtr<CacheReadAttrs> n = make_object<CacheReadAttrs>();
  n->i = Downcast<Integer>(from[0]);
  n->storage_scope = Downcast<String>(from[1]);
  return InstAttrs(std::move(n));
}

InstAttrs CacheWriteAttrs::Import(const Array<ObjectRef>& record) {
  CHECK_EQ(record.size(), 4);
  Array<ObjectRef> from = Downcast<Array<ObjectRef>>(record[3]);
  CHECK_EQ(from.size(), 2);
  ObjectPtr<CacheWriteAttrs> n = make_object<CacheWriteAttrs>();
  n->i = Downcast<Integer>(from[0]);
  n->storage_scope = Downcast<String>(from[1]);
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

/**************** Import/Export for empty instructions ****************/

#define TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY(AttrsType)                                  \
  void AttrsType::Export(Array<ObjectRef>* record, const Optional<Array<ObjectRef>>& decision) \
      const {                                                                                  \
    CHECK(!decision.defined());                                                                \
  }                                                                                            \
  InstAttrs AttrsType::Import(const Array<ObjectRef>& record) {                                \
    CHECK_EQ(record.size(), 3);                                                                \
    return InstAttrs(make_object<AttrsType>());                                                \
  }

TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY(GetProducersAttrs);
TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY(GetConsumersAttrs);
TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY(GetAxesAttrs);
TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY(GetReadBuffersAttrs);
TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY(GetWriteBuffersAttrs);
TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY(GetRootBlocksAttrs);
TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY(GetLeafBlocksAttrs);
TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY(FuseAttrs);
TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY(SplitAttrs);
TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY(ReorderAttrs);
TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY(ComputeAtAttrs);
TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY(ReverseComputeAtAttrs);
TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY(ComputeInlineAttrs);
TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY(ReverseComputeInlineAttrs);
TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY(DecomposeReductionAttrs);
TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY(EnterPostProcAttrs);
TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY(ParallelAttrs);
TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY(VectorizeAttrs);

#undef TVM_META_SCHEDULE_INST_EXPORT_IMPORT_EMPTY

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(BlockRVNode);
TVM_REGISTER_NODE_TYPE(LoopRVNode);
TVM_REGISTER_NODE_TYPE(BufferRVNode);
TVM_REGISTER_OBJECT_TYPE(InstAttrsNode);
TVM_REGISTER_NODE_TYPE(InstructionNode);
TVM_REGISTER_NODE_TYPE(SamplePerfectTileAttrs);
TVM_REGISTER_NODE_TYPE(SampleTileFactorAttrs);
TVM_REGISTER_NODE_TYPE(SampleIntAttrs);
TVM_REGISTER_NODE_TYPE(SampleCategoricalAttrs);
TVM_REGISTER_NODE_TYPE(SampleComputeLocationAttrs);
TVM_REGISTER_NODE_TYPE(GetProducersAttrs);
TVM_REGISTER_NODE_TYPE(GetConsumersAttrs);
TVM_REGISTER_NODE_TYPE(GetBlockAttrs);
TVM_REGISTER_NODE_TYPE(GetAxesAttrs);
TVM_REGISTER_NODE_TYPE(GetReadBuffersAttrs);
TVM_REGISTER_NODE_TYPE(GetWriteBuffersAttrs);
TVM_REGISTER_NODE_TYPE(GetRootBlocksAttrs);
TVM_REGISTER_NODE_TYPE(GetLeafBlocksAttrs);
TVM_REGISTER_NODE_TYPE(MarkLoopAttrs);
TVM_REGISTER_NODE_TYPE(MarkBlockAttrs);
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
TVM_REGISTER_NODE_TYPE(EnterPostProcAttrs);
TVM_REGISTER_NODE_TYPE(ParallelAttrs);
TVM_REGISTER_NODE_TYPE(VectorizeAttrs);

TVM_REGISTER_GLOBAL("meta_schedule.LoopRVComputeInlineRV").set_body_typed(LoopRV::ComputeInlineRV);
TVM_REGISTER_GLOBAL("meta_schedule.LoopRVComputeRootRV").set_body_typed(LoopRV::ComputeRootRV);

}  // namespace meta_schedule
}  // namespace tvm
