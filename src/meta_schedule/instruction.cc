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

#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt_functor.h>

#include "./schedule.h"
// #include "./utils.h"

namespace tvm {
namespace meta_schedule {

using BlockRV = tir::BlockRV;
using LoopRV = tir::LoopRV;
using ExprRV = tir::ExprRV;

/**************** Constructors ****************/

Instruction::Instruction(Array<Optional<ObjectRef>> inputs, Array<ObjectRef> outputs,
                         InstAttrs inst_attrs) {
  ObjectPtr<InstructionNode> n = make_object<InstructionNode>();
  n->inputs = std::move(inputs);
  n->outputs = std::move(outputs);
  n->inst_attrs = std::move(inst_attrs);
  data_ = std::move(n);
}

/**************** Instruction  ****************/

Array<ObjectRef> InstructionNode::Serialize(const Map<ObjectRef, String>& rv_names,
                                            const Optional<ObjectRef>& decision) const {
  Array<ObjectRef> record;
  record.reserve(4);
  // record[0]: inst_attrs::_name
  record.push_back(inst_attrs->GetName());
  // record[1]: inputs
  {
    Array<ObjectRef> names;
    names.reserve(this->inputs.size());
    for (const Optional<ObjectRef>& opt_rv : this->inputs) {
      if (!opt_rv.defined()) {
        names.push_back(String("None"));
        continue;
      }
      ObjectRef rv = opt_rv.value();
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
  // record[2]: outputs
  {
    Array<ObjectRef> names;
    names.reserve(this->outputs.size());
    for (const ObjectRef& rv : this->outputs) {
      ICHECK(rv.defined());
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
  // record[3...]: allocated for inst_attrs
  inst_attrs->Serialize(&record, decision);
  return record;
}

Array<ObjectRef> InstructionNode::Deserialize(const Array<ObjectRef>& record,
                                              Map<String, ObjectRef>* named_rvs,
                                              const Schedule& sch) {
#define TVM_META_SCHEDULE_INST_VTABLE_ENTRY(AttrsType) \
  { String(AttrsType::_name), AttrsType::Deserialize }
  static const std::unordered_map<
      String, std::function<InstAttrs(const Array<ObjectRef>&, Optional<ObjectRef>*)>>
      vtable = {
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(SamplePerfectTileAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(SampleCategoricalAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(SampleComputeLocationAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(GetProducersAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(GetConsumersAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(GetBlockAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(GetAxesAttrs),
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
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(TensorizeAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(ParallelAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(VectorizeAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(UnrollAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(RFactorAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(BindAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(SetScopeAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(StorageAlignAttrs),
          TVM_META_SCHEDULE_INST_VTABLE_ENTRY(EnterPostProcAttrs),
      };
#undef TVM_META_SCHEDULE_INST_VTABLE_ENTRY
  CHECK_GE(record.size(), 3);
  // Step 1. Extract inst_attrs::_name <= record[0]
  String attrs_name = Downcast<String>(record[0]);
  // Step 2. Extract record_inputs <= record[1], then translate record_inputs to inputs
  Array<Optional<ObjectRef>> inputs;
  {
    const ArrayNode* record_inputs = record[1].as<ArrayNode>();
    inputs.reserve(record_inputs->size());
    for (const ObjectRef& obj : *record_inputs) {
      if (const auto* integer = obj.as<IntImmNode>()) {
        inputs.push_back(GetRef<Integer>(integer));
      } else if (const auto* str_obj = obj.as<StringObj>()) {
        String str = GetRef<String>(str_obj);
        ICHECK(named_rvs->count(str)) << "IndexError: Cannot find variable: " << str;
        inputs.push_back(named_rvs->at(str));
      } else {
        LOG(FATAL) << "TypeError: Cannot deal with type '" << obj->GetTypeKey()
                   << "' for input: " << obj;
      }
    }
  }
  // Step 3. Extract record_outputs <= record[2]
  Array<String> record_outputs = Downcast<Array<String>>(record[2]);
  // Step 4. Extract inst_attrs <= record[3...]
  Optional<ObjectRef> decision = NullOpt;
  InstAttrs inst_attrs = vtable.at(attrs_name)(record, &decision);
  // Step 6. Calculate the new outputs, and translate record_outputs to outputs
  Array<ObjectRef> outputs = inst_attrs->Apply(sch, inputs, decision);
  ICHECK_EQ(record_outputs.size(), outputs.size());
  int n = record_outputs.size();
  for (int i = 0; i < n; ++i) {
    named_rvs->Set(record_outputs[i], outputs[i]);
  }
  return outputs;
}

void InstructionNode::AsPython(std::ostream& os, const Map<ObjectRef, String>& rv_names,
                               const Optional<ObjectRef>& decision) const {
  auto rename_expr = [&rv_names](const tir::Var& var) -> Optional<PrimExpr> {
    if (Optional<String> name = rv_names.Get(var)) {
      return tir::Var(name.value(), var.dtype());
    }
    LOG(FATAL) << "ValueError: Variable '" << var << "' is not defined in the schedule.";
    throw;
  };
  auto rv2name = [&rename_expr, &rv_names](const ObjectRef& obj) -> String {
    if (Optional<String> name = rv_names.Get(obj)) {
      return name.value();
    }
    const auto* prim_expr = obj.as<PrimExprNode>();
    ICHECK(prim_expr) << "TypeError: Cannot handle type: " << obj->GetTypeKey();
    std::ostringstream oss;
    oss << tir::Substitute(GetRef<PrimExpr>(prim_expr), rename_expr);
    return oss.str();
  };
  Array<String> input_names;
  {
    input_names.reserve(inputs.size());
    for (const ObjectRef& v : inputs) {
      input_names.push_back(rv2name(v));
    }
  }
  Array<String> output_names;
  {
    output_names.reserve(outputs.size());
    for (const ObjectRef& v : outputs) {
      output_names.push_back(rv2name(v));
    }
  }
  inst_attrs->AsPython(os, input_names, output_names, decision);
}

/**************** Make  ****************/
/**************** (Make) Sampling  ****************/

Instruction SamplePerfectTileAttrs::Make(const LoopRV& loop, int n, int max_innermost_factor,
                                         const Array<tir::Var>& outputs) {
  ObjectPtr<SamplePerfectTileAttrs> n_ = make_object<SamplePerfectTileAttrs>();
  n_->n = n;
  n_->max_innermost_factor = max_innermost_factor;
  return Instruction(/*inputs=*/{loop},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/InstAttrs(std::move(n_)));
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
  Array<Optional<ObjectRef>> inputs;
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

Instruction ComputeAtAttrs::Make(const BlockRV& block, const LoopRV& loop,
                                 bool preserve_unit_loop) {
  ObjectPtr<ComputeAtAttrs> n = make_object<ComputeAtAttrs>();
  n->preserve_unit_loop = preserve_unit_loop;
  return Instruction(/*inputs=*/{block, loop},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction ReverseComputeAtAttrs::Make(const BlockRV& block, const LoopRV& loop,
                                        bool preserve_unit_loop) {
  ObjectPtr<ReverseComputeAtAttrs> n = make_object<ReverseComputeAtAttrs>();
  n->preserve_unit_loop = preserve_unit_loop;
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
  Array<Optional<ObjectRef>> inputs{loop};
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

Instruction BlockizeAttrs::Make(const LoopRV& loop, const BlockRV& output) {
  ObjectPtr<BlockizeAttrs> n = make_object<BlockizeAttrs>();
  return Instruction(/*inputs=*/{loop},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction DecomposeReductionAttrs::Make(const BlockRV& block, const Optional<LoopRV>& loop,
                                          const BlockRV& output) {
  ObjectPtr<DecomposeReductionAttrs> n = make_object<DecomposeReductionAttrs>();
  return Instruction(/*inputs=*/{block, loop},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction TensorizeAttrs::Make(const LoopRV& loop, const String& tensor_intrin_name) {
  ObjectPtr<TensorizeAttrs> n = make_object<TensorizeAttrs>();
  n->tensor_intrin_name = tensor_intrin_name;
  return Instruction(/*inputs=*/{loop},
                     /*outputs=*/{},
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

Instruction UnrollAttrs::Make(const LoopRV& loop) {
  return Instruction(/*inputs=*/{loop},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(make_object<UnrollAttrs>()));
}

Instruction RFactorAttrs::Make(const LoopRV& loop, int factor_axis, const BlockRV& output) {
  ObjectPtr<RFactorAttrs> n = make_object<RFactorAttrs>();
  n->factor_axis = factor_axis;
  return Instruction(/*inputs=*/{loop},
                     /*outputs=*/{output},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction BindAttrs::Make(const LoopRV& loop, const String& thread_axis) {
  ObjectPtr<BindAttrs> n = make_object<BindAttrs>();
  n->thread_axis = thread_axis;
  return Instruction(/*inputs=*/{loop},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction SetScopeAttrs::Make(const BlockRV& block, int i, const String& storage_scope) {
  ObjectPtr<SetScopeAttrs> n = make_object<SetScopeAttrs>();
  n->i = i;
  n->storage_scope = storage_scope;
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction StorageAlignAttrs::Make(const BlockRV& block, int buffer_index, int axis, int factor,
                                    int offset) {
  ObjectPtr<StorageAlignAttrs> n = make_object<StorageAlignAttrs>();
  n->buffer_index = buffer_index;
  n->axis = axis;
  n->factor = factor;
  n->offset = offset;
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(std::move(n)));
}

Instruction EnterPostProcAttrs::Make() {
  return Instruction(/*inputs=*/{},
                     /*outputs=*/{},
                     /*attrs=*/InstAttrs(make_object<EnterPostProcAttrs>()));
}

/**************** Apply  ****************/

#define TVM_META_SCHEDULE_INST_CAST(CastType, VarName, Input)                    \
  ICHECK(!Input.defined() || Input->IsInstance<CastType::ContainerType>())       \
      << "TypeError: Cannot downcast to '" << CastType::ContainerType::_type_key \
      << "' from: " << Input->GetTypeKey();                                      \
  CastType VarName = Input.defined() ? Downcast<CastType>(Input) : CastType(nullptr);

template <class T>
Array<ObjectRef> AdaptOutputs(const Array<T>& outputs) {
  return {outputs.begin(), outputs.end()};
}

/**************** (Apply) Sampling  ****************/

Array<ObjectRef> SamplePerfectTileAttrs::Apply(const Schedule& sch,
                                               const Array<Optional<ObjectRef>>& inputs,
                                               const Optional<ObjectRef>& decision) const {
  ICHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  Optional<Array<Integer>> casted_decision = NullOpt;
  if (decision.defined()) {
    const auto* array = decision.as<ArrayNode>();
    Array<Integer> result;
    result.reserve(array->size());
    for (const ObjectRef& obj : *array) {
      result.push_back(Downcast<Integer>(obj));
    }
    casted_decision = std::move(result);
  }
  return AdaptOutputs(sch->SamplePerfectTile(loop, n, max_innermost_factor, casted_decision));
}

Array<ObjectRef> SampleCategoricalAttrs::Apply(const Schedule& sch,
                                               const Array<Optional<ObjectRef>>& inputs,
                                               const Optional<ObjectRef>& decision) const {
  ICHECK_EQ(inputs.size(), 0);
  Optional<Integer> casted_decision = NullOpt;
  if (decision.defined()) {
    casted_decision = Downcast<Integer>(decision);
  }
  return {sch->SampleCategorical(candidates, probs, casted_decision)};
}

Array<ObjectRef> SampleComputeLocationAttrs::Apply(const Schedule& sch,
                                                   const Array<Optional<ObjectRef>>& inputs,
                                                   const Optional<ObjectRef>& decision) const {
  ICHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  Optional<Integer> casted_decision = NullOpt;
  if (decision.defined()) {
    casted_decision = Downcast<Integer>(decision);
  }
  return {sch->SampleComputeLocation(block, casted_decision)};
}

/**************** (Apply) Block/Loop Relationship  ****************/

Array<ObjectRef> GetProducersAttrs::Apply(const Schedule& sch,
                                          const Array<Optional<ObjectRef>>& inputs,
                                          const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  return AdaptOutputs(sch->GetProducers(block));
}

Array<ObjectRef> GetConsumersAttrs::Apply(const Schedule& sch,
                                          const Array<Optional<ObjectRef>>& inputs,
                                          const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  return AdaptOutputs(sch->GetConsumers(block));
}

Array<ObjectRef> GetBlockAttrs::Apply(const Schedule& sch,  //
                                      const Array<Optional<ObjectRef>>& inputs,
                                      const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 0);
  return {sch->GetBlock(name)};
}

Array<ObjectRef> GetAxesAttrs::Apply(const Schedule& sch,  //
                                     const Array<Optional<ObjectRef>>& inputs,
                                     const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  return AdaptOutputs(sch->GetAxes(block));
}

/**************** (Apply) Scheduling Primitives  ****************/

Array<ObjectRef> MarkLoopAttrs::Apply(const Schedule& sch,  //
                                      const Array<Optional<ObjectRef>>& inputs,
                                      const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK(inputs.size() == 1 || inputs.size() == 2);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  if (ann_val == "") {
    ICHECK_EQ(inputs.size(), 2);
    TVM_META_SCHEDULE_INST_CAST(PrimExpr, val, inputs[1]);
    sch->MarkLoop(loop, ann_key, val);
  } else {
    ICHECK_EQ(inputs.size(), 1);
    sch->MarkLoop(loop, ann_key, tir::StringImm(ann_val));
  }
  return {};
}

Array<ObjectRef> MarkBlockAttrs::Apply(const Schedule& sch,
                                       const Array<Optional<ObjectRef>>& inputs,
                                       const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 2);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  TVM_META_SCHEDULE_INST_CAST(PrimExpr, ann_val, inputs[1]);
  sch->MarkBlock(block, ann_key, ann_val);
  return {};
}

Array<ObjectRef> FuseAttrs::Apply(const Schedule& sch,  //
                                  const Array<Optional<ObjectRef>>& inputs,
                                  const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  int n_loops = inputs.size();
  Array<LoopRV> loops;
  loops.reserve(n_loops);
  for (int i = 0; i < n_loops; ++i) {
    TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[i]);
    loops.push_back(loop);
  }
  return {sch->Fuse(loops)};
}

Array<ObjectRef> SplitAttrs::Apply(const Schedule& sch,  //
                                   const Array<Optional<ObjectRef>>& inputs,
                                   const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  CHECK_GE(inputs.size(), 3);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  Array<Optional<PrimExpr>> factors;
  for (int i = 1, n = inputs.size(); i < n; ++i) {
    if (inputs[i].defined()) {
      TVM_META_SCHEDULE_INST_CAST(PrimExpr, factor, inputs[i]);
      factors.push_back(factor);
    } else {
      factors.push_back(NullOpt);
    }
  }
  return AdaptOutputs(sch->Split(loop, factors));
}

Array<ObjectRef> ReorderAttrs::Apply(const Schedule& sch,  //
                                     const Array<Optional<ObjectRef>>& inputs,
                                     const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  Array<LoopRV> after_axes;
  for (const ObjectRef& obj : inputs) {
    if (const auto* loop = obj.as<tir::LoopRVNode>()) {
      after_axes.push_back(GetRef<LoopRV>(loop));
    } else {
      LOG(FATAL) << "TypeError: Expects LoopRV, but gets: " << obj->GetTypeKey();
    }
  }
  sch->Reorder(after_axes);
  return {};
}

Array<ObjectRef> ComputeAtAttrs::Apply(const Schedule& sch,
                                       const Array<Optional<ObjectRef>>& inputs,
                                       const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 2);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[1]);
  sch->ComputeAt(block, loop, this->preserve_unit_loop);
  return {};
}

Array<ObjectRef> ReverseComputeAtAttrs::Apply(const Schedule& sch,
                                              const Array<Optional<ObjectRef>>& inputs,
                                              const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 2);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[1]);
  sch->ReverseComputeAt(block, loop, this->preserve_unit_loop);
  return {};
}

Array<ObjectRef> ComputeInlineAttrs::Apply(const Schedule& sch,
                                           const Array<Optional<ObjectRef>>& inputs,
                                           const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  sch->ComputeInline(block);
  return {};
}

Array<ObjectRef> ReverseComputeInlineAttrs::Apply(const Schedule& sch,
                                                  const Array<Optional<ObjectRef>>& inputs,
                                                  const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  sch->ReverseComputeInline(block);
  return {};
}

Array<ObjectRef> CacheReadAttrs::Apply(const Schedule& sch,
                                       const Array<Optional<ObjectRef>>& inputs,
                                       const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  return {sch->CacheRead(block, i, storage_scope)};
}

Array<ObjectRef> CacheWriteAttrs::Apply(const Schedule& sch,
                                        const Array<Optional<ObjectRef>>& inputs,
                                        const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  return {sch->CacheWrite(block, i, storage_scope)};
}

Array<ObjectRef> BlockizeAttrs::Apply(const Schedule& sch,  //
                                      const Array<Optional<ObjectRef>>& inputs,
                                      const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  return {sch->Blockize(loop)};
}

Array<ObjectRef> DecomposeReductionAttrs::Apply(const Schedule& sch,
                                                const Array<Optional<ObjectRef>>& inputs,
                                                const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 2);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[1]);
  return {sch->DecomposeReduction(block, loop)};
}

Array<ObjectRef> TensorizeAttrs::Apply(const Schedule& sch,
                                       const Array<Optional<ObjectRef>>& inputs,
                                       const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  sch->Tensorize(loop, this->tensor_intrin_name);
  return {};
}

Array<ObjectRef> ParallelAttrs::Apply(const Schedule& sch,  //
                                      const Array<Optional<ObjectRef>>& inputs,
                                      const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  sch->Parallel(loop);
  return {};
}

Array<ObjectRef> VectorizeAttrs::Apply(const Schedule& sch,
                                       const Array<Optional<ObjectRef>>& inputs,
                                       const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  sch->Vectorize(loop);
  return {};
}

Array<ObjectRef> UnrollAttrs::Apply(const Schedule& sch,  //
                                    const Array<Optional<ObjectRef>>& inputs,
                                    const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  sch->Unroll(loop);
  return {};
}

Array<ObjectRef> RFactorAttrs::Apply(const Schedule& sch,  //
                                     const Array<Optional<ObjectRef>>& inputs,
                                     const Optional<ObjectRef>& decision) const {
  CHECK(!decision.defined());
  CHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  return {sch->RFactor(loop, factor_axis)};
}

Array<ObjectRef> BindAttrs::Apply(const Schedule& sch,  //
                                  const Array<Optional<ObjectRef>>& inputs,
                                  const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(LoopRV, loop, inputs[0]);
  sch->Bind(loop, thread_axis);
  return {};
}

Array<ObjectRef> SetScopeAttrs::Apply(const Schedule& sch,  //
                                      const Array<Optional<ObjectRef>>& inputs,
                                      const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  sch->SetScope(block, i, storage_scope);
  return {};
}

Array<ObjectRef> StorageAlignAttrs::Apply(const Schedule& sch,  //
                                          const Array<Optional<ObjectRef>>& inputs,
                                          const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 1);
  TVM_META_SCHEDULE_INST_CAST(BlockRV, block, inputs[0]);
  sch->StorageAlign(block, buffer_index, axis, factor, offset);
  return {};
}

Array<ObjectRef> EnterPostProcAttrs::Apply(const Schedule& sch,
                                           const Array<Optional<ObjectRef>>& inputs,
                                           const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  ICHECK_EQ(inputs.size(), 0);
  sch->EnterPostProc();
  return {};
}

#undef TVM_META_SCHEDULE_INST_CAST

/**************** AsPython  ****************/

struct PythonAPICall {
  String method_name;
  std::vector<String> arg_names;
  std::vector<String> args;
  Optional<String> output;

  explicit PythonAPICall(const String& method_name) : method_name(method_name), output(NullOpt) {}

  void AddArgAttr(const String& arg_name, const ObjectRef& arg) {
    std::ostringstream os;
    os << arg;
    arg_names.push_back(arg_name);
    args.push_back(os.str());
  }

  void AddArgAttr(const String& arg_name, int arg) {
    arg_names.push_back(arg_name);
    args.push_back(std::to_string(arg));
  }

  void AddArgInput(const String& arg_name, const String& arg) {
    arg_names.push_back(arg_name);
    args.push_back(arg);
  }

  void AddArgInputList(const String& arg_name, const Array<String>& arg) {
    std::ostringstream oss;
    oss << '[';
    for (int i = 0, n = arg.size(); i < n; ++i) {
      if (i > 0) {
        oss << ", ";
      }
      oss << arg[i];
    }
    oss << ']';
    arg_names.push_back(arg_name);
    args.push_back(oss.str());
  }

  void AddDecision(const Optional<ObjectRef>& decision) {
    if (decision.defined()) {
      std::ostringstream os;
      os << decision;
      arg_names.push_back("decision");
      args.push_back(os.str());
    }
  }

  void AddOutput(const String& single_output) { output = single_output; }

  void AddOutputs(const Array<String>& outputs) {
    if (outputs.empty()) {
      return;
    }
    if (outputs.size() == 1) {
      output = outputs[0] + ",";
      return;
    }
    std::ostringstream oss;
    oss << outputs[0];
    for (int i = 1, n = outputs.size(); i < n; ++i) {
      oss << ", " << outputs[i];
    }
    output = oss.str();
  }

  void Print(std::ostream& os) const {
    if (output.defined()) {
      os << output.value() << " = ";
    }
    os << "sch." << method_name << '(';
    int n = args.size();
    for (int i = 0; i < n; ++i) {
      if (i > 0) {
        os << ", ";
      }
      os << arg_names[i] << '=' << args[i];
    }
    os << ')';
  }
};

/**************** (AsPython) Sampling  ****************/

void SamplePerfectTileAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                                      const Array<String>& outputs,
                                      const Optional<ObjectRef>& decision) const {
  PythonAPICall py("sample_perfect_tile");
  py.AddArgAttr("n", this->n);
  py.AddArgInput("loop", inputs[0]);
  py.AddArgAttr("max_innermost_factor", this->max_innermost_factor);
  py.AddDecision(decision);
  py.AddOutputs(outputs);
  py.Print(os);
}

void SampleCategoricalAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                                      const Array<String>& outputs,
                                      const Optional<ObjectRef>& decision) const {
  PythonAPICall py("sample_categorical");
  py.AddArgAttr("candidates", this->candidates);
  py.AddArgAttr("probs", this->probs);
  py.AddDecision(decision);
  py.AddOutput(outputs[0]);
  py.Print(os);
}

void SampleComputeLocationAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                                          const Array<String>& outputs,
                                          const Optional<ObjectRef>& decision) const {
  PythonAPICall py("sample_compute_location");
  py.AddArgInput("block", inputs[0]);
  py.AddDecision(decision);
  py.AddOutput(outputs[0]);
  py.Print(os);
}

/**************** (AsPython) Block/Loop Relationship ****************/

void GetProducersAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                                 const Array<String>& outputs,
                                 const Optional<ObjectRef>& decision) const {
  PythonAPICall py("get_producers");
  py.AddArgInput("block", inputs[0]);
  py.AddOutputs(outputs);
  py.Print(os);
}

void GetConsumersAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                                 const Array<String>& outputs,
                                 const Optional<ObjectRef>& decision) const {
  PythonAPICall py("get_consumers");
  py.AddArgInput("block", inputs[0]);
  py.AddOutputs(outputs);
  py.Print(os);
}

void GetBlockAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                             const Array<String>& outputs,
                             const Optional<ObjectRef>& decision) const {
  PythonAPICall py("get_block");
  py.AddArgAttr("name", this->name);
  py.AddOutput(outputs[0]);
  py.Print(os);
}

void GetAxesAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                            const Array<String>& outputs,
                            const Optional<ObjectRef>& decision) const {
  PythonAPICall py("get_axes");
  py.AddArgInput("block", inputs[0]);
  py.AddOutputs(outputs);
  py.Print(os);
}

/**************** (AsPython) Scheduling Primitives ****************/

void MarkLoopAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                             const Array<String>& outputs,
                             const Optional<ObjectRef>& decision) const {
  PythonAPICall py("mark_loop");
  if (ann_val.empty()) {
    py.AddArgInput("loop", inputs[0]);
    py.AddArgAttr("ann_key", this->ann_key);
    py.AddArgInput("ann_val", inputs[1]);
  } else {
    py.AddArgInput("loop", inputs[0]);
    py.AddArgAttr("ann_key", this->ann_key);
    py.AddArgAttr("ann_val", this->ann_val);
  }
  py.Print(os);
}

void MarkBlockAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                              const Array<String>& outputs,
                              const Optional<ObjectRef>& decision) const {
  PythonAPICall py("mark_block");
  py.AddArgInput("block", inputs[0]);
  py.AddArgAttr("ann_key", this->ann_key);
  py.AddArgInput("ann_val", inputs[1]);
  py.Print(os);
}

void FuseAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                         const Array<String>& outputs, const Optional<ObjectRef>& decision) const {
  PythonAPICall py("fuse");
  py.AddArgInputList("loops", inputs);
  py.AddOutput(outputs[0]);
  py.Print(os);
}

void SplitAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                          const Array<String>& outputs, const Optional<ObjectRef>& decision) const {
  PythonAPICall py("split");
  py.AddArgInput("loop", inputs[0]);
  py.AddArgInputList("factors", {inputs.begin() + 1, inputs.end()});
  py.AddOutputs(outputs);
  py.Print(os);
}

void ReorderAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                            const Array<String>& outputs,
                            const Optional<ObjectRef>& decision) const {
  PythonAPICall py("reorder");
  py.AddArgInputList("after_axes", inputs);
  py.Print(os);
}

void ComputeAtAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                              const Array<String>& outputs,
                              const Optional<ObjectRef>& decision) const {
  PythonAPICall py("compute_at");
  py.AddArgInput("block", inputs[0]);
  py.AddArgInput("loop", inputs[1]);
  py.AddArgAttr("preserve_unit_loop", this->preserve_unit_loop);
  py.Print(os);
}

void ReverseComputeAtAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                                     const Array<String>& outputs,
                                     const Optional<ObjectRef>& decision) const {
  PythonAPICall py("reverse_compute_at");
  py.AddArgInput("block", inputs[0]);
  py.AddArgInput("loop", inputs[1]);
  py.AddArgAttr("preserve_unit_loop", this->preserve_unit_loop);
  py.Print(os);
}

void ComputeInlineAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                                  const Array<String>& outputs,
                                  const Optional<ObjectRef>& decision) const {
  PythonAPICall py("compute_inline");
  py.AddArgInput("block", inputs[0]);
  py.Print(os);
}

void ReverseComputeInlineAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                                         const Array<String>& outputs,
                                         const Optional<ObjectRef>& decision) const {
  PythonAPICall py("reverse_compute_inline");
  py.AddArgInput("block", inputs[0]);
  py.Print(os);
}

void CacheReadAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                              const Array<String>& outputs,
                              const Optional<ObjectRef>& decision) const {
  PythonAPICall py("cache_read");
  py.AddArgInput("block", inputs[0]);
  py.AddArgAttr("i", this->i);
  py.AddArgAttr("storage_scope", this->storage_scope);
  py.AddOutput(outputs[0]);
  py.Print(os);
}

void CacheWriteAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                               const Array<String>& outputs,
                               const Optional<ObjectRef>& decision) const {
  PythonAPICall py("cache_write");
  py.AddArgInput("block", inputs[0]);
  py.AddArgAttr("i", this->i);
  py.AddArgAttr("storage_scope", this->storage_scope);
  py.AddOutput(outputs[0]);
  py.Print(os);
}

void BlockizeAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                             const Array<String>& outputs,
                             const Optional<ObjectRef>& decision) const {
  PythonAPICall py("blockize");
  py.AddArgInput("loop", inputs[0]);
  py.AddOutput(outputs[0]);
  py.Print(os);
}

void DecomposeReductionAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                                       const Array<String>& outputs,
                                       const Optional<ObjectRef>& decision) const {
  PythonAPICall py("decompose_reduction");
  py.AddArgInput("block", inputs[0]);
  py.AddArgInput("loop", inputs[1]);
  py.AddOutput(outputs[0]);
  py.Print(os);
}

void TensorizeAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                              const Array<String>& outputs,
                              const Optional<ObjectRef>& decision) const {
  PythonAPICall py("tensorize");
  py.AddArgInput("loop", inputs[0]);
  py.AddArgAttr("tensor_intrin_name", this->tensor_intrin_name);
  py.Print(os);
}

void ParallelAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                             const Array<String>& outputs,
                             const Optional<ObjectRef>& decision) const {
  PythonAPICall py("parallel");
  py.AddArgInput("loop", inputs[0]);
  py.Print(os);
}

void VectorizeAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                              const Array<String>& outputs,
                              const Optional<ObjectRef>& decision) const {
  PythonAPICall py("vectorize");
  py.AddArgInput("loop", inputs[0]);
  py.Print(os);
}

void UnrollAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                           const Array<String>& outputs,
                           const Optional<ObjectRef>& decision) const {
  PythonAPICall py("unroll");
  py.AddArgInput("loop", inputs[0]);
  py.Print(os);
}

void RFactorAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                            const Array<String>& outputs,
                            const Optional<ObjectRef>& decision) const {
  PythonAPICall py("rfactor");
  py.AddArgInput("loop", inputs[0]);
  py.AddArgAttr("factor_axis", this->factor_axis);
  py.AddOutput(outputs[0]);
  py.Print(os);
}

void BindAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                         const Array<String>& outputs, const Optional<ObjectRef>& decision) const {
  PythonAPICall py("bind");
  py.AddArgInput("loop", inputs[0]);
  py.AddArgAttr("thread_axis", this->thread_axis);
  py.Print(os);
}

void SetScopeAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                             const Array<String>& outputs,
                             const Optional<ObjectRef>& decision) const {
  PythonAPICall py("set_scope");
  py.AddArgInput("block", inputs[0]);
  py.AddArgAttr("i", this->i);
  py.AddArgAttr("storage_scope", this->storage_scope);
  py.Print(os);
}

void StorageAlignAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                                 const Array<String>& outputs,
                                 const Optional<ObjectRef>& decision) const {
  PythonAPICall py("storage_align");
  py.AddArgInput("block", inputs[0]);
  py.AddArgAttr("buffer_index", buffer_index);
  py.AddArgAttr("axis", axis);
  py.AddArgAttr("factor", factor);
  py.AddArgAttr("offset", offset);
  py.Print(os);
}

void EnterPostProcAttrs::AsPython(std::ostream& os, const Array<String>& inputs,
                                  const Array<String>& outputs,
                                  const Optional<ObjectRef>& decision) const {
  os << "# Postprocessing";
}

/**************** Serialize  ****************/
/**************** (Serialize) Sampling  ****************/

void SamplePerfectTileAttrs::Serialize(Array<ObjectRef>* record,
                                       const Optional<ObjectRef>& decision) const {
  record->push_back(Integer(n));
  record->push_back(Integer(max_innermost_factor));
  if (decision.defined()) {
    record->push_back(decision.value());
  }
}

void SampleCategoricalAttrs::Serialize(Array<ObjectRef>* record,
                                       const Optional<ObjectRef>& decision) const {
  record->push_back(candidates);
  record->push_back(probs);
  if (decision.defined()) {
    record->push_back(decision.value());
  }
}

void SampleComputeLocationAttrs::Serialize(Array<ObjectRef>* record,
                                           const Optional<ObjectRef>& decision) const {
  if (decision.defined()) {
    record->push_back(decision.value());
  }
}

/**************** (Serialize) Block/Loop Relationship  ****************/

void GetBlockAttrs::Serialize(Array<ObjectRef>* record, const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  record->push_back(name);
}

/**************** (Serialize) Scheduling Primitives  ****************/

void MarkLoopAttrs::Serialize(Array<ObjectRef>* record, const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  record->push_back(this->ann_key);
  record->push_back(this->ann_val);
}

void MarkBlockAttrs::Serialize(Array<ObjectRef>* record,
                               const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  record->push_back(this->ann_key);
}

void ComputeAtAttrs::Serialize(Array<ObjectRef>* record,
                               const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  record->push_back(Bool(this->preserve_unit_loop));
}

void ReverseComputeAtAttrs::Serialize(Array<ObjectRef>* record,
                                      const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  record->push_back(Bool(this->preserve_unit_loop));
}

void CacheReadAttrs::Serialize(Array<ObjectRef>* record,
                               const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  record->push_back(Integer(i));
  record->push_back(this->storage_scope);
}

void CacheWriteAttrs::Serialize(Array<ObjectRef>* record,
                                const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  record->push_back(Integer(i));
  record->push_back(this->storage_scope);
}

void BlockizeAttrs::Serialize(Array<ObjectRef>* record, const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
}

void TensorizeAttrs::Serialize(Array<ObjectRef>* record,
                               const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  record->push_back(this->tensor_intrin_name);
}

void RFactorAttrs::Serialize(Array<ObjectRef>* record, const Optional<ObjectRef>& decision) const {
  CHECK(!decision.defined());
  record->push_back(Integer(factor_axis));
}

void BindAttrs::Serialize(Array<ObjectRef>* record, const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  record->push_back(this->thread_axis);
}

void SetScopeAttrs::Serialize(Array<ObjectRef>* record, const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  record->push_back(Integer(i));
  record->push_back(this->storage_scope);
}

void StorageAlignAttrs::Serialize(Array<ObjectRef>* record,
                                  const Optional<ObjectRef>& decision) const {
  ICHECK(!decision.defined());
  record->push_back(Integer(this->buffer_index));
  record->push_back(Integer(this->axis));
  record->push_back(Integer(this->factor));
  record->push_back(Integer(this->offset));
}

/**************** Deserialize  ****************/
/**************** (Deserialize) Sampling  ****************/

InstAttrs SamplePerfectTileAttrs::Deserialize(const Array<ObjectRef>& record,
                                              Optional<ObjectRef>* decision) {
  ObjectPtr<SamplePerfectTileAttrs> n = make_object<SamplePerfectTileAttrs>();
  n->n = Downcast<Integer>(record[3]);
  n->max_innermost_factor = Downcast<Integer>(record[4]);
  if (record.size() > 5) {
    *decision = Downcast<Array<Integer>>(record[5]);
  }
  return InstAttrs(std::move(n));
}

InstAttrs SampleCategoricalAttrs::Deserialize(const Array<ObjectRef>& record,
                                              Optional<ObjectRef>* decision) {
  ObjectPtr<SampleCategoricalAttrs> n = make_object<SampleCategoricalAttrs>();
  n->candidates = Downcast<Array<Integer>>(record[3]);
  n->probs = Downcast<Array<FloatImm>>(record[4]);
  if (record.size() > 5) {
    *decision = Downcast<Integer>(record[5]);
  }
  return InstAttrs(std::move(n));
}

InstAttrs SampleComputeLocationAttrs::Deserialize(const Array<ObjectRef>& record,
                                                  Optional<ObjectRef>* decision) {
  if (record.size() > 3) {
    *decision = Downcast<Integer>(record[3]);
  }
  return InstAttrs(make_object<SampleComputeLocationAttrs>());
}

/**************** (Deserialize) Block/Loop Relationship  ****************/

InstAttrs GetBlockAttrs::Deserialize(const Array<ObjectRef>& record,
                                     Optional<ObjectRef>* decision) {
  ObjectPtr<GetBlockAttrs> n = make_object<GetBlockAttrs>();
  n->name = Downcast<String>(record[3]);
  return InstAttrs(std::move(n));
}

/**************** (Deserialize) Scheduling Primitives  ****************/

InstAttrs MarkLoopAttrs::Deserialize(const Array<ObjectRef>& record,
                                     Optional<ObjectRef>* decision) {
  ObjectPtr<MarkLoopAttrs> n = make_object<MarkLoopAttrs>();
  n->ann_key = Downcast<String>(record[3]);
  n->ann_val = Downcast<String>(record[4]);
  return InstAttrs(std::move(n));
}

InstAttrs MarkBlockAttrs::Deserialize(const Array<ObjectRef>& record,
                                      Optional<ObjectRef>* decision) {
  ObjectPtr<MarkBlockAttrs> n = make_object<MarkBlockAttrs>();
  n->ann_key = Downcast<String>(record[3]);
  return InstAttrs(std::move(n));
}

InstAttrs ComputeAtAttrs::Deserialize(const Array<ObjectRef>& record,
                                      Optional<ObjectRef>* decision) {
  ObjectPtr<ComputeAtAttrs> n = make_object<ComputeAtAttrs>();
  n->preserve_unit_loop = Downcast<Bool>(record[3]);
  return InstAttrs(std::move(n));
}

InstAttrs ReverseComputeAtAttrs::Deserialize(const Array<ObjectRef>& record,
                                             Optional<ObjectRef>* decision) {
  ObjectPtr<ReverseComputeAtAttrs> n = make_object<ReverseComputeAtAttrs>();
  n->preserve_unit_loop = Downcast<Bool>(record[3]);
  return InstAttrs(std::move(n));
}
InstAttrs CacheReadAttrs::Deserialize(const Array<ObjectRef>& record,
                                      Optional<ObjectRef>* decision) {
  ObjectPtr<CacheReadAttrs> n = make_object<CacheReadAttrs>();
  n->i = Downcast<Integer>(record[3]);
  n->storage_scope = Downcast<String>(record[4]);
  return InstAttrs(std::move(n));
}

InstAttrs CacheWriteAttrs::Deserialize(const Array<ObjectRef>& record,
                                       Optional<ObjectRef>* decision) {
  ObjectPtr<CacheWriteAttrs> n = make_object<CacheWriteAttrs>();
  n->i = Downcast<Integer>(record[3]);
  n->storage_scope = Downcast<String>(record[4]);
  return InstAttrs(std::move(n));
}

InstAttrs BlockizeAttrs::Deserialize(const Array<ObjectRef>& record,
                                     Optional<ObjectRef>* decision) {
  ObjectPtr<BlockizeAttrs> n = make_object<BlockizeAttrs>();
  return InstAttrs(std::move(n));
}

InstAttrs TensorizeAttrs::Deserialize(const Array<ObjectRef>& record,
                                      Optional<ObjectRef>* decision) {
  ObjectPtr<TensorizeAttrs> n = make_object<TensorizeAttrs>();
  n->tensor_intrin_name = Downcast<String>(record[3]);
  return InstAttrs(std::move(n));
}

InstAttrs RFactorAttrs::Deserialize(const Array<ObjectRef>& record, Optional<ObjectRef>* decision) {
  ObjectPtr<RFactorAttrs> n = make_object<RFactorAttrs>();
  n->factor_axis = Downcast<Integer>(record[3]);
  return InstAttrs(std::move(n));
}

InstAttrs BindAttrs::Deserialize(const Array<ObjectRef>& record, Optional<ObjectRef>* decision) {
  ObjectPtr<BindAttrs> n = make_object<BindAttrs>();
  n->thread_axis = Downcast<String>(record[3]);
  return InstAttrs(std::move(n));
}

InstAttrs SetScopeAttrs::Deserialize(const Array<ObjectRef>& record,
                                     Optional<ObjectRef>* decision) {
  ObjectPtr<SetScopeAttrs> n = make_object<SetScopeAttrs>();
  n->i = Downcast<Integer>(record[3]);
  n->storage_scope = Downcast<String>(record[4]);
  return InstAttrs(std::move(n));
}

InstAttrs StorageAlignAttrs::Deserialize(const Array<ObjectRef>& record,
                                         Optional<ObjectRef>* decision) {
  ObjectPtr<StorageAlignAttrs> n = make_object<StorageAlignAttrs>();
  n->buffer_index = Downcast<Integer>(record[3]);
  n->axis = Downcast<Integer>(record[4]);
  n->factor = Downcast<Integer>(record[5]);
  n->factor = Downcast<Integer>(record[6]);
  return InstAttrs(std::move(n));
}

/**************** Deserialize/Serialize for empty instructions ****************/

#define TVM_META_SCHEDULE_INST_IO_EMPTY(AttrsType)                                                 \
  void AttrsType::Serialize(Array<ObjectRef>* record, const Optional<ObjectRef>& decision) const { \
    ICHECK(!decision.defined());                                                                   \
  }                                                                                                \
  InstAttrs AttrsType::Deserialize(const Array<ObjectRef>& record,                                 \
                                   Optional<ObjectRef>* decision) {                                \
    return InstAttrs(make_object<AttrsType>());                                                    \
  }

TVM_META_SCHEDULE_INST_IO_EMPTY(GetProducersAttrs);
TVM_META_SCHEDULE_INST_IO_EMPTY(GetConsumersAttrs);
TVM_META_SCHEDULE_INST_IO_EMPTY(GetAxesAttrs);
TVM_META_SCHEDULE_INST_IO_EMPTY(FuseAttrs);
TVM_META_SCHEDULE_INST_IO_EMPTY(SplitAttrs);
TVM_META_SCHEDULE_INST_IO_EMPTY(ReorderAttrs);
TVM_META_SCHEDULE_INST_IO_EMPTY(ComputeInlineAttrs);
TVM_META_SCHEDULE_INST_IO_EMPTY(ReverseComputeInlineAttrs);
TVM_META_SCHEDULE_INST_IO_EMPTY(DecomposeReductionAttrs);
TVM_META_SCHEDULE_INST_IO_EMPTY(EnterPostProcAttrs);
TVM_META_SCHEDULE_INST_IO_EMPTY(ParallelAttrs);
TVM_META_SCHEDULE_INST_IO_EMPTY(VectorizeAttrs);
TVM_META_SCHEDULE_INST_IO_EMPTY(UnrollAttrs);

#undef TVM_META_SCHEDULE_INST_Serialize_IMPORT_EMPTY

/**************** FFI ****************/

TVM_REGISTER_OBJECT_TYPE(InstAttrsNode);
TVM_REGISTER_NODE_TYPE(InstructionNode);
TVM_REGISTER_NODE_TYPE(SamplePerfectTileAttrs);
TVM_REGISTER_NODE_TYPE(SampleCategoricalAttrs);
TVM_REGISTER_NODE_TYPE(SampleComputeLocationAttrs);
TVM_REGISTER_NODE_TYPE(GetProducersAttrs);
TVM_REGISTER_NODE_TYPE(GetConsumersAttrs);
TVM_REGISTER_NODE_TYPE(GetBlockAttrs);
TVM_REGISTER_NODE_TYPE(GetAxesAttrs);
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
TVM_REGISTER_NODE_TYPE(TensorizeAttrs);
TVM_REGISTER_NODE_TYPE(ParallelAttrs);
TVM_REGISTER_NODE_TYPE(VectorizeAttrs);
TVM_REGISTER_NODE_TYPE(UnrollAttrs);
TVM_REGISTER_NODE_TYPE(RFactorAttrs);
TVM_REGISTER_NODE_TYPE(BindAttrs);
TVM_REGISTER_NODE_TYPE(SetScopeAttrs);
TVM_REGISTER_NODE_TYPE(StorageAlignAttrs);
TVM_REGISTER_NODE_TYPE(EnterPostProcAttrs);

}  // namespace meta_schedule
}  // namespace tvm
