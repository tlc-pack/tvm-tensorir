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
#include "./trace.h"  // NOLINT(build/include)

#include <tvm/tir/stmt_functor.h>

#include "./schedule.h"

namespace tvm {
namespace meta_schedule {

/**************** Constructor ****************/

Trace::Trace() { data_ = make_object<TraceNode>(); }

Trace::Trace(Array<Instruction> insts, Map<Instruction, Array<ObjectRef>> decisions) {
  ObjectPtr<TraceNode> n = make_object<TraceNode>();
  n->insts = std::move(insts);
  n->decisions = std::move(decisions);
  data_ = std::move(n);
}

/**************** Mutation ****************/

void TraceNode::Append(const Instruction& inst) { insts.push_back(inst); }

void TraceNode::Append(const Instruction& inst, const Array<ObjectRef>& decision) {
  insts.push_back(inst);
  decisions.Set(inst, decision);
}

Optional<Instruction> TraceNode::Pop() {
  if (insts.empty()) {
    return NullOpt;
  }
  Instruction inst = insts.back();
  insts.pop_back();
  if (decisions.count(inst)) {
    decisions.erase(inst);
  }
  return inst;
}

/**************** Serialization ****************/

void TraceNode::Apply(const Schedule& sch) const {
  // Maps an old random variable to its corresponding new random variable in the re-sampling
  std::unordered_map<const Object*, const Object*> var_map;
  // Utility function to convert an old tir::Var to the new one, according to `var_map`
  auto f_var_convert = [&var_map](const tir::Var& var) -> Optional<PrimExpr> {
    const Object* src = var.get();
    if (!var_map.count(src)) {
      return NullOpt;
    }
    const Object* dst = var_map.at(var.get());
    CHECK(dst->IsInstance<tir::VarNode>());
    return GetRef<tir::Var>(static_cast<const tir::VarNode*>(dst));
  };
  // Utility function to convert an old expression to the one, according to `var_map`
  auto f_var_map = [&var_map, &f_var_convert](const ObjectRef& obj) -> ObjectRef {
    if (const auto* expr = obj.as<PrimExprNode>()) {
      return tir::Substitute(GetRef<PrimExpr>(expr), f_var_convert);
    }
    const Object* src = obj.get();
    CHECK(var_map.count(src));
    const Object* dst = var_map.at(src);
    return GetRef<ObjectRef>(dst);
  };
  // Redo all the instructions in the trace
  for (const Instruction& inst : this->insts) {
    // Stop before post-processing
    if (inst->inst_attrs->IsInstance<EnterPostProcAttrs>()) {
      break;
    }
    // Step 1. Extract old inputs and construct new inputs
    const Array<ObjectRef>& old_inputs = inst->inputs;
    Array<ObjectRef> new_inputs;
    {
      new_inputs.reserve(old_inputs.size());
      for (const ObjectRef& old_input : old_inputs) {
        new_inputs.push_back(f_var_map(old_input));
      }
    }
    // Step 2. Apply the instruction to the schedule to get new outputs
    Array<ObjectRef> new_outputs =
        inst->inst_attrs->ApplyToSchedule(sch, new_inputs, decisions.Get(inst));
    // Step 3. Step up the correspondence between old outputs and construct new outputs
    {
      const Array<ObjectRef>& old_outputs = inst->outputs;
      CHECK_EQ(old_outputs.size(), new_outputs.size()) << "ValueError: Output size mismatch";
      for (int i = 0, n = new_outputs.size(); i < n; ++i) {
        var_map[old_outputs[i].get()] = new_outputs[i].get();
      }
    }
  }
}

ObjectRef TraceNode::Serialize() const {
  Map<ObjectRef, String> rv_names;
  // Allocate names for random variables
  for (const Instruction& inst : this->insts) {
    for (const ObjectRef& output : inst->outputs) {
      int i = rv_names.size();
      CHECK(!rv_names.count(output));
      if (output->IsInstance<BlockRVNode>()) {
        rv_names.Set(output, "b" + std::to_string(i));
      } else if (output->IsInstance<LoopRVNode>()) {
        rv_names.Set(output, "l" + std::to_string(i));
      } else if (output->IsInstance<BufferRVNode>()) {
        rv_names.Set(output, "c" + std::to_string(i));
      } else if (output->IsInstance<tir::VarNode>()) {
        rv_names.Set(output, "v" + std::to_string(i));
      } else {
        LOG(FATAL) << "TypeError: Cannot recognize the type of the random variable: "
                   << output->GetTypeKey();
        throw;
      }
    }
  }
  // Export to JSON
  Array<ObjectRef> json;
  for (const Instruction& inst : this->insts) {
    if (inst->inst_attrs->IsInstance<EnterPostProcAttrs>()) {
      break;
    }
    json.push_back(inst->Export(rv_names, decisions.Get(inst)));
  }
  return json;
}

void TraceNode::Deserialize(const ObjectRef& json, const Schedule& sch) {
  const ArrayNode* array = json.as<ArrayNode>();
  CHECK(array) << "TypeError: Expects Array, but gets: " << json->GetTypeKey();
  // Random variables created on the fly
  Map<String, ObjectRef> named_rvs;
  // For each instruction
  for (auto iter = array->begin(), end = array->end(); iter != end; ++iter) {
    // Extract the serialized JSON array for the instruction
    const ArrayNode* inst = (*iter).as<ArrayNode>();
    CHECK(inst) << "TypeError: Expects Array, but gets: " << (*iter)->GetTypeKey();
    // Deserialize it
    Instruction::ImportToSchedule(sch, GetRef<Array<ObjectRef>>(inst), &named_rvs);
  }
}

/**************** Def-Use ****************/

struct DefUseSites {
  int def;
  std::vector<int> use;
};

std::unordered_map<const Object*, DefUseSites> ExtractDefUseSites(const Trace& trace) {
  std::unordered_map<const Object*, DefUseSites> result;
  int i = 0;
  for (const Instruction& inst : trace->insts) {
    // Stop analysis on postprocssing
    if (trace->insts[i]->inst_attrs->IsInstance<EnterPostProcAttrs>()) {
      break;
    }
    // Record def
    for (const ObjectRef& def : inst->inputs) {
      if (IsRV(def)) {
        result[def.get()].def = i;
      }
    }
    // Record use
    for (const ObjectRef& use : inst->outputs) {
      if (IsRV(use)) {
        result[use.get()].use.push_back(i);
      } else if (IsRVExpr(use)) {
        tir::PostOrderVisit(use, [&result, &i](const ObjectRef& obj) {
          if (const auto* var = obj.as<tir::VarNode>()) {
            result[var].use.push_back(i);
          }
        });
      }
    }
    ++i;
  }
  return result;
}

/**************** Dead code elimination ****************/

Trace DeadCodeElimination(const Trace& trace) {
  std::unordered_map<const Object*, DefUseSites> def_use = ExtractDefUseSites(trace);
  // Step 1. Calculate number of instructions
  int n_inst = trace->insts.size();
  for (int i = 0; i < n_inst; ++i) {
    if (trace->insts[i]->inst_attrs->IsInstance<EnterPostProcAttrs>()) {
      n_inst = i;
    }
  }
  // Step 2. Check in reverse order if the instruction is dead
  std::vector<int> dead(n_inst, 0);
  for (int i = n_inst - 1; i >= 0; --i) {
    // Never remove effectful instructions
    if (!trace->insts[i]->inst_attrs->IsPure()) {
      continue;
    }
    bool all_outputs_dead = true;
    // For each output, check if it is dead
    for (const ObjectRef& output : trace->insts[i]->outputs) {
      // Skip the non-random-variables
      if (!def_use.count(output.get())) {
        continue;
      }
      // Check if all the instructions using the output are dead
      bool all_use_dead = true;
      for (int use_site : def_use.at(output.get()).use) {
        if (!dead[use_site]) {
          all_use_dead = false;
          break;
        }
      }
      if (!all_use_dead) {
        all_outputs_dead = false;
        break;
      }
    }
    if (all_outputs_dead) {
      dead[i] = 1;
    }
  }
  Trace result;
  for (int i = 0; i < n_inst; ++i) {
    if (dead[i]) {
      continue;
    }
    Instruction inst = trace->insts[i];
    result->insts.push_back(inst);
    if (Optional<Array<ObjectRef>> decision = trace->decisions.Get(inst)) {
      result->decisions.Set(inst, decision.value());
    }
  }
  return result;
}

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(TraceNode);

}  // namespace meta_schedule
}  // namespace tvm
