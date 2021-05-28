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

#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt_functor.h>

#include "./schedule.h"

namespace tvm {
namespace meta_schedule {

/**************** Constructor ****************/

Trace::Trace() { data_ = make_object<TraceNode>(); }

Trace::Trace(Array<Instruction> insts, Map<Instruction, ObjectRef> decisions) {
  ObjectPtr<TraceNode> n = make_object<TraceNode>();
  n->insts = std::move(insts);
  n->decisions = std::move(decisions);
  data_ = std::move(n);
}

/**************** Utilities ****************/

int IndexPostproc(const Array<Instruction>& insts) {
  int i = 0;
  for (const Instruction& inst : insts) {
    if (inst->inst_attrs->IsInstance<EnterPostProcAttrs>()) {
      return i;
    }
    ++i;
  }
  return -1;
}

inline bool IsRV(const ObjectRef& obj) {
  if (obj->IsInstance<IntImmNode>() || obj->IsInstance<FloatImmNode>()) {
    return false;
  }
  return obj->IsInstance<BlockRVNode>() || obj->IsInstance<LoopRVNode>() ||
         obj->IsInstance<tir::VarNode>();
}

inline bool IsRVExpr(const ObjectRef& obj) { return obj->IsInstance<PrimExprNode>(); }

/**************** Mutation ****************/

void TraceNode::Append(const Instruction& inst) { insts.push_back(inst); }

void TraceNode::Append(const Instruction& inst, const ObjectRef& decision) {
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
  this->Apply(sch,
              [this](const Instruction& inst,
                     const Array<Optional<ObjectRef>>& inputs) -> Optional<ObjectRef> {
                // Keep the original decision
                return this->decisions.Get(inst);
              });
}

void TraceNode::Apply(const Schedule& sch,
                      const std::function<Optional<ObjectRef>(
                          const Instruction& inst, const Array<Optional<ObjectRef>>& inputs)>&
                          decision_provider) const {
  // Maps an old random variable to its corresponding new random variable in the re-sampling
  std::unordered_map<const Object*, const Object*> var_map;
  // Utility function to convert an old tir::Var to the new one, according to `var_map`
  auto f_var_convert = [&var_map](const tir::Var& var) -> Optional<PrimExpr> {
    const Object* src = var.get();
    if (!var_map.count(src)) {
      return NullOpt;
    }
    const Object* dst = var_map.at(var.get());
    ICHECK(dst->IsInstance<tir::VarNode>());
    return GetRef<tir::Var>(static_cast<const tir::VarNode*>(dst));
  };
  // Utility function to convert an old expression to the one, according to `var_map`
  auto f_var_map = [&var_map, &f_var_convert](const ObjectRef& obj) -> ObjectRef {
    if (const auto* expr = obj.as<PrimExprNode>()) {
      return tir::Substitute(GetRef<PrimExpr>(expr), f_var_convert);
    }
    const Object* src = obj.get();
    ICHECK(var_map.count(src));
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
    const Array<Optional<ObjectRef>>& old_inputs = inst->inputs;
    Array<Optional<ObjectRef>> new_inputs;
    {
      new_inputs.reserve(old_inputs.size());
      for (const Optional<ObjectRef>& old_input : old_inputs) {
        if (old_input.defined()) {
          new_inputs.push_back(f_var_map(old_input.value()));
        } else {
          new_inputs.push_back(NullOpt);
        }
      }
    }
    // Step 2. Apply the instruction to the schedule to get new outputs
    Array<ObjectRef> new_outputs =
        inst->inst_attrs->Apply(sch, new_inputs, decision_provider(inst, new_inputs));
    // Step 3. Step up the correspondence between old outputs and construct new outputs
    {
      const Array<ObjectRef>& old_outputs = inst->outputs;
      ICHECK_EQ(old_outputs.size(), new_outputs.size()) << "ValueError: Output size mismatch";
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
      ICHECK(!rv_names.count(output));
      if (output->IsInstance<BlockRVNode>()) {
        rv_names.Set(output, "b" + std::to_string(i));
      } else if (output->IsInstance<LoopRVNode>()) {
        rv_names.Set(output, "l" + std::to_string(i));
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
    json.push_back(inst->Serialize(rv_names, decisions.Get(inst)));
  }
  return std::move(json);
}

void TraceNode::Deserialize(const ObjectRef& json, const Schedule& sch) {
  const ArrayNode* array = json.as<ArrayNode>();
  ICHECK(array) << "TypeError: Expects Array, but gets: " << json->GetTypeKey();
  // Random variables created on the fly
  Map<String, ObjectRef> named_rvs{{"None", ObjectRef(nullptr)}};
  // For each instruction
  for (auto iter = array->begin(), end = array->end(); iter != end; ++iter) {
    // Extract the serialized JSON array for the instruction
    const ArrayNode* inst = (*iter).as<ArrayNode>();
    ICHECK(inst) << "TypeError: Expects Array, but gets: " << (*iter)->GetTypeKey();
    // Deserialize it
    InstructionNode::Deserialize(GetRef<Array<ObjectRef>>(inst), &named_rvs, sch);
  }
}

/**************** Def-Use ****************/

/*! \brief An ad hoc data structure for def-use analysis */
struct DefUseSites {
  /*! \brief The index of the instruction that defines the random variable */
  int def;
  /*! \brief The indices of the instructions that use the random variable */
  std::vector<int> use;

  static std::unordered_map<const Object*, DefUseSites> Extract(const Array<Instruction>& insts) {
    std::unordered_map<const Object*, DefUseSites> result;
    int n_insts = insts.size();
    for (int i = 0; i < n_insts; ++i) {
      const Instruction& inst = insts[i];
      // Record def
      for (const ObjectRef& def : inst->outputs) {
        if (IsRV(def)) {
          result[def.get()].def = i;
        }
      }
      // Record use
      for (const Optional<ObjectRef>& opt_use : inst->inputs) {
        if (!opt_use.defined()) {
          continue;
        }
        ObjectRef use = opt_use.value();
        // Case 1. If the use is a random variable
        if (IsRV(use)) {
          result[use.get()].use.push_back(i);
          continue;
        }
        // Case 2. If the use is a PrimExpr containing random variables
        if (!IsRVExpr(use)) {
          continue;
        }
        tir::PostOrderVisit(use, [&result, i](const ObjectRef& obj) {
          if (const auto* var = obj.as<tir::VarNode>()) {
            result[var].use.push_back(i);
          }
        });
      }
    }
    return result;
  }
};

/**************** AsPython ****************/

Array<String> TraceNode::AsPython() const {
  Map<ObjectRef, String> rv_names{{ObjectRef(nullptr), "None"}};
  // Allocate names for random variables
  for (const Instruction& inst : this->insts) {
    for (const ObjectRef& output : inst->outputs) {
      int i = rv_names.size();
      ICHECK(!rv_names.count(output));
      if (output->IsInstance<BlockRVNode>()) {
        rv_names.Set(output, "b" + std::to_string(i));
      } else if (output->IsInstance<LoopRVNode>()) {
        rv_names.Set(output, "l" + std::to_string(i));
      } else if (output->IsInstance<tir::VarNode>()) {
        rv_names.Set(output, "v" + std::to_string(i));
      } else {
        LOG(FATAL) << "TypeError: Cannot recognize the type of the random variable: "
                   << output->GetTypeKey();
        throw;
      }
    }
  }
  Array<String> result;
  for (const Instruction& inst : this->insts) {
    std::ostringstream oss;
    inst->AsPython(oss, rv_names, decisions.Get(inst));
    result.push_back(oss.str());
  }
  return result;
}

String TraceNode::Stringify() const {
  std::ostringstream os;
  for (const String& line : AsPython()) {
    os << line << '\n';
  }
  return os.str();
}

/**************** New trace creators ****************/

Trace TraceNode::WithDecision(const Instruction& inst,    //
                              const ObjectRef& decision,  //
                              bool remove_postproc) const {
  int i = remove_postproc ? IndexPostproc(this->insts) : -1;
  Array<Instruction> new_insts =
      (i == -1) ? Array<Instruction>{this->insts.begin(), this->insts.end()}
                : Array<Instruction>{this->insts.begin(), this->insts.begin() + i};
  Map<Instruction, ObjectRef> new_decisions{this->decisions.begin(), this->decisions.end()};
  new_decisions.Set(inst, decision);
  return Trace(new_insts, new_decisions);
}

Trace TraceNode::Simplified(bool remove_postproc) const {
  std::unordered_map<const Object*, DefUseSites> def_use = DefUseSites::Extract(this->insts);
  // Step 1. Calculate number of instructions
  int n_inst = remove_postproc ? IndexPostproc(this->insts) : -1;
  if (n_inst == -1) {
    n_inst = this->insts.size();
  }
  // Step 2. Check in reverse order if the instruction is dead
  std::vector<int> inst_dead(n_inst, 0);
  for (int i = n_inst - 1; i >= 0; --i) {
    const Instruction& inst = this->insts[i];
    // Never remove effectful instructions
    if (!inst->inst_attrs->IsPure()) {
      continue;
    }
    // Check if there is any variable defined by `inst`
    bool all_defs_dead = true;
    for (const ObjectRef& def : inst->outputs) {
      if (IsRV(def) && !def_use[def.get()].use.empty()) {
        // There is an RV used afterwards
        all_defs_dead = false;
        break;
      }
    }
    // If all defined variables are dead, then this instruction is dead
    if (!all_defs_dead) {
      continue;
    }
    inst_dead[i] = 1;
    // For each variable used by the instruction, remove their use site
    for (const Optional<ObjectRef>& opt_use : inst->inputs) {
      if (!opt_use.defined()) {
        continue;
      }
      ObjectRef use = opt_use.value();
      // Case 1. If the use is a random variable
      if (IsRV(use)) {
        def_use[use.get()].use.pop_back();
        continue;
      }
      // Case 2. If the use is a PrimExpr containing random variables
      if (!IsRVExpr(use)) {
        continue;
      }
      tir::PostOrderVisit(use, [&def_use](const ObjectRef& obj) {
        if (const auto* var = obj.as<tir::VarNode>()) {
          def_use[var].use.pop_back();
        }
      });
    }
  }
  // Construct the result trace
  Trace result;
  for (int i = 0; i < n_inst; ++i) {
    if (!inst_dead[i]) {
      const Instruction& inst = this->insts[i];
      result->insts.push_back(inst);
      if (Optional<ObjectRef> decision = this->decisions.Get(inst)) {
        result->decisions.Set(inst, decision.value());
      }
    }
  }
  return result;
}

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(TraceNode);

TVM_REGISTER_GLOBAL("meta_schedule.Trace")
    .set_body_typed([](Optional<Array<Instruction>> insts,
                       Optional<Map<Instruction, ObjectRef>> decisions) {
      return Trace(insts.value_or({}), decisions.value_or({}));
    });
TVM_REGISTER_GLOBAL("meta_schedule.TraceAppend")
    .set_body_typed([](Trace self, Instruction inst, Optional<ObjectRef> decision) {
      if (decision.defined()) {
        self->Append(inst, decision.value());
      } else {
        self->Append(inst);
      }
    });
TVM_REGISTER_GLOBAL("meta_schedule.TracePop").set_body_method<Trace>(&TraceNode::Pop);
TVM_REGISTER_GLOBAL("meta_schedule.TraceApply").set_body_typed([](Trace self, Schedule sch) {
  self->Apply(sch);
});
TVM_REGISTER_GLOBAL("meta_schedule.TraceSerialize").set_body_method<Trace>(&TraceNode::Serialize);
TVM_REGISTER_GLOBAL("meta_schedule.TraceDeserialize").set_body_typed(TraceNode::Deserialize);
TVM_REGISTER_GLOBAL("meta_schedule.TraceAsPython").set_body_method<Trace>(&TraceNode::AsPython);

}  // namespace meta_schedule
}  // namespace tvm
