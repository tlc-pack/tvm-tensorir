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
#include <tvm/tir/schedule/inst.h>
#include <tvm/tir/stmt_functor.h>

#include "../../node/attr_registry.h"

namespace tvm {
namespace tir {

Inst::Inst(InstKind kind, Array<ObjectRef> inputs, Array<ObjectRef> attrs,
           Array<ObjectRef> outputs) {
  ObjectPtr<InstNode> n = make_object<InstNode>();
  n->kind = std::move(kind);
  n->inputs = std::move(inputs);
  n->attrs = std::move(attrs);
  n->outputs = std::move(outputs);
  this->data_ = std::move(n);
}

using InstKindRegistry = AttrRegistry<InstKindRegEntry, InstKind>;

InstKind InstKind::Get(const String& inst_kind_name) {
  const InstKindRegEntry* reg = InstKindRegistry::Global()->Get(inst_kind_name);
  ICHECK(reg != nullptr) << "AttributeError: Instruction kind " << inst_kind_name
                         << " is not registered";
  return reg->inst_kind_;
}

InstKindRegEntry::InstKindRegEntry(uint32_t reg_index) {
  ObjectPtr<InstKindNode> n = make_object<InstKindNode>();
  n->reg_index_ = reg_index;
  this->inst_kind_ = InstKind(std::move(n));
}

InstKindRegEntry& InstKindRegEntry::RegisterOrGet(const String& name) {
  return InstKindRegistry::Global()->RegisterOrGet(name);
}

/********** PythonAPICall **********/

void AsPythonString(const ObjectRef& obj, std::ostringstream* _os) {
  std::ostringstream& os = *_os;
  if (const auto* str = obj.as<runtime::StringObj>()) {
    os << str->data;
  } else if (const auto* int_imm = obj.as<IntImmNode>()) {
    os << int_imm->value;
  } else if (const auto* float_imm = obj.as<FloatImmNode>()) {
    os.precision(17);
    os << float_imm->value;
  } else if (const auto* array = obj.as<ArrayNode>()) {
    os << '[';
    bool is_first = true;
    for (const ObjectRef& e : *array) {
      if (is_first) {
        is_first = false;
      } else {
        os << ", ";
      }
      AsPythonString(e, _os);
    }
    os << ']';
  } else {
    LOG(FATAL) << "ValueError: Cannot translate type '" << obj->GetTypeKey()
               << "' to python. Its value is: " << obj;
    throw;
  }
}

void PythonAPICall::Input(String arg_name, int arg) {
  arg_names_.emplace_back(std::move(arg_name));
  args_.push_back(std::to_string(arg));
}

void PythonAPICall::Input(String arg_name, int64_t arg) {
  arg_names_.emplace_back(std::move(arg_name));
  args_.push_back(std::to_string(arg));
}

void PythonAPICall::Input(String arg_name, double arg) {
  arg_names_.emplace_back(std::move(arg_name));
  std::ostringstream os;
  os.precision(17);
  os << arg;
  args_.push_back(os.str());
}

void PythonAPICall::Input(String arg_name, String arg) {
  arg_names_.emplace_back(std::move(arg_name));
  args_.emplace_back(std::move(arg));
}

void PythonAPICall::Input(String arg_name, ObjectRef arg) {
  arg_names_.emplace_back(std::move(arg_name));
  std::ostringstream os;
  AsPythonString(arg, &os);
  args_.push_back(os.str());
}

void PythonAPICall::Decision(ObjectRef decision) {
  if (decision.defined()) {
    this->Input("decision", decision);
  }
}

void PythonAPICall::SingleOutput(Array<String> unit_array) {
  ICHECK_EQ(unit_array.size(), 1);
  this->output_ = unit_array[0];
}

void PythonAPICall::OutputList(Array<String> outputs) {
  if (outputs.empty()) {
    return;
  }
  if (outputs.size() == 1) {
    this->output_ = outputs[0] + ",";
    return;
  }
  std::ostringstream os;
  os << outputs[0];
  for (int i = 1, n = outputs.size(); i < n; ++i) {
    os << ", " << outputs[i];
  }
  this->output_ = os.str();
}

String PythonAPICall::Str() const {
  std::ostringstream os;
  if (output_.defined()) {
    os << output_.value() << " = ";
  }
  os << "sch." << method_name_ << '(';
  int n = args_.size();
  for (int i = 0; i < n; ++i) {
    if (i > 0) {
      os << ", ";
    }
    if (arg_names_[i].empty()) {
      os << args_[i];
    } else {
      os << arg_names_[i] << '=' << args_[i];
    }
  }
  os << ')';
  return os.str();
}

/******** Instruction traits ********/

struct EnterPostProcTraits : public UnpackedInstTraits<EnterPostProcTraits> {
  static constexpr const char* kName = "EnterPostProc";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 0;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch) { return sch->EnterPostProc(); }

  static String UnpackedAsPython(Array<String> outputs) {
    PythonAPICall py("enter_postproc");
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND(EnterPostProcTraits);

/**************** Repr ****************/

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<InstNode>([](const ObjectRef& obj, ReprPrinter* p) {
      const auto* self = obj.as<InstNode>();
      ICHECK_NOTNULL(self);
      Array<ObjectRef> inputs;
      inputs.reserve(self->inputs.size());
      for (const ObjectRef& obj : self->inputs) {
        if (!obj.defined()) {
          inputs.push_back(String("None"));
        } else if (obj->IsInstance<BlockRVNode>() || obj->IsInstance<LoopRVNode>()) {
          inputs.push_back(String("_"));
        } else if (const auto* str_obj = obj.as<StringObj>()) {
          inputs.push_back(String('"' + std::string(str_obj->data) + '"'));
        } else if (obj->IsInstance<IntImmNode>() || obj->IsInstance<FloatImmNode>()) {
          inputs.push_back(obj);
        } else if (const auto* expr = obj.as<PrimExprNode>()) {
          PrimExpr new_expr =
              Substitute(GetRef<PrimExpr>(expr), [](const Var& var) -> Optional<PrimExpr> {
                ObjectPtr<VarNode> new_var = make_object<VarNode>(*var.get());
                new_var->name_hint = "_";
                return Var(new_var);
              });
          std::ostringstream os;
          os << new_expr;
          inputs.push_back(String(os.str()));
        } else {
          LOG(FATAL) << "TypeError: Stringifying is not supported for type: " << obj->GetTypeKey();
          throw;
        }
      }
      p->stream << self->kind->f_as_python(
          /*inputs=*/inputs,
          /*attrs=*/self->attrs,
          /*decision=*/NullOpt,  //
          /*outputs=*/Array<String>(self->outputs.size(), String("_")));
    });

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(InstNode);
TVM_REGISTER_NODE_TYPE(InstKindNode);

TVM_REGISTER_GLOBAL("tir.schedule.InstKindGet").set_body_typed(InstKind::Get);

}  // namespace tir
}  // namespace tvm
