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

const InstKind& InstKind::Get(const String& inst_kind_name) {
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

void PythonAPICall::Input(String arg_name, String arg) {
  arg_names_.emplace_back(std::move(arg_name));
  args_.emplace_back(std::move(arg));
}

void PythonAPICall::InputList(String arg_name, const Array<String>& arg) {
  arg_names_.emplace_back(std::move(arg_name));
  std::ostringstream os;
  os << '[';
  for (int i = 0, n = arg.size(); i < n; ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << arg[i];
  }
  os << ']';
  args_.push_back(os.str());
}

void PythonAPICall::Attr(String arg_name, int arg) {
  arg_names_.emplace_back(std::move(arg_name));
  args_.push_back(std::to_string(arg));
}

void PythonAPICall::Attr(String arg_name, int64_t arg) {
  arg_names_.emplace_back(std::move(arg_name));
  args_.push_back(std::to_string(arg));
}

void PythonAPICall::Attr(String arg_name, const ObjectRef& arg) {
  arg_names_.emplace_back(std::move(arg_name));
  std::ostringstream os;
  os << arg;
  args_.push_back(os.str());
}

void PythonAPICall::Decision(const Optional<ObjectRef>& decision) {
  if (decision.defined()) {
    arg_names_.push_back("decision");
    std::ostringstream os;
    os << decision;
    args_.push_back(os.str());
  }
}

void PythonAPICall::Output(String single_output) { this->output_ = std::move(single_output); }

void PythonAPICall::Outputs(const Array<String>& outputs) {
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

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(InstNode);
TVM_REGISTER_NODE_TYPE(InstKindNode);

}  // namespace tir
}  // namespace tvm
