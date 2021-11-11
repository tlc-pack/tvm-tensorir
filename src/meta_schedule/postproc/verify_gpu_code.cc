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
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

class VerifyGPUCodeNode : public PostprocNode {
 public:
  Target target_;

  static Integer Extract(const Target& target, const char* name) {
    ICHECK(target.defined());

    if (Optional<Integer> v = target->GetAttr<Integer>(name)) {
      return v.value();
    }
    LOG(FATAL) << "AttributedError: \"" << name << "\" is not defined in the target";
    throw;
  }

  static bool VerifyGPU(const tir::PrimFunc& func, const Target& target) {
    Map<String, PrimExpr> constraints{
        {"max_shared_memory_per_block", Extract(target, "shared_memory_per_block")},
        {"max_local_memory_per_block", Extract(target, "registers_per_block")},
        {"max_threads_per_block", Extract(target, "max_threads_per_block")},
        {"max_vthread", Integer(8)},
        {"max_vector_bytes", Integer(16)}};
    return tir::VerifyGPUCode(func, constraints);
  }

  void InitializeWithTuneContext(const TuneContext& context) final {
    ICHECK(context->target != nullptr);
    this->target_ = context->target.value();
  }

  bool Apply(const tir::Schedule& sch) final {
    IRModule mod = sch->mod();
    try {
      mod = LowerModule(std::move(mod));
    } catch (const dmlc::Error& e) {
      return false;
    }
    for (const auto& kv : mod->functions) {
      if (const auto* func = kv.second.as<tir::PrimFuncNode>()) {
        if (!VerifyGPU(GetRef<tir::PrimFunc>(func), this->target_)) {
          return false;
        }
      }
    }
    return true;
  }

  static constexpr const char* _type_key = "meta_schedule.VerifyGPUCode";
  TVM_DECLARE_FINAL_OBJECT_INFO(VerifyGPUCodeNode, PostprocNode);
};

Postproc Postproc::VerifyGPUCode() {
  ObjectPtr<VerifyGPUCodeNode> n = make_object<VerifyGPUCodeNode>();
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(VerifyGPUCodeNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocVerifyGPUCode").set_body_typed(Postproc::VerifyGPUCode);

}  // namespace meta_schedule
}  // namespace tvm
