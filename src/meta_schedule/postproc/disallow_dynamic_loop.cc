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

class DisallowDynamicLoopsNode : public PostprocNode {
 public:
  void InitializeWithTuneContext(const TuneContext& context) final {}

  bool Apply(const tir::Schedule& schedule) final {
    bool has_dyn_ext = false;
    auto f_visit = [&has_dyn_ext](const ObjectRef& obj) -> bool {
      if (has_dyn_ext) {
        return false;
      }
      if (const auto* loop = obj.as<tir::ForNode>()) {
        if (!loop->extent->IsInstance<IntImmNode>()) {
          has_dyn_ext = true;
          return false;
        }
      }
      return true;
    };
    tir::PreOrderVisit(FindEntryFunc(schedule->mod())->body, f_visit);
    return !has_dyn_ext;
  }

  static constexpr const char* _type_key = "meta_schedule.DisallowDynamicLoops";
  TVM_DECLARE_FINAL_OBJECT_INFO(DisallowDynamicLoopsNode, PostprocNode);
};

Postproc Postproc::DisallowDynamicLoops() {
  ObjectPtr<DisallowDynamicLoopsNode> n = make_object<DisallowDynamicLoopsNode>();
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(DisallowDynamicLoopsNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocDisallowDynamicLoops")
    .set_body_typed(Postproc::DisallowDynamicLoops);

}  // namespace meta_schedule
}  // namespace tvm
