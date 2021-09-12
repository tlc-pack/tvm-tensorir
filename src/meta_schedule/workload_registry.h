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
#ifndef SRC_META_SCHEDULE_WORKLOAD_REGISTERY_H_
#define SRC_META_SCHEDULE_WORKLOAD_REGISTERY_H_

#include <tvm/ir/module.h>

#include <unordered_map>
#include <vector>

namespace tvm {
namespace meta_schedule {

class WorkloadTokenNode : public runtime::Object {
 public:
  IRModule mod;
  String shash;
  int64_t token_id_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("mod", &mod);
    v->Visit("shash", &shash);
    // `token_id_` is not visited
  }

  ObjectRef AsJSON() const;

  static constexpr const char* _type_key = "meta_schedule.WorkloadToken";
  TVM_DECLARE_FINAL_OBJECT_INFO(WorkloadTokenNode, runtime::Object);
};

class WorkloadToken : public runtime::ObjectRef {
 public:
  TVM_DLL WorkloadToken(IRModule mod, String shash, int64_t token_id);

  TVM_DLL static WorkloadToken FromJSON(const ObjectRef& json_obj, int64_t token_id);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(WorkloadToken, runtime::ObjectRef, WorkloadTokenNode);
};

class WorkloadRegistryNode : public runtime::Object {
 public:
  String path;
  std::unordered_map<IRModule, int64_t, tvm::StructuralHash, tvm::StructuralEqual> mod2token_id_;
  std::vector<WorkloadToken> workload_tokens_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("path", &path);
    // `mod2token_id_` is not visited
    // `workload_tokens_` is not visited
  }

  static constexpr const char* _type_key = "meta_schedule.WorkloadRegistry";
  TVM_DECLARE_FINAL_OBJECT_INFO(WorkloadRegistryNode, runtime::Object);

 public:
  TVM_DLL WorkloadToken LookupOrAdd(const IRModule& mod);

  TVM_DLL int64_t Size() const;

  TVM_DLL WorkloadToken At(int64_t token_id) const;
};

class WorkloadRegistry : public runtime::ObjectRef {
 public:
  TVM_DLL WorkloadRegistry(String path, bool allow_missing);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(WorkloadRegistry, runtime::ObjectRef,
                                                    WorkloadRegistryNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_WORKLOAD_REGISTERY_H_
