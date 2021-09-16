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
#include "./workload_registry.h"

#include <dmlc/memory_io.h>
#include <tvm/node/serialization.h>

#include "../support/base64.h"
#include "./utils.h"

namespace tvm {
namespace meta_schedule {

WorkloadToken::WorkloadToken(IRModule mod, String shash, int64_t token_id) {
  ObjectPtr<WorkloadTokenNode> n = runtime::make_object<WorkloadTokenNode>();
  n->mod = mod;
  n->shash = shash;
  n->token_id_ = token_id;
  data_ = std::move(n);
}

ObjectRef WorkloadTokenNode::AsJSON() const {
  // Convert `this->mod` to JSON
  std::string json_mod = tvm::SaveJSON(this->mod);
  // Dump the JSON string to base64
  std::string b64_mod = Base64Encode(json_mod);
  // Output
  return Array<ObjectRef>{this->shash, String(b64_mod)};
}

WorkloadToken WorkloadToken::FromJSON(const ObjectRef& json_obj, int64_t token_id) {
  IRModule mod{nullptr};
  String shash{ObjectPtr<runtime::StringObj>(nullptr)};
  try {
    const ArrayNode* json_array = json_obj.as<ArrayNode>();
    CHECK(json_array && json_array->size() == 2);
    // Load json[0] => shash
    shash = Downcast<String>(json_array->at(0));
    // Load json[1] => mod
    {
      String b64_mod = Downcast<String>(json_array->at(1));
      std::string json_mod = Base64Decode(b64_mod);
      mod = Downcast<IRModule>(LoadJSON(json_mod));
    }
    // Verify SHash(mod) == shash
    String recalc_shash = GetSHash(mod);
    CHECK_EQ(recalc_shash, shash) << "ValueError: Structural hash changed. Given: " << shash
                                  << "; Recalculated: " << recalc_shash;
  } catch (const std::runtime_error& e) {  // includes tvm::Error and dmlc::Error
    LOG(FATAL) << "ValueError: Unable to parse the JSON object: " << json_obj
               << "\nThe error is: " << e.what();
  }
  return WorkloadToken(mod, shash, token_id);
}

WorkloadToken WorkloadRegistryNode::LookupOrAdd(const IRModule& mod) {
  // Insert `mod` into the lookup table if it doesn't exist
  decltype(this->mod2token_id_)::iterator it;
  bool inserted = false;
  std::tie(it, inserted) = this->mod2token_id_.insert({mod, -1});
  int64_t& token_id = it->second;
  // Case 1. Insertion fails: it already exists in the lookup table
  if (inserted == false) {
    return this->workload_tokens_.at(token_id);
  }
  // Case 2. Insertion succeeds: add to the lookup table
  token_id = this->workload_tokens_.size();
  WorkloadToken token(mod, GetSHash(mod), token_id);
  this->workload_tokens_.push_back(token);
  JSONFileAppendLine(this->path, JSONObj2Str(token->AsJSON()));
  return token;
}

int64_t WorkloadRegistryNode::Size() const { return this->workload_tokens_.size(); }

WorkloadToken WorkloadRegistryNode::At(int64_t token_id) const {
  return this->workload_tokens_.at(token_id);
}

WorkloadRegistry::WorkloadRegistry(String path, bool allow_missing) {
  ObjectPtr<WorkloadRegistryNode> n = runtime::make_object<WorkloadRegistryNode>();
  // Set `path`
  n->path = path;
  // Set `workload_tokens_`, `mod2token_id_`
  Array<ObjectRef> json_objs = JSONStr2Obj(JSONFileReadLines(path, allow_missing));
  n->workload_tokens_.reserve(json_objs.size());
  n->mod2token_id_.reserve(json_objs.size());
  for (const ObjectRef& json_obj : json_objs) {
    int64_t token_id = n->workload_tokens_.size();
    WorkloadToken token = WorkloadToken::FromJSON(json_obj, token_id);
    n->mod2token_id_.emplace(token->mod, token_id);
    n->workload_tokens_.push_back(token);
  }
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(WorkloadTokenNode);
TVM_REGISTER_NODE_TYPE(WorkloadRegistryNode);
TVM_REGISTER_GLOBAL("meta_schedule.WorkloadToken")
    .set_body_typed([](IRModule mod, String shash, int64_t token_id) {
      return WorkloadToken(mod, shash, token_id);
    });
TVM_REGISTER_GLOBAL("meta_schedule.WorkloadTokenAsJSON")
    .set_body_method<WorkloadToken>(&WorkloadTokenNode::AsJSON);
TVM_REGISTER_GLOBAL("meta_schedule.WorkloadTokenFromJSON").set_body_typed(&WorkloadToken::FromJSON);
TVM_REGISTER_GLOBAL("meta_schedule.WorkloadRegistry")
    .set_body_typed([](String path, bool allow_missing) {
      return WorkloadRegistry(path, allow_missing);
    });
TVM_REGISTER_GLOBAL("meta_schedule.WorkloadRegistryLookupOrAdd")
    .set_body_method<WorkloadRegistry>(&WorkloadRegistryNode::LookupOrAdd);
TVM_REGISTER_GLOBAL("meta_schedule.WorkloadRegistrySize")
    .set_body_method<WorkloadRegistry>(&WorkloadRegistryNode::Size);
TVM_REGISTER_GLOBAL("meta_schedule.WorkloadRegistryAt")
    .set_body_method<WorkloadRegistry>(&WorkloadRegistryNode::At);

}  // namespace meta_schedule
}  // namespace tvm
