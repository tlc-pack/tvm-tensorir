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
#ifndef SRC_META_SCHEDULE_SEARCH_POLICY_H_
#define SRC_META_SCHEDULE_SEARCH_POLICY_H_

#include <tvm/runtime/object.h>

namespace tvm {
namespace meta_schedule {

class SearchTask;
class ProgramMeasurer;
class Schedule;

class SearchPolicyNode : public runtime::Object {
 public:
  virtual ~SearchPolicyNode() = default;
  virtual Schedule Search(SearchTask task, ProgramMeasurer measurer, int verbose) = 0;

  static constexpr const char* _type_key = "meta_schedule.SearchPolicy";
  TVM_DECLARE_BASE_OBJECT_INFO(SearchPolicyNode, Object);
};

class SearchPolicy : public runtime::ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SearchPolicy, ObjectRef, SearchPolicyNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SEARCH_POLICY_H_
