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
#include "../measure.h"
#include "../search.h"

namespace tvm {
namespace meta_schedule {

using runtime::TypedPackedFunc;

/********** Definition for ScheduleFn **********/

class ScheduleFnNode : public SearchSpaceNode {
 public:
  runtime::TypedPackedFunc<void(Schedule)> sch_fn;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // sch_fn is not visited
  }

  ~ScheduleFnNode() = default;

  Schedule SampleByReplay(const SearchTask& task) override;
  Array<Schedule> GetSupport(const SearchTask& task) override;

  static constexpr const char* _type_key = "meta_schedule.ScheduleFn";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleFnNode, SearchSpaceNode);
};

class ScheduleFn : public SearchSpace {
 public:
  explicit ScheduleFn(PackedFunc sch_fn);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScheduleFn, SearchSpace, ScheduleFnNode);
};

/********** Constructor **********/

ScheduleFn::ScheduleFn(PackedFunc sch_fn) {
  ObjectPtr<ScheduleFnNode> n = make_object<ScheduleFnNode>();
  n->sch_fn = sch_fn;
  data_ = std::move(n);
}

/********** Sampling **********/

Schedule ScheduleFnNode::SampleByReplay(const SearchTask& task) {
  Schedule sch(task->func);
  this->sch_fn(sch);
  return sch;
}

Array<Schedule> ScheduleFnNode::GetSupport(const SearchTask& task) {
  return {SampleByReplay(task)};
}

/********** FFI **********/

struct Internal {
  static ScheduleFn New(PackedFunc sch_fn) { return ScheduleFn(sch_fn); }
};

TVM_REGISTER_NODE_TYPE(ScheduleFnNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleFn").set_body_typed(Internal::New);

}  // namespace meta_schedule
}  // namespace tvm
