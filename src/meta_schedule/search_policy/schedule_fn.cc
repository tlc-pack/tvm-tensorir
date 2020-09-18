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
#include "../search_policy.h"
#include "../search_task.h"

namespace tvm {
namespace meta_schedule {

using runtime::TypedPackedFunc;

/********** Definition for ScheduleFn **********/

class ScheduleFnNode : public SearchPolicyNode {
 public:
  String sch_fn;
  runtime::TypedPackedFunc<void(Schedule)> sch_fn_;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("sch_fn", &sch_fn); }

  ~ScheduleFnNode() = default;

  Schedule Search(SearchTask task, ProgramMeasurer measurer, int verbose) override;

  static constexpr const char* _type_key = "meta_schedule.ScheduleFn";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleFnNode, SearchPolicyNode);
};

class ScheduleFn : public SearchPolicy {
 public:
  explicit ScheduleFn(String sch_fn);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScheduleFn, SearchPolicy, ScheduleFnNode);
};

/********** Registration and Constructor **********/

ScheduleFn::ScheduleFn(String sch_fn) {
  const auto* fn = runtime::Registry::Get(sch_fn);
  CHECK(fn != nullptr) << "AttributeError: Cannot find packed function: " << sch_fn;
  ObjectPtr<ScheduleFnNode> n = make_object<ScheduleFnNode>();
  n->sch_fn = std::move(sch_fn);
  n->sch_fn_ = *fn;
  data_ = std::move(n);
}

/********** Searching **********/

Schedule ScheduleFnNode::Search(SearchTask task, ProgramMeasurer measurer, int verbose) {
  measurer->Reset();
  for (int iteration = 0; iteration < 1; ++iteration) {
    Schedule sch(task->func);
    this->sch_fn_(sch);
    MeasureInput measure_input(task, sch);
    Array<MeasureResult> measure_results = measurer->Measure({measure_input}, verbose);
    CHECK_EQ(measure_results.size(), 1);
    MeasureResult measure_result = measure_results[0];
    if (measure_result->error_no != 0) {
      LOG(INFO) << "[Failed] error_msg = " << measure_result->error_msg;
    } else {
      LOG(INFO) << "[Success] measure_result = " << measure_result;
    }
  }
  return Schedule(nullptr);
}

/********** Searching **********/

struct Internal {
  static ScheduleFn CreateScheduleFn(String sch_fn) { return ScheduleFn(sch_fn); }
};

TVM_REGISTER_NODE_TYPE(ScheduleFnNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleFn").set_body_typed(Internal::CreateScheduleFn);

}  // namespace meta_schedule
}  // namespace tvm
