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
  runtime::TypedPackedFunc<void(Schedule)> sch_fn;
  int num_iterations;
  int batch_size;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // sch_fn is not visited
    v->Visit("num_iterations", &num_iterations);
    v->Visit("batch_size", &batch_size);
  }

  ~ScheduleFnNode() = default;

  Schedule Search(SearchTask task, ProgramMeasurer measurer, int verbose) override;

  static constexpr const char* _type_key = "meta_schedule.ScheduleFn";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleFnNode, SearchPolicyNode);
};

class ScheduleFn : public SearchPolicy {
 public:
  explicit ScheduleFn(PackedFunc sch_fn, int batch_size, int num_iterations);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScheduleFn, SearchPolicy, ScheduleFnNode);
};

/********** Registration and Constructor **********/

ScheduleFn::ScheduleFn(PackedFunc sch_fn, int num_iterations, int batch_size) {
  ObjectPtr<ScheduleFnNode> n = make_object<ScheduleFnNode>();
  n->sch_fn = sch_fn;
  n->num_iterations = num_iterations;
  n->batch_size = batch_size;
  data_ = std::move(n);
}

/********** Searching **********/

Schedule ScheduleFnNode::Search(SearchTask task, ProgramMeasurer measurer, int verbose) {
  measurer->Reset();
  for (int iter_id = 0; iter_id < num_iterations;) {
    Array<MeasureInput> measure_inputs;
    measure_inputs.reserve(batch_size);
    for (int batch_id = 0; batch_id < batch_size && iter_id < num_iterations;
         ++batch_id, ++iter_id) {
      Schedule sch(task->func);
      this->sch_fn(sch);
      measure_inputs.push_back(MeasureInput(task, sch));
    }
    measurer->BatchMeasure(measure_inputs, this->batch_size, verbose);
  }
  return measurer->best_sch.defined() ? measurer->best_sch.value() : Schedule(nullptr);
}

/********** Searching **********/

struct Internal {
  static ScheduleFn CreateScheduleFn(PackedFunc sch_fn, int batch_size, int num_iterations) {
    return ScheduleFn(sch_fn, batch_size, num_iterations);
  }
};

TVM_REGISTER_NODE_TYPE(ScheduleFnNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleFn").set_body_typed(Internal::CreateScheduleFn);

}  // namespace meta_schedule
}  // namespace tvm
