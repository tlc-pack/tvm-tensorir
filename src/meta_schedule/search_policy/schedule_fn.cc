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

/********** Definition for ScheduleFn **********/

class ScheduleFnNode : public SearchPolicyNode {
 public:
  String sch_fn;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("sch_fn", &sch_fn);
    v->Visit("num_measure_trials", &num_measure_trials);
    v->Visit("num_measures_per_round", &num_measures_per_round);
    v->Visit("early_stopping", &early_stopping);
  }

  ~ScheduleFnNode() = default;

  Schedule Search(SearchTask task, ProgramMeasurer measurer, int verbose) override;

  static constexpr const char* _type_key = "meta_schedule.ScheduleFn";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleFnNode, SearchPolicyNode);
};

class ScheduleFn : public SearchPolicy {
 public:
  explicit ScheduleFn(String sch_fn, int num_measure_trials, int num_measures_per_round,
                      int early_stopping);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScheduleFn, SearchPolicy, ScheduleFnNode);
};

/********** Registration and Constructor **********/

TVM_REGISTER_NODE_TYPE(ScheduleFnNode);

ScheduleFn::ScheduleFn(String sch_fn, int num_measure_trials, int num_measures_per_round,
                       int early_stopping) {
  ObjectPtr<ScheduleFnNode> n = make_object<ScheduleFnNode>();
  n->sch_fn = std::move(sch_fn);
  n->num_measure_trials = num_measure_trials;
  n->num_measures_per_round = num_measures_per_round;
  n->early_stopping = early_stopping;
  data_ = std::move(n);
}

/********** Searching **********/

Schedule ScheduleFnNode::Search(SearchTask task, ProgramMeasurer measurer, int verbose) {
  // TODO(@junrushao19994)
}

/********** Searching **********/

struct Internal {
  static ScheduleFn CreateScheduleFn(String sch_fn, int num_measure_trials,
                                     int num_measures_per_round, int early_stopping) {
    return ScheduleFn(sch_fn, num_measure_trials, num_measures_per_round, early_stopping);
  }
};

TVM_REGISTER_GLOBAL("meta_schedule.ScheduleFn").set_body_typed(Internal::CreateScheduleFn);

}  // namespace meta_schedule
}  // namespace tvm
