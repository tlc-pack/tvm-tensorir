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
#include <tvm/target/target.h>
#include <tvm/tir/function.h>

#include "./measure.h"

namespace tvm {
namespace meta_schedule {

/********** SearchTask **********/

class SearchTaskNode : public Object {
 public:
  tir::PrimFunc workload;
  Target target;
  Target target_host;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("workload", &workload);
    v->Visit("target", &target);
    v->Visit("target_host", &target_host);
  }

  static constexpr const char* _type_key = "meta_schedule.SearchTask";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleNode, Object);
};

class SearchTask : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(SearchTask, ObjectRef, SearchTaskNode);
};

/********** SearchPolicy **********/

class SearchPolicyNode : public Object {
 public:
  SearchTask task;
  int num_measure_trials;
  int num_measures_per_round;
  int early_stopping;
  int verbose;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("task", &task);
    v->Visit("num_measure_trials", &num_measure_trials);
    v->Visit("num_measures_per_round", &num_measures_per_round);
    v->Visit("early_stopping", &early_stopping);
    v->Visit("verbose", &verbose);
  }

  static constexpr const char* _type_key = "meta_schedule.SearchPolicy";
  TVM_DECLARE_FINAL_OBJECT_INFO(SearchPolicyNode, Object);
};

class SearchPolicy : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(SearchPolicy, ObjectRef, SearchPolicyNode);
};

/********** Search API **********/

TVM_DLL void Search(const SearchPolicy& policy, const ProgramBuilder& builder,
                    const ProgramRunner& runner);

}  // namespace meta_schedule
}  // namespace tvm
