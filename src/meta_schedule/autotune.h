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
#ifndef SRC_META_SCHEDULE_AUTOTUNE_H_
#define SRC_META_SCHEDULE_AUTOTUNE_H_

#include <utility>

#include "./cost_model.h"
#include "./database.h"
#include "./measure.h"
#include "./measure_callback.h"
#include "./search.h"
#include "./space/postproc.h"

namespace tvm {
namespace meta_schedule {

class TuneContextNode : public runtime::Object {
 public:
  Optional<SearchTask> task;
  Optional<SearchSpace> space;
  Optional<SearchStrategy> strategy;
  Optional<ProgramBuilder> builder;
  Optional<ProgramRunner> runner;
  Optional<Database> database;
  Optional<CostModel> cost_model;
  Array<Postproc> postprocs;
  Array<MeasureCallback> measure_callbacks;
  int num_threads;

  Sampler::TRandomState rand_state;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("task", &task);
    v->Visit("space", &space);
    v->Visit("strategy", &strategy);
    v->Visit("builder", &builder);
    v->Visit("runner", &runner);
    v->Visit("database", &database);
    v->Visit("cost_model", &cost_model);
    v->Visit("postprocs", &postprocs);
    v->Visit("measure_callbacks", &measure_callbacks);
    v->Visit("num_threads", &num_threads);
    // `sampler` is not visited
  }

  void Init(Optional<Integer> seed = NullOpt);

  bool Postprocess(const Schedule& sch);

  Array<MeasureResult> Measure(const Array<MeasureInput>& measure_inputs);

  static constexpr const char* _type_key = "meta_schedule.TuneContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(TuneContextNode, runtime::Object);
};

class TuneContext : public runtime::ObjectRef {
 public:
  explicit TuneContext(const Optional<SearchTask>& task,                 //
                       const Optional<SearchSpace>& space,               //
                       const Optional<SearchStrategy>& strategy,         //
                       const Optional<ProgramBuilder>& builder,          //
                       const Optional<ProgramRunner>& runner,            //
                       const Optional<Database>& database,               //
                       const Optional<CostModel>& cost_model,            //
                       const Array<Postproc>& postprocs,                 //
                       const Array<MeasureCallback>& measure_callbacks,  //
                       int num_threads,                                  //
                       Optional<Integer> seed) {
    ObjectPtr<TuneContextNode> n = make_object<TuneContextNode>();
    n->task = task;
    n->space = space;
    n->strategy = strategy;
    n->builder = builder;
    n->runner = runner;
    n->database = database;
    n->cost_model = cost_model;
    n->postprocs = postprocs;
    n->measure_callbacks = measure_callbacks;
    n->num_threads = num_threads;
    // `n->sampler` is not initialized
    data_ = std::move(n);
    (*this)->Init(seed);
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TuneContext, runtime::ObjectRef,
                                                    TuneContextNode);
};

TVM_DLL Optional<Schedule> Autotune(SearchTask task,                           //
                                    Optional<SearchSpace> space,               //
                                    Optional<SearchStrategy> strategy,         //
                                    Optional<ProgramBuilder> builder,          //
                                    Optional<ProgramRunner> runner,            //
                                    Database database,                         //
                                    Optional<CostModel> cost_model,            //
                                    Array<Postproc> postprocs,                 //
                                    Array<MeasureCallback> measure_callbacks,  //
                                    int num_threads,                           //
                                    Optional<Integer> seed);

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_AUTOTUNE_H_
