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
#include "./autotune.h"  // NOLINT(build/include)

#include "./utils.h"

namespace tvm {
namespace meta_schedule {

void TuneContextNode::Init(Optional<Integer> seed) {
  if (seed.defined() && seed.value() != -1) {
    Sampler(&this->rand_state).Seed(seed.value()->value);
  } else {
    Sampler(&this->rand_state).Seed(std::random_device()());
  }
  if (task.defined()) {
    task.value()->Init(this);
  }
  if (space.defined()) {
    space.value()->Init(this);
  }
  if (strategy.defined()) {
    strategy.value()->Init(this);
  }
  if (builder.defined()) {
    builder.value()->Init(this);
  }
  if (runner.defined()) {
    runner.value()->Init(this);
  }
  if (database.defined()) {
    database.value()->Init(this);
  }
  if (cost_model.defined()) {
    cost_model.value()->Init(this);
  }
  for (const DMLC_ATTRIBUTE_UNUSED Postproc& postproc : postprocs) {
    // TODO(@junrushao1994): support PostprocNode::Init
  }
  for (const DMLC_ATTRIBUTE_UNUSED MeasureCallback& callback : measure_callbacks) {
    callback->Init(this);
  }
}

bool TuneContextNode::Postprocess(const Schedule& sch) {
  sch->EnterPostproc();
  for (const Postproc& postproc : postprocs) {
    if (!postproc->Apply(task.value(), sch, &rand_state)) {
      return false;
    }
  }
  return true;
}

Array<MeasureResult> TuneContextNode::Measure(const Array<MeasureInput>& measure_inputs) {
  constexpr int verbose = 1;
  ICHECK(this->task.defined()) << "ValueError: `task` is not defined";
  ICHECK(this->builder.defined()) << "ValueError: `builder` is not defined";
  ICHECK(this->runner.defined()) << "ValueError: `runner` is not defined";
  ICHECK(this->database.defined()) << "ValueError: `database` is not defined";
  DatabaseNode* db = this->database.value().operator->();
  int n = measure_inputs.size();
  // Build & Measure
  Array<BuildResult> build_results = builder.value()->Build(measure_inputs, verbose);
  Array<MeasureResult> measure_results =
      runner.value()->Run(measure_inputs, build_results, verbose);
  // Add to database
  for (int i = 0; i < n; ++i) {
    const MeasureInput& measure_input = measure_inputs[i];
    const MeasureResult& measure_result = measure_results[i];
    MeasureErrorNO error_no = static_cast<MeasureErrorNO>(measure_result->error_no);
    if (error_no == MeasureErrorNO::kNoError) {
      db->Add(measure_input->sch->trace().value(), Repr(measure_input->sch),
              AsVector<FloatImm, double>(measure_result->costs), measure_input->task);
    }
    for (const MeasureCallback& callback : this->measure_callbacks) {
      callback->Callback(measure_input, measure_result);
    }
  }
  return measure_results;
}

Optional<Schedule> Autotune(SearchTask task,                           //
                            Optional<SearchSpace> space,               //
                            Optional<SearchStrategy> strategy,         //
                            Optional<ProgramBuilder> builder,          //
                            Optional<ProgramRunner> runner,            //
                            Database database,                         //
                            Optional<CostModel> cost_model,            //
                            Array<Postproc> postprocs,                 //
                            Array<MeasureCallback> measure_callbacks,  //
                            int num_threads,                           //
                            Optional<Integer> seed) {
  TuneContext tune_context(task, space, strategy, builder, runner, database, cost_model, postprocs,
                           measure_callbacks, num_threads, seed);
  if (strategy.defined()) {
    strategy.value()->Search();
  }
  DatabaseNode::Entry best = database->GetBest(task);
  if (!best.trace.defined()) {
    return NullOpt;
  }
  Schedule sch = Schedule::Traced(/*mod=*/IRModule({{GlobalVar("main"), task->workload}}),  //
                                  /*seed=*/-1,
                                  /*debug_mode=*/false,
                                  /*error_render_level=*/tir::ScheduleErrorRenderLevel::kDetail);
  best.trace.value()->ApplyToSchedule(sch, true);
  if (!tune_context->Postprocess(sch)) {
    LOG(FATAL) << "ValueError: The best schedule cannot be postprocessed all of a sudden";
  }
  return sch;
}

/**************** FFI ****************/

struct Internal {
  static TuneContext New(Optional<SearchTask> task,                 //
                         Optional<SearchSpace> space,               //
                         Optional<SearchStrategy> strategy,         //
                         Optional<ProgramBuilder> builder,          //
                         Optional<ProgramRunner> runner,            //
                         Optional<Database> database,               //
                         Optional<CostModel> cost_model,            //
                         Array<Postproc> postprocs,                 //
                         Array<MeasureCallback> measure_callbacks,  //
                         int num_threads,                           //
                         Optional<Integer> seed) {
    return TuneContext(task, space, strategy, builder, runner, database, cost_model, postprocs,
                       measure_callbacks, num_threads, seed);
  }
  static void Init(TuneContext self, Optional<Integer> seed) { self->Init(seed); }
};

TVM_REGISTER_NODE_TYPE(TuneContextNode);
TVM_REGISTER_GLOBAL("meta_schedule.TuneContext").set_body_typed(Internal::New);
TVM_REGISTER_GLOBAL("meta_schedule.TuneContextInit").set_body_typed(Internal::Init);

}  // namespace meta_schedule
}  // namespace tvm
