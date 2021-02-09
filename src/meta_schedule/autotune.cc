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

namespace tvm {
namespace meta_schedule {

void TuneContextNode::Init(Optional<Integer> seed) {
  if (seed.defined()) {
    this->sampler.Seed(seed.value()->value);
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
  for (const Postproc& postproc : postprocs) {
    // TODO(@junrushao1994): support PostprocNode::Init
  }
  for (const MeasureCallback& callback : measure_callbacks) {
    callback->Init(this);
  }
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
