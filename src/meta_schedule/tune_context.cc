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
#include "./tune_context.h"

#include <utility>

namespace tvm {
namespace meta_schedule {

TuneContext::TuneContext(Optional<IRModule> workload,                         //
                         Optional<SpaceGenerator> space_generator,            //
                         Optional<SearchStrategy> search_strategy,            //
                         Optional<Database> database,                         //
                         Optional<CostModel> cost_model,                      //
                         Optional<Target> target,                             //
                         Optional<Array<PostProc>> postprocs,                 //
                         Optional<Array<MeasureCallback>> measure_callbacks,  //
                         String name,                                         //
                         TRandState seed,                                     //
                         int num_threads,                                     //
                         int verbose) {
  ObjectPtr<TuneContextNode> n = make_object<TuneContextNode>();
  n->workload = workload;
  n->space_generator = space_generator;
  n->search_strategy = search_strategy;
  n->database = database;
  n->cost_model = cost_model;
  n->target = target;
  n->postprocs = postprocs;
  n->measure_callbacks = measure_callbacks;
  n->name = name;
  n->seed = seed;  // AWAIT(zxybazh): Initialize the random seed.
  n->num_threads = num_threads;
  n->verbose = verbose;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("meta_schedule.TuneContext")
    .set_body_typed([](Optional<IRModule> workload,                         //
                       Optional<SpaceGenerator> space_generator,            //
                       Optional<SearchStrategy> search_strategy,            //
                       Optional<Database> database,                         //
                       Optional<CostModel> cost_model,                      //
                       Optional<Target> target,                             //
                       Optional<Array<PostProc>> postprocs,                 //
                       Optional<Array<MeasureCallback>> measure_callbacks,  //
                       String name,                                         //
                       TRandState seed,                                     //
                       int num_threads,                                     //
                       int verbose) -> TuneContext {
      return TuneContext(workload, space_generator, search_strategy, database, cost_model, target,
                         postprocs, measure_callbacks, name, seed, num_threads, verbose);
    });

TVM_REGISTER_NODE_TYPE(TuneContextNode);

}  // namespace meta_schedule
}  // namespace tvm
