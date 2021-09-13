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
#include <random>
#include <utility>

#include "./utils.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Constructor function of TuneContext class.
 * \param mod The mod to be optimized.
 * \param space_generator The design space generator.
 * \param search_strategy The search strategy to be used.
 * \param database The database for querying and storage.
 * \param cost_model The cost model for estimation.
 * \param target The target to be optimized for.
 * \param postprocs The post processing functions.
 * \param measure_callbacks The measure callback functions.
 * \param task_name The name of the tuning task.
 * \param rand_state The random state.
 * \param num_threads The number of threads to be used.
 * \param verbose The verbosity level.
 */
TuneContext::TuneContext(Optional<IRModule> mod,                                    //
                         Optional<Target> target,                                   //
                         Optional<SpaceGenerator> space_generator,                  //
                         Optional<SearchStrategy> search_strategy,                  //
                         Optional<Database> database,                               //
                         Optional<CostModel> cost_model,                            //
                         Optional<Array<Postproc>> postprocs,                       //
                         Optional<Array<MeasureCallback>> measure_callbacks,        //
                         Optional<String> task_name,                                //
                         support::LinearCongruentialEngine::TRandState rand_state,  //
                         int num_threads,                                           //
                         int verbose) {
  ObjectPtr<TuneContextNode> n = make_object<TuneContextNode>();
  n->mod = mod;
  n->target = target;
  n->space_generator = space_generator;
  n->search_strategy = search_strategy;
  n->database = database;
  n->cost_model = cost_model;
  n->postprocs = postprocs;
  n->measure_callbacks = measure_callbacks;
  n->task_name = task_name;
  if (rand_state == -1) {
    rand_state = std::random_device()();
  }
  support::LinearCongruentialEngine(&n->rand_state).Seed(rand_state);
  n->num_threads = num_threads;
  n->verbose = verbose;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("meta_schedule.TuneContext")
    .set_body_typed([](Optional<IRModule> mod,                                    //
                       Optional<Target> target,                                   //
                       Optional<SpaceGenerator> space_generator,                  //
                       Optional<SearchStrategy> search_strategy,                  //
                       Optional<Database> database,                               //
                       Optional<CostModel> cost_model,                            //
                       Optional<Array<Postproc>> postprocs,                       //
                       Optional<Array<MeasureCallback>> measure_callbacks,        //
                       Optional<String> task_name,                                //
                       support::LinearCongruentialEngine::TRandState rand_state,  //
                       int num_threads,                                           //
                       int verbose) -> TuneContext {
      return TuneContext(mod, target, space_generator, search_strategy, database, cost_model,
                         postprocs, measure_callbacks, task_name, rand_state, num_threads, verbose);
    });

TVM_REGISTER_NODE_TYPE(TuneContextNode);

}  // namespace meta_schedule
}  // namespace tvm
