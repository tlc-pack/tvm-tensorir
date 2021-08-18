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

#include "tune_context.h"

namespace tvm {
namespace meta_schedule {

void TuneContextNode::PostProcessFunc() {}
void TuneContextNode::MeasureCallbackFunc() {}

TVM_REGISTER_OBJECT_TYPE(TuneContextNode);
TVM_REGISTER_GLOBAL("meta_schedule.TuneContext")
    .set_body_typed([](Optional<IRModule> workload,                         //
                       Optional<SpaceGenerator> space_generator,            //
                       Optional<SearchStrategy> search_strategy,            //
                       Optional<Database> database,                         //
                       Optional<CostModel> cost_model,                      //
                       Optional<Target> target,                             //
                       Optional<Array<PostProc>> post_procs,                //
                       Optional<Array<MeasureCallback>> measure_callbacks,  //
                       String name,                                         //
                       TRandState seed,                                     //
                       int num_threads,                                     //
                       int verbose) -> TuneContext {
      return TuneContext(workload, space_generator, search_strategy, database, cost_model, target,
                         post_procs, measure_callbacks, name, seed, num_threads, verbose);
    });
TVM_REGISTER_GLOBAL("tvm.meta_schedule.TuneContextPostProcess")
    .set_body_method<TuneContext>(&TuneContextNode::PostProcessFunc);
TVM_REGISTER_GLOBAL("tvm.meta_schedule.TuneContextMeasureCallback")
    .set_body_method<TuneContext>(&TuneContextNode::MeasureCallbackFunc);

}  // namespace meta_schedule
}  // namespace tvm
