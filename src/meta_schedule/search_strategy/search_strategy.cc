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

#include "../search_strategy.h"

#include "../tune_context.h"

namespace tvm {
namespace meta_schedule {

SearchStrategy SearchStrategy::PySearchStrategy(
    PySearchStrategyNode::FInitializeWithTuneContext f_initialize_with_tune_context,  //
    PySearchStrategyNode::FGenerateMeasureCandidates f_generate_measure_candidates,   //
    PySearchStrategyNode::FNotifyRunnerResults f_notify_runner_results,               //
    PySearchStrategyNode::FPreTuning f_pre_tuning,                                    //
    PySearchStrategyNode::FPostTuning f_post_tuning) {
  ObjectPtr<PySearchStrategyNode> n = make_object<PySearchStrategyNode>();
  n->f_initialize_with_tune_context = f_initialize_with_tune_context;
  n->f_generate_measure_candidates = f_generate_measure_candidates;
  n->f_notify_runner_results = f_notify_runner_results;
  n->f_pre_tuning = f_pre_tuning;
  n->f_post_tuning = f_post_tuning;
  return SearchStrategy(n);
}

TVM_REGISTER_OBJECT_TYPE(SearchStrategyNode);
TVM_REGISTER_NODE_TYPE(PySearchStrategyNode);

TVM_REGISTER_GLOBAL("meta_schedule.PySearchStrategy")
    .set_body_typed(SearchStrategy::PySearchStrategy);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategyInitializeWithTuneContext")
    .set_body_method<SearchStrategy>(&SearchStrategyNode::InitializeWithTuneContext);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategyGenerateMeasureCandidates")
    .set_body_method<SearchStrategy>(&SearchStrategyNode::GenerateMeasureCandidates);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategyNotifyRunnerResults")
    .set_body_method<SearchStrategy>(&SearchStrategyNode::NotifyRunnerResults);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategyPreTuning")
    .set_body_method<SearchStrategy>(&SearchStrategyNode::PreTuning);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategyPostTuning")
    .set_body_method<SearchStrategy>(&SearchStrategyNode::PostTuning);

}  // namespace meta_schedule
}  // namespace tvm
