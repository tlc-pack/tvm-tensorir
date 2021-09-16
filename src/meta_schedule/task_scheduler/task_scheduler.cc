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
#include "../task_scheduler.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Send the measure candidates to builder.
 * \param builder The builder to send the candidates to.
 * \param context The tuning context.
 * \param candidates The measure candidates.
 * \return An array of the builder results.
 */
Array<BuilderResult> SendToBuilder(const Builder& builder,  //
                                   const TuneContext& context,
                                   const Array<MeasureCandidate>& candidates) {
  Target target = context->target.value();
  Array<BuilderInput> inputs;
  inputs.reserve(candidates.size());
  for (const MeasureCandidate& candidate : candidates) {
    inputs.push_back(BuilderInput(candidate->sch->mod(), target));
  }
  return builder->Build(inputs);
}

/*!
 * \brief Send the built measure candidates to runner.
 * \param runner The runner to send the candidates to.
 * \param context The tuning context.
 * \param candidates The mesure candidates.
 * \param builder_results The builder results.
 * \return An array of the runner results.
 */
Array<RunnerFuture> SendToRunner(const Runner& runner,  //
                                 const TuneContext& context,
                                 const Array<MeasureCandidate>& candidates,
                                 const Array<BuilderResult>& builder_results) {
  Target target = context->target.value();
  ICHECK_EQ(candidates.size(), builder_results.size());
  int n = candidates.size();
  int n_build_errors = 0;
  Array<RunnerInput> inputs;
  inputs.reserve(n);
  for (int i = 0; i < n; ++i) {
    const MeasureCandidate& candidate = candidates[i];
    const BuilderResult& builder_result = builder_results[i];
    if (builder_result->error_msg.defined()) {
      ++n_build_errors;
      continue;
    }
    inputs.push_back(RunnerInput(/*artifact_path=*/builder_result->artifact_path.value(),
                                 /*device_type=*/target->kind->name,
                                 /*args_info=*/candidate->args_info));
  }
  Array<RunnerFuture> futures = runner->Run(inputs);
  if (n_build_errors == 0) {
    return futures;
  }
  Array<RunnerFuture> results;
  results.reserve(n);
  for (int i = 0, j = 0; i < n; ++i) {
    const BuilderResult& builder_result = builder_results[i];
    if (builder_result->error_msg.defined()) {
      results.push_back(RunnerFuture(
          /*f_done=*/[]() -> bool { return true; },
          /*f_result=*/
          [msg = builder_result->error_msg]() -> RunnerResult {
            return RunnerResult(NullOpt, msg);
          }));
    } else {
      results.push_back(futures[j++]);
    }
  }
  return results;
}

void TaskSchedulerNode::Tune() {
  for (const TuneContext& task : this->tasks) {
    CHECK(task->mod.defined()) << "ValueError: Require `context.mod`, but it is not defined";
    CHECK(task->space_generator.defined())
        << "ValueError: Require `context.space_generator`, but it is not defined";
    CHECK(task->search_strategy.defined())
        << "ValueError: Require `context.search_strategy`, but it is not defined";
    IRModule mod = task->mod.value();
    SpaceGenerator space = task->space_generator.value();
    SearchStrategy strategy = task->search_strategy.value();
    space->InitializeWithTuneContext(task);
    strategy->InitializeWithTuneContext(task);
    strategy->PreTuning(space->GenerateDesignSpace(mod));
  }
  for (int task_id; (task_id = this->NextTaskId()) != -1;) {
    TuneContext task = tasks[task_id];
    ICHECK(!task->is_stopped);
    ICHECK(!task->runner_futures.defined());
    SearchStrategy strategy = task->search_strategy.value();
    if (Optional<Array<MeasureCandidate>> candidates = strategy->GenerateMeasureCandidates()) {
      Array<BuilderResult> builder_results = SendToBuilder(this->builder, task, candidates.value());
      task->runner_futures = SendToRunner(this->runner, task, candidates.value(), builder_results);
    } else {
      SetTaskStopped(task_id);
    }
  }
  int n_tasks = this->tasks.size();
  for (int task_id = 0; task_id < n_tasks; ++task_id) {
    TuneContext task = tasks[task_id];
    this->JoinRunningTask(task_id);
    task->search_strategy.value()->PostTuning();
  }
}

void TaskSchedulerNode::SetTaskStopped(int task_id) {
  TuneContext task = tasks[task_id];
  ICHECK(!task->is_stopped);
  task->is_stopped = true;
}

bool TaskSchedulerNode::IsTaskRunning(int task_id) {
  TuneContext task = tasks[task_id];
  if (task->is_stopped || !task->runner_futures.defined()) {
    return false;
  }
  for (const RunnerFuture future : task->runner_futures.value()) {
    if (!future->Done()) {
      return true;
    }
  }
  this->JoinRunningTask(task_id);
  return false;
}

void TaskSchedulerNode::JoinRunningTask(int task_id) {
  TuneContext task = tasks[task_id];
  ICHECK(task->runner_futures.defined());
  Array<RunnerFuture> futures = task->runner_futures.value();
  int n = futures.size();
  Array<RunnerResult> results;
  results.reserve(n);
  for (const RunnerFuture future : task->runner_futures.value()) {
    results.push_back(future->Result());
  }
  task->search_strategy.value()->NotifyRunnerResults(results);
  task->runner_futures = NullOpt;
  // TODO(@zxybazh,@junrushao1994): add those records to the database
}

TVM_REGISTER_OBJECT_TYPE(TaskSchedulerNode);

}  // namespace meta_schedule
}  // namespace tvm
