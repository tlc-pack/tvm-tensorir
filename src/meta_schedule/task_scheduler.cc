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

#include "task_scheduler.h"

#include "schedule.h"

namespace tvm {
namespace meta_schedule {

// Round robin task scheduler
void TaskSchedulerNode::SortAllTasks() {}

void TaskSchedulerNode::TuneAllTasks() {
  int num_unfinished = tasks.size();
  for (TaskWithContext task : tasks) {
    ICHECK(task.tune_context->workload.defined()) << "Workload not set";
    ICHECK(task.tune_context->space_generator.defined()) << "Design space generator not set";
    ICHECK(task.tune_context->search_strategy.defined()) << "Search strategy not set";

    Array<Trace> design_spaces =
        task.tune_context->space_generator.value()->Generate(task.tune_context->workload.value());
    task.tune_context->search_strategy.value()->PreTuning(design_spaces);
  }
  while (num_unfinished > 0) {
    SortAllTasks();
    for (TaskWithContext task : tasks) {
      if (task.is_finished) continue;

      if (task.runner_callback == nullptr) {
        Optional<runtime::Array<BuildInput>> measure_candidates =
            task.tune_context->search_strategy.value()->GenerateMeasureCandidates();
        if (measure_candidates.defined()) {
          ICHECK(builder.defined()) << "Builder not set";
          Array<BuildResult> builds = builder.value()->Build(measure_candidates.value());
          ICHECK(runner.defined()) << "Runner not set";
          task.runner_callback = runner.value()->Run(builds);
        } else {
          task.is_finished = true;
          task.tune_context->search_strategy.value()->PostTuning();
          --num_unfinished;
        }
      } else if (Optional<Array<MeasureResult>> results = task.runner_callback()) {
        task.runner_callback = RunnerFuture(nullptr);
        // Search strategy already ICHECKed when called before.
        task.tune_context->search_strategy.value()->NotifyMeasureResults(results.value());
      }
    }
  }
}

TVM_REGISTER_OBJECT_TYPE(TaskSchedulerNode);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerTuneAllTasks")
    .set_body_method<TaskScheduler>(&TaskSchedulerNode::TuneAllTasks);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerSortAllTasks")
    .set_body_method<TaskScheduler>(&TaskSchedulerNode::SortAllTasks);

}  // namespace meta_schedule
}  // namespace tvm
