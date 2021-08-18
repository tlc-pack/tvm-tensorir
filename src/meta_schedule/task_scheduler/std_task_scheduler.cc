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

using RunnerFuture = RunnerNode::RunnerFuture;

/**************** TaskWithContext ****************/

class TaskWithContextNode : public runtime::Object {
 public:
  bool is_finished;
  TuneContext tune_context;
  RunnerNode::RunnerFuture runner_callback;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("is_finished", &is_finished);
    v->Visit("tune_context", &tune_context);
    // runner_callback is not visited because it is not serializable.
  }

  static constexpr const char* _type_key = "meta_schedule.TaskWithContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(TaskWithContextNode, Object);
};

class TaskWithContext : public runtime::ObjectRef {
 public:
  explicit TaskWithContext(const TuneContext& tune_context) {
    ObjectPtr<TaskWithContextNode> n = make_object<TaskWithContextNode>();
    n->is_finished = false;
    n->tune_context = std::move(tune_context);
    n->runner_callback = RunnerFuture(nullptr);
    data_ = std::move(n);
  }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TaskWithContext, ObjectRef,
                                                    TaskWithContextNode);
};

TVM_REGISTER_NODE_TYPE(TaskWithContextNode);

/**************** StandardTaskScheduler ****************/

class StandardTaskSchedulerNode : public TaskSchedulerNode {
 public:
  /*! \brief Tasks of the scheduler. */
  Array<TaskWithContext> tasks;

  // Round robin task scheduler
  void SortAllTasks() {}

  void TuneAllTasks() {
    int num_unfinished = tasks.size();
    for (TaskWithContext task : tasks) {
      ICHECK(task->tune_context->workload.defined()) << "Workload not set";
      ICHECK(task->tune_context->space_generator.defined()) << "Design space generator not set";
      ICHECK(task->tune_context->search_strategy.defined()) << "Search strategy not set";

      Array<tir::Trace> design_spaces = task->tune_context->space_generator.value()->Generate(
          task->tune_context->workload.value());
      task->tune_context->search_strategy.value()->PreTuning(design_spaces);
    }
    while (num_unfinished > 0) {
      SortAllTasks();
      for (TaskWithContext task : tasks) {
        if (task->is_finished) continue;

        if (task->runner_callback == nullptr) {
          Optional<runtime::Array<BuildInput>> measure_candidates =
              task->tune_context->search_strategy.value()->GenerateMeasureCandidates();
          if (measure_candidates.defined()) {
            ICHECK(builder.defined()) << "Builder not set";
            Array<BuildResult> builds = builder.value()->Build(measure_candidates.value());
            ICHECK(runner.defined()) << "Runner not set";
            task->runner_callback = runner.value()->Run(builds);
          } else {
            task->is_finished = true;
            task->tune_context->search_strategy.value()->PostTuning();
            --num_unfinished;
          }
        } else if (Optional<Array<MeasureResult>> results = task->runner_callback()) {
          task->runner_callback = RunnerFuture(nullptr);
          // Search strategy already ICHECKed when called before.
          task->tune_context->search_strategy.value()->NotifyMeasureResults(results.value());
        }
      }
    }
  }

  static constexpr const char* _type_key = "meta_schedule.StandardTaskScheduler";
  TVM_DECLARE_FINAL_OBJECT_INFO(StandardTaskSchedulerNode, TaskSchedulerNode);
};

/*!
 * \brief Managed reference to TaskScheduler.
 * \sa TaskSchedulerNode
 */
class StandardTaskScheduler : public TaskScheduler {
 public:
  explicit StandardTaskScheduler(Array<TuneContext> tune_contexts,  //
                                 Optional<Builder> builder,         //
                                 Optional<Runner> runner) {
    ObjectPtr<StandardTaskSchedulerNode> n = make_object<StandardTaskSchedulerNode>();
    n->tasks.reserve(tune_contexts.size());
    for (size_t i = 0; i < tune_contexts.size(); ++i) {
      n->tasks.push_back(TaskWithContext(tune_contexts[i]));
    }
    n->builder = std::move(builder);
    n->runner = std::move(runner);
    data_ = std::move(n);
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(StandardTaskScheduler, TaskScheduler,
                                                    StandardTaskSchedulerNode);
};

TaskScheduler TaskScheduler::StandardTaskScheduler(Array<TuneContext> tune_contexts,  //
                                                   Optional<Builder> builder,         //
                                                   Optional<Runner> runner) {
  return StandardTaskScheduler(tune_contexts, builder, runner);
}

TVM_REGISTER_NODE_TYPE(StandardTaskSchedulerNode);
TVM_REGISTER_GLOBAL("meta_schedule.StandardTaskScheduler")
    .set_body_typed(TaskScheduler::StandardTaskScheduler);

}  // namespace meta_schedule
}  // namespace tvm
