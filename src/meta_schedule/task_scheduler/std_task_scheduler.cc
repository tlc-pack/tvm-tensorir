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

/*! \brief Each task's resource class for task scheduling.  */
class TaskWithContextNode : public runtime::Object {
 public:
  /*! \brief Whether the task has been finished. */
  bool is_finished;
  /*! \brief The tuning context for the task. */
  TuneContext tune_context;
  /*! \brief The function returned by runner to asynchronously fetch runner results. */
  RunnerFuture runner_callback;

  /*!
   * \brief Visitor for variables in python.
   * \note required for non-abstract classes.
   */
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("is_finished", &is_finished);
    v->Visit("tune_context", &tune_context);
    // runner_callback is not visited because it is not serializable.
  }

  static constexpr const char* _type_key = "meta_schedule.TaskWithContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(TaskWithContextNode, Object);
};

/*!
 * \brief Managed reference to TaskWithContextNode.
 * \sa TaskWithContextNode
 */
class TaskWithContext : public runtime::ObjectRef {
 public:
  /*! \brief Constructor */
  TVM_DLL explicit TaskWithContext(const TuneContext& tune_context) {
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

/*! \brief The default class for task scheduling. */
class StandardTaskSchedulerNode : public TaskSchedulerNode {
 public:
  /*! \brief The array of all task resources. */
  Array<TaskWithContext> tasks;

  /*!
   * \brief Visitor for variables in python.
   * \note required for non-abstract classes.
   */
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("tasks", &tasks);
    v->Visit("builder", &builder);
    v->Visit("runner", &runner);
  }

  /*! \brief Round-robin style task scheduling. */
  void SortAllTasks() {}

  /*! \brief Auto tuning all the tasks. */
  void TuneAllTasks() {
    // Count number of all the tasks.
    int num_unfinished = tasks.size();
    // Pretune all the tasks.
    for (TaskWithContext task : tasks) {
      // Checks for optional fields.
      ICHECK(task->tune_context->mod.defined()) << "Workload not set";
      ICHECK(task->tune_context->space_generator.defined()) << "Design space generator not set";
      ICHECK(task->tune_context->search_strategy.defined()) << "Search strategy not set";
      // Pretune the task.
      Array<tir::Schedule> design_spaces =
          task->tune_context->space_generator.value()->GenerateDesignSpace(
              task->tune_context->mod.value());
      task->tune_context->search_strategy.value()->PreTuning(design_spaces);
    }
    // The auto tuning loop.
    while (num_unfinished > 0) {
      // Sort all tasks according to their priority.
      SortAllTasks();
      // Enumerate all the tasks.
      for (TaskWithContext task : tasks) {
        // Check if the task is finished.
        if (task->is_finished) continue;
        // Check if the task has been sent to the runner.
        if (task->runner_callback == RunnerFuture(nullptr)) {
          // If not, generate the next batch of measure candidates.
          Optional<runtime::Array<BuilderInput>> measure_candidates =
              task->tune_context->search_strategy.value()->GenerateMeasureCandidates();
          if (measure_candidates.defined()) {
            // If candidates are generated, send them to the builder.
            ICHECK(builder.defined()) << "Builder not set";
            Array<BuilderResult> build_results = builder.value()->Build(measure_candidates.value());
            ICHECK(runner.defined()) << "Runner not set";
            // Send the builds to the runner, and get the runner's callback.
            Array<RunnerInput> runner_inputs;
            for (BuilderResult build_result : build_results) {
              if (build_result->artifact_path.defined()) {
                runner_inputs.push_back(RunnerInput(build_result->artifact_path.value()));
              }
            }
            task->runner_callback = runner.value()->Run(runner_inputs);
          } else {
            // If no measure candidates are generated, the task is finished.
            task->is_finished = true;
            // Posttune the task.
            task->tune_context->search_strategy.value()->PostTuning();
            // Remove the task from the unfinished tasks.
            --num_unfinished;
          }
        } else if (Optional<Array<RunnerResult>> results = task->runner_callback()) {
          // Clear the runner's callback.
          task->runner_callback = RunnerFuture(nullptr);
          // Search strategy already been checked when called before.
          // Update the search strategy status with the runner results.
          task->tune_context->search_strategy.value()->NotifyRunnerResults(results.value());
        }
      }
    }
  }

  static constexpr const char* _type_key = "meta_schedule.StandardTaskScheduler";
  TVM_DECLARE_FINAL_OBJECT_INFO(StandardTaskSchedulerNode, TaskSchedulerNode);
};

/*!
 * \brief Expose the StandardTaskScheduler's constructor function as a member function of
 *  TaskScheduler.
 * \param tune_contexts The array of tuning contexts for different tasks.
 * \param builder The builder of the scheduler.
 * \param runner The runner of the scheduler.
 * \return The constructed StandardTaskScheduler object but in TaskScheduler type.
 */
TaskScheduler TaskScheduler::StandardTaskScheduler(Array<TuneContext> tune_contexts,  //
                                                   Optional<Builder> builder,         //
                                                   Optional<Runner> runner) {
  ObjectPtr<StandardTaskSchedulerNode> n = make_object<StandardTaskSchedulerNode>();
  n->tasks.reserve(tune_contexts.size());
  for (size_t i = 0; i < tune_contexts.size(); ++i) {
    n->tasks.push_back(TaskWithContext(tune_contexts[i]));
  }
  n->builder = std::move(builder);
  n->runner = std::move(runner);
  return TaskScheduler(n);
}

TVM_REGISTER_NODE_TYPE(StandardTaskSchedulerNode);

TVM_REGISTER_GLOBAL("meta_schedule.StandardTaskScheduler")
    .set_body_typed(TaskScheduler::StandardTaskScheduler);

}  // namespace meta_schedule
}  // namespace tvm
