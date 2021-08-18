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

#include <tvm/ir/module.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/object.h>

#include "./builder.h"
#include "./runner.h"
#include "./tune_context.h"

#ifndef SRC_META_SCHEDULE_TASK_SCHEDULER_H_
#define SRC_META_SCHEDULE_TASK_SCHEDULER_H_

namespace tvm {
namespace meta_schedule {

struct TaskWithContext {
  bool is_finished;
  TuneContext tune_context;
  RunnerNode::RunnerFuture runner_callback;

  explicit TaskWithContext(const TuneContext& tune_context)
      : is_finished(false), tune_context(tune_context), runner_callback(nullptr) {}
};

class TaskSchedulerNode : public runtime::Object {
 public:
  using RunnerFuture = RunnerNode::RunnerFuture;
  using FTuneAllTasks = runtime::TypedPackedFunc<void()>;
  using FSortAllTasks = runtime::TypedPackedFunc<void()>;

  /*! \brief Virtual destructor */
  virtual ~TaskSchedulerNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*! \brief Tasks of the scheduler. */
  std::vector<TaskWithContext> tasks;

  Optional<Builder> builder;
  Optional<Runner> runner;

  /*! \brief Run auot-tuning on all tasks. */
  virtual void TuneAllTasks();

  /*! \brief Sort all tuning tasks, together with the runner_callback functions. */
  virtual void SortAllTasks();

  static constexpr const char* _type_key = "meta_schedule.TaskScheduler";
  TVM_DECLARE_BASE_OBJECT_INFO(TaskSchedulerNode, Object);
};

/*!
 * \brief Managed reference to TaskScheduler Node
 * \sa TuneContextNode
 */
class TaskScheduler : public runtime::ObjectRef {
 public:
  /*! \brief Constructor */
  explicit TaskScheduler(Array<TuneContext> tune_contexts,  //
                         Optional<Builder> builder,         //
                         Optional<Runner> runner) {
    ObjectPtr<TaskSchedulerNode> n = make_object<TaskSchedulerNode>();
    n->builder = builder;
    n->runner = runner;
    n->tasks.reserve(tune_contexts.size());
    for (size_t i = 0; i < tune_contexts.size(); ++i) {
      n->tasks.push_back(TaskWithContext(tune_contexts[i]));
    }
    data_ = std::move(n);
  }

  static TaskScheduler PyTaskScheduler(Array<TuneContext> tune_contexts,                      //
                                       Optional<Builder> builder,                             //
                                       Optional<Runner> runner,                               //
                                       TaskSchedulerNode::FSortAllTasks sort_all_tasks_func,  //
                                       TaskSchedulerNode::FTuneAllTasks tune_all_tasks_func);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TaskScheduler, ObjectRef, TaskSchedulerNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_TASK_SCHEDULER_H_
