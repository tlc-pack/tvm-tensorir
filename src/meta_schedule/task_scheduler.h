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
#ifndef SRC_META_SCHEDULE_TASK_SCHEDULER_H_
#define SRC_META_SCHEDULE_TASK_SCHEDULER_H_

#include <tvm/ir/module.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/object.h>

#include "./builder.h"
#include "./runner.h"
#include "./tune_context.h"

namespace tvm {
namespace meta_schedule {

/*! \brief The abstract interface of task schedulers. */
class TaskSchedulerNode : public runtime::Object {
 public:
  /*! \brief The function type of `TuneAllTasks` method. */
  using FTuneAllTasks = runtime::TypedPackedFunc<void()>;
  /*! \brief The function type of `SortAllTasks` method. */
  using FSortAllTasks = runtime::TypedPackedFunc<void()>;

  virtual ~TaskSchedulerNode() = default;

  /*! \brief The builder of the scheduler. */
  Optional<Builder> builder;
  /*! \brief The runner of the scheduler. */
  Optional<Runner> runner;

  /*! \brief Sort all tuning tasks with certain priority. */
  virtual void SortAllTasks() = 0;
  /*! \brief Run auto-tuning on all tasks. */
  virtual void TuneAllTasks() = 0;

  static constexpr const char* _type_key = "meta_schedule.TaskScheduler";
  TVM_DECLARE_BASE_OBJECT_INFO(TaskSchedulerNode, Object);
};

/*!
 * \brief Managed reference to TaskSchedulerNode.
 * \sa TaskSchedulerNode
 */
class TaskScheduler : public runtime::ObjectRef {
 public:
  /*!
   * \brief Member function to create the StandardTaskScheduler class.
   * \param tune_contexts The array of tuning contexts for different tasks.
   * \param builder The builder of the task scheduler.
   * \param runner The runner of the task scheduler.
   * \return The constructed StandardTaskScheduler object but in TaskScheduler type.
   */
  static TaskScheduler StandardTaskScheduler(Array<TuneContext> tune_contexts,  //
                                             Optional<Builder> builder,         //
                                             Optional<Runner> runner);
  /*!
   * \brief Member function to create the python side customizable PyTaskScheduler class.
   * \param tune_contexts The array of tuning contexts for different tasks.
   * \param builder The builder of the task scheduler.
   * \param runner The runner of the task scheduler.
   * \param sort_all_tasks_func The function to sort all tuning tasks with certain priority.
   * \param tune_all_tasks_func The function to run auto-tuning on all tasks.
   * \return The constructed PyTaskScheduler object but in TaskScheduler type.
   */
  static TaskScheduler PyTaskScheduler(Array<TuneContext> tune_contexts,                      //
                                       Optional<Builder> builder,                             //
                                       Optional<Runner> runner,                               //
                                       TaskSchedulerNode::FSortAllTasks sort_all_tasks_func,  //
                                       TaskSchedulerNode::FTuneAllTasks tune_all_tasks_func);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TaskScheduler, ObjectRef, TaskSchedulerNode);
};

/*! \brief The python side customizable class for task scheduling. */
class PyTaskSchedulerNode : public TaskSchedulerNode {
 public:
  /*! \brief The array of tuning contexts for different tasks. */
  Array<TuneContext> tune_contexts;
  /*! \brief Pointer to the `SortAllTasks` funcion in python. */
  FSortAllTasks sort_all_tasks_func;
  /*! \brief Pointer to the `TuneAllTasks` funcion in python. */
  FTuneAllTasks tune_all_tasks_func;

  /*!
   * \brief Visitor for variables in python.
   * \note required for non-abstract classes.
   */
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tune_contexts", &tune_contexts);
    v->Visit("builder", &builder);
    v->Visit("runner", &runner);
  }

  /*! \brief Use the given function pointer to override the `SortAllTasks` function. */
  void SortAllTasks() override { this->sort_all_tasks_func(); }

  /*! \brief Use the given function pointer to override the `TuneAllTasks` function. */
  void TuneAllTasks() override { this->tune_all_tasks_func(); }

  static constexpr const char* _type_key = "meta_schedule.PyTaskScheduler";
  TVM_DECLARE_FINAL_OBJECT_INFO(PyTaskSchedulerNode, TaskSchedulerNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_TASK_SCHEDULER_H_
