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
#ifndef TVM_META_SCHEDULE_TASK_SCHEDULER_H_
#define TVM_META_SCHEDULE_TASK_SCHEDULER_H_

#include "./builder.h"
#include "./runner.h"
#include "./tune_context.h"

namespace tvm {
namespace meta_schedule {

/*! \brief The abstract interface of task schedulers. */
class TaskSchedulerNode : public runtime::Object {
 public:
  /*! \brief The tasks to be tuned */
  Array<TuneContext> tasks;
  /*! \brief The builder of the scheduler. */
  Builder builder{nullptr};
  /*! \brief The runner of the scheduler. */
  Runner runner{nullptr};
  /*! \brief The database of the scheduler. */
  Database database{nullptr};

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tasks", &tasks);
    v->Visit("builder", &builder);
    v->Visit("runner", &runner);
    v->Visit("database", &database);
  }

  static constexpr const char* _type_key = "meta_schedule.TaskScheduler";
  TVM_DECLARE_BASE_OBJECT_INFO(TaskSchedulerNode, Object);

 public:
  /*! \brief The default desctructor. */
  virtual ~TaskSchedulerNode() = default;

  /*! \brief Auto-tuning. */
  virtual void Tune();

 protected:
  /*!
   * \brief Set specific task to be stopped.
   * \param task_id The task id to be stopped.
   */
  virtual void SetTaskStopped(int task_id);

  /*!
   * \brief Check whether the task is running.
   * \param task_id The task id to be checked.
   */
  virtual bool IsTaskRunning(int task_id);

  /*!
   * \brief Wait until the task is finished.
   * \param task_id The task id to be joined.
   */
  virtual void JoinRunningTask(int task_id);

  /*!
   * \brief Fetch the next task by id.
   * \return The next task id.
   */
  virtual int NextTaskId() = 0;
};

/*!
 * \brief Managed reference to TaskSchedulerNode.
 * \sa TaskSchedulerNode
 */
class TaskScheduler : public runtime::ObjectRef {
 public:
  /*!
   * \brief Create a task scheduler that fetches tasks in a round-robin fashion.
   * \param tasks The tasks to be tuned.
   * \param builder The builder of the scheduler.
   * \param runner The runner of the scheduler.
   * \param database The database of the scheduler.
   */
  TVM_DLL TaskScheduler RoundRobin(Array<TuneContext> tasks, Builder builder, Runner runner,
                                   Database database);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TaskScheduler, ObjectRef, TaskSchedulerNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_TASK_SCHEDULER_H_
