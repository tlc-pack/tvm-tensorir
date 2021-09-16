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

class TaskNode : public runtime::Object {
 public:
  TuneContext context{nullptr};
  bool is_stopped;
  Optional<Array<RunnerFuture>> running;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("context", &context);
    v->Visit("is_stopped", &is_stopped);
    v->Visit("running", &running);
  }

  static constexpr const char* _type_key = "meta_schedule.Task";
  TVM_DECLARE_BASE_OBJECT_INFO(TaskNode, Object);
};

class Task : public runtime::ObjectRef {
 public:
  TVM_DLL explicit Task(TuneContext context);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Task, ObjectRef, TaskNode);
};

/*! \brief The abstract interface of task schedulers. */
class TaskSchedulerNode : public runtime::Object {
 public:
  /*! \brief The tasks to be tuned */
  Array<Task> tasks;
  /*! \brief The builder of the scheduler. */
  Builder builder;
  /*! \brief The runner of the scheduler. */
  Runner runner;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tasks", &tasks);
    v->Visit("builder", &builder);
    v->Visit("runner", &runner);
  }

  static constexpr const char* _type_key = "meta_schedule.TaskScheduler";
  TVM_DECLARE_BASE_OBJECT_INFO(TaskSchedulerNode, Object);

 public:
  virtual ~TaskSchedulerNode() = default;

  virtual void Tune();

 protected:
  virtual bool IsTaskRunning(int task_id);

  virtual void JoinRunningTask(int task_id);

  virtual int NextTaskId() = 0;
};

/*!
 * \brief Managed reference to TaskSchedulerNode.
 * \sa TaskSchedulerNode
 */
class TaskScheduler : public runtime::ObjectRef {
 public:
  TVM_DLL TaskScheduler RoundRobin(Array<Task> tasks, Builder builder, Runner runner);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TaskScheduler, ObjectRef, TaskSchedulerNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_TASK_SCHEDULER_H_
