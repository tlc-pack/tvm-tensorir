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

/**************** PyTaskScheduler ****************/

class PyTaskSchedulerNode : public TaskSchedulerNode {
 public:
  Array<TuneContext> tune_contexts;
  /*! \brief Pointer to the task sorting function in python */
  FSortAllTasks sort_all_tasks_func;
  /*! \brief Pointer to the autotuning function in python */
  FTuneAllTasks tune_all_tasks_func;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("tune_contexts", &tune_contexts); }

  void SortAllTasks() override { this->sort_all_tasks_func(); }

  void TuneAllTasks() override { this->tune_all_tasks_func(); }

  static constexpr const char* _type_key = "meta_schedule.PyTaskScheduler";
  TVM_DECLARE_FINAL_OBJECT_INFO(PyTaskSchedulerNode, TaskSchedulerNode);
};

/*!
 * \brief Managed reference to PyTaskScheduler.
 * \sa PyTaskSchedulerNode
 */
class PyTaskScheduler : public TaskScheduler {
 public:
  explicit PyTaskScheduler(Array<TuneContext> tune_contexts,                      //
                           Optional<Builder> builder,                             //
                           Optional<Runner> runner,                               //
                           TaskSchedulerNode::FSortAllTasks sort_all_tasks_func,  //
                           TaskSchedulerNode::FTuneAllTasks tune_all_tasks_func) {
    ObjectPtr<PyTaskSchedulerNode> n = make_object<PyTaskSchedulerNode>();
    n->tune_contexts = std::move(tune_contexts);
    n->builder = std::move(builder);
    n->runner = std::move(runner);
    n->tune_all_tasks_func = std::move(tune_all_tasks_func);
    n->sort_all_tasks_func = std::move(sort_all_tasks_func);
    data_ = std::move(n);
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PyTaskScheduler, TaskScheduler,
                                                    PyTaskSchedulerNode);
};

TaskScheduler TaskScheduler::PyTaskScheduler(
    Array<TuneContext> tune_contexts,                      //
    Optional<Builder> builder,                             //
    Optional<Runner> runner,                               //
    TaskSchedulerNode::FSortAllTasks sort_all_tasks_func,  //
    TaskSchedulerNode::FTuneAllTasks tune_all_tasks_func) {
  return meta_schedule::PyTaskScheduler(tune_contexts, builder, runner, tune_all_tasks_func,
                                        sort_all_tasks_func);
}

TVM_REGISTER_NODE_TYPE(PyTaskSchedulerNode);
TVM_REGISTER_GLOBAL("meta_schedule.PyTaskScheduler").set_body_typed(TaskScheduler::PyTaskScheduler);

}  // namespace meta_schedule
}  // namespace tvm
