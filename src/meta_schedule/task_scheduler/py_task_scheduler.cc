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

  /*! \brief Class name `PyTaskScheduler`. */
  static constexpr const char* _type_key = "meta_schedule.PyTaskScheduler";
  TVM_DECLARE_FINAL_OBJECT_INFO(PyTaskSchedulerNode, TaskSchedulerNode);  // Concrete class
};

/*!
 * \brief Managed reference to PyTaskScheduler.
 * \sa PyTaskSchedulerNode
 */
class PyTaskScheduler : public TaskScheduler {
 public:
  /*! \brief Constructor function of PyTaskScheduler class. */
  TVM_DLL explicit PyTaskScheduler(Array<TuneContext> tune_contexts,                      //
                                   Optional<Builder> builder,                             //
                                   Optional<Runner> runner,                               //
                                   TaskSchedulerNode::FSortAllTasks sort_all_tasks_func,  //
                                   TaskSchedulerNode::FTuneAllTasks tune_all_tasks_func) {
    // Make a new PyTaskScheduler object.
    ObjectPtr<PyTaskSchedulerNode> n = make_object<PyTaskSchedulerNode>();
    // Copy the given arguments and function pointers.
    n->tune_contexts = std::move(tune_contexts);
    n->builder = std::move(builder);
    n->runner = std::move(runner);
    n->tune_all_tasks_func = std::move(tune_all_tasks_func);
    n->sort_all_tasks_func = std::move(sort_all_tasks_func);
    data_ = std::move(n);
  }

  /*! \brief Declare reference relationship. */
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PyTaskScheduler, TaskScheduler,
                                                    PyTaskSchedulerNode);
};

/*!
 * \brief Expose the PyTaskScheduler constructor function as a member function of TaskScheduler.
 * \param tune_contexts The array of tuning contexts for different tasks.
 * \param builder The builder of the scheduler.
 * \param runner The runner of the scheduler.
 * \param sort_all_tasks_func The function to sort all tuning tasks with certain priority.
 * \param tune_all_tasks_func The function to run auto-tuning on all tasks.
 * \return The constructed PyTaskScheduler object but in TaskScheduler type.
 */
TaskScheduler TaskScheduler::PyTaskScheduler(
    Array<TuneContext> tune_contexts,                      //
    Optional<Builder> builder,                             //
    Optional<Runner> runner,                               //
    TaskSchedulerNode::FSortAllTasks sort_all_tasks_func,  //
    TaskSchedulerNode::FTuneAllTasks tune_all_tasks_func) {
  return meta_schedule::PyTaskScheduler(tune_contexts, builder, runner, tune_all_tasks_func,
                                        sort_all_tasks_func);
}

TVM_REGISTER_NODE_TYPE(PyTaskSchedulerNode);  // Concrete class
/*! \brief Register TaskScheduler's `PyTaskScheduler` function to global registry. */
TVM_REGISTER_GLOBAL("meta_schedule.PyTaskScheduler").set_body_typed(TaskScheduler::PyTaskScheduler);

}  // namespace meta_schedule
}  // namespace tvm
