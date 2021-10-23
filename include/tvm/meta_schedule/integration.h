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
#ifndef TVM_META_SCHEDULE_INTEGRATION_H_
#define TVM_META_SCHEDULE_INTEGRATION_H_

#include <tvm/meta_schedule/database.h>
#include <tvm/support/with.h>

#include <unordered_set>

namespace tvm {
namespace meta_schedule {

/**************** ExtractedTask ****************/

/*!
 * \brief A tuning task extracted from the high-level IR
 */
class ExtractedTaskNode : public runtime::Object {
 public:
  /*! \brief The name of the task extracted */
  String task_name;
  /*! \brief The high-level IR */
  IRModule mod;
  /*! \brief A list of low-level IRs that the high-level IR could potentially dispatch to */
  Array<IRModule> dispatched;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("task_name", &task_name);
    v->Visit("mod", &mod);
    v->Visit("dispatched", &dispatched);
  }

  static constexpr const char* _type_key = "meta_schedule.ExtractedTask";
  TVM_DECLARE_FINAL_OBJECT_INFO(ExtractedTaskNode, runtime::Object);
};

/*!
 * \brief Managed reference to ExtractedTaskNode
 * \sa ExtractedTaskNode
 */
class ExtractedTask : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor. The name of the task extracted
   * \brief The high-level IR
   * \brief A list of low-level IRs that the high-level IR could potentially dispatch to
   */
  explicit ExtractedTask(String task_name, IRModule mod, Array<IRModule> dispatched);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ExtractedTask, runtime::ObjectRef, ExtractedTaskNode);
};

/**************** IntegrationContext ****************/

/*!
 * \brief A context manager interface for the integration
 */
class IntegrationContextNode : public runtime::Object {
 public:
  /*! \brief Default destructor */
  virtual ~IntegrationContextNode() = default;
  /*!
   * \brief The entry point of the integration
   * \param task_name The name of the task
   * \param mod The high-level IR
   * \param dispatched A list of low-level IRs that the high-level IR could potentially dispatch to.
   * NullOpt means the dispatch needs to be done in the context.
   * \return There are different types of the output
   * 1) NullOpt if there is no feedback hint
   * 2) tir::PrimFunc if `mod` should be lowered to a PrimFunc
   * 3) relay::Function if `mod` should be dispatched to BYOC workflow
   * 4) IRModule for unified dispatch
   */
  virtual Optional<ObjectRef> Query(runtime::String task_name, IRModule mod,
                                    Optional<Array<IRModule>> dispatched) = 0;

  static constexpr const char* _type_key = "meta_schedule.IntegrationContext";
  TVM_DECLARE_BASE_OBJECT_INFO(IntegrationContextNode, runtime::Object);
};

/*!
 * \brief Managed reference to IntegrationContextNode
 * \sa IntegrationContextNode
 */
class IntegrationContext : public runtime::ObjectRef {
  friend class IntegrationContextInternal;
  friend class With<IntegrationContext>;

 public:
  /*! \brief Default destructor */
  virtual ~IntegrationContext() = default;
  /*!
   * \brief The context manager in the current scope
   * \return The IntegrationContext in the current scope. NullOpt if it's currently not under any
   * IntegrationContext.
   */
  static Optional<IntegrationContext> Current();
  /*!
   * \brief The entry point of the integration workflow. The compilation process of the high-level
   * IR should call this method for task extraction and for feedback hints
   * \param task_name The name of the task
   * \param mod The high-level IR
   * \param dispatched A list of low-level IRs that the high-level IR could potentially dispatch to
   * \return There are different types of the output
   * 1) NullOpt if there is no feedback hint
   * 2) tir::PrimFunc if `mod` should be lowered to a PrimFunc
   * 3) relay::Function if `mod` should be dispatched to BYOC workflow
   * 4) IRModule for unified dispatch
   */
  static Optional<ObjectRef> EntryPoint(runtime::String task_name, IRModule mod,
                                        Optional<Array<IRModule>> dispatched);

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IntegrationContext, runtime::ObjectRef,
                                                    IntegrationContextNode);

 protected:
  /*! \brief Default constructor */
  IntegrationContext() = default;
  /*! \brief Entering the scope of the context manager */
  void EnterWithScope();
  /*! \brief Exiting the scope of the context manager */
  void ExitWithScope();
};

/**************** TaskExtraction ****************/

/*!
 * \brief An integration context for task extraction
 */
class TaskExtractionNode : public IntegrationContextNode {
 public:
  /*! \brief The extracted tasks */
  Array<ExtractedTask> tasks{nullptr};

  void VisitAttrs(AttrVisitor* v) { v->Visit("tasks", &tasks); }

  // Inherited from base class
  Optional<ObjectRef> Query(runtime::String task_name, IRModule mod,
                            Optional<Array<IRModule>> dispatched) final;

  static constexpr const char* _type_key = "meta_schedule.TaskExtraction";
  TVM_DECLARE_FINAL_OBJECT_INFO(TaskExtractionNode, IntegrationContextNode);
};

/*!
 * \brief Managed reference to TaskExtractionNode
 * \sa TaskExtractionNode
 */
class TaskExtraction : public IntegrationContext {
 public:
  /*! \brief The path to a cache file storing extracted tasks */
  TaskExtraction();
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TaskExtraction, IntegrationContext,
                                                    TaskExtractionNode);
};

/**************** ApplyHistoryBest ****************/

/*!
 * \brief An integration context that allows application of historically best records from a
 * database
 */
class ApplyHistoryBestNode : public IntegrationContextNode {
 public:
  /*! \brief The database to be queried from */
  Database database{nullptr};

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("database", &database);  //
  }

  // Inherited from base class
  Optional<ObjectRef> Query(runtime::String task_name, IRModule mod,
                            Optional<Array<IRModule>> dispatched) final;

  static constexpr const char* _type_key = "meta_schedule.ApplyHistoryBest";
  TVM_DECLARE_FINAL_OBJECT_INFO(ApplyHistoryBestNode, IntegrationContextNode);
};

/*!
 * \brief Managed reference to ApplyHistoryBestNode
 * \sa ApplyHistoryBestNode
 */
class ApplyHistoryBest : public IntegrationContext {
 public:
  /*!
   * \brief Constructor
   * \param database The database to be queried from
   */
  explicit ApplyHistoryBest(Database database);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ApplyHistoryBest, IntegrationContext,
                                                    ApplyHistoryBestNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_INTEGRATION_H_
