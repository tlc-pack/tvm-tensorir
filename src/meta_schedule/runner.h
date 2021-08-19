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
#ifndef SRC_META_SCHEDULE_RUNNER_H_
#define SRC_META_SCHEDULE_RUNNER_H_

#include <tvm/ir/module.h>
#include <tvm/target/target.h>

#include "./builder.h"

namespace tvm {
namespace meta_schedule {

/*! \brief The class for runner's input type. */
class RunnerInputNode : public runtime::Object {
 public:
  /*! \brief The binary artifact to be uploaded to the remote runner. */
  String artifact_path;

  /*!
   * \brief Visitor for variables in python.
   * \note required for non-abstract classes.
   */
  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("artifact_path", &artifact_path); }

  /*! \brief Class name `RunnerInput`. */
  static constexpr const char* _type_key = "meta_schedule.RunnerInput";
  TVM_DECLARE_FINAL_OBJECT_INFO(RunnerInputNode, runtime::Object);  // Concrete class
};

/*!
 * \brief Managed reference to RunnerInputNode.
 * \sa RunnerInputNode
 */
class RunnerInput : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor function of RunnerInput class.
   * \param artifact_path The binary artifact to be uploaded to the remote runner.
   */
  TVM_DLL explicit RunnerInput(String artifact_path) {
    ObjectPtr<RunnerInputNode> n = make_object<RunnerInputNode>();
    n->artifact_path = artifact_path;
    data_ = std::move(n);
  }

  /*! \brief Declare reference relationship. */
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(RunnerInput, runtime::ObjectRef, RunnerInputNode);
};

/*! \brief The class for runner's result type. */
class RunnerResultNode : public runtime::Object {
 public:
  /*! \brief The running time obtained via profiling, if it runs successfully. */
  Array<FloatImm> run_secs;
  /*! \brief The error message, if error occurs. */
  Optional<String> error_msg;

  /*!
   * \brief Visitor for variables in python.
   * \note required for non-abstract classes.
   */
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("run_secs", &run_secs);
    v->Visit("error_msg", &error_msg);
  }

  /*! \brief Class name `RunnerResult`. */
  static constexpr const char* _type_key = "meta_schedule.RunnerResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(RunnerResultNode, runtime::Object);  // Concrete class
};

/*!
 * \brief Managed reference to RunnerResultNode.
 * \sa RunnerResultNode
 */
class RunnerResult : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor function of RunnerResult class.
   * \param run_secs The running time obtained via profiling, if it runs successfully.
   * \param error_msg The error message, if error occurs.
   */
  TVM_DLL explicit RunnerResult(Array<FloatImm> run_secs, Optional<String> error_msg);
  /*! \brief Declare reference relationship. */
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(RunnerResult, runtime::ObjectRef, RunnerResultNode);
};

/*! \brief The class for runner. */
class RunnerNode : public runtime::Object {
 public:
  /*! \brief The function type to asynchronously fetch runner's runner results. */
  using RunnerFuture = runtime::TypedPackedFunc<Optional<Array<RunnerResult>>()>;

  /*! \brief Virtual destructor, required for abstract class. */
  virtual ~RunnerNode() = default;
  /*!
   * \brief Virtual function to run and profile a batch of binary artifacts.
   * \param runner_input The batch of binary artifacts.
   * \return The callback function to fetch the runner results.
   */
  virtual RunnerFuture Run(const Array<RunnerInput>& runner_input) = 0;

  /*! \brief Class name `Runner` */
  static constexpr const char* _type_key = "meta_schedule.Runner";
  TVM_DECLARE_BASE_OBJECT_INFO(RunnerNode, runtime::Object);  // Abstract class
};

/*!
 * \brief Managed reference to RunnerNode.
 * \sa RunnerNode
 */
class Runner : public runtime::ObjectRef {
 public:
  /*! \brief Declare reference relationship. */
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Runner, runtime::ObjectRef, RunnerNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_RUNNER_H_