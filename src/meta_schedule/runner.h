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

/*! \brief The class for runner's measure result type. */
class MeasureResultNode : public runtime::Object {
 public:
  /*! \brief The running result in gflops. */
  Optional<FloatImm> gflops;

  /*!
   * \brief Visitor for variables in python.
   * \note required for non-abstract classes.
   */
  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("gflops", &gflops); }

  /*! \brief Class name `MeasureResult`. */
  static constexpr const char* _type_key = "meta_schedule.MeasureResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(MeasureResultNode, runtime::Object);  // Concrete class
};

/*! \brief The class for runner's measure result type. */
class MeasureResult : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor function of MeasureResult class.
   * \param gflops The running result in gflops.
   */
  TVM_DLL explicit MeasureResult(Optional<FloatImm> gflops);
  /*! \brief Declare reference relationship. */
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(MeasureResult, runtime::ObjectRef, MeasureResultNode);
};

/*! \brief The class for runner. */
class RunnerNode : public runtime::Object {
 public:
  /*! \brief The function type to asynchronously fetch runner's measure results. */
  using RunnerFuture = runtime::TypedPackedFunc<Optional<Array<MeasureResult>>()>;

  /*! \brief Virtual destructor, required for abstract class. */
  virtual ~RunnerNode() = default;
  /*!
   * \brief Virtual function to send builds to runenr and return the callback function.
   * \param build_results The builds returned from the builder.
   * \return The callback function to fetch the measure results.
   */
  virtual RunnerFuture Run(const Array<BuildResult>& build_results) = 0;

  /*! \brief Class name `Runner` */
  static constexpr const char* _type_key = "meta_schedule.Runner";
  TVM_DECLARE_BASE_OBJECT_INFO(RunnerNode, runtime::Object);  // Abstract class
};

/*!
 * \brief Managed reference to RunnerNode
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