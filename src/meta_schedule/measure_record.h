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

/*!
 * Taken and modified from Ansor
 * \brief Distributed measurement infrastructure to measure the runtime costs of tensor programs.
 * These functions are responsible for building the tvm module, uploading it to remote devices,
 * recording the running time costs, and checking the correctness of the output.
 *
 * The measurement is separated into two steps: build and run.
 * A builder builds the executable binary files and a runner runs the binary files to get the
 * measurement results. The flow of data structures is
 *
 *                 `ProgramBuilder`                 `ProgramRunner`
 * `MeasureInput` -----------------> `BuildResult` ----------------> `MeasureResult`
 *
 *
 * The core functions is implemented in python to utilize python's multiprocessing
 * and error handling (see also `python/tvm/meta_schedule/measure.py`).
 * This c++ file is just a wrapper for the python functions.
 */

#ifndef SRC_META_SCHEDULE_MEASURE_RECORD_H_
#define SRC_META_SCHEDULE_MEASURE_RECORD_H_

#include "./schedule.h"
#include "./search.h"

namespace tvm {
namespace meta_schedule {

/*! \brief The error code of one measurement */
enum class MeasureErrorNO : int {
  /*! \brief No error. */
  kNoError = 0,
  /*! \brief Errors happen when apply transform steps from init state. */
  kInstantiationError = 1,
  /*! \brief Errors happen when compiling code on host. (when build module) */
  kCompileHostError = 2,
  /*! \brief Errors happen when compiling code on device. (when load module) */
  kCompileDeviceError = 3,
  /*! \brief Errors happen when run program on device. */
  kRuntimeDeviceError = 4,
  /*! \brief Answer is wrong when compared to a reference output. */
  kWrongAnswerError = 5,
  /*! \brief Timeout during compilation. */
  kBuildTimeoutError = 6,
  /*! \brief Timeout during run. */
  kRunTimeoutError = 7,
  /*! \brief Unknown error. */
  kUnknownError = 8,
};

/*!
 * \brief Convert MeasureErrorNO to string
 * \param error_no The MeasureErrorNO to be converted
 * \return The string correpsonding to the given MeasureErrorNO
 */
inline const char* MeasureErrorNOToStr(MeasureErrorNO error_no) {
  static const char* names[] = {
      "NoError",
      "InstantiationError",
      "CompileHostError",
      "CompileDeviceError",
      "RuntimeDeviceError",
      "WrongAnswerError",
      "BuildTimeoutError",
      "RunTimeoutError",
      "UnknownError",
  };
  return names[static_cast<int>(error_no)];
}

/********** MeasureInput **********/

// Forward declaration
class MeasureInput;

/*! \brief Store the input of a measurement */
class MeasureInputNode : public Object {
 public:
  /*! \brief The task to be measured */
  SearchTask task;
  /*! \brief Concrete schedule of the task */
  Schedule sch;

  Optional<Array<IntImm>> variant;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("task", &task);
    v->Visit("sch", &sch);
    v->Visit("variant", &variant);
  }

  /*! \brief Do shallow copy. */
  MeasureInput Copy() const;

  static constexpr const char* _type_key = "meta_schedule.MeasureInput";
  TVM_DECLARE_FINAL_OBJECT_INFO(MeasureInputNode, Object);
};

/*!
 * \brief Managed reference to MeasureInputNode.
 * \sa MeasureInputNode
 */
class MeasureInput : public ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param task The task to be measured
   * \param state Concrete schedule of the task
   */
  explicit MeasureInput(SearchTask task, Schedule sch,
                        Optional<Array<IntImm>> variant = NullOpt);

  TVM_DEFINE_OBJECT_REF_METHODS(MeasureInput, ObjectRef, MeasureInputNode);
};

/********** BuildResult **********/

/*! \brief Store the result of a build. */
class BuildResultNode : public Object {
 public:
  /*! \brief The filename of built binary file. */
  String filename;
  /*! \brief The error code. (0 means no error, see MeasureErrorNO) */
  int error_no;
  /*! \brief The error message if there is any error. */
  String error_msg;
  /*! \brief The time cost of build. */
  double time_cost;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("filename", &filename);
    v->Visit("error_no", &error_no);
    v->Visit("error_msg", &error_msg);
    v->Visit("time_cost", &time_cost);
  }

  static constexpr const char* _type_key = "meta_schedule.BuildResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(BuildResultNode, Object);
};

/*!
 * \brief Managed reference to BuildResultNode.
 * \sa BuildResultNode
 */
class BuildResult : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param filename The filename of built binary file.
   * \param error_no The error code.
   * \param error_msg The error message if there is any error.
   * \param time_cost The time cost of build.
   */
  explicit BuildResult(String filename, int error_no, String error_msg, double time_cost);
  TVM_DEFINE_OBJECT_REF_METHODS(BuildResult, ObjectRef, BuildResultNode);
};

/********** MeasureResult **********/

class MeasureResult;

/*! \brief Store the results of a measurement. */
class MeasureResultNode : public Object {
 public:
  /*! \brief The time costs of execution. */
  Array<FloatImm> costs;
  /*! \brief The error code. (0 means no error, see MeasureErrorNO) */
  int error_no;
  /*! \brief The error message if there is any error. */
  String error_msg;
  /*! \brief The time cost of build and run. */
  double all_cost;
  /*! \brief The time stamps of this measurement. */
  double timestamp;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("costs", &costs);
    v->Visit("error_no", &error_no);
    v->Visit("error_msg", &error_msg);
    v->Visit("all_cost", &all_cost);
    v->Visit("timestamp", &timestamp);
  }

  /*! \brief Do shallow copy. */
  MeasureResult Copy() const;

  /*! \brief The average cost */
  double MeanCost() const;

  static constexpr const char* _type_key = "meta_schedule.MeasureResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(MeasureResultNode, Object);
};

/*!
 * \brief Managed reference to MeasureResultNode.
 * \sa MeasureResultNode
 */
class MeasureResult : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param costs The time costs of execution.
   * \param error_no The error code.
   * \param error_msg The error message if there is any error.
   * \param all_cost The time cost of build and run.
   * \param timestamp The time stamps of this measurement.
   */
  explicit MeasureResult(Array<FloatImm> costs, int error_no, String error_msg, double all_cost,
                         double timestamp);

  TVM_DEFINE_OBJECT_REF_METHODS(MeasureResult, ObjectRef, MeasureResultNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_MEASURE_RECORD_H_
