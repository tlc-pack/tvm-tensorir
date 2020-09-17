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
 * Taken and modified from Ansor
 * The core functions is implemented in python to utilize python's multiprocessing
 * and error handling (see also `python/tvm/meta_schedule/measure.py`).
 * This c++ file is just a wrapper for the python functions.
 */

#ifndef SRC_META_SCHEDULE_MEASURE_H_
#define SRC_META_SCHEDULE_MEASURE_H_

#include <string>
#include <unordered_map>
#include <utility>

#include "./schedule.h"

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

/********** MeasureInput **********/

// Inputs and results of one measurement

class MeasureInput;

/*! \brief Store the input of a measurement */
class MeasureInputNode : public Object {
 public:
  /*! \brief The search task. */
  Schedule sch{nullptr};

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("sch", &sch); }

  /*! \brief Do shallow copy. */
  MeasureInput copy() const;

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
   * \brief The constructor.
   * \param task The SearchTask of this measure.
   * \param state The State to be measured.
   */
  explicit MeasureInput(Schedule sch);

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
  BuildResult(String filename, int error_no, String error_msg, double time_cost);
  TVM_DEFINE_OBJECT_REF_METHODS(BuildResult, ObjectRef, BuildResultNode);
};

/********** MeasureResult **********/

class MeasureResult;

/*! \brief Store the results of a measurement. */
class MeasureResultNode : public Object {
 public:
  /*! \brief The time costs of execution. */
  Array<PrimExpr> costs;
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
  MeasureResult copy() const;

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
  MeasureResult(Array<PrimExpr> costs, int error_no, String error_msg, double all_cost,
                double timestamp);

  TVM_DEFINE_OBJECT_REF_METHODS(MeasureResult, ObjectRef, MeasureResultNode);
};

/********** ProgramBuilder **********/

/*! \brief ProgramBuilder that builds the programs */
class ProgramBuilderNode : public Object {
 public:
  /*! \brief The number of build processes to run in parallel */
  int n_parallel;
  /*! \brief Timeout of a build */
  int timeout;

  virtual ~ProgramBuilderNode() = default;

  /*!
   * \brief Build programs and return results.
   * \param inputs An Array of MeasureInput.
   * \param verbose Verbosity level. 0 for silent, 1 to output information during program
   * building.
   * \return An Array of MeasureResult.
   */
  virtual Array<BuildResult> Build(const Array<MeasureInput>& inputs, int verbose) = 0;

  static constexpr const char* _type_key = "meta_schedule.ProgramBuilder";
  TVM_DECLARE_BASE_OBJECT_INFO(ProgramBuilderNode, Object);
};

/*!
 * \brief Managed reference to ProgramBuilderNode.
 * \sa ProgramBuilderNode
 */
class ProgramBuilder : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ProgramBuilder, ObjectRef, ProgramBuilderNode);
};

/********** ProgramRunner **********/

/*! \brief ProgramRunner that runs the built programs and measure the time cost. */
class ProgramRunnerNode : public Object {
 public:
  /*! \brief Timeout of a run. */
  int timeout;
  /*! \brief The number of times to run the generated code for taking average. */
  int number;
  /*! \brief The number of times to repeat the measurement. */
  int repeat;
  /*! \brief The minimum duration of one repeat in milliseconds. */
  int min_repeat_ms;
  /*! \brief The cool down interval between two measurements. */
  double cooldown_interval;
  /*! \brief Whether to flush cache on CPU between repeated measurements. */
  bool enable_cpu_cache_flush;

  virtual ~ProgramRunnerNode() = default;

  /*!
   * \brief Run measurement and return results.
   * \param inputs An Array of MeasureInput.
   * \param build_results An Array of BuildResult.
   * \param verbose Verbosity level. 0 for silent, 1 to output information during program
   * running.
   * \return An Array of MeasureResult.
   */
  virtual Array<MeasureResult> Run(const Array<MeasureInput>& inputs,
                                   const Array<BuildResult>& build_results, int verbose) = 0;

  static constexpr const char* _type_key = "meta_schedule.ProgramRunner";
  TVM_DECLARE_BASE_OBJECT_INFO(ProgramRunnerNode, Object);
};

/*!
 * \brief Managed reference to ProgramRunnerNode.
 * \sa ProgramRunnerNode
 */
class ProgramRunner : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ProgramRunner, ObjectRef, ProgramRunnerNode);
};

/********** LocalBuilder: ProgramBuilder **********/

/*! \brief LocalBuilder use local CPU cores to build programs in parallel */
class LocalBuilderNode : public ProgramBuilderNode {
 public:
  /*! \brief Build function. */
  String build_func;

  ~LocalBuilderNode() = default;

  Array<BuildResult> Build(const Array<MeasureInput>& inputs, int verbose) override;

  static constexpr const char* _type_key = "meta_schedule.LocalBuilder";
  TVM_DECLARE_FINAL_OBJECT_INFO(LocalBuilderNode, ProgramBuilderNode);
};

/*!
 * \brief Managed reference to LocalBuilderNode.
 * \sa LocalBuilderNode
 */
class LocalBuilder : public ProgramBuilder {
 public:
  /*!
   * \brief The constructor.
   * \param timeout The timeout limit (in second) for each build thread.
   * This will be used in a wrapper of the multiprocessing.Process.join().
   * \param n_parallel The number of threads used to build in parallel.
   * \param build_func The name of the registered build function.
   */
  LocalBuilder(int timeout, int n_parallel, const String& build_func);

  TVM_DEFINE_OBJECT_REF_METHODS(LocalBuilder, ProgramBuilder, LocalBuilderNode);
};

/********** RPCRunner: ProgramRunner **********/

/*!
 * \brief RPCRunner that uses RPC call to measures the time cost of programs on remote devices.
 * Or sometime we may need to use RPC even in local running to insulate the thread environment.
 * (e.g. running CUDA programs)
 */
class RPCRunnerNode : public ProgramRunnerNode {
 public:
  /*! \brief The key of the device registered in the RPC tracker. */
  String key;
  /*! \brief The host address of the RPC Tracker. */
  String host;
  /*! \brief The port of the RPC Tracker. */
  int port;
  /*! \brief The priority of this run request, larger is more prior. */
  int priority;
  /*! \brief The number of tasks run in parallel. */
  int n_parallel;

  ~RPCRunnerNode() = default;

  Array<MeasureResult> Run(const Array<MeasureInput>& inputs,
                           const Array<BuildResult>& build_results, int verbose) override;

  static constexpr const char* _type_key = "meta_schedule.RPCRunner";
  TVM_DECLARE_FINAL_OBJECT_INFO(RPCRunnerNode, ProgramRunnerNode);
};

/*!
 * \brief Managed reference to RPCRunnerNode.
 * \sa RPCRunnerNode
 */
class RPCRunner : public ProgramRunner {
 public:
  /*!
   * \brief The constructor. See the corresponding class in python/tvm/meta_schedule/measure.py
   * for more detailed parameter explanation.
   * \param key The key of the device registered in the RPC tracker.
   * \param host The host address of the RPC Tracker.
   * \param port The port of RPC Tracker.
   * \param priority The priority of this run request, larger is more prior.
   * \param n_parallel The number of tasks run in parallel.
   * \param timeout Timeout of a run.
   * \param number The number of times to run the generated code for taking average.
   * \param repeat The number of times to repeat the measurement.
   * \param min_repeat_ms The minimum duration of one repeat in milliseconds.
   * \param cooldown_interval The cool down interval between two measurements.
   * \param enable_cpu_cache_flush Whether to flush cache on CPU between repeated measurements.
   */
  RPCRunner(const String& key, const String& host, int port, int priority, int n_parallel,
            int timeout, int number, int repeat, int min_repeat_ms, double cooldown_interval,
            bool enable_cpu_cache_flush);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(RPCRunner, ProgramRunner, RPCRunnerNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_MEASURE_H_
