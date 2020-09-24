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

#ifndef SRC_META_SCHEDULE_MEASURE_H_
#define SRC_META_SCHEDULE_MEASURE_H_

#include <string>
#include <unordered_map>
#include <utility>

#include "./schedule.h"
#include "./search_task.h"

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

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("task", &task);
    v->Visit("sch", &sch);
  }

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
   * \brief Constructor
   * \param task The task to be measured
   * \param state Concrete schedule of the task
   */
  explicit MeasureInput(SearchTask task, Schedule sch);

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
  /* TODO(@junrushao1994): use avg_cost instead */
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
  explicit MeasureResult(Array<PrimExpr> costs, int error_no, String error_msg, double all_cost,
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
  /*! \brief Virtual destructor */
  virtual ~ProgramBuilderNode() = default;
  /*!
   * \brief Build programs and return results.
   * \param inputs An Array of MeasureInput.
   * \param verbose Verbosity level. 0 for silent, 1 to output information during program
   * building.
   * \return An Array of MeasureResult.
   */
  virtual Array<BuildResult> Build(const Array<MeasureInput>& inputs, int verbose) const = 0;

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
  /*! \brief Virtual destructor */
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
                                   const Array<BuildResult>& build_results, int verbose) const = 0;

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
  /*! \brief Build function, can be `tar` or `ndk`. */
  String build_func;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("n_parallel", &n_parallel);
    v->Visit("timeout", &timeout);
    v->Visit("build_func", &build_func);
  }

  /*! \brief Default destructor */
  ~LocalBuilderNode() = default;

  Array<BuildResult> Build(const Array<MeasureInput>& inputs, int verbose) const override;

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
   * \param timeout The timeout limit (in second) for each build process.
   * This will be used in a wrapper of the multiprocessing.Process.join().
   * \param n_parallel The number of threads used to build in parallel.
   * \param build_func The name of the registered build function.
   */
  explicit LocalBuilder(int timeout, int n_parallel, String build_func);

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
  /*! \brief The RPC tracker information. */
  String tracker;
  /*! \brief The priority of this run request, larger is more prior. */
  int priority;
  /*! \brief The number of tasks run in parallel. */
  int n_parallel;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("timeout", &timeout);
    v->Visit("number", &number);
    v->Visit("repeat", &repeat);
    v->Visit("min_repeat_ms", &min_repeat_ms);
    v->Visit("cooldown_interval", &cooldown_interval);
    v->Visit("enable_cpu_cache_flush", &enable_cpu_cache_flush);
    v->Visit("tracker", &tracker);
    v->Visit("priority", &priority);
    v->Visit("n_parallel", &n_parallel);
  }

  /*! \brief Default destructor */
  ~RPCRunnerNode() = default;

  Array<MeasureResult> Run(const Array<MeasureInput>& inputs,
                           const Array<BuildResult>& build_results, int verbose) const override;

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
  explicit RPCRunner(String tracker, int priority, int n_parallel, int timeout, int number,
                     int repeat, int min_repeat_ms, double cooldown_interval,
                     bool enable_cpu_cache_flush);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(RPCRunner, ProgramRunner, RPCRunnerNode);
};

/********** MeasureCallback **********/

/*! \brief Bass class of measurement callbacks */
class MeasureCallbackNode : public Object {
 public:
  virtual ~MeasureCallbackNode() = default;
  /*!
   * \brief Callback function that will be called on measurement input/result pairs
   * after each measurement batch.
   * \param inputs An Array of MeasureInput.
   * \param results An Array of MeasureResult.
   */
  virtual void Callback(const Array<MeasureInput>& inputs, const Array<MeasureResult>& results) = 0;
  static constexpr const char* _type_key = "meta_schedule.MeasureCallback";
  TVM_DECLARE_BASE_OBJECT_INFO(MeasureCallbackNode, Object);
};

/*!
 * \brief Managed reference to MeasureCallbackNode.
 * \sa MeasureCallbackNode
 */
class MeasureCallback : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MeasureCallback, ObjectRef, MeasureCallbackNode);
};

/********** ProgramMeasurer **********/

/*!
 * \brief Measurer that measures the time costs of tvm programs
 * This class combines ProgramBuilder and ProgramRunner and provides a simpler API */
class ProgramMeasurerNode : public Object {
 public:
  /*! \brief The ProgramBuilder to build each program. */
  ProgramBuilder builder;
  /*! \brief The ProgramRunner to measure each program. */
  ProgramRunner runner;
  /*! \brief MeasureCallback to be called after each measure batch. */
  Array<MeasureCallback> callbacks;
  /*! \brief Number of samples that have been measured. */
  int num_measured;
  /*! \brief The best running time (the smaller the better). */
  double best_time_cost;
  /*! \brief The index of the samples that the best schedule is in. */
  int best_index;
  /*! \brief The best schedule found so far. */
  Optional<Schedule> best_sch;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("builder", &builder);
    v->Visit("runner", &runner);
    v->Visit("callbacks", &callbacks);
    v->Visit("num_measured", &num_measured);
    v->Visit("best_time_cost", &best_time_cost);
    v->Visit("best_index", &best_index);
    v->Visit("best_sch", &best_sch);
  }

  /*!
   * \brief Do measurement and update the records maintained internally
   * \param measure_inputs The programs to be measured
   * \param batch_size Number of programs to be measured in one batch
   * \param verbose Flag for verbose mode
   * \return The measured result
   */
  Array<MeasureResult> BatchMeasure(const Array<MeasureInput>& measure_inputs, int batch_size,
                                    int verbose);
  /*! \brief Reset book keeping variables */
  void Reset();

  static constexpr const char* _type_key = "meta_schedule.ProgramMeasurer";
  TVM_DECLARE_FINAL_OBJECT_INFO(ProgramMeasurerNode, Object);

 private:
  /*!
   * \brief Measure the inputs without modifying the internal states
   * \param measure_inputs The inputs to be measured
   * \param verbose Flag for verbose mode
   * \return The measured result
   */
  Array<MeasureResult> PureMeasure(const Array<MeasureInput>& measure_inputs, int verbose) const;
};

/*!
 * \brief Managed reference to ProgramMeasurerNode.
 * \sa ProgramMeasurerNode
 */
class ProgramMeasurer : public ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param builder The program builder
   * \param runner The program runner
   * \param callbacks The callbacks invoked after measurement
   * \param num_measure The default number of measurement that have been conducted
   * \param best_time_cost The current best time for a single measurement
   * \param best_index The index of the samples that the best schedule is in.
   * \param best_sch The best schedule found so far
   */
  explicit ProgramMeasurer(ProgramBuilder builder, ProgramRunner runner,
                           Array<MeasureCallback> callbacks, int num_measured,
                           double best_time_cost, int best_index, Optional<Schedule> best_sch);
  /*!
   * \brief Simplified constructor
   * \param builder The program builder
   * \param runner The program runner
   * \param callbacks The callbacks invoked after measurement
   */
  explicit ProgramMeasurer(ProgramBuilder builder, ProgramRunner runner,
                           Array<MeasureCallback> callbacks);

  /*! \brief The maximum time cost*/
  static const constexpr double MAX_TIME_COST = 1e10;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ProgramMeasurer, ObjectRef, ProgramMeasurerNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_MEASURE_H_
