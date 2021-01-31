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

#include "./database.h"
#include "./measure_record.h"
#include "./schedule.h"

namespace tvm {
namespace meta_schedule {

static constexpr const char* kLogVersion = "v0.0.1";

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
  /*! \brief The key of the device registered in the RPC tracker. */
  String key;
  /*! \brief The host address of the RPC tracker. */
  String host;
  /*! \brief The port of RPC tracker */
  int port;
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
    v->Visit("key", &key);
    v->Visit("host", &host);
    v->Visit("port", &port);
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
  explicit RPCRunner(String key, String host, int port, int priority, int n_parallel, int timeout,
                     int number, int repeat, int min_repeat_ms, double cooldown_interval,
                     bool enable_cpu_cache_flush);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(RPCRunner, ProgramRunner, RPCRunnerNode);
};

/********** MeasureCallback **********/

/*! \brief Bass class of measurement callbacks */
class MeasureCallbackNode : public Object {
 public:
  virtual ~MeasureCallbackNode() = default;
  /*!
   * \brief Initialize the callback
   * \param task The search task
   */
  virtual void Init(const SearchTask& task) = 0;
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
  /*! \brief The database of measured programs. */
  Database db;
  /*! \brief Number of measurement done so far */
  int num_measures;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("builder", &builder);
    v->Visit("runner", &runner);
    v->Visit("callbacks", &callbacks);
    v->Visit("db", &db);
    v->Visit("num_measures", &num_measures);
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
  /*!
   * \brief Initialize the measurer
   * \param task The search task
   */
  void Init(const SearchTask& task);

  /*!
   * \brief Initialize the measurer
   * \param task The search task
   * \return The best schedule so far
   */
  Optional<Schedule> GetBest(const SearchTask& task) const;

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
   * \param db The database of measured programs.
   * \param num_measures Number of measurement done so far
   */
  explicit ProgramMeasurer(ProgramBuilder builder, ProgramRunner runner,
                           Array<MeasureCallback> callbacks, Database db, int num_measures);
  /*!
   * \brief Simplified constructor
   * \param builder The program builder
   * \param runner The program runner
   * \param callbacks The callbacks invoked after measurement
   */
  explicit ProgramMeasurer(ProgramBuilder builder, ProgramRunner runner,
                           Array<MeasureCallback> callbacks);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ProgramMeasurer, ObjectRef, ProgramMeasurerNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_MEASURE_H_
