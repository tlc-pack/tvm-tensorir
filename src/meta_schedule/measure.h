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

#include <tvm/runtime/ndarray.h>

#include "./database.h"
#include "./measure_record.h"
#include "./schedule.h"

namespace tvm {
namespace meta_schedule {

class TuneContextNode;

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
  /*! \brief Initialize the program builder */
  virtual void Init(TuneContextNode* tune_context) {}
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
  /*!
   * \brief Callback function to create arguments for functions to measure. This can be used for
   * sparse workloads when we cannot use random tensors for measurement.
   */
  using FCreateArgs = runtime::TypedPackedFunc<Array<runtime::NDArray>(TVMContext)>;

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
  /*! \brief Callback functions to create arguments */
  FCreateArgs f_create_args;
  /*! \brief Virtual destructor */
  virtual ~ProgramRunnerNode() = default;
  /*! \brief Initialize the program runner */
  virtual void Init(TuneContextNode* tune_context) {}
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
  using FCreateArgs = ProgramRunnerNode::FCreateArgs;
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ProgramRunner, ObjectRef, ProgramRunnerNode);
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
  /*! \brief Initialize the callback */
  virtual void Init(TuneContextNode* tune_context) {}
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
