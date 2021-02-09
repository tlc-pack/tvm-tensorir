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
#include "./measure.h"  // NOLINT(build/include)

#include <tvm/node/serialization.h>

#include <algorithm>

#include "./utils.h"

namespace tvm {
namespace meta_schedule {

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

  Array<BuildResult> Build(const Array<MeasureInput>& inputs, int verbose) const override {
    if (const auto* f = runtime::Registry::Get("meta_schedule.local_builder.build")) {
      Array<BuildResult> results = (*f)(inputs, timeout, n_parallel, build_func, verbose);
      return results;
    }
    LOG(FATAL) << "meta_schedule.local_builder.build is not registered. "
               << "This is a function registered in Python, "
               << "make sure the TVM Python runtime has been loaded successfully.";
    throw;
  }

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
  explicit LocalBuilder(int timeout, int n_parallel, String build_func) {
    if (build_func != "tar" && build_func != "ndk") {
      LOG(FATAL) << "ValueError: Unknown build_func in LocalBuilder: " << build_func;
    }
    ObjectPtr<LocalBuilderNode> n = make_object<LocalBuilderNode>();
    n->timeout = timeout;
    n->n_parallel = n_parallel;
    n->build_func = std::move(build_func);
    data_ = std::move(n);
  }

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(LocalBuilder, ProgramBuilder, LocalBuilderNode);
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
                           const Array<BuildResult>& build_results, int verbose) const override {
    if (const auto* f = runtime::Registry::Get("meta_schedule.rpc_runner.run")) {
      Array<MeasureResult> results = (*f)(
          inputs, build_results, key, host, port, priority, n_parallel, timeout, number, repeat,
          min_repeat_ms, cooldown_interval, enable_cpu_cache_flush, f_create_args, verbose);
      return results;
    }
    LOG(FATAL) << "meta_schedule.rpc_runner.run is not registered. "
               << "This is a function registered in Python, "
               << "make sure the TVM Python runtime has been loaded successfully.";
    throw;
  }

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
   * \param f_create_args Callback function to create arguments.
   */
  explicit RPCRunner(String key, String host, int port, int priority, int n_parallel, int timeout,
                     int number, int repeat, int min_repeat_ms, double cooldown_interval,
                     bool enable_cpu_cache_flush, FCreateArgs f_create_args) {
    ObjectPtr<RPCRunnerNode> n = make_object<RPCRunnerNode>();
    n->key = std::move(key);
    n->host = std::move(host);
    n->port = port;
    n->priority = priority;
    n->timeout = timeout;
    n->n_parallel = n_parallel;
    n->number = number;
    n->repeat = repeat;
    n->min_repeat_ms = min_repeat_ms;
    n->cooldown_interval = cooldown_interval;
    n->enable_cpu_cache_flush = enable_cpu_cache_flush;
    n->f_create_args = f_create_args;
    data_ = std::move(n);
  }

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(RPCRunner, ProgramRunner, RPCRunnerNode);
};

/********** ProgramMeasurer **********/

ProgramMeasurer::ProgramMeasurer(ProgramBuilder builder, ProgramRunner runner,
                                 Array<MeasureCallback> callbacks, Database db, int num_measures) {
  ObjectPtr<ProgramMeasurerNode> n = make_object<ProgramMeasurerNode>();
  n->builder = std::move(builder);
  n->runner = std::move(runner);
  n->callbacks = std::move(callbacks);
  n->db = std::move(db);
  n->num_measures = num_measures;
  data_ = std::move(n);
}

ProgramMeasurer::ProgramMeasurer(ProgramBuilder builder, ProgramRunner runner,
                                 Array<MeasureCallback> callbacks)
    : ProgramMeasurer(builder, runner, callbacks, InMemoryDB(NullOpt), /*num_measures=*/0) {}

void ProgramMeasurerNode::Init(const SearchTask& task) {
  for (const MeasureCallback& callback : callbacks) {
    callback->Init(task);
  }
  this->db = InMemoryDB(task->log_file);
  this->db->Init(task);
}

Optional<Schedule> ProgramMeasurerNode::GetBest(const SearchTask& task) const {
  Optional<Trace> trace = db->GetBest().trace;
  if (!trace.defined()) {
    return NullOpt;
  }
  Schedule sch(task->workload);
  trace.value()->Apply(sch);
  return sch;
}

Array<MeasureResult> ProgramMeasurerNode::PureMeasure(const Array<MeasureInput>& measure_inputs,
                                                      int verbose) const {
  Array<BuildResult> build_results = builder->Build(measure_inputs, verbose);
  Array<MeasureResult> measure_results = runner->Run(measure_inputs, build_results, verbose);
  return measure_results;
}

Array<MeasureResult> ProgramMeasurerNode::BatchMeasure(const Array<MeasureInput>& measure_inputs,
                                                       int batch_size, int verbose) {
  Array<MeasureResult> measure_results;
  int n_samples = measure_inputs.size();
  for (int st = 0; st < n_samples; st += batch_size) {
    int ed = std::min(st + batch_size, n_samples);
    Array<MeasureInput> batch_measure_inputs(measure_inputs.begin() + st,
                                             measure_inputs.begin() + ed);
    Array<MeasureResult> batch_measure_results = this->PureMeasure(batch_measure_inputs, verbose);
    for (int i = 0; i < ed - st; ++i) {
      const MeasureInput& measure_input = batch_measure_inputs[i];
      const MeasureResult& measure_result = batch_measure_results[i];
      double flop_ct = measure_input->task->flop_ct;
      const String& task_name = measure_input->task->task_name;
      MeasureErrorNO error_no = static_cast<MeasureErrorNO>(measure_result->error_no);
      ++num_measures;
      if (error_no == MeasureErrorNO::kNoError) {
        double avg_time_cost = FloatArrayMean(measure_result->costs);
        db->Add(measure_input->sch->trace, Repr(measure_input->sch),
                AsVector<FloatImm, double>(measure_result->costs));
        double best_time_cost = db->GetBest().MeanTime();
        StdCout(verbose) << std::fixed << std::setprecision(4)  //
                         << '[' << task_name << "] #" << num_measures
                         << "\tTime: " << (avg_time_cost * 1000) << " ms, "
                         << (flop_ct / avg_time_cost / 1e9) << " GFLOPs"
                         << "\tBest time: " << (best_time_cost * 1000) << " ms, "
                         << (flop_ct / best_time_cost / 1e9) << " GFLOPs" << std::endl;
      } else if (error_no == MeasureErrorNO::kRunTimeoutError ||
                 error_no == MeasureErrorNO::kBuildTimeoutError) {
        double best_time_cost = db->GetBest().MeanTime();
        StdCout(verbose) << std::fixed << std::setprecision(4)  //
                         << '[' << task_name << "] #" << num_measures
                         << "\tError: " << MeasureErrorNOToStr(error_no)
                         << "\tBest time: " << (best_time_cost * 1000) << " ms, "
                         << (flop_ct / best_time_cost / 1e9) << " GFLOPs" << std::endl;
      } else {
        double best_time_cost = db->GetBest().MeanTime();
        StdCout(verbose) << std::fixed << std::setprecision(4)  //
                         << '[' << task_name << "] #" << num_measures
                         << "\tError: " << MeasureErrorNOToStr(error_no)
                         << "\tBest time: " << (best_time_cost * 1000) << " ms, "
                         << (flop_ct / best_time_cost / 1e9) << " GFLOPs" << std::endl
                         << measure_result->error_msg << "\n"
                         << "The IR is:\n"
                         << Repr(measure_input->sch) << "\nSchedule is:\n"
                         << measure_input->sch->trace->Stringify();
      }
    }
    measure_results.insert(measure_results.end(), batch_measure_results.begin(),
                           batch_measure_results.end());
    for (const MeasureCallback& callback : this->callbacks) {
      callback->Callback(measure_inputs, measure_results);
    }
  }
  return measure_results;
}

/********** FFI **********/

struct Internal {
  /********** Constructors **********/
  /*!
   * \brief The constructor.
   * \param timeout The timeout limit (in second) for each build process.
   * \param n_parallel The number of threads used to build in parallel.
   * \param build_func The name of the registered build function.
   * \return The LocalBuilder constructed
   * \sa LocalBuilder::LocalBuilder
   */
  static LocalBuilder LocalBuilderNew(int timeout, int n_parallel, String build_func) {
    return LocalBuilder(timeout, n_parallel, build_func);
  }
  /*!
   * \brief Constructor of RPCRunner
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
   * \return The RPCRunner constructed
   * \sa RPCRunnerNew::RPCRunnerNew
   */
  static RPCRunner RPCRunnerNew(String key, String host, int port, int priority, int n_parallel,
                                int timeout, int number, int repeat, int min_repeat_ms,
                                double cooldown_interval, bool enable_cpu_cache_flush,
                                ProgramRunner::FCreateArgs f_create_args) {
    return RPCRunner(key, host, port, priority, n_parallel, timeout, number, repeat, min_repeat_ms,
                     cooldown_interval, enable_cpu_cache_flush, f_create_args);
  }
  /*!
   * \brief Constructor of ProgramMeasurerNew
   * \param builder The program builder
   * \param runner The program runner
   * \param callbacks The callbacks invoked after measurement
   * \return The ProgramMeasurer constructed
   * \sa ProgramMeasurer::ProgramMeasurer
   */
  static ProgramMeasurer ProgramMeasurerNew(ProgramBuilder builder, ProgramRunner runner,
                                            Array<MeasureCallback> callbacks) {
    return ProgramMeasurer(builder, runner, callbacks);
  }
  /********** Member methods **********/
  /*!
   * \brief Invoke ProgramBuilder::Build
   * \param builder The program builder
   * \param inputs An Array of MeasureInput
   * \param verbose Verbosity level. 0 for silent, 1 to output information during program building
   * \return An Array of MeasureResult
   * \sa ProgramBuilder::Build
   */
  static Array<BuildResult> ProgramBuilderBuild(ProgramBuilder builder, Array<MeasureInput> inputs,
                                                int verbose) {
    return builder->Build(inputs, verbose);
  }
  /*!
   * \brief Invoke ProgramRunner::Run
   * \param runner The program runner
   * \param inputs An Array of MeasureInput
   * \param build_results An Array of BuildResult
   * \param verbose Verbosity level. 0 for silent, 1 to output information during program running
   * \return An Array of MeasureResult
   * \sa ProgramRunner::Run
   */
  static Array<MeasureResult> ProgramRunnerRun(ProgramRunner runner, Array<MeasureInput> inputs,
                                               Array<BuildResult> build_results, int verbose) {
    return runner->Run(inputs, build_results, verbose);
  }
};

TVM_REGISTER_OBJECT_TYPE(ProgramRunnerNode);
TVM_REGISTER_OBJECT_TYPE(ProgramBuilderNode);
TVM_REGISTER_NODE_TYPE(LocalBuilderNode);
TVM_REGISTER_NODE_TYPE(RPCRunnerNode);
TVM_REGISTER_OBJECT_TYPE(MeasureCallbackNode);
TVM_REGISTER_NODE_TYPE(ProgramMeasurerNode);

TVM_REGISTER_GLOBAL("meta_schedule.LocalBuilder").set_body_typed(Internal::LocalBuilderNew);
TVM_REGISTER_GLOBAL("meta_schedule.RPCRunner").set_body_typed(Internal::RPCRunnerNew);
TVM_REGISTER_GLOBAL("meta_schedule.ProgramMeasurer").set_body_typed(Internal::ProgramMeasurerNew);
TVM_REGISTER_GLOBAL("meta_schedule.ProgramBuilderBuild")
    .set_body_typed(Internal::ProgramBuilderBuild);
TVM_REGISTER_GLOBAL("meta_schedule.ProgramRunnerRun").set_body_typed(Internal::ProgramRunnerRun);

}  // namespace meta_schedule
}  // namespace tvm
