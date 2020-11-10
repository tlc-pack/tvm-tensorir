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
#include <dmlc/memory_io.h>  // NOLINT(build/include)
#include <tvm/node/serialization.h>
#include <algorithm>

#include "../support/base64.h"
#include "./measure.h"
#include "./utils.h"

namespace tvm {
namespace meta_schedule {

/********** Constructors **********/

LocalBuilder::LocalBuilder(int timeout, int n_parallel, String build_func) {
  if (build_func != "tar" && build_func != "ndk") {
    LOG(FATAL) << "ValueError: Unknown build_func in LocalBuilder: " << build_func;
  }
  ObjectPtr<LocalBuilderNode> n = make_object<LocalBuilderNode>();
  n->timeout = timeout;
  n->n_parallel = n_parallel;
  n->build_func = std::move(build_func);
  data_ = std::move(n);
}

RPCRunner::RPCRunner(String key, String host, int port, int priority, int n_parallel, int timeout,
                     int number, int repeat, int min_repeat_ms, double cooldown_interval,
                     bool enable_cpu_cache_flush) {
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
  data_ = std::move(n);
}

ProgramMeasurer::ProgramMeasurer(ProgramBuilder builder, ProgramRunner runner,
                                 Array<MeasureCallback> callbacks, int num_measured,
                                 double best_time_cost, int best_index,
                                 Optional<Schedule> best_sch) {
  ObjectPtr<ProgramMeasurerNode> n = make_object<ProgramMeasurerNode>();
  n->builder = std::move(builder);
  n->runner = std::move(runner);
  n->callbacks = std::move(callbacks);
  n->num_measured = num_measured;
  n->best_time_cost = best_time_cost;
  n->best_index = best_index;
  n->best_sch = std::move(best_sch);
  data_ = std::move(n);
}

ProgramMeasurer::ProgramMeasurer(ProgramBuilder builder, ProgramRunner runner,
                                 Array<MeasureCallback> callbacks)
    : ProgramMeasurer(builder, runner, callbacks, /*num_measured=*/0,
                      /*best_time_cost=*/kMaxTimeCost, /*best_index=*/-1, /*best_sch=*/NullOpt) {}

/********** LocalBuilder **********/

Array<BuildResult> LocalBuilderNode::Build(const Array<MeasureInput>& inputs, int verbose) const {
  if (const auto* f = runtime::Registry::Get("meta_schedule.local_builder.build")) {
    Array<BuildResult> results = (*f)(inputs, timeout, n_parallel, build_func, verbose);
    return results;
  }
  LOG(FATAL) << "meta_schedule.local_builder.build is not registered. "
             << "This is a function registered in Python, "
             << "make sure the TVM Python runtime has been loaded successfully.";
  throw;
}

/********** RPCRunner **********/

Array<MeasureResult> RPCRunnerNode::Run(const Array<MeasureInput>& inputs,
                                        const Array<BuildResult>& build_results,
                                        int verbose) const {
  if (const auto* f = runtime::Registry::Get("meta_schedule.rpc_runner.run")) {
    Array<MeasureResult> results =
        (*f)(inputs, build_results, key, host, port, priority, n_parallel, timeout, number, repeat,
             min_repeat_ms, cooldown_interval, enable_cpu_cache_flush, verbose);
    return results;
  }
  LOG(FATAL) << "meta_schedule.rpc_runner.run is not registered. "
             << "This is a function registered in Python, "
             << "make sure the TVM Python runtime has been loaded successfully.";
  throw;
}

/********** ProgramMeasurer **********/

void ProgramMeasurerNode::Init(const SearchTask& task) {
  num_measured = 0;
  best_time_cost = ProgramMeasurer::kMaxTimeCost;
  best_index = -1;
  best_sch = NullOpt;
  for (const MeasureCallback& callback : callbacks) {
    callback->Init(task);
  }
  // Loading existing logs from file
  if (!task->filename.defined()) {
    return;
  }
  static const auto* f_deserialize = runtime::Registry::Get("meta_schedule._deserialize_json");
  CHECK(f_deserialize) << "IndexError: Cannot find packed function \""
                          "meta_schedule._deserialize_json\", which should be registered in python";
  std::ifstream ifs(task->filename.value());
  if (!ifs.is_open() || ifs.fail()) {
    return;
  }
  std::vector<Array<ObjectRef>> records;
  for (std::string line; std::getline(ifs, line);) {
    if (line.empty() || line[0] == '#' || line[0] == '/' || std::isspace(line[0])) {
      continue;
    }
    Array<ObjectRef> record = (*f_deserialize)(line);
    records.push_back(record);
  }
  LOG(INFO) << "Found " << records.size() << " record(s) in the file: " << task->filename.value()
            << ". Now loading...";
  for (const Array<ObjectRef>& record : records) {
    CHECK_EQ(record.size(), 7);
    String task_name = Downcast<String>(record[0]);
    if (task_name != task->task_name) {
      continue;
    }
    String log_version = Downcast<String>(record[5]);
    if (log_version != String(kLogVersion)) {
      continue;
    }
    // TODO(@junrushao1994): structural equality of target
    Map<String, ObjectRef> target = Downcast<Map<String, ObjectRef>>(record[1]);
    if (Target(target)->str() != task->target->str()) {
      continue;
    }
    Map<String, ObjectRef> target_host = Downcast<Map<String, ObjectRef>>(record[2]);
    if (Target(target_host)->str() != task->target_host->str()) {
      continue;
    }
    tir::PrimFunc orig_func{nullptr};
    {
      std::string prim_func_b64 = Downcast<String>(record[6]);
      dmlc::MemoryStringStream mstrm(&prim_func_b64);
      support::Base64InStream b64strm(&mstrm);
      std::string parsed;
      b64strm.InitPosition();
      dmlc::Stream* strm = &b64strm;
      strm->Read(&parsed);
      orig_func = Downcast<tir::PrimFunc>(LoadJSON(parsed));
    }
    Array<ObjectRef> trace = Downcast<Array<ObjectRef>>(record[4]);
    Schedule sch = ScheduleNode::Import(trace, orig_func, /*seed=*/NullOpt);
    Array<FloatImm> costs = Downcast<Array<FloatImm>>(record[3]);
    double avg_time_cost = FloatArrayMean(costs);
    ++num_measured;
    if (avg_time_cost < best_time_cost) {
      best_time_cost = avg_time_cost;
      best_index = num_measured;
      best_sch = sch;
    }
  }
  LOG(INFO) << "Loaded " << num_measured
            << " valid records from the file: " << task->filename.value()
            << ". Best time cost: " << best_time_cost;
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
      ++num_measured;
      const MeasureInput& measure_input = batch_measure_inputs[i];
      const MeasureResult& measure_result = batch_measure_results[i];
      MeasureErrorNO error_no = static_cast<MeasureErrorNO>(measure_result->error_no);
      if (error_no == MeasureErrorNO::kNoError) {
        double avg_time_cost = FloatArrayMean(measure_result->costs);
        if (avg_time_cost < best_time_cost) {
          best_time_cost = avg_time_cost;
          best_index = num_measured;
          best_sch = measure_input->sch;
        }
        StdCout(verbose) << std::fixed << std::setprecision(4) << "#" << num_measured
                         << "\tTime: " << avg_time_cost << "\tBest time: " << best_time_cost
                         << std::endl;
      } else if (error_no == MeasureErrorNO::kRunTimeoutError ||
                 error_no == MeasureErrorNO::kBuildTimeoutError) {
        StdCout(verbose) << std::fixed << std::setprecision(4) << "#" << num_measured
                         << "\tError: " << MeasureErrorNOToStr(error_no)
                         << "\tBest time: " << best_time_cost << std::endl
                         << measure_result->error_msg << "\n";
      } else {
        StdCout(verbose) << std::fixed << std::setprecision(4) << "#" << num_measured
                         << "\tError: " << MeasureErrorNOToStr(error_no)
                         << "\tBest time: " << best_time_cost << std::endl
                         << measure_result->error_msg << "\n"
                         << "The IR is:\n"
                         << Repr(measure_input->sch);
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
                                double cooldown_interval, bool enable_cpu_cache_flush) {
    return RPCRunner(key, host, port, priority, n_parallel, timeout, number, repeat, min_repeat_ms,
                     cooldown_interval, enable_cpu_cache_flush);
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
