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

#include <tvm/node/node.h>

#include <algorithm>

#include "./utils.h"

namespace tvm {
namespace meta_schedule {

/********** Constructors **********/

MeasureInput::MeasureInput(SearchTask task, Schedule sch) {
  ObjectPtr<MeasureInputNode> n = make_object<MeasureInputNode>();
  n->task = std::move(task);
  n->sch = std::move(sch);
  data_ = std::move(n);
}

BuildResult::BuildResult(String filename, int error_no, String error_msg, double time_cost) {
  ObjectPtr<BuildResultNode> n = make_object<BuildResultNode>();
  n->filename = std::move(filename);
  n->error_no = error_no;
  n->error_msg = std::move(error_msg);
  n->time_cost = time_cost;
  data_ = std::move(n);
}

MeasureResult::MeasureResult(Array<PrimExpr> costs, int error_no, String error_msg, double all_cost,
                             double timestamp) {
  ObjectPtr<MeasureResultNode> n = make_object<MeasureResultNode>();
  n->costs = std::move(costs);
  n->error_no = error_no;
  n->error_msg = std::move(error_msg);
  n->all_cost = all_cost;
  n->timestamp = timestamp;
  data_ = std::move(n);
}

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

RPCRunner::RPCRunner(String tracker, int priority, int n_parallel, int timeout, int number,
                     int repeat, int min_repeat_ms, double cooldown_interval,
                     bool enable_cpu_cache_flush) {
  ObjectPtr<RPCRunnerNode> n = make_object<RPCRunnerNode>();
  n->tracker = std::move(tracker);
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
                      /*best_time_cost=*/MAX_TIME_COST, /*best_index=*/-1, /*best_sch=*/NullOpt) {}

/********** Shallow copy functions **********/

MeasureInput MeasureInputNode::copy() const {
  ObjectPtr<MeasureInputNode> n = make_object<MeasureInputNode>();
  n->sch = sch;
  return MeasureInput(n);
}

MeasureResult MeasureResultNode::copy() const {
  ObjectPtr<MeasureResultNode> n = make_object<MeasureResultNode>();
  n->costs = costs;
  n->error_no = error_no;
  n->error_msg = error_msg;
  n->all_cost = all_cost;
  n->timestamp = timestamp;
  return MeasureResult(n);
}

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
        (*f)(inputs, build_results, tracker, priority, n_parallel, timeout, number, repeat,
             min_repeat_ms, cooldown_interval, enable_cpu_cache_flush, verbose);
    return results;
  }
  LOG(FATAL) << "meta_schedule.rpc_runner.run is not registered. "
             << "This is a function registered in Python, "
             << "make sure the TVM Python runtime has been loaded successfully.";
  throw;
}

/********** ProgramMeasurer **********/

void ProgramMeasurerNode::Reset() {
  num_measured = 0;
  best_time_cost = ProgramMeasurer::MAX_TIME_COST;
  best_index = -1;
  best_sch = NullOpt;
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
        StdCout(verbose) << std::fixed << std::setprecision(2) << "#" << num_measured
                         << "\tTime: " << avg_time_cost << "\tBest time: " << best_time_cost
                         << std::endl;
      } else {
        StdCout(verbose) << std::fixed << std::setprecision(2) << "#" << num_measured
                         << "\tError: " << MeasureErrorNOToStr(error_no)
                         << "\tBest time: " << best_time_cost << std::endl;
      }
    }
    measure_results.insert(measure_results.end(), batch_measure_results.begin(),
                           batch_measure_results.end());
  }
  return measure_results;
}

/********** Printing functions **********/

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MeasureResultNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const MeasureResultNode*>(ref.get());
      if (node->error_no == static_cast<int>(MeasureErrorNO::kNoError)) {
        p->stream << "MeasureResult(cost:[";
        auto old_config = p->stream.precision(4);
        for (size_t i = 0; i < node->costs.size(); ++i) {
          auto pf = node->costs[i].as<FloatImmNode>();
          CHECK(pf != nullptr);
          p->stream << pf->value;
          if (i != node->costs.size() - 1) {
            p->stream << ",";
          }
        }
        p->stream.precision(old_config);
        p->stream << "], ";
        p->stream << "error_no:" << 0 << ", "
                  << "all_cost:" << node->all_cost << ", "
                  << "Tstamp:" << node->timestamp << ")";
      } else {
        p->stream << "MeasureResult("
                  << "error_type:"
                  << MeasureErrorNOToStr(static_cast<MeasureErrorNO>(node->error_no)) << ", "
                  << "error_msg:" << node->error_msg << ", "
                  << "all_cost:" << node->all_cost << ", "
                  << "Tstamp:" << node->timestamp << ")";
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BuildResultNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const BuildResultNode*>(ref.get());
      p->stream << "BuildResult(" << node->filename << ", " << node->error_no << ", "
                << node->time_cost << ")";
    });

/********** FFI **********/

struct Internal {
  /********** Constructors **********/
  /*!
   * \brief Constructor of MeasureInput
   * \param task The task to be measured
   * \param state Concrete schedule of the task
   * \return The MeasureInput constructed
   * \sa MeasureInput::MeasureInput
   */
  static MeasureInput MeasureInputNew(SearchTask task, Schedule sch) {
    return MeasureInput(task, sch);
  }
  /*!
   * \brief Constructor of BuildResult
   * \param filename The filename of built binary file.
   * \param error_no The error code.
   * \param error_msg The error message if there is any error.
   * \param time_cost The time cost of build.
   * \return The BuildResult constructed
   * \sa BuildResult::BuildResult
   */
  static BuildResult BuildResultNew(String filename, int error_no, String error_msg,
                                    double time_cost) {
    return BuildResult(filename, error_no, error_msg, time_cost);
  }
  /*!
   * \brief Constructor of MeasureResult
   * \param costs The time costs of execution.
   * \param error_no The error code.
   * \param error_msg The error message if there is any error.
   * \param all_cost The time cost of build and run.
   * \param timestamp The time stamps of this measurement.
   * \return The MeasureResult constructed
   * \sa MeasureResult::MeasureResult
   */
  static MeasureResult MeasureResultNew(Array<PrimExpr> costs, int error_no, String error_msg,
                                        double all_cost, double timestamp) {
    return MeasureResult(costs, error_no, error_msg, all_cost, timestamp);
  }
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
  static RPCRunner RPCRunnerNew(String tracker, int priority, int n_parallel, int timeout,
                                int number, int repeat, int min_repeat_ms, double cooldown_interval,
                                bool enable_cpu_cache_flush) {
    return RPCRunner(tracker, priority, n_parallel, timeout, number, repeat, min_repeat_ms,
                     cooldown_interval, enable_cpu_cache_flush);
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

TVM_REGISTER_NODE_TYPE(MeasureInputNode);
TVM_REGISTER_NODE_TYPE(BuildResultNode);
TVM_REGISTER_NODE_TYPE(MeasureResultNode);
TVM_REGISTER_OBJECT_TYPE(ProgramRunnerNode);
TVM_REGISTER_OBJECT_TYPE(ProgramBuilderNode);
TVM_REGISTER_NODE_TYPE(LocalBuilderNode);
TVM_REGISTER_NODE_TYPE(RPCRunnerNode);
TVM_REGISTER_OBJECT_TYPE(MeasureCallbackNode);
TVM_REGISTER_NODE_TYPE(ProgramMeasurerNode);

TVM_REGISTER_GLOBAL("meta_schedule.MeasureInput").set_body_typed(Internal::MeasureInputNew);
TVM_REGISTER_GLOBAL("meta_schedule.BuildResult").set_body_typed(Internal::BuildResultNew);
TVM_REGISTER_GLOBAL("meta_schedule.MeasureResult").set_body_typed(Internal::MeasureResultNew);
TVM_REGISTER_GLOBAL("meta_schedule.LocalBuilder").set_body_typed(Internal::LocalBuilderNew);
TVM_REGISTER_GLOBAL("meta_schedule.RPCRunner").set_body_typed(Internal::RPCRunnerNew);
TVM_REGISTER_GLOBAL("meta_schedule.ProgramBuilderBuild")
    .set_body_typed(Internal::ProgramBuilderBuild);
TVM_REGISTER_GLOBAL("meta_schedule.ProgramRunnerRun").set_body_typed(Internal::ProgramRunnerRun);

}  // namespace meta_schedule
}  // namespace tvm
