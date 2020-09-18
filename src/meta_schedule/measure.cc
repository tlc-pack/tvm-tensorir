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

namespace tvm {
namespace meta_schedule {

TVM_REGISTER_NODE_TYPE(MeasureInputNode);
TVM_REGISTER_NODE_TYPE(BuildResultNode);
TVM_REGISTER_NODE_TYPE(MeasureResultNode);
TVM_REGISTER_OBJECT_TYPE(ProgramRunnerNode);
TVM_REGISTER_OBJECT_TYPE(ProgramBuilderNode);
TVM_REGISTER_NODE_TYPE(LocalBuilderNode);
TVM_REGISTER_NODE_TYPE(RPCRunnerNode);
TVM_REGISTER_OBJECT_TYPE(MeasureCallbackNode);
TVM_REGISTER_NODE_TYPE(ProgramMeasurerNode);

static const char* ErrorNoToStr[] = {
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
  n->key = key;
  n->host = host;
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
                                 Array<MeasureCallback> callbacks) {
  ObjectPtr<ProgramMeasurerNode> n = make_object<ProgramMeasurerNode>();
  n->builder = std::move(builder);
  n->runner = std::move(runner);
  n->callbacks = std::move(callbacks);
  data_ = std::move(n);
}

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

Array<MeasureResult> ProgramMeasurerNode::Measure(const Array<MeasureInput>& measure_inputs,
                                                  int verbose) const {
  Array<BuildResult> build_results = builder->Build(measure_inputs, verbose);
  Array<MeasureResult> measure_results = runner->Run(measure_inputs, build_results, verbose);
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
                  << "error_type:" << ErrorNoToStr[node->error_no] << ", "
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
  static MeasureInput CreateMeasureInput(SearchTask task, Schedule sch) {
    return MeasureInput(task, sch);
  }
  static BuildResult CreateBuildResult(String filename, int error_no, String error_msg,
                                       double time_cost) {
    return BuildResult(filename, error_no, error_msg, time_cost);
  }
  static MeasureResult CreateMeasureResult(Array<PrimExpr> costs, int error_no, String error_msg,
                                           double all_cost, double timestamp) {
    return MeasureResult(costs, error_no, error_msg, all_cost, timestamp);
  }
  static LocalBuilder CreateLocalBuilder(int timeout, int n_parallel, String build_func) {
    return LocalBuilder(timeout, n_parallel, build_func);
  }
  static RPCRunner CreateRPCRunner(String key, String host, int port, int priority, int n_parallel,
                                   int timeout, int number, int repeat, int min_repeat_ms,
                                   double cooldown_interval, bool enable_cpu_cache_flush) {
    return RPCRunner(key, host, port, priority, n_parallel, timeout, number, repeat, min_repeat_ms,
                     cooldown_interval, enable_cpu_cache_flush);
  }
  /********** Member methods **********/
  static Array<BuildResult> ProgramBuilderBuild(ProgramBuilder builder, Array<MeasureInput> inputs,
                                                int verbose) {
    return builder->Build(inputs, verbose);
  }
  static Array<MeasureResult> ProgramRunnerRun(ProgramRunner runner, Array<MeasureInput> inputs,
                                               Array<BuildResult> build_results, int verbose) {
    return runner->Run(inputs, build_results, verbose);
  }
};

TVM_REGISTER_GLOBAL("meta_schedule.MeasureInput").set_body_typed(Internal::CreateMeasureInput);
TVM_REGISTER_GLOBAL("meta_schedule.BuildResult").set_body_typed(Internal::CreateBuildResult);
TVM_REGISTER_GLOBAL("meta_schedule.MeasureResult").set_body_typed(Internal::CreateMeasureResult);
TVM_REGISTER_GLOBAL("meta_schedule.LocalBuilder").set_body_typed(Internal::CreateLocalBuilder);
TVM_REGISTER_GLOBAL("meta_schedule.RPCRunner").set_body_typed(Internal::CreateRPCRunner);
TVM_REGISTER_GLOBAL("meta_schedule.ProgramBuilderBuild")
    .set_body_typed(Internal::ProgramBuilderBuild);
TVM_REGISTER_GLOBAL("meta_schedule.ProgramRunnerRun").set_body_typed(Internal::ProgramRunnerRun);

}  // namespace meta_schedule
}  // namespace tvm
