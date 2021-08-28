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

#include <tvm/ir/expr.h>

#include "./arg_info.h"

namespace tvm {
namespace meta_schedule {

class RunnerInputNode : public runtime::Object {
 public:
  String artifact_path;
  String device_type;
  Array<ArgInfo> args_info;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("artifact_path", &artifact_path);
    v->Visit("device_type", &device_type);
    v->Visit("args_info", &args_info);
  }

  static constexpr const char* _type_key = "meta_schedule.RunnerInput";
  TVM_DECLARE_FINAL_OBJECT_INFO(RunnerInputNode, runtime::Object);
};

class RunnerInput : public runtime::ObjectRef {
 public:
  TVM_DLL RunnerInput(String artifact_path, String device_type, Array<ArgInfo> args_info);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(RunnerInput, runtime::ObjectRef, RunnerInputNode);
};

class RunnerResultNode : public runtime::Object {
 public:
  Optional<Array<FloatImm>> run_sec;
  Optional<String> error_msg;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("run_sec", &run_sec);
    v->Visit("error_msg", &error_msg);
  }

  static constexpr const char* _type_key = "meta_schedule.RunnerResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(RunnerResultNode, runtime::Object);
};

class RunnerResult : public runtime::ObjectRef {
 public:
  TVM_DLL RunnerResult(Optional<Array<FloatImm>> run_sec, Optional<String> error_msg);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(RunnerResult, runtime::ObjectRef, RunnerResultNode);
};

class RunnerFutureNode : public runtime::Object {
 public:
  // https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future
  using FDone = runtime::TypedPackedFunc<bool()>;
  using FResult = runtime::TypedPackedFunc<RunnerResult()>;

  FDone f_done;
  FResult f_result;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `f_done` is not visited
    // `f_result` is not visited
  }

  bool Done() const { return f_done(); }
  RunnerResult Result() const { return f_result(); }

  static constexpr const char* _type_key = "meta_schedule.RunnerFuture";
  TVM_DECLARE_FINAL_OBJECT_INFO(RunnerFutureNode, runtime::Object);
};

class RunnerFuture : public runtime::ObjectRef {
 public:
  using FDone = RunnerFutureNode::FDone;
  using FResult = RunnerFutureNode::FResult;

  TVM_DLL RunnerFuture(FDone f_done, FResult f_result);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(RunnerFuture, runtime::ObjectRef,
                                                    RunnerFutureNode);
};

class RunnerNode : public runtime::Object {
 public:
  using FRun = runtime::TypedPackedFunc<Array<RunnerFuture>(Array<RunnerInput>)>;

  virtual ~RunnerNode() = default;
  virtual Array<RunnerFuture> Run(Array<RunnerInput> runner_inputs) = 0;

  static constexpr const char* _type_key = "meta_schedule.Runner";
  TVM_DECLARE_BASE_OBJECT_INFO(RunnerNode, runtime::Object);
};

class Runner : public runtime::ObjectRef {
 public:
  using FRun = RunnerNode::FRun;
  TVM_DLL static Runner PyRunner(FRun f_run);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Runner, runtime::ObjectRef, RunnerNode);
};

class PyRunnerNode : public RunnerNode {
 public:
  FRun f_run;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `f_run` is not visited
  }

  Array<RunnerFuture> Run(Array<RunnerInput> runner_inputs) final { return f_run(runner_inputs); }

  static constexpr const char* _type_key = "meta_schedule.PyRunner";
  TVM_DECLARE_FINAL_OBJECT_INFO(PyRunnerNode, runtime::Object);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_RUNNER_H_
