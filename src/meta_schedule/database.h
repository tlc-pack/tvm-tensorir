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
#ifndef SRC_META_SCHEDULE_DATABASE_H_
#define SRC_META_SCHEDULE_DATABASE_H_

#include <tvm/target/target.h>
#include <tvm/tir/schedule/trace.h>

#include "./arg_info.h"
#include "./workload_registry.h"

namespace tvm {
namespace meta_schedule {

class TuneContext;

class TuningRecordNode : public runtime::Object {
 public:
  tir::Trace trace;
  Array<FloatImm> run_secs;
  WorkloadToken workload{nullptr};
  Target target;
  Array<ArgInfo> args_info;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("trace", &trace);
    v->Visit("run_secs", &run_secs);
    v->Visit("workload", &workload);
    v->Visit("target", &target);
    v->Visit("args_info", &args_info);
  }

  ObjectRef AsJSON() const;

  static constexpr const char* _type_key = "meta_schedule.TuningRecord";
  TVM_DECLARE_FINAL_OBJECT_INFO(TuningRecordNode, runtime::Object);
};

class TuningRecord : public runtime::ObjectRef {
 public:
  TVM_DLL TuningRecord(tir::Trace trace, Array<FloatImm> run_secs, WorkloadToken workload,
                       Target target, Array<ArgInfo> args_info);
  TVM_DLL static TuningRecord FromJSON(const ObjectRef& json_obj, const WorkloadRegistry& reg);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TuningRecord, runtime::ObjectRef, TuningRecordNode);
};

class DatabaseNode : public runtime::Object {
 public:
  virtual ~DatabaseNode() = default;
  virtual void InitializeWithTuneContext(const TuneContext& tune_context) = 0;
  virtual void Add(const TuningRecord& record) = 0;
  virtual Array<TuningRecord> GetTopK(const WorkloadToken& workload, int top_k) = 0;
  virtual WorkloadToken LookupOrAdd(const IRModule& mod) = 0;
  virtual int64_t Size() = 0;

  static constexpr const char* _type_key = "meta_schedule.Database";
  TVM_DECLARE_FINAL_OBJECT_INFO(DatabaseNode, runtime::Object);
};

// TOOD: add PyDatabase

class Database : public runtime::ObjectRef {
 public:
  TVM_DLL static Database DefaultDatabase(String record_path, String workload_path,
                                          bool allow_missing);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Database, runtime::ObjectRef, DatabaseNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif
