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
#include "../database.h"

#include <tvm/tir/schedule/schedule.h>

#include "../tune_context.h"

namespace tvm {
namespace meta_schedule {

TuningRecord::TuningRecord(tir::Trace trace, Array<FloatImm> run_secs, WorkloadToken workload,
                           Target target, Array<ArgInfo> args_info) {
  ObjectPtr<TuningRecordNode> n = make_object<TuningRecordNode>();
  n->trace = trace;
  n->run_secs = run_secs;
  n->workload = workload;
  n->target = target;
  n->args_info = args_info;
  this->data_ = n;
}

Database Database::PyDatabase(
    PyDatabaseNode::FInitializeWithTuneContext f_initialize_with_tune_context,  //
    PyDatabaseNode::FAdd f_add,                                                 //
    PyDatabaseNode::FGetTopK f_get_top_k,                                       //
    PyDatabaseNode::FLookupOrAdd f_lookup_or_add,                               //
    PyDatabaseNode::FSize f_size) {
  ObjectPtr<PyDatabaseNode> n = make_object<PyDatabaseNode>();
  n->f_initialize_with_tune_context = std::move(f_initialize_with_tune_context);
  n->f_add = std::move(f_add);
  n->f_get_top_k = std::move(f_get_top_k);
  n->f_lookup_or_add = std::move(f_lookup_or_add);
  n->f_size = std::move(f_size);
  return Database(n);
}

ObjectRef TuningRecordNode::AsJSON() const {
  Array<ObjectRef> json_args_info;
  json_args_info.reserve(args_info.size());
  for (const ArgInfo& arg_info : args_info) {
    json_args_info.push_back(arg_info->AsJSON());
  }
  return Array<ObjectRef>{trace->AsJSON(false),          //
                          run_secs,                      //
                          Integer(workload->token_id_),  //
                          target->Export(),              //
                          json_args_info};
}

TuningRecord TuningRecord::FromJSON(const ObjectRef& json_obj, const WorkloadRegistry& reg) {
  tir::Trace trace{nullptr};
  Array<FloatImm> run_secs{nullptr};
  WorkloadToken workload{nullptr};
  Target target{nullptr};
  Array<ArgInfo> args_info;
  try {
    const ArrayNode* json_array = json_obj.as<ArrayNode>();
    CHECK(json_array && json_array->size() == 5);
    // Load json[1] => run_secs
    run_secs = Downcast<Array<FloatImm>>(json_array->at(1));
    // Load json[2] => workload
    workload = reg->At(Downcast<Integer>(json_array->at(2)));
    // Load json[3] => target
    target = Target(Downcast<Map<String, ObjectRef>>(json_array->at(3)));
    // Load json[4] => args_info
    {
      const ArrayNode* json_args_info = json_array->at(4).as<ArrayNode>();
      args_info.reserve(json_args_info->size());
      for (const ObjectRef& json_arg_info : *json_args_info) {
        args_info.push_back(ArgInfo::FromJSON(json_arg_info));
      }
    }
    // Load json[0] => trace
    {
      const ObjectRef& json_trace = json_array->at(0);
      tir::Schedule sch =
          tir::Schedule::Traced(workload->mod, -1, 0, tir::ScheduleErrorRenderLevel::kNone);
      tir::Trace::ApplyJSONToSchedule(json_trace, sch);
      trace = sch->trace().value();
    }
  } catch (const std::runtime_error& e) {  // includes tvm::Error and dmlc::Error
    LOG(FATAL) << "ValueError: Unable to parse the JSON object: " << json_obj
               << "\nThe error is: " << e.what();
  }
  return TuningRecord(trace, run_secs, workload, target, args_info);
}

TVM_REGISTER_NODE_TYPE(TuningRecordNode);
TVM_REGISTER_NODE_TYPE(PyDatabaseNode);
TVM_REGISTER_OBJECT_TYPE(DatabaseNode);

TVM_REGISTER_GLOBAL("meta_schedule.TuningRecord")
    .set_body_typed([](tir::Trace trace, Array<FloatImm> run_secs, WorkloadToken workload,
                       Target target, Array<ArgInfo> args_info) {
      return TuningRecord(trace, run_secs, workload, target, args_info);
    });
TVM_REGISTER_GLOBAL("meta_schedule.TuningRecordAsJSON")  //
    .set_body_method<TuningRecord>(&TuningRecordNode::AsJSON);
TVM_REGISTER_GLOBAL("meta_schedule.TuningRecordFromJSON")  //
    .set_body_typed(TuningRecord::FromJSON);
TVM_REGISTER_GLOBAL("meta_schedule.DatabasePyDatabase")  //
    .set_body_typed(Database::PyDatabase);

TVM_REGISTER_GLOBAL("meta_schedule.DatabaseInitializeWithTuneContext")  //
    .set_body_method<Database>(&DatabaseNode::InitializeWithTuneContext);
TVM_REGISTER_GLOBAL("meta_schedule.DatabaseAdd")  //
    .set_body_method<Database>(&DatabaseNode::Add);
TVM_REGISTER_GLOBAL("meta_schedule.DatabaseGetTopK")  //
    .set_body_method<Database>(&DatabaseNode::GetTopK);
TVM_REGISTER_GLOBAL("meta_schedule.DatabaseLookupOrAdd")  //
    .set_body_method<Database>(&DatabaseNode::LookupOrAdd);
TVM_REGISTER_GLOBAL("meta_schedule.DatabaseSize")  //
    .set_body_method<Database>(&DatabaseNode::Size);

}  // namespace meta_schedule
}  // namespace tvm
