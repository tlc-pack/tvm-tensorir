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
#include "./measure_callback.h"  // NOLINT(build/include)

#include <dmlc/memory_io.h>
#include <tvm/node/serialization.h>
#include <tvm/runtime/registry.h>

#include <fstream>

#include "../support/base64.h"

namespace tvm {
namespace meta_schedule {

/********** RecordToFile **********/

RecordToFile::RecordToFile() { data_ = make_object<RecordToFileNode>(); }

void RecordToFileNode::Init(const SearchTask& task) {
  ICHECK(!task->log_file.value_or("").empty())
      << "ValueError: log_file is not specified in SeachTask";
  this->log_file = task->log_file.value();
  this->task_name = task->task_name;
  this->target = task->target->Export();
  this->target_host = task->target_host->Export();
  {
    std::string prim_func_json = SaveJSON(task->workload);
    std::string prim_func_b64;
    dmlc::MemoryStringStream m_stream(&prim_func_b64);
    support::Base64OutStream b64strm(&m_stream);
    dmlc::Stream* strm = &b64strm;
    strm->Write(prim_func_json);
    b64strm.Finish();
    this->prim_func_b64 = prim_func_b64;
  }
}

void RecordToFileNode::Callback(const Array<MeasureInput>& inputs,
                                const Array<MeasureResult>& results) {
  static const auto* f_serialize = runtime::Registry::Get("meta_schedule._serialize_json");
  ICHECK(f_serialize) << "IndexError: Cannot find packed function \""
                         "meta_schedule._serialize_json\", which should be registered in python";
  CHECK(!this->log_file.empty()) << "ValueError: empty log_file for measure logs";
  std::ofstream ofs(this->log_file, std::ofstream::app);
  ICHECK_EQ(inputs.size(), results.size());
  for (int i = 0, n = inputs.size(); i < n; ++i) {
    const MeasureInput& measure_input = inputs[i];
    const MeasureResult& measure_result = results[i];
    MeasureErrorNO error_no = static_cast<MeasureErrorNO>(measure_result->error_no);
    if (error_no != MeasureErrorNO::kNoError) {
      continue;
    }
    Array<ObjectRef> result{
        this->task_name,                                    // record[0]
        this->target,                                       // record[1]
        this->target_host,                                  // record[2]
        measure_result->costs,                              // record[3]
        measure_input->sch->trace().value()->AsJSON(true),  // record[4]
        String(kLogVersion),                                // record[5]
        this->prim_func_b64,                                // record[6]
    };
    String record = (*f_serialize)(result);
    ofs << record << std::endl;
  }
}

/********** FFI **********/

struct Internal {
  static RecordToFile RecordToFileNew() { return RecordToFile(); }
};

TVM_REGISTER_NODE_TYPE(RecordToFileNode);
TVM_REGISTER_GLOBAL("meta_schedule.RecordToFile").set_body_typed(Internal::RecordToFileNew);

}  // namespace meta_schedule
}  // namespace tvm
