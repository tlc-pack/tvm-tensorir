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

#include "../support/base64.h"
#include "./measure_callback.h"

namespace tvm {
namespace meta_schedule {

static constexpr const char* kLogVersion = "v0.0.1";

/********** RecordToFile **********/

RecordToFile::RecordToFile() { data_ = make_object<RecordToFileNode>(); }

void RecordToFileNode::Init(const SearchTask& task) {
  CHECK(!task->filename.value_or("").empty())
      << "ValueError: filename is not specified in SeachTask";
  this->filename = task->filename.value();
  this->task_name = task->task_name;
  this->target = task->target->Export();
  this->target_host = task->target_host->Export();
  {
    std::string prim_func_json = SaveJSON(task->func);
    std::string prim_func_b64;
    dmlc::MemoryStringStream mstrm(&prim_func_b64);
    support::Base64OutStream b64strm(&mstrm);
    dmlc::Stream* strm = &b64strm;
    strm->Write(prim_func_json);
    b64strm.Finish();
    this->prim_func_b64 = prim_func_b64;
  }
}

void RecordToFileNode::Callback(const Array<MeasureInput>& inputs,
                                const Array<MeasureResult>& results) {
  static const auto* f_serialize = runtime::Registry::Get("meta_schedule._serialize_json");
  CHECK(f_serialize) << "IndexError: Cannot find packed function \""
                        "meta_schedule._serialize_json\", which should be registered in python";
  CHECK(!this->filename.empty()) << "ValueError: empty filename for measure logs";
  std::ofstream ofs(this->filename, std::ofstream::app);
  CHECK_EQ(inputs.size(), results.size());
  for (int i = 0, n = inputs.size(); i < n; ++i) {
    Array<ObjectRef> result{
        this->task_name,           //
        this->target,              //
        this->target_host,         //
        results[i]->costs,         //
        inputs[i]->sch->Export(),  //
        String(kLogVersion),       //
        this->prim_func_b64,       //
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
