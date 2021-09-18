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
#include <set>

#include "../database.h"
#include "../tune_context.h"
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

/*! \brief The struct defining comparison function of sorting by mean run seconds. */
struct SortByMeanRunSecs {
  static double Mean(const Array<FloatImm>& a) {
    ICHECK(!a.empty());
    double sum = 0.0;
    for (const FloatImm& i : a) {
      sum += i->value;
    }
    return sum / a.size();
  }

  bool operator()(const TuningRecord& a, const TuningRecord& b) const {
    double a_time = Mean(a->run_secs);
    double b_time = Mean(b->run_secs);
    return a_time < b_time;
  }
};

/*! \brief The default database implementation. */
class JSONFileDatabaseNode : public DatabaseNode {
 public:
  /*! \brief The path to store or load database records. */
  String record_path;
  /*! \brief The workload registry. */
  WorkloadRegistry reg{nullptr};
  /*! \brief The database records organized using a set. */
  std::multiset<TuningRecord, SortByMeanRunSecs> records_;

  static constexpr const char* _type_key = "meta_schedule.JSONFileDatabase";
  TVM_DECLARE_FINAL_OBJECT_INFO(JSONFileDatabaseNode, DatabaseNode);

 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("record_path", &record_path);
    v->Visit("reg", &reg);
    // `records_` is not visited
  }

  void InitializeWithTuneContext(const TuneContext& tune_context) final {
    // do nothing
  }

  WorkloadToken LookupOrAdd(const IRModule& mod) final { return reg->LookupOrAdd(mod); }

  void Add(const TuningRecord& record) final {
    this->records_.insert(record);
    JSONFileAppendLine(this->record_path, JSONObj2Str(record->AsJSON()));
  }

  int64_t Size() final { return this->records_.size(); }

  Array<TuningRecord> GetTopK(const WorkloadToken& workload, int top_k) final {
    CHECK(top_k > 0) << "top_k must be positive";
    Array<TuningRecord> results;
    results.reserve(top_k);
    int counter = 0;
    for (const TuningRecord& record : this->records_)
      if (record->workload->shash == workload->shash) {
        results.push_back(record);
        if (++counter == top_k) {
          break;
        }
      }
    return results;
  }
};

Database Database::JSONFileDatabase(String record_path, String workload_path, bool allow_missing) {
  ObjectPtr<JSONFileDatabaseNode> n = make_object<JSONFileDatabaseNode>();
  n->record_path = record_path;
  n->reg = WorkloadRegistry(workload_path, allow_missing);
  Array<ObjectRef> json_objs = JSONStr2Obj(JSONFileReadLines(record_path, allow_missing));
  for (const ObjectRef& json_obj : json_objs) {
    n->records_.insert(TuningRecord::FromJSON(json_obj, n->reg));
  }
  return Database(n);
}

TVM_REGISTER_NODE_TYPE(JSONFileDatabaseNode);
TVM_REGISTER_GLOBAL("meta_schedule.DatabaseJSONFileDatabase")
    .set_body_typed(Database::JSONFileDatabase);

}  // namespace meta_schedule
}  // namespace tvm
