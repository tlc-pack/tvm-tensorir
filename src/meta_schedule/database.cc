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
#include "./database.h"  // NOLINT(build/include)

#include <dmlc/memory_io.h>
#include <tvm/node/serialization.h>
#include <tvm/support/parallel_for.h>

#include <map>

#include "../support/base64.h"
#include "./measure.h"
#include "./measure_record.h"
#include "./utils.h"

namespace tvm {
namespace meta_schedule {
namespace json_io {

/*!
 * \brief Read a specific file line by line, parse them into JSON-like TVM objects
 * \param path Path to the file
 * \return Lines of the file, NullOpt if fails to open the file
 */
Optional<Array<ObjectRef>> LoadTuningRecords(const String& path) {
  // Open the file
  std::ifstream ifs(path);
  if (!ifs.is_open() || ifs.fail()) {
    return NullOpt;
  }
  // Read lines from the file
  int count = 0;
  std::ostringstream os;
  os << '[';
  bool is_first = true;
  for (std::string line; std::getline(ifs, line);) {
    if (!line.empty() && line[0] != '#' && line[0] != '/' && !std::isspace(line[0])) {
      if (is_first) {
        is_first = false;
      } else {
        os << ',';
      }
      ++count;
      os << line;
    }
  }
  os << ']';
  // Deserialize the tuning records to JSON-like TVM objects
  static const auto* f_deserialize =
      runtime::Registry::Get("meta_schedule._deserialize_tuning_records");
  LOG(INFO) << "Found " << count << " tuning record(s)";
  LOG(INFO) << "Deserializing JSON tuning records...";
  ICHECK(f_deserialize)
      << "IndexError: Cannot find packed function \""
         "meta_schedule._deserialize_tuning_records\", which should be registered in python";
  ObjectRef parsed = (*f_deserialize)(os.str());
  const ArrayNode* array = parsed.as<runtime::ArrayNode>();
  ICHECK(array);
  return GetRef<Array<ObjectRef>>(array);
}

/*!
 * \brief Convert a tuning record, represented by JSON-like TVM object, to a database entry
 * \param record_obj The tuning record
 * \param task The search task
 */
Database::Entry RecordToEntry(const ObjectRef& record_obj, SearchTask& task) {
  const auto* record = record_obj.as<ArrayNode>();
  ICHECK_EQ(record->size(), 9);
  String task_name = Downcast<String>(record->at(0));
  Map<String, ObjectRef> target = Downcast<Map<String, ObjectRef>>(record->at(1));
  Map<String, ObjectRef> target_host = Downcast<Map<String, ObjectRef>>(record->at(2));
  Array<String> shape_vars = Downcast<Array<String>>(record->at(3));
  Array<IntImm> variant = Downcast<Array<IntImm>>(record->at(4));
  Array<FloatImm> times = Downcast<Array<FloatImm>>(record->at(5));
  ObjectRef trace_obj = record->at(6);
  String log_version = Downcast<String>(record->at(7));
  // TODO(@junrushao1994): structural equality of target
  if (task_name != task->task_name ||                  //
      log_version != String(kLogVersion) ||            //
      Target(target)->str() != task->target->str() ||  //
      Target(target_host)->str() != task->target_host->str()) {
    return Database::Entry{NullOpt, String(""), {}};
  }
  tir::PrimFunc orig_func{nullptr};
  {
    std::string prim_func_b64 = Downcast<String>(record->at(8));
    dmlc::MemoryStringStream m_stream(&prim_func_b64);
    support::Base64InStream b64strm(&m_stream);
    std::string parsed;
    b64strm.InitPosition();
    dmlc::Stream* strm = &b64strm;
    strm->Read(&parsed);
    orig_func = Downcast<tir::PrimFunc>(LoadJSON(parsed));
  }

  task = SearchTask(orig_func, task_name, Target(target), Target(target_host), NullOpt);
  Schedule sch(orig_func);
  TraceNode::Deserialize(trace_obj, sch);
  InfoMap<std::vector<double>> info2gflops; 
  WorkloadInfo info;
  if (variant.defined()) info.shape_variant = variant;
  info2gflops[info] = CalculateGFlops(sch, shape_vars, variant, AsVector<FloatImm, double>(times));
  return Database::Entry{sch->trace, Repr(sch), info2gflops};
}

}  // namespace json_io

namespace in_memory_db {

struct EntryHasher {
  size_t operator()(const Database::Entry& entry) const { return std::hash<String>()(entry.repr); }
};

struct EntryPtrComparator {
  bool operator()(Database::Entry* a, Database::Entry* b) const {
    double a_gflops = a->MeanGFlops();
    double b_gflops = b->MeanGFlops();
    if (a_gflops != b_gflops) {
      return a_gflops > b_gflops;
    }
    return a->repr.compare(b->repr) < 0;
  }
};

struct SearchTaskHasher {
  size_t operator()(const SearchTask& task) const {
    size_t hash = std::hash<String>()(task->task_name);
    hash ^= std::hash<String>()(task->target->str());
    return hash;
  }
};
struct SearchTaskEqual {
  TVM_DLL bool operator()(const SearchTask& task1, const SearchTask& task2) const {
    return task1->task_name == task2->task_name && task1->target->str() == task2->target->str();
  }
};

class InMemoryDBNode : public DatabaseNode {
 public:
  /*! \brief Virtual destructor */
  ~InMemoryDBNode() = default;

  void Init() override {
    if (!path.defined()) {
      LOG(INFO) << "Path to tuning logs is not specified - No file is used.";
      return;
    }
    LOG(INFO) << "Loading tuning records from: " << path.value();
    if (Optional<Array<ObjectRef>> opt_loaded = json_io::LoadTuningRecords(path.value())) {
      Array<ObjectRef> loaded = opt_loaded.value();
      int n_loaded = loaded.size();
      LOG(INFO) << "Converting tuning records to meta schedule trace...";
      std::vector<Entry> records(n_loaded);
      std::vector<SearchTask> tasks(n_loaded);
      auto worker = [&loaded, &records, &tasks](int thread_id, int i) -> void {
        const ObjectRef& record_obj = loaded[i];
        records[i] = json_io::RecordToEntry(record_obj, tasks[i]);
      };
      support::parallel_persist_for(0, n_loaded, worker);
      int total_valid = 0;
      LOG(INFO) << "Finish read records.";
      for (int i = 0; i < n_loaded; ++i) {
        const Entry& entry = records[i];
        if (entry.trace.defined()) {
          ++total_valid;
          this->Add(entry.trace.value(), entry.repr, entry.gflops);
        }
      }
      if (total_valid > 0) {
        LOG(INFO) << "Loaded " << total_valid << " valid record(s). "
                  << "Best gflops: " << this->GetBest().MeanGFlops() << ".";
                  // << (task->flop_ct / this->best.MeanTime() / 1e9) << " GFLOPs";
      } else {
        LOG(INFO) << "No valid records found.";
      }
    } else {
      LOG(INFO) << "Nothing is loaded. File does not exist or cannot be opened";
    }
  }

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("path", &path);
    // `best` is not visited
    // `entries_` is not visited
    // `sorted_` is not visited
  }

  /*!
   * \brief Add a schedule into the database
   * \param trace The trace of a schedule to be added
   * \param repr The string representation of the schedule
   * \param time The running time of the schedule
   */
  // void Add(const Database::Entry& entry) override {
  //   ICHECK(!entry.times.empty());
  //   entries_[entry.repr].push_back(entry);
  // }

  Entry MergeEntry(const Entry& a, const Entry& b) {
    ICHECK_EQ(a.repr, b.repr);
    InfoMap<std::vector<double>> gflops = a.gflops;
    for (auto kv : b.gflops) {
      auto it = gflops.find(kv.first);
      if (it == gflops.end()) {
        gflops[kv.first] = kv.second;
      } else {
        for (size_t i = 0; i < kv.second.size(); ++i) {
          gflops[kv.first].push_back(kv.second[i]);
        }
      } 
    }
    return Entry{a.trace, a.repr, gflops};
  }

  void Add(const Trace& trace, const String& repr,
           const InfoMap<std::vector<double>>& gflops) override {
    ICHECK(!gflops.empty());

    Entry entry{trace, repr, gflops}; 
    if (entries_.find(repr) != entries_.end()) {
      entry = MergeEntry(entry, entries_.at(repr));
    }
    entries_[repr] = entry;
    sorted_.insert(&entries_[repr]);
    if (!best.trace.defined() || best.MeanGFlops() < entry.MeanGFlops()) {
      best = entry;
    }
  }

  void Add(const Trace& trace, const Schedule& sch,
           const std::vector<double> times,
           const Optional<Array<String>>& shape_vars,
           const Optional<Array<IntImm>>& shape_variant) override {
    InfoMap<std::vector<double>> info2gflops; 
    WorkloadInfo info;
    if (shape_variant.defined()) info.shape_variant = shape_variant.value();
    info2gflops[info] = CalculateGFlops(sch, shape_vars, shape_variant, times);
    this->Add(trace, Repr(sch), info2gflops);
  }

  // void Add(const Trace& trace, const String& repr, const std::vector<double>& times,
  //          Optional<Array<String>> shape_vars, Optional<Array<IntImm>> shape_varaint) override {
  //   ICHECK(!times.empty());
  //   Database::Entry& entry = entries_[repr];
  //   double time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  //   if (!entry.repr.empty()) {
  //     if (entry.MeanTime() >= time) {
  //       sorted_.erase(&entry);
  //     } else {
  //       return;
  //     }
  //   }
  //   entry.trace = trace;
  //   entry.repr = repr;
  //   entry.times = times;
  //   entry.shape_vars = shape_vars;
  //   entry.shape_variant = shape_variant;
  //   sorted_.insert(&entry);
  //   if (!best.trace.defined() || best.MeanTime() > time) {
  //     best = entry;
  //   }
  // }

  /*!
   * \brief Check if a schedule already exists in the database
   * \param repr The string representation of the schedule
   * \return A boolean indicating if the schedule exists in the database
   */
  bool Has(const String& repr, const SearchTask& task) const override {
    if (entries_.count(task) == 0) {
      return false;
    }
    return entries_.at(task).count(repr) != 0;
  }

  /*!
   * \brief Get the top-k entries
   * \param repr The string representation of the schedule
   */
  std::vector<Entry> GetTopK(int top_k, const SearchTask& task) const override {
    std::vector<Entry> result;
    result.reserve(top_k);
    if (sorted_.count(task) == 0) {
      return result;
    }
    auto iter = sorted_.at(task).cbegin();
    for (int i = 0; i < top_k && iter != sorted_.at(task).cend(); ++i, ++iter) {
      result.push_back(**iter);
    }
    return result;
  }

  Entry GetBest() const override { 
    return best; 
  }

  int Size() const override {
    int size = 0;
    for (auto& kv : entries_) {
      size += kv.second.size();
    }
    return size;
  }

 public:
  /*! \brief Path to the file that stores tuning records in JSON format */
  Optional<String> path;
  /*! \brief The best entry so far */
  std::unordered_map<SearchTask, Entry, SearchTaskHasher, SearchTaskEqual> best;

 private:
  /*! \brief All the measured states, de-duplicated by the string repr */
  std::unordered_map<SearchTask, std::unordered_map<String, Database::Entry>, SearchTaskHasher,
                     SearchTaskEqual>
      entries_;
  /*! \brief All the measured states */
  std::unordered_map<SearchTask, std::multiset<Database::Entry*, EntryPtrComparator>,
                     SearchTaskHasher, SearchTaskEqual>
      sorted_;
};

class InMemoryDB : public Database {
 public:
  explicit InMemoryDB(const Optional<String>& path) {
    ObjectPtr<InMemoryDBNode> n = make_object<InMemoryDBNode>();
    n->path = path;
    data_ = std::move(n);
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(InMemoryDB, Database, InMemoryDBNode);
};

TVM_REGISTER_NODE_TYPE(InMemoryDBNode);
TVM_REGISTER_GLOBAL("meta_schedule.GetBest").set_body_typed([](InMemoryDB self, SearchTask task) {
  if (self->best.count(task)) {
    return self->best.at(task).trace;
  } else {
    return Optional<Trace>(NullOpt);
  }
});
}  // namespace in_memory_db

Database InMemoryDB(Optional<String> path) { return in_memory_db::InMemoryDB(path); }
Database InitMemoryDB(String path) {
  auto db = InMemoryDB(path);
  db->Init();
  return db;
}
tir::PrimFunc ApplyTrace(Trace trace, SearchTask task, SearchSpace space) {
  Schedule sch(task->workload);
  trace->Apply(sch);
  if (!space->Postprocess(task, sch, nullptr)) {
    LOG(FATAL) << "ValueError: The best schedule cannot be postprocessed all of a sudden";
  }
  return GetOnlyFunc(sch->mod());
}
TVM_REGISTER_GLOBAL("meta_schedule.ApplyTrace").set_body_typed(ApplyTrace);
TVM_REGISTER_OBJECT_TYPE(DatabaseNode);
TVM_REGISTER_GLOBAL("meta_schedule.GetInMemoryDB").set_body_typed(InitMemoryDB);

}  // namespace meta_schedule
}  // namespace tvm
