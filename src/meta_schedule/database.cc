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

#include <map>

namespace tvm {
namespace meta_schedule {
namespace in_memory_db {

struct EntryHasher {
  size_t operator()(const Database::Entry& entry) const { return std::hash<String>()(entry.repr); }
};

struct EntryPtrComparator {
  bool operator()(Database::Entry* a, Database::Entry* b) const {
    if (a->time != b->time) {
      return a->time < b->time;
    }
    return a->repr.compare(b->repr) < 0;
  }
};

class InMemoryDBNode : public DatabaseNode {
 public:
  /*! \brief Virtual destructor */
  ~InMemoryDBNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Add a schedule into the database
   * \param trace The trace of a schedule to be added
   * \param repr The string representation of the schedule
   * \param time The running time of the schedule
   */
  void Add(const Trace& trace, const String& repr, double time) override {
    Database::Entry& entry = entries_[repr];
    if (!entry.repr.empty()) {
      if (entry.time >= time) {
        sorted_.erase(&entry);
      } else {
        return;
      }
    }
    entry.trace = trace->WithNoPostproc();
    entry.repr = repr;
    entry.time = time;
    sorted_.insert(&entry);
  }

  /*!
   * \brief Check if a schedule already exists in the database
   * \param repr The string representation of the schedule
   * \return A boolean indicating if the schedule exists in the database
   */
  bool Has(const String& repr) const override { return entries_.count(repr) != 0; }

  /*!
   * \brief Get the top-k entries
   * \param repr The string representation of the schedule
   */
  std::vector<Entry> GetTopK(int top_k) const {
    std::vector<Entry> result;
    result.reserve(top_k);
    auto iter = sorted_.cbegin();
    for (int i = 0; i < top_k && iter != sorted_.cend(); ++i, ++iter) {
      result.push_back(**iter);
    }
    return result;
  }

 private:
  /*! \brief All the measured states, de-duplicated by the string repr */
  std::unordered_map<String, Database::Entry> entries_;
  /*! \brief All the measured states */
  std::multiset<Database::Entry*, EntryPtrComparator> sorted_;
};

class InMemoryDB : public Database {
 public:
  InMemoryDB() { data_ = make_object<InMemoryDBNode>(); }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(InMemoryDB, Database, InMemoryDBNode);
};

TVM_REGISTER_NODE_TYPE(InMemoryDBNode);

}  // namespace in_memory_db

Database InMemoryDB() { return in_memory_db::InMemoryDB(); }

TVM_REGISTER_OBJECT_TYPE(DatabaseNode);

}  // namespace meta_schedule
}  // namespace tvm
