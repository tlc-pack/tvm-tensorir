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

#include <limits>
#include <numeric>
#include <vector>

#include "./trace.h"
#include "./utils.h"

namespace tvm {
namespace meta_schedule {

class SearchTask;
class TuneContextNode;

struct WorkloadInfo {
  Array<IntImm> shape_variant;

  bool operator==(const WorkloadInfo &other) const {
    if (shape_variant.size() != other.shape_variant.size()) return false;
    for (size_t i = 0; i < shape_variant.size(); ++i) {
      if (shape_variant[i]->value != other.shape_variant[i]->value) {
        return false;
      }
    } 
    return true;
  }
};

struct WklInfoHasher {
  std::size_t operator()(const WorkloadInfo& k) const {
    std::size_t seed = k.shape_variant.size();
    for(auto& k : k.shape_variant) {
      seed ^= k->value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

template<typename T>
using InfoMap = std::unordered_map<WorkloadInfo, T, WklInfoHasher>;


/*! \brief An abstract database storing all the tuning records. */
class DatabaseNode : public runtime::Object {
 public:
  /*! \brief A database entry */
  struct Entry {
    /*! \brief The trace */
    Optional<Trace> trace;
    /*! \brief The string representation of the schedule */
    String repr;
    /*! \brief The running time of the schedule */
    InfoMap<std::vector<double>> times;
  };

  /*! \brief Virtual destructor */
  virtual ~DatabaseNode() = default;

  /*! \brief Initialize the database */
  virtual void Init() = 0;

  /*! \brief Initialize the database */
  virtual void Init(TuneContextNode* tune_context) {}

  /*!
   * \brief Add a schedule into the database
   * \param trace The trace of a schedule to be added
   * \param repr The string representation of the schedule
   * \param times The running time of the schedule
   */
  virtual void Add(const Trace& trace, const String& repr,
                   const InfoMap<std::vector<double>>& gflops,
                   const SearchTask& task) = 0;

  virtual void Add(const Trace& trace, const Schedule& sch,
                   const std::vector<double> times,
                   const Optional<Array<IntImm>>& shape_variant,
                   const SearchTask& task) = 0;

  /*!
   * \brief Check if a schedule already exists in the database
   * \param repr The string representation of the schedule
   * \return A boolean indicating if the schedule exists in the database
   */
  virtual bool Has(const String& repr, const SearchTask& task) const = 0;

  /*!
   * \brief Get the top-k entries
   * \param top_k The top-k entries to be queried
   * \return A list of at most `top_k` elements
   */
  virtual std::vector<Entry> GetTopK(int top_k, const SearchTask& task) const = 0;

  /*!
   * \brief Get the best entry
   * \return An entry, nullable
   */
  virtual Entry GetBest(const SearchTask& task) = 0;

  /*!
   * \brief Number of records in the database
   * \return An integer, number of measures so far
   */
  virtual int Size() const = 0;

  /*! \brief The maximum time cost*/
  static const constexpr double kMaxTimeCost = 1e10;
};

/*!
 * \brief Managed refernce to DatabaseNode
 * \sa DatabaseNode
 */
class Database : public runtime::ObjectRef {
 public:
  using Entry = DatabaseNode::Entry;
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Database, runtime::ObjectRef, DatabaseNode);
};


double MeanGFlops(SearchTask task, const DatabaseNode::Entry& entry);

/*!
 * \brief Create an in-memory database
 * \param path Path to the file that stores tuning records in JSON format
 * \return The database created
 */
TVM_DLL Database InMemoryDB(Optional<String> path);

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_DATABASE_H_
