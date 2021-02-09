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

namespace tvm {
namespace meta_schedule {

class SearchTask;
class TuneContextNode;

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
    std::vector<double> times;

    double MeanTime() const {
      if (times.empty()) {
        return kMaxTimeCost;
      }
      return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    }
  };

  /*! \brief Virtual destructor */
  virtual ~DatabaseNode() = default;

  /*! \brief Initialize the database */
  virtual void Init(const SearchTask& task) = 0;

  /*! \brief Initialize the database */
  virtual void Init(TuneContextNode* tune_context) {}

  /*!
   * \brief Add a schedule into the database
   * \param trace The trace of a schedule to be added
   * \param repr The string representation of the schedule
   * \param times The running time of the schedule
   */
  virtual void Add(const Trace& trace, const String& repr, const std::vector<double>& times) = 0;

  /*!
   * \brief Check if a schedule already exists in the database
   * \param repr The string representation of the schedule
   * \return A boolean indicating if the schedule exists in the database
   */
  virtual bool Has(const String& repr) const = 0;

  /*!
   * \brief Get the top-k entries
   * \param top_k The top-k entries to be queried
   * \return A list of at most `top_k` elements
   */
  virtual std::vector<Entry> GetTopK(int top_k) const = 0;

  /*!
   * \brief Get the best entry
   * \return An entry, nullable
   */
  virtual Entry GetBest() const = 0;

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

/*!
 * \brief Create an in-memory database
 * \param path Path to the file that stores tuning records in JSON format
 * \return The database created
 */
TVM_DLL Database InMemoryDB(Optional<String> path);

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_DATABASE_H_
