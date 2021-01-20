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
#include <vector>

#include "./schedule.h"

namespace tvm {
namespace meta_schedule {

/*! \brief An abstract database storing all the tuning records. */
class DatabaseNode : public runtime::Object {
 public:
  /*! \brief A database entry */
  struct Entry {
    /*! \brief The schedule */
    Schedule sch;
    /*! \brief The string representation of the schedule */
    String repr;
    /*! \brief The running time of the schedule */
    double time;
  };

  /*! \brief Virtual destructor */
  virtual ~DatabaseNode() = default;

  /*!
   * \brief Add a schedule into the database
   * \param sch The schedule to be added
   * \param repr The string representation of the schedule
   * \param time The running time of the schedule
   */
  virtual void Add(const Schedule& sch, const String& repr, double time) = 0;

  /*!
   * \brief Check if a schedule already exists in the database
   * \param repr The string representation of the schedule
   * \return A boolean indicating if the schedule exists in the database
   */
  virtual bool Has(const String& repr) const = 0;

  /*!
   * \brief Get the top-k entries
   * \param repr The string representation of the schedule
   */
  virtual std::vector<Entry> GetTopK(int top_k) const = 0;
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
 * \return The database created
 */
TVM_DLL Database InMemoryDB();

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_DATABASE_H_
