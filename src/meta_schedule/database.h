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

/*! \brief The class of tuning records. */
class TuningRecordNode : public runtime::Object {
 public:
  /*! \brief The trace tuned. */
  tir::Trace trace;
  /*! \brief The profiling result in seconds. */
  Array<FloatImm> run_secs;
  /*! \brief The workload token. */
  WorkloadToken workload{nullptr};
  /*! \brief The target for tuning. */
  Target target;
  /*! \brief The argument information. */
  Array<ArgInfo> args_info;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("trace", &trace);
    v->Visit("run_secs", &run_secs);
    v->Visit("workload", &workload);
    v->Visit("target", &target);
    v->Visit("args_info", &args_info);
  }

  /*!
   * \brief Export the tuning record to a JSON string.
   * \return An array containing the trace, running secs, workload token id, serialized target, and
   *  argument information.
   */
  ObjectRef AsJSON() const;

  static constexpr const char* _type_key = "meta_schedule.TuningRecord";
  TVM_DECLARE_FINAL_OBJECT_INFO(TuningRecordNode, runtime::Object);
};

/*!
 * \brief The managed reference of TuningRecordNode.
 * \sa TuningRecordNode
 */
class TuningRecord : public runtime::ObjectRef {
 public:
  /*!
   \brief Constructor of a tuning record.
   \param trace The trace of the tuning record.
   \param run_secs The running time of the tuning record.
   \param workload The workload of the tuning record.
   \param target The target of the tuning record.
   \param args_info The argument information of the tuning record.
  */
  TVM_DLL explicit TuningRecord(tir::Trace trace, Array<FloatImm> run_secs, WorkloadToken workload,
                                Target target, Array<ArgInfo> args_info);
  /*!
   * \brief Create a tuning record from a json object.
   * \param json The json object.
   * \param reg The workload registry.
   * \return The tuning record created.
   */
  TVM_DLL static TuningRecord FromJSON(const ObjectRef& json_obj, const WorkloadRegistry& reg);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TuningRecord, runtime::ObjectRef, TuningRecordNode);
};

/* \brief The abstract interface of database. */
class DatabaseNode : public runtime::Object {
 public:
  /*! \brief Default destructor */
  virtual ~DatabaseNode() = default;
  /*!
   * \brief Initialize the database with tuning context.
   * \param tune_context The tuning context for initialization.
   */
  virtual void InitializeWithTuneContext(const TuneContext& tune_context) = 0;
  /*!
   * \brief Add a tuning record to the database.
   * \param record The tuning record to be added.
   */
  virtual void Add(const TuningRecord& record) = 0;
  /*!
   * \brief Get the top K tuning records of given workload from the database.
   * \param workload The workload to be searched for.
   * \param top_k The number of top records to be returned.
   * \return An array of top K tuning records for the given workload.
   */
  virtual Array<TuningRecord> GetTopK(const WorkloadToken& workload, int top_k) = 0;
  /*!
   * \brief Look up or add workload to the database if missing.
   * \param mod The IRModule to be searched for or added.
   * \return The workload token of the given IRModule.
   */
  virtual WorkloadToken LookupOrAdd(const IRModule& mod) = 0;
  /*! \brief Get the size of the database. */
  virtual int64_t Size() = 0;

  static constexpr const char* _type_key = "meta_schedule.Database";
  TVM_DECLARE_FINAL_OBJECT_INFO(DatabaseNode, runtime::Object);
};

// TOOD: add PyDatabase

/*!
 * \brief Managed reference to DatabaseNode.
 * \sa DatabaseNode
 */
class Database : public runtime::ObjectRef {
 public:
  /*!
   * \brief Default constructor
   * \param record_path The path to the database file.
   * \param workload_path The path to the workload registry file.
   * \param allow_missing_files Whether to create new file when the given path is not found.
   */
  TVM_DLL static Database DefaultDatabase(String record_path, String workload_path,
                                          bool allow_missing);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Database, runtime::ObjectRef, DatabaseNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif
