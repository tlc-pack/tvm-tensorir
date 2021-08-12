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
#ifndef TVM_META_SCHEDULE_SEARCH_H_
#define TVM_META_SCHEDULE_SEARCH_H_

#include <tvm/target/target.h>

#include "../tir/schedule/primitive.h"
#include "./schedule.h"

namespace tvm {
namespace meta_schedule {

class ProgramMeasurer;

class TuneContextNode;

/********** SearchTask **********/

/*! \brief Descrption of a search task */
class SearchTaskNode : public Object {
 public:
  /*! \brief The function to be optimized */
  tir::PrimFunc workload;
  /*! \brief Name of this search task */
  String task_name;
  /*! \brief The target to be built at */
  Target target;
  /*! \brief The target host to be built at */
  Target target_host;
  /*! \brief The file to load/store search logs */
  Optional<String> log_file;
  /*! \brief The number of floating point operations in the task */
  double flop_ct;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("workload", &workload);
    v->Visit("task_name", &task_name);
    v->Visit("target", &target);
    v->Visit("target_host", &target_host);
    v->Visit("log_file", &log_file);
    v->Visit("flop_ct", &flop_ct);
  }

  void Init(TuneContextNode* tune_context) {}

  static constexpr const char* _type_key = "meta_schedule.SearchTask";
  TVM_DECLARE_FINAL_OBJECT_INFO(SearchTaskNode, Object);
};

/*!
 * \brief Managed reference to SearchTaskNode
 * \sa SearchTaskNode
 */
class SearchTask : public ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param workload The function to be optimized
   * \param task_name Name of this search task
   * \param target The target to be built at
   * \param target_host The target host to be built at
   * \param log_file The file to load/store search logs
   */
  explicit SearchTask(tir::PrimFunc workload, String task_name, Target target, Target target_host,
                      Optional<String> log_file);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SearchTask, ObjectRef, SearchTaskNode);
};

/********** SearchSpace **********/

/*!
 * \brief Description and abstraction of a search space.
 * The search space could be specified by manually written schedule function,
 * generated via loop analysis, ansor-like rules that apply to each block, etc.
 */
class SearchSpaceNode : public runtime::Object {
 public:
  /*! \brief Virtual destructor */
  virtual ~SearchSpaceNode() = default;
  /*! \brief Initialize the search space */
  virtual void Init(const SearchTask& task) {}
  /*! \brief Initialize the search space */
  virtual void Init(TuneContextNode* tune_context) {}
  /*!
   * \brief Apply postprocessors onto the schedule
   * \param task The search task
   * \param sch The schedule to be postprocessed
   * \param rand_state The random state for sampling
   */
  virtual bool Postprocess(const SearchTask& task, const Schedule& sch,
                           tir::TRandState* rand_state) = 0;
  /*!
   * \brief Sample a schedule out of the search space
   * \param task The search task to be sampled from
   * \return The schedule sampled
   */
  virtual Schedule SampleSchedule(const SearchTask& task, tir::TRandState* rand_state) = 0;
  /*!
   * \brief Get support of the search space
   * \param task The search task to be sampled from
   * \return The support of the search space. Any point from the search space should along to one of
   * the traces returned
   */
  virtual Array<Schedule> GetSupport(const SearchTask& task, tir::TRandState* rand_state) = 0;

  static constexpr const char* _type_key = "meta_schedule.SearchSpace";
  TVM_DECLARE_BASE_OBJECT_INFO(SearchSpaceNode, Object);
};

/*!
 * \brief Managed reference to SearchSpaceNode
 * \sa SearchSpaceNode
 */
class SearchSpace : public runtime::ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SearchSpace, ObjectRef, SearchSpaceNode);
};

/********** SearchStrategy **********/

/*!
 * \brief The search strategy for exploring the search space.
 * It could be always replay the sampling function, or saving several traces
 * from the sample function and then do lightweight-metropolis-hastings, or integrate those with
 * evolutionary search, etc.
 */
class SearchStrategyNode : public Object {
 public:
  /*! \brief Virtual destructor */
  virtual ~SearchStrategyNode() = default;
  /*! \brief Initialize the search strategy */
  virtual void Init(const SearchTask& task) {}
  /*! \brief Initialize the search strategy */
  virtual void Init(TuneContextNode* tune_context) {}
  /*!
   * \brief Explore the search space and find the best schedule
   * \param task The search task
   * \param space The search space
   * \param measurer The measurer that builds, runs and profiles sampled programs
   * \param verbose Whether or not in verbose mode
   * \return The best schedule found, NullOpt if no valid schedule is found
   */
  virtual Optional<Schedule> Search(const SearchTask& task, const SearchSpace& space,
                                    const ProgramMeasurer& measurer, tir::TRandState* rand_state,
                                    int verbose) = 0;
  /*! \brief Explore the search space */
  virtual void Search() { LOG(FATAL) << "NotImplemented"; }

  static constexpr const char* _type_key = "meta_schedule.SearchStrategy";
  TVM_DECLARE_BASE_OBJECT_INFO(SearchStrategyNode, Object);
};

/*!
 * \brief Managed reference to SearchStrategyNode
 * \sa SearchStrategyNode
 */
class SearchStrategy : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SearchStrategy, ObjectRef, SearchStrategyNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_SEARCH_H_
