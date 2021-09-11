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
#ifndef SRC_META_SCHEDULE_TUNE_CONTEXT_H_
#define SRC_META_SCHEDULE_TUNE_CONTEXT_H_

#include <tvm/ir/module.h>
#include <tvm/support/random_engine.h>
#include <tvm/target/target.h>

#include "./search_strategy.h"
#include "./space_generator.h"

namespace tvm {
namespace meta_schedule {

using Database = ObjectRef;
using CostModel = ObjectRef;
using Postproc = ObjectRef;
using MeasureCallback = ObjectRef;

/*! \brief The tuning context. */
class TuneContextNode : public runtime::Object {
 public:
  /*! \brief The mod to be optimized. */
  Optional<IRModule> mod;
  /*! \brief The target to be optimized for. */
  Optional<Target> target;
  /*! \brief The design space generator. */
  Optional<SpaceGenerator> space_generator;
  /*! \brief The search strategy to be used. */
  Optional<SearchStrategy> search_strategy;
  /*! \brief The database for querying and storage. */
  Optional<Database> database;
  /*! \brief The cost model for estimation. */
  Optional<CostModel> cost_model;
  /*! \brief The post processing functions. */
  Optional<Array<Postproc>> postprocs;
  /*! \brief The measure callback functions. */
  Optional<Array<MeasureCallback>> measure_callbacks;
  /*! \brief The name of the tuning task. */
  Optional<String> task_name;
  /*! \brief The random state. */
  support::LinearCongruentialEngine::TRandState rand_state;
  /*! \brief The number of threads to be used. */
  int num_threads;
  /*! \brief The verbosity level. */
  int verbose;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("mod", &mod);
    v->Visit("target", &target);
    v->Visit("space_generator", &space_generator);
    v->Visit("search_strategy", &search_strategy);
    v->Visit("database", &database);
    v->Visit("cost_model", &cost_model);
    v->Visit("postprocs", &postprocs);
    v->Visit("measure_callbacks", &measure_callbacks);
    v->Visit("task_name", &task_name);
    v->Visit("rand_state", &rand_state);
    v->Visit("num_threads", &num_threads);
    v->Visit("verbose", &verbose);
  }

  static constexpr const char* _type_key = "meta_schedule.TuneContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(TuneContextNode, Object);
};

/*!
 * \brief Managed reference to TuneContextNode.
 * \sa TuneContextNode
 */
class TuneContext : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor.
   * \param mod The mod to be optimized.
   * \param target The target to be optimized for.
   * \param space_generator The design space generator.
   * \param search_strategy The search strategy to be used.
   * \param database The database for querying and storage.
   * \param cost_model The cost model for estimation.
   * \param postprocs The post processing functions.
   * \param measure_callbacks The measure callback functions.
   * \param task_name The task_name of the tuning task.
   * \param rand_state The random state.
   * \param num_threads The number of threads to be used.
   * \param verbose The verbosity level.
   */
  TVM_DLL explicit TuneContext(Optional<IRModule> mod,                                    //
                               Optional<Target> target,                                   //
                               Optional<SpaceGenerator> space_generator,                  //
                               Optional<SearchStrategy> search_strategy,                  //
                               Optional<Database> database,                               //
                               Optional<CostModel> cost_model,                            //
                               Optional<Array<Postproc>> postprocs,                       //
                               Optional<Array<MeasureCallback>> measure_callbacks,        //
                               Optional<String> task_name,                                //
                               support::LinearCongruentialEngine::TRandState rand_state,  //
                               int num_threads,                                           //
                               int verbose);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TuneContext, ObjectRef, TuneContextNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_TUNE_CONTEXT_H_
