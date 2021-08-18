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

#include <tvm/ir/module.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/object.h>
#include <tvm/target/target.h>

#include "./builder.h"
#include "./runner.h"
#include "./search_strategy.h"
#include "./space_generator.h"

#ifndef SRC_META_SCHEDULE_TUNE_CONTEXT_H_
#define SRC_META_SCHEDULE_TUNE_CONTEXT_H_

namespace tvm {
namespace meta_schedule {

/*! \brief Type defintions */
using TRandState = int64_t;  // TODO(zxybazh): Merge with Sampling PR.
using Database = ObjectRef;
using CostModel = ObjectRef;
using PostProc = ObjectRef;
using MeasureCallback = ObjectRef;

class TuneContextNode : public runtime::Object {
 public:
  /*! \brief The function to be optimized */
  Optional<IRModule> workload;
  /*! \brief The design space generator. */
  Optional<SpaceGenerator> space_generator;
  /*! \brief The search strategy to be used. */
  Optional<SearchStrategy> search_strategy;
  /*! \brief The database for querying and storage. */
  Optional<Database> database;
  /*! \brief The cost model for estimation. */
  Optional<CostModel> cost_model;
  /*! \brief The target for builder. */
  Optional<Target> target;
  /* \brief The post processing functions. */
  Optional<Array<PostProc>> postprocs;
  /*! \brief The measurement call back functions. */
  Optional<Array<MeasureCallback>> measure_callbacks;
  /*! \brief The name of the tuning task. */
  String name;
  /*! \brief The seed value of random state. */
  TRandState seed;
  /*! \brief The number of threads to use. */
  int num_threads;
  /*! \brief The value of verbose. */
  int verbose;

  // TODO(zxybazh): Convenient functions for post-processing and measure callbacks.

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("workload", &workload);
    v->Visit("space_generator", &space_generator);
    v->Visit("search_strategy", &search_strategy);
    v->Visit("database", &database);
    v->Visit("cost_model", &cost_model);
    v->Visit("target", &target);
    v->Visit("postprocs", &postprocs);
    v->Visit("measure_callbacks", &measure_callbacks);
    v->Visit("name", &name);
    v->Visit("seed", &seed);
    v->Visit("num_threads", &num_threads);
    v->Visit("verbose", &verbose);
  }

  static constexpr const char* _type_key = "meta_schedule.TuneContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(TuneContextNode, Object);
};

/*!
 * \brief Managed reference to TuneContext Node
 * \sa TuneContextNode
 */
class TuneContext : public runtime::ObjectRef {
 public:
  TuneContext() = default;

  explicit TuneContext(Optional<IRModule> workload,                         //
                       Optional<SpaceGenerator> space_generator,            //
                       Optional<SearchStrategy> search_strategy,            //
                       Optional<Database> database,                         //
                       Optional<CostModel> cost_model,                      //
                       Optional<Target> target,                             //
                       Optional<Array<PostProc>> postprocs,                 //
                       Optional<Array<MeasureCallback>> measure_callbacks,  //
                       String name,                                         //
                       TRandState seed,                                     //
                       int num_threads,                                     //
                       int verbose);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TuneContext, ObjectRef, TuneContextNode);
};  // namespace meta_schedule

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_TUNE_CONTEXT_H_
