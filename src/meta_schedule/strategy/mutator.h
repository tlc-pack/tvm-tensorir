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
#ifndef SRC_META_SCHEDULE_STRATEGY_MUTATOR_H_
#define SRC_META_SCHEDULE_STRATEGY_MUTATOR_H_

#include "../schedule.h"
#include "../search.h"

namespace tvm {
namespace meta_schedule {

/********** Mutator **********/

/*! \brief A mutation rule for the genetic algorithm */
class MutatorNode : public Object {
 public:
  /*! \brief The mutator application function */
  using FApply = runtime::TypedPackedFunc<Optional<Trace>(SearchTask, Trace, void*)>;

  /*! \brief Name of the mutator */
  String name;
  /*! \brief A packed function that applies the mutator */
  FApply apply_;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("name", &name); }

  /*!
   * \brief Mutate the schedule by applying the mutation
   * \param task The search task
   * \param trace The trace to be mutated
   * \param rand_state The random state for sampling
   * \return The new schedule after mutation, NullOpt if mutation fails
   */
  Optional<Trace> Apply(const SearchTask& task, const Trace& trace, tir::TRandState* rand_state);

  static constexpr const char* _type_key = "meta_schedule.Mutator";
  TVM_DECLARE_BASE_OBJECT_INFO(MutatorNode, Object);
};

/*!
 * \brief Managed refernce to MutatorNode
 * \sa MutatorNode
 */
class Mutator : public ObjectRef {
 public:
  using FApply = MutatorNode::FApply;

  /*!
   * \brief Constructing with name and a packed function
   * \param name Name of the mutator
   * \param apply The application function
   */
  explicit Mutator(String name, FApply apply);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Mutator, ObjectRef, MutatorNode);
};

/********** Built-in Mutators **********/

/*!
 * \brief Create a mutator that randomly mutate the tile size
 * \return The mutator created
 */
TVM_DLL Mutator MutateTileSize();

/*!
 * \brief Create a mutator that randomly mutate the outcome of SampleComputeLocation
 * \return The mutator created
 */
TVM_DLL Mutator MutateComputeLocation();

/*!
 * \brief Create a mutator that randomly mutate the depth of auto unroll
 * \return The mutator created
 */
TVM_DLL Mutator MutateAutoUnroll();

/*!
 * \brief Create a mutator that randomly mutate the max extent of parallelization
 * \return The mutator created
 */
TVM_DLL Mutator MutateParallel(const int& max_jobs_per_core);

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_STRATEGY_MUTATOR_H_
