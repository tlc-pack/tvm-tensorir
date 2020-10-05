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
#ifndef SRC_META_SCHEDULE_SEARCH_STRATEGY_MUTATOR_H_
#define SRC_META_SCHEDULE_SEARCH_STRATEGY_MUTATOR_H_

#include "../schedule.h"
#include "../search.h"

namespace tvm {
namespace meta_schedule {

/********** Mutator **********/

/*! \brief A mutation rule for the genetic algorithm */
class MutatorNode : public Object {
 public:
  /*! \brief The probability weight of choosing this rule */
  double p;

  /*! \brief Base destructor */
  virtual ~MutatorNode() = default;

  /*!
   * \brief Mutate the schedule by applying the mutation
   * \param task The search task
   * \param sch The schedule to be mutated
   * \param sampler The random number sampler
   * \return The new schedule after mutation, NullOpt if mutation fails
   */
  virtual Optional<Schedule> Apply(const SearchTask& task, const Schedule& sch,
                                   Sampler* sampler) = 0;

  static constexpr const char* _type_key = "meta_schedule.Mutator";
  TVM_DECLARE_BASE_OBJECT_INFO(MutatorNode, Object);
};

/*!
 * \brief Managed refernce to MutatorNode
 * \sa MutatorNode
 */
class Mutator : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Mutator, ObjectRef, MutatorNode);
};

/********** MutateTileSize **********/

/*! \brief Mutate the sampled tile size by re-factorized two axes */
class MutateTileSizeNode : public MutatorNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("p", &p); }

  /*! \brief Default destructor */
  ~MutateTileSizeNode() = default;

  /*!
   * \brief Mutate the schedule by applying the mutation
   * \param task The search task
   * \param sch The schedule to be mutated
   * \param sampler The random number sampler
   * \return The new schedule after mutation, NullOpt if mutation fails
   */
  Optional<Schedule> Apply(const SearchTask& task, const Schedule& sch, Sampler* sampler);

  static constexpr const char* _type_key = "meta_schedule.MutateTileSize";
  TVM_DECLARE_BASE_OBJECT_INFO(MutateTileSizeNode, MutatorNode);
};

/*!
 * \brief Managed refernce to MutateTileSizeNode
 * \sa MutateTileSizeNode
 */
class MutateTileSize : public Mutator {
 public:
  /*!
   * \brief Constructor
   * \param p The probability mass that this rule is selected
   */
  explicit MutateTileSize(double p);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MutateTileSize, Mutator, MutateTileSizeNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SEARCH_STRATEGY_MUTATOR_H_
