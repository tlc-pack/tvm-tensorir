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
#ifndef SRC_META_SCHEDULE_COST_MODEL_H_
#define SRC_META_SCHEDULE_COST_MODEL_H_

#include <vector>

#include "./measure_record.h"
#include "./schedule.h"

namespace tvm {
namespace meta_schedule {

/*! \brief The base class for cost model */
class CostModelNode : public Object {
 public:
  /*! \brief Base destructor */
  virtual ~CostModelNode() = default;

  /*!
   * \brief Update the cost model according to new measurement results (training data).
   * \param inputs The measure inputs
   * \param results The measure results
   */
  virtual void Update(const Array<MeasureInput>& inputs, const Array<MeasureResult>& results) = 0;

  /*!
   * \brief Predict the scores of states
   * \param task The search task of states
   * \param states The input states
   * \return The predicted scores for all states
   */
  virtual std::vector<double> Predict(const SearchTask& task, const Array<Schedule>& states) = 0;

  static constexpr const char* _type_key = "meta_schedule.CostModel";
  TVM_DECLARE_BASE_OBJECT_INFO(CostModelNode, Object);
};

/*!
 * \brief Managed reference to CostModelNode.
 * \sa CostModelNode
 */
class CostModel : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(CostModel, ObjectRef, CostModelNode);
};

/*! \brief The cost model returning random value for all predictions */
class RandomModelNode : public CostModelNode {
 public:
  /*! \brief A sampler for generating random numbers */
  Sampler sampler_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // sampler_ is not visited
  }

  void Update(const Array<MeasureInput>& inputs, const Array<MeasureResult>& results) override;

  std::vector<double> Predict(const SearchTask& task, const Array<Schedule>& states) override;

  static constexpr const char* _type_key = "meta_schedule.RandomModel";
  TVM_DECLARE_FINAL_OBJECT_INFO(RandomModelNode, CostModelNode);
};

/*!
 * \brief Managed reference to RandomModelNode.
 * \sa RandomModelNode
 */
class RandomModel : public CostModel {
 public:
  RandomModel();
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(RandomModel, CostModel, RandomModelNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_COST_MODEL_H_
