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

class TuneContextNode;

/*! \brief The base class for cost model */
class CostModelNode : public Object {
 public:
  /*! \brief Base destructor */
  virtual ~CostModelNode() = default;

  /*! \brief Initialize the database */
  virtual void Init(TuneContextNode* tune_context) {}

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

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_COST_MODEL_H_
