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
#include "../cost_model.h"
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

/**************** RandCostModel ****************/

/*! \brief The cost model returning random value for all predictions */
class RandCostModelNode : public CostModelNode {
 public:
  /*! \brief A random state for sampler to generate random numbers */
  tir::TRandState rand_state;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // sampler is not visited
  }

  /*!
   * \brief Update the cost model according to new measurement results (training data).
   * \param inputs The measure inputs
   * \param results The measure results
   */
  void Update(const Array<MeasureInput>& inputs, const Array<MeasureResult>& results) override {}

  /*!
   * \brief Predict the scores of states
   * \param task The search task of states
   * \param states The input states
   * \return The predicted scores for all states
   */
  std::vector<double> Predict(const SearchTask& task, const Array<Schedule>& states) override {
    return tir::SampleUniform(&rand_state, states.size(), 0.0, 1.0);
  }

  static constexpr const char* _type_key = "meta_schedule.RandCostModel";
  TVM_DECLARE_FINAL_OBJECT_INFO(RandCostModelNode, CostModelNode);
};

/*!
 * \brief Managed reference to RandCostModelNode.
 * \sa RandCostModelNode
 */
class RandCostModel : public CostModel {
 public:
  explicit RandCostModel(int seed = -1) {
    ObjectPtr<RandCostModelNode> n = make_object<RandCostModelNode>();
    if (seed == -1) seed = std::random_device()();
    tir::RandEngine(&n->rand_state).Seed(seed);
    data_ = std::move(n);
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(RandCostModel, CostModel, RandCostModelNode);
};

/**************** FFI ****************/

struct Internal {
  static RandCostModel New(Optional<Integer> seed) {
    return seed.defined() ? RandCostModel(seed.value()->value) : RandCostModel();
  }
};

TVM_REGISTER_NODE_TYPE(RandCostModelNode);
TVM_REGISTER_GLOBAL("meta_schedule.RandCostModel").set_body_typed(Internal::New);

}  // namespace meta_schedule
}  // namespace tvm
