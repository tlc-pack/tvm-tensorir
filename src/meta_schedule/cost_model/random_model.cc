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

/**************** RandomModel ****************/

/*! \brief The cost model returning random value for all predictions */
class RandomModelNode : public CostModelNode {
 public:
  /*! \brief A sampler for generating random numbers */
  Sampler sampler;

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
  Array<FloatImm> Predict(const SearchTask& task, const Array<Schedule>& states) override;

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
  explicit RandomModel(Optional<Integer> seed);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(RandomModel, CostModel, RandomModelNode);
};

/**************** Constructors ****************/

RandomModel::RandomModel() : RandomModel(NullOpt) {}

RandomModel::RandomModel(Optional<Integer> seed) {
  ObjectPtr<RandomModelNode> n = make_object<RandomModelNode>();
  if (seed.defined()) {
    n->sampler.Seed(seed.value());
  }
  data_ = std::move(n);
}

/**************** RandomModel ****************/

Array<FloatImm> RandomModelNode::Predict(const SearchTask& task, const Array<Schedule>& states) {
  std::vector<double> result = sampler.SampleUniform(states.size(), 0.0, 1.0);
  return AsArray<double, FloatImm>()(result);
}

/**************** FFI ****************/

struct Internal {
  static RandomModel New(Optional<Integer> seed) { return RandomModel(seed); }
};

TVM_REGISTER_NODE_TYPE(RandomModelNode);
TVM_REGISTER_GLOBAL("meta_schedule.RandomModel").set_body_typed(Internal::New);

}  // namespace meta_schedule
}  // namespace tvm
