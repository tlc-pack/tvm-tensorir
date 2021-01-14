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

/**************** PyCostModel ****************/

/*! \brief The cost model returning random value for all predictions */
class PyCostModelNode : public CostModelNode {
 public:
  using FUpdate = void(Array<MeasureInput>, Array<MeasureResult>);
  using FPredict = void(SearchTask, Array<Schedule>, void*);

  /*! \brief Pointer to the update funcion in python */
  runtime::TypedPackedFunc<FUpdate> update_func;
  /*! \brief Pointer to the predict funcion in python */
  runtime::TypedPackedFunc<FPredict> predict_func;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `update_func` is not visited
    // `predict_func` is not visited
  }

  /*!
   * \brief Update the cost model according to new measurement results (training data).
   * \param inputs The measure inputs
   * \param results The measure results
   */
  void Update(const Array<MeasureInput>& inputs, const Array<MeasureResult>& results) override {
    this->update_func(inputs, results);
  }

  /*!
   * \brief Predict the scores of states
   * \param task The search task of states
   * \param states The input states
   * \return The predicted scores for all states
   */
  std::vector<double> Predict(const SearchTask& task, const Array<Schedule>& states) override {
    std::vector<double> result(states.size(), 0.0);
    this->predict_func(task, states, result.data());
    return result;
  }

  static constexpr const char* _type_key = "meta_schedule.PyCostModel";
  TVM_DECLARE_FINAL_OBJECT_INFO(PyCostModelNode, CostModelNode);
};

/*!
 * \brief Managed reference to PyCostModelNode.
 * \sa PyCostModelNode
 */
class PyCostModel : public CostModel {
 public:
  using FUpdate = PyCostModelNode::FUpdate;
  using FPredict = PyCostModelNode::FPredict;

  explicit PyCostModel(runtime::TypedPackedFunc<FUpdate> update_func,
                       runtime::TypedPackedFunc<FPredict> predict_func) {
    ObjectPtr<PyCostModelNode> n = make_object<PyCostModelNode>();
    n->update_func = std::move(update_func);
    n->predict_func = std::move(predict_func);
    data_ = std::move(n);
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PyCostModel, CostModel, PyCostModelNode);
};

/**************** FFI ****************/

struct Internal {
  static PyCostModel New(runtime::PackedFunc update_func, runtime::PackedFunc predict_func) {
    return PyCostModel(update_func, predict_func);
  }
};

TVM_REGISTER_NODE_TYPE(PyCostModelNode);
TVM_REGISTER_GLOBAL("meta_schedule.PyCostModel").set_body_typed(Internal::New);

}  // namespace meta_schedule
}  // namespace tvm
