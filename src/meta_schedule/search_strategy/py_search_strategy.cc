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

#include "../search_strategy.h"
#include "../tune_context.h"

namespace tvm {
namespace meta_schedule {

/**************** PySearchStrategy ****************/

/*! \brief The cost model returning random value for all predictions */
class PySearchStrategyNode : public SearchStrategyNode {
 public:
  /*! \brief Pointer to the TuneContext init funcion in python */
  FInitializeWithTuneContext initialize_with_tune_context_func;
  /*! \brief Pointer to the funcion to generate measure candidates in python */
  FGenerateMeasureCandidates generate_measure_candidates_func;
  /*! \brief Pointer to the function to update results in python */
  FNotifyMeasureResults notify_measure_results_func;
  /*! \brief Pointer to the pretuning function in python */
  FPreTuning pre_tuning_func;
  /*! \brief Pointer to the posttuning function in python */
  FPostTuning post_tuning_func;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  void InitializeWithTuneContext(const TuneContext& context) override {
    this->initialize_with_tune_context_func(context);
  }

  Optional<runtime::Array<BuildInput>> GenerateMeasureCandidates() override {
    return this->generate_measure_candidates_func();
  }

  void NotifyMeasureResults(const Array<MeasureResult>& results) override {
    this->notify_measure_results_func(results);
  }

  void PreTuning(const Array<tir::Schedule>& design_spaces) override {
    this->pre_tuning_func(design_spaces);
  }

  void PostTuning() override { this->post_tuning_func(); }

  static constexpr const char* _type_key = "meta_schedule.PySearchStrategy";
  TVM_DECLARE_FINAL_OBJECT_INFO(PySearchStrategyNode, SearchStrategyNode);
};

/*!
 * \brief Managed reference to PySearchStrategyNode.
 * \sa PySearchStrategyNode
 */
class PySearchStrategy : public SearchStrategy {
 public:
  explicit PySearchStrategy(
      SearchStrategyNode::FInitializeWithTuneContext initialize_with_tune_context_func,
      SearchStrategyNode::FGenerateMeasureCandidates generate_measure_candidates_func,
      SearchStrategyNode::FNotifyMeasureResults notify_measure_results_func,
      SearchStrategyNode::FPreTuning pre_tuning_func,
      SearchStrategyNode::FPostTuning post_tuning_func) {
    ObjectPtr<PySearchStrategyNode> n = make_object<PySearchStrategyNode>();
    n->initialize_with_tune_context_func = std::move(initialize_with_tune_context_func);
    n->generate_measure_candidates_func = std::move(generate_measure_candidates_func);
    n->notify_measure_results_func = std::move(notify_measure_results_func);
    n->pre_tuning_func = std::move(pre_tuning_func);
    n->post_tuning_func = std::move(post_tuning_func);
    data_ = std::move(n);
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PySearchStrategy, SearchStrategy,
                                                    PySearchStrategyNode);
};

SearchStrategy SearchStrategy::PySearchStrategy(
    SearchStrategyNode::FInitializeWithTuneContext initialize_with_tune_context_func,
    SearchStrategyNode::FGenerateMeasureCandidates generate_measure_candidates_func,
    SearchStrategyNode::FNotifyMeasureResults notify_measure_results_func,
    SearchStrategyNode::FPreTuning pre_tuning_func,
    SearchStrategyNode::FPostTuning post_tuning_func) {
  return meta_schedule::PySearchStrategy(
      initialize_with_tune_context_func, generate_measure_candidates_func,
      notify_measure_results_func, pre_tuning_func, post_tuning_func);
}

TVM_REGISTER_NODE_TYPE(PySearchStrategyNode);
TVM_REGISTER_GLOBAL("meta_schedule.PySearchStrategy")
    .set_body_typed(SearchStrategy::PySearchStrategy);

}  // namespace meta_schedule
}  // namespace tvm