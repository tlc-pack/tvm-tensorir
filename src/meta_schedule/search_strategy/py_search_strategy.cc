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

/*! \brief The python side customizable class for measure candidate generation */
class PySearchStrategyNode : public SearchStrategyNode {
 public:
  /*! \brief Pointer to the `Init...` function python. */
  FInitializeWithTuneContext initialize_with_tune_context_func;
  /*! \brief Pointer to the `Generate...` function python. */
  FGenerateMeasureCandidates generate_measure_candidates_func;
  /*! \brief Pointer to the `Notify...` function python. */
  FNotifyMeasureResults notify_measure_results_func;
  /*! \brief Pointer to the `PreTuning` function python. */
  FPreTuning pre_tuning_func;
  /*! \brief Pointer to the `PostTuning` function python. */
  FPostTuning post_tuning_func;

  /*! \brief Visitor for variables in python (required). */
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Use the given function pointer to override the `Init...` function.
   * \param context The TuneContext object for initialization.
   */
  void InitializeWithTuneContext(const TuneContext& context) override {
    this->initialize_with_tune_context_func(context);
  }

  /*!
   * \brief Use the given function pointer to override the `Gene...` function.
   * \return The next batch of candidates for measurements generated from the design space. Return
   *  nullptr when the search strategy is done.
   */
  Optional<runtime::Array<BuildInput>> GenerateMeasureCandidates() override {
    return this->generate_measure_candidates_func();
  }

  /*!
   * \brief Use the given function pointer to override the `Notify...` function.
   * \param results The measurement results of candidates generated from the search strategy.
   */
  void NotifyMeasureResults(const Array<MeasureResult>& results) override {
    this->notify_measure_results_func(results);
  }

  /*!
   * \brief Use the given function pointer to override the `PreTuning` function.
   * \param design_spaces The given design spaces for measure candidates generation.
   */
  void PreTuning(const Array<tir::Trace>& design_spaces) override {
    this->pre_tuning_func(design_spaces);
  }
  /*! \brief Use the given function pointer to override the `PostTuning` function. */
  void PostTuning() override { this->post_tuning_func(); }

  /*! \brief Class name `PySearchStrategy` */
  static constexpr const char* _type_key = "meta_schedule.PySearchStrategy";
  TVM_DECLARE_FINAL_OBJECT_INFO(PySearchStrategyNode, SearchStrategyNode);  // Concrete class
};

/*!
 * \brief Managed reference to PySearchStrategyNode.
 * \sa PySearchStrategyNode
 */
class PySearchStrategy : public SearchStrategy {
 public:
  /*! \brief Constructor function of PySearchStrategy class. */
  explicit PySearchStrategy(
      SearchStrategyNode::FInitializeWithTuneContext initialize_with_tune_context_func,
      SearchStrategyNode::FGenerateMeasureCandidates generate_measure_candidates_func,
      SearchStrategyNode::FNotifyMeasureResults notify_measure_results_func,
      SearchStrategyNode::FPreTuning pre_tuning_func,
      SearchStrategyNode::FPostTuning post_tuning_func) {
    // Make a new PySearchStrategyNode object.
    ObjectPtr<PySearchStrategyNode> n = make_object<PySearchStrategyNode>();
    // Copy the given function pointers.
    n->initialize_with_tune_context_func = std::move(initialize_with_tune_context_func);
    n->generate_measure_candidates_func = std::move(generate_measure_candidates_func);
    n->notify_measure_results_func = std::move(notify_measure_results_func);
    n->pre_tuning_func = std::move(pre_tuning_func);
    n->post_tuning_func = std::move(post_tuning_func);
    data_ = std::move(n);
  }

  /*! \brief Declare reference relationship. */
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PySearchStrategy, SearchStrategy,
                                                    PySearchStrategyNode);
};

/*!
 * \brief Expose the PySearchStrategy constructor function as a member function of SearchStrategy.
 * \param initialize_with_tune_context_func The function pointer to the `Init...` function python.
 * \param generate_measure_candidates_func The function pointer to the `Gene...` function python.
 * \param notify_measure_results_func The function pointer to the `Notify...` function python.
 * \param pre_tuning_func The function pointer to the `PreTuning` function python.
 * \param post_tuning_func The function pointer to the `PostTuning` function python.
 * \return The constructed PySearchStrategy object but in SearchStrategy type.
 */
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

TVM_REGISTER_NODE_TYPE(PySearchStrategyNode);  // Concrete class
/*! \brief Register SearchStrategy's `PySearchStrategy` function to global registry. */
TVM_REGISTER_GLOBAL("meta_schedule.PySearchStrategy")
    .set_body_typed(SearchStrategy::PySearchStrategy);

}  // namespace meta_schedule
}  // namespace tvm
