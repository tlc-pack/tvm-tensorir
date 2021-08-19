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

#ifndef SRC_META_SCHEDULE_SEARCH_STRATEGY_H_
#define SRC_META_SCHEDULE_SEARCH_STRATEGY_H_

#include <tvm/ir/module.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/schedule/trace.h>

#include "./builder.h"
#include "./runner.h"

namespace tvm {
namespace meta_schedule {

// Forward declaration
class TuneContext;

/*! \brief The mutable class of search strategy, i.e, measure candidates generation. */
class SearchStrategyNode : public runtime::Object {
 public:
  /*! \brief The function type of `InitializeWithTuneContext` method. */
  using FInitializeWithTuneContext = runtime::TypedPackedFunc<void(const TuneContext&)>;
  /*! \brief The function type of `GenerateMeasureCandidates` method. */
  using FGenerateMeasureCandidates =
      runtime::TypedPackedFunc<Optional<runtime::Array<BuildInput>>()>;
  /*! \brief The function type of `NotifyRunnerResults` method. */
  using FNotifyRunnerResults = runtime::TypedPackedFunc<void(const Array<RunnerResult>&)>;
  /*! \brief The function type of `PreTuning` method. */
  using FPreTuning = runtime::TypedPackedFunc<void(const Array<tir::Trace>&)>;
  /*! \brief The function type of `PostTuning` method. All typedefs are used for customization. */
  using FPostTuning = runtime::TypedPackedFunc<void()>;

  /*! \brief Virtual destructor, required for abstract class. */
  virtual ~SearchStrategyNode() = default;

  /*!
   * \brief Virtual function to initialize the search strategy with TuneContext.
   * \param context The TuneContext object for initialization.
   */
  virtual void InitializeWithTuneContext(const TuneContext& context) = 0;

  /*!
   * \brief Virtual function to generate candidates from design space for measurement.
   * \return The next batch of candidates for measurements generated from the design space. Return
   *  nullptr when the search strategy is done.
   */
  virtual Optional<runtime::Array<BuildInput>> GenerateMeasureCandidates() = 0;

  /*!
   * \brief Virtual function to update the search strategy with meansurements from the runners.
   * \param results The runner's results of candidates generated from the search strategy.
   */
  virtual void NotifyRunnerResults(const Array<RunnerResult>& results) = 0;

  /*!
   * \brief Virtual function to prepare the search strategy status before tuning.
   * \param design_spaces The given design spaces for measure candidates generation.
   */
  virtual void PreTuning(const Array<tir::Trace>& design_spaces) = 0;

  /*! \brief Virtual function to do post tuning work. */
  virtual void PostTuning() = 0;

  /*! \brief Class name `SearchStrategy` */
  static constexpr const char* _type_key = "meta_schedule.SearchStrategy";
  TVM_DECLARE_BASE_OBJECT_INFO(SearchStrategyNode, Object);  // Abstract class
};

/*!
 * \brief Managed reference to SearchStrategyNode.
 * \sa SearchStrategyNode
 */
class SearchStrategy : public runtime::ObjectRef {
 public:
  /*! \brief Declare reference relationship. */
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SearchStrategy, ObjectRef, SearchStrategyNode);

  /*!
   * \brief Member function to create the python side customizable PySearchStrategy class.
   * \param initialize_with_tune_context_func The function pointer to the `Init...` function.
   * \param generate_measure_candidates_func The function pointer to the `Generate...` function.
   * \param notify_runner_results_func The function pointer to the `Notify...` function.
   * \param pre_tuning_func The function pointer to the `PreTuning` function.
   * \param post_tuning_func The function pointer to the `PostTuning` function.
   * \return The constructed PySpaceGenerator object but in SpaceGenerator type.
   */
  static SearchStrategy PySearchStrategy(
      SearchStrategyNode::FInitializeWithTuneContext initialize_with_tune_context_func,
      SearchStrategyNode::FGenerateMeasureCandidates generate_measure_candidates_func,
      SearchStrategyNode::FNotifyRunnerResults notify_runner_results_func,
      SearchStrategyNode::FPreTuning pre_tuning_func,
      SearchStrategyNode::FPostTuning post_tuning_func);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SEARCH_STRATEGY_H_
