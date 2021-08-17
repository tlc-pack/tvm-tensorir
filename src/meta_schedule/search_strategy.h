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
#include <tvm/runtime/container.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>

#include "builder.h"
#include "runner.h"
#include "schedule.h"

namespace tvm {
namespace meta_schedule {

class TuneContext;

class SearchStrategyNode : public runtime::Object {
 public:
  using FInitializeWithTuneContext = runtime::TypedPackedFunc<void(const TuneContext&)>;
  using FGenerateMeasureCandidates =
      runtime::TypedPackedFunc<Optional<runtime::Array<BuildInput>>()>;
  using FNotifyMeasureResults = runtime::TypedPackedFunc<void(const Array<MeasureResult>&)>;
  using FPreTuning = runtime::TypedPackedFunc<void(const Array<Schedule>&)>;
  using FPostTuning = runtime::TypedPackedFunc<void()>;

  /*! \brief Virtual destructor */
  virtual ~SearchStrategyNode() = default;

  /*! \brief Initialize the search strategy with TuneContext */
  virtual void InitializeWithTuneContext(const TuneContext& context) = 0;

  /*!
   * \brief Generate the candidates from design space in tune context for measurement
   * \return The next batch of candidates for measurements generated from the design space
   */
  virtual Optional<runtime::Array<BuildInput>> GenerateMeasureCandidates() = 0;

  /*!
   * \brief Update the search strategy with meansurement results from the runners
   * \param results The measurement results of candidates generated from the search strategy
   */
  virtual void NotifyMeasureResults(const Array<MeasureResult>& results) = 0;

  virtual void PreTuning(const Array<Schedule>& design_spaces) = 0;

  virtual void PostTuning() = 0;

  static constexpr const char* _type_key = "meta_schedule.SearchStrategy";
  TVM_DECLARE_BASE_OBJECT_INFO(SearchStrategyNode, Object);
};

/*!
 * \brief Managed reference to Search Strategy Generator Node
 * \sa SearchStrategyNode
 */
class SearchStrategy : public runtime::ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SearchStrategy, ObjectRef, SearchStrategyNode);
  static SearchStrategy PySearchStrategy(
      SearchStrategyNode::FInitializeWithTuneContext initialize_with_tune_context_func,
      SearchStrategyNode::FGenerateMeasureCandidates generate_measure_candidates_func,
      SearchStrategyNode::FNotifyMeasureResults notify_measure_results_func,
      SearchStrategyNode::FPreTuning pre_tuning_func,
      SearchStrategyNode::FPostTuning post_tuning_func);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SEARCH_STRATEGY_H_
