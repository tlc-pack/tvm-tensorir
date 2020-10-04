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

#include "./search.h"  // NOLINT(build/include)

#include "./measure.h"

namespace tvm {
namespace meta_schedule {

using runtime::PackedFunc;

/********** Constructors **********/

RulePackedArgs::RulePackedArgs(Schedule schedule) : RulePackedArgs({schedule}, {}) {}

RulePackedArgs::RulePackedArgs(Array<Schedule> proceed, Array<Schedule> skipped) {
  ObjectPtr<RulePackedArgsNode> n = make_object<RulePackedArgsNode>();
  n->proceed = std::move(proceed);
  n->skipped = std::move(skipped);
  data_ = std::move(n);
}

SearchRule::SearchRule(String name, PackedFunc apply) {
  ObjectPtr<SearchRuleNode> n = make_object<SearchRuleNode>();
  n->name = std::move(name);
  n->apply_ = apply;
  data_ = std::move(n);
}

SearchRule::SearchRule(String name, SearchRuleNode::FApply apply) {
  ObjectPtr<SearchRuleNode> n = make_object<SearchRuleNode>();
  n->name = std::move(name);
  n->apply_ = std::move(apply);
  data_ = std::move(n);
}

/********** SearchRule **********/

RulePackedArgs SearchRuleNode::Apply(Schedule schedule, BlockRV block) const {
  return Apply(RulePackedArgs(schedule), block);
}

RulePackedArgs SearchRuleNode::Apply(RulePackedArgs schedules, BlockRV block) const {
  Array<Schedule> skipped = schedules->skipped;
  Array<Schedule> proceed;
  for (const Schedule& sch : schedules->proceed) {
    RulePackedArgs results = apply_(sch, block);
    proceed.insert(proceed.end(), results->proceed.begin(), results->proceed.end());
    skipped.insert(skipped.end(), results->skipped.begin(), results->skipped.end());
  }
  return RulePackedArgs(proceed, skipped);
}

SearchRule SearchRule::Compose(const String& name, const std::vector<SearchRule>& rules) {
  auto apply = [rules](Schedule schedule, BlockRV block) -> RulePackedArgs {
    RulePackedArgs results(schedule);
    for (const SearchRule& rule : rules) {
      results = rule->Apply(results, block);
    }
    return results;
  };
  return SearchRule(name, SearchRuleNode::FApply(apply));
}

/********** Search **********/

Optional<Schedule> AutoTune(SearchTask task, SearchSpace space, SearchStrategy strategy,
                            ProgramBuilder builder, ProgramRunner runner,
                            Array<MeasureCallback> measure_callbacks, int verbose) {
  return strategy->Search(task, space, ProgramMeasurer(builder, runner, measure_callbacks),
                          verbose);
}

/********** FFI **********/

struct Internal {
  /*!
   * \brief Sample a schedule out of the search space, calls SearchSpaceNode::SampleSchedule
   * \param space The specific space
   * \param task The search task to be sampled from
   * \return The schedule sampled
   * \sa SearchSpaceNode::SampleSchedule
   */
  static Schedule SearchSpaceSampleSchedule(SearchSpace space, SearchTask task) {
    return space->SampleSchedule(task);
  }
  /*!
   * \brief Get support of the search space, calls SearchSpaceNode::GetSupport
   * \param space The specific space
   * \param task The search task to be sampled from
   * \return The support of the search space. Any point from the search space should along to one of
   * the traces returned
   * \sa SearchSpaceNode::GetSupport
   */
  static Array<Schedule> SearchSpaceGetSupport(SearchSpace space, SearchTask task) {
    return space->GetSupport(task);
  }
  /*!
   * \brief Explore the search space and find the best schedule
   * \param strategy The strategy to explore the search space
   * \param task The search task
   * \param space The search space
   * \param measurer The measurer that builds, runs and profiles sampled programs
   * \param verbose Whether or not in verbose mode
   * \return The best schedule found, NullOpt if no valid schedule is found
   */
  static Optional<Schedule> SearchStrategySearch(SearchStrategy strategy, SearchTask task,
                                                 SearchSpace space, ProgramMeasurer measurer,
                                                 int verbose) {
    return strategy->Search(task, space, measurer, verbose);
  }
  /*!
   * \brief Constructor of RulePackedArgs
   * \param proceed The arguments the rule should apply to
   * \param skipped The arguments the rule should skip
   * \sa RulePackedArgs::RulePackedArgs
   */
  static RulePackedArgs RulePackedArgsNew(Array<Schedule> proceed, Array<Schedule> skipped) {
    return RulePackedArgs(proceed, skipped);
  }
  /*!
   * \brief Constructor of SearchRule
   * \param name Name of the search rule
   * \param apply The application function
   * \sa SearchRule::SearchRule
   */
  static SearchRule SearchRuleNew(String name, PackedFunc apply) { return SearchRule(name, apply); }
  /*!
   * \brief Apply the rule with a single schedule
   * \param rule The search rule to be called
   * \param schedule Where the schedule snippets should be generated
   * \param block The block the rule applies on
   * \sa SearchRuleNode::Apply
   */
  static RulePackedArgs SearchRuleCall(SearchRule rule, Schedule sch, BlockRV block) {
    return rule->Apply(sch, block);
  }
  /*!
   * \brief Composing search rules sequentially into a single rule
   * \param name Name of the new composite search rule
   * \param rules The rules provided sequentially
   * \return The composite rule
   * \sa SearchRule::Compose
   */
  static SearchRule Compose(String name, Array<SearchRule> rules) {
    return SearchRule::Compose(name, {rules.begin(), rules.end()});
  }
};

TVM_REGISTER_NODE_TYPE(RulePackedArgsNode);
TVM_REGISTER_NODE_TYPE(SearchRuleNode);
TVM_REGISTER_NODE_TYPE(SearchTaskNode);
TVM_REGISTER_OBJECT_TYPE(SearchSpaceNode);
TVM_REGISTER_OBJECT_TYPE(SearchStrategyNode);

TVM_REGISTER_GLOBAL("meta_schedule.SearchSpaceSampleSchedule")
    .set_body_typed(Internal::SearchSpaceSampleSchedule);
TVM_REGISTER_GLOBAL("meta_schedule.SearchSpaceGetSupport")
    .set_body_typed(Internal::SearchSpaceGetSupport);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategySearch")
    .set_body_typed(Internal::SearchStrategySearch);
TVM_REGISTER_GLOBAL("meta_schedule.RulePackedArgs").set_body_typed(Internal::RulePackedArgsNew);
TVM_REGISTER_GLOBAL("meta_schedule.SearchRule").set_body_typed(Internal::SearchRuleNew);
TVM_REGISTER_GLOBAL("meta_schedule.SearchRuleCall").set_body_typed(Internal::SearchRuleCall);
TVM_REGISTER_GLOBAL("meta_schedule.SearchRuleCompose").set_body_typed(Internal::Compose);
TVM_REGISTER_GLOBAL("meta_schedule.AutoTune").set_body_typed(AutoTune);

}  // namespace meta_schedule
}  // namespace tvm
