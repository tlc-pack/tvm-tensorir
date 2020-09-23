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
#ifndef SRC_META_SCHEDULE_SEARCH_H_
#define SRC_META_SCHEDULE_SEARCH_H_

#include <vector>

#include "./measure.h"
#include "./schedule.h"

namespace tvm {
namespace meta_schedule {

/********** SearchSpace **********/

/*!
 * \brief Description and abstraction of a search space.
 * The search space could be specified by manually written schedule function,
 * generated via loop analysis, ansor-like rules that apply to each block, etc.
 */
class SearchSpaceNode : public runtime::Object {
 public:
  /*! \brief Virtual destructor */
  virtual ~SearchSpaceNode() = default;
  /*!
   * \brief Sample a schedule out of the search space
   * \param task The search task to be sampled from
   * \return The schedule sampled
   */
  virtual Schedule SampleSchedule(const SearchTask& task) = 0;
  /*!
   * \brief Get support of the search space
   * \param task The search task to be sampled from
   * \return The support of the search space. Any point from the search space should along to one of
   * the traces returned
   */
  virtual Array<Schedule> GetSupport(const SearchTask& task) = 0;

  static constexpr const char* _type_key = "meta_schedule.SearchSpace";
  TVM_DECLARE_BASE_OBJECT_INFO(SearchSpaceNode, Object);
};

/*!
 * \brief Managed reference to SearchSpaceNode
 * \sa SearchSpaceNode
 */
class SearchSpace : public runtime::ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SearchSpace, ObjectRef, SearchSpaceNode);
};

/********** SearchStrategy **********/

/*!
 * \brief The search strategy for exploring the search space.
 * It could be always replay the sampling function, or saving several traces
 * from the sample function and then do lightweight-metropolis-hastings, or integrate those with
 * evolutionary search, etc.
 */
class SearchStrategyNode : public Object {
 public:
  /*! \brief Virtual destructor */
  virtual ~SearchStrategyNode() = default;
  /*!
   * \brief Explore the search space and find the best schedule
   * \param task The search task
   * \param space The search space
   * \param measurer The measurer that builds, runs and profiles sampled programs
   * \param verbose Whether or not in verbose mode
   * \return The best schedule found, NullOpt if no valid schedule is found
   */
  virtual Optional<Schedule> Search(const SearchTask& task, const SearchSpace& space,
                                    const ProgramMeasurer& measurer, int verbose) = 0;

  static constexpr const char* _type_key = "meta_schedule.SearchStrategy";
  TVM_DECLARE_BASE_OBJECT_INFO(SearchStrategyNode, Object);
};

/*!
 * \brief Managed reference to SearchStrategyNode
 * \sa SearchStrategyNode
 */
class SearchStrategy : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SearchStrategy, ObjectRef, SearchStrategyNode);
};

/********** RulePackedArgs **********/

/*!
 * \brief Input/output arguments of a SearchRule
 * \sa SearchRule
 * \sa SearchRuleNode
 */
class RulePackedArgsNode : public Object {
 public:
  /*! \brief The arguments the rule should apply to */
  Array<Schedule> proceed;
  /*! \brief The arguments the rule should skip */
  Array<Schedule> skipped;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("proceed", &proceed);
    v->Visit("skipped", &skipped);
  }

  static constexpr const char* _type_key = "meta_schedule.RulePackedArgs";
  TVM_DECLARE_FINAL_OBJECT_INFO(RulePackedArgsNode, Object);
};

/*!
 * \brief Managed reference to RulePackedArgsNode
 * \sa RulePackedArgs
 */
class RulePackedArgs : public ObjectRef {
 public:
  /*! \brief Constructing the packed args using a single schedule */
  explicit RulePackedArgs(Schedule schedule);
  /*!
   * \brief Constructor
   * \param proceed The arguments the rule should apply to
   * \param skipped The arguments the rule should skip
   */
  explicit RulePackedArgs(Array<Schedule> proceed, Array<Schedule> skipped);

  TVM_DEFINE_OBJECT_REF_METHODS(RulePackedArgs, ObjectRef, RulePackedArgsNode);
};

/********** SearchRule **********/

/*!
 * \brief A rule that applies to a block and generates a snippet of schedule on it
 */
class SearchRuleNode : public Object {
 public:
  using FApply = runtime::TypedPackedFunc<RulePackedArgs(Schedule, BlockRV)>;
  /*! \brief Name of the rule */
  String name;
  /*! \brief A packed function that applies the rule */
  FApply apply_;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("name", &name); }
  /*!
   * \brief Apply the rule with a single schedule
   * \param schedule Where the schedule snippets should be generated
   * \param block The block the rule applies on
   */
  RulePackedArgs Apply(Schedule schedule, BlockRV block) const;
  /*!
   * \brief Apply the rule with a list of schedules
   * \param schedules Where the schedule snippets should be generated
   * \param block The block the rule applies on
   */
  RulePackedArgs Apply(RulePackedArgs schedules, BlockRV block) const;

  static constexpr const char* _type_key = "meta_schedule.SearchRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(SearchRuleNode, Object);
};

/*!
 * \brief Managed reference to SearchRuleNode
 * \sa SearchRuleNode
 */
class SearchRule : public ObjectRef {
 public:
  /*!
   * \brief Constructing with name and a packed function
   * \param name Name of the search rule
   * \param apply The application function
   */
  explicit SearchRule(String name, runtime::PackedFunc apply);
  /*!
   * \brief Constructing with name and a typed packed function
   * \param name Name of the search rule
   * \param apply The application function
   */
  explicit SearchRule(String name, SearchRuleNode::FApply apply);
  /*!
   * \brief Composing search rules sequentially into a single rule
   * \param name Name of the new composite search rule
   * \param rules The rules provided sequentially
   */
  TVM_DLL static SearchRule Compose(const String& name, const std::vector<SearchRule>& rules);

  TVM_DEFINE_OBJECT_REF_METHODS(SearchRule, ObjectRef, SearchRuleNode);
};

/********** Search **********/

/*!
 * \brief The entry function for auto tuning
 * \param task The search task
 * \param space The search space
 * \param strategy The search strategy
 * \param builder Program builder used to run TIR build process
 * \param runner Program runner used to run the TIR profiling process, or interact with RPC tracker
 * \param measure_callbacks The callbacks to be triggered after each batch of meansuring
 * \param verbose Flag for the verbose mode
 * \return The best schedule found, NullOpt if no valid schedule is found in the search space
 */
TVM_DLL Optional<Schedule> AutoTune(SearchTask task, SearchSpace space, SearchStrategy strategy,
                                    ProgramBuilder builder, ProgramRunner runner,
                                    Array<MeasureCallback> measure_callbacks, int verbose);

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SEARCH_H_
