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

#ifndef SRC_META_SCHEDULE_SEARCH_SPACE_SEARCH_RULE_H_
#define SRC_META_SCHEDULE_SEARCH_SPACE_SEARCH_RULE_H_

#include <vector>

#include "../schedule.h"

namespace tvm {
namespace meta_schedule {

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
   * \return The composite rule
   */
  TVM_DLL static SearchRule Compose(const String& name, const std::vector<SearchRule>& rules);

  TVM_DEFINE_OBJECT_REF_METHODS(SearchRule, ObjectRef, SearchRuleNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SEARCH_SPACE_SEARCH_RULE_H_
