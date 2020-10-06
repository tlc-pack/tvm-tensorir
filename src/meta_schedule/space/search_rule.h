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

#ifndef SRC_META_SCHEDULE_SPACE_SEARCH_RULE_H_
#define SRC_META_SCHEDULE_SPACE_SEARCH_RULE_H_

#include <vector>

#include "../schedule.h"
#include "../search.h"

namespace tvm {
namespace meta_schedule {

/********** SearchRule **********/

/*!
 * \brief A rule that applies to a block and generates a snippet of schedule on it.
 * the SearchRule API is designed with the following signature:
 *
 *     (task: SearchTask, sch: Schedule, block: BlockRV, info: Dict[str, Any])
 *        -> Dict[Schedule, Any]
 *
 * \note The input schedule becomes invalid after calling SearchRule API, because
 * it is possible to be mutated.
 */
class SearchRuleNode : public Object {
 public:
  /*! \brief Dictionary holding context-related information */
  using TContextInfo = Optional<Map<String, ObjectRef>>;
  /*! \brief Return type of SearchRule */
  using TReturn = Map<Schedule, TContextInfo>;
  /*! \brief The SearchRule application function */
  using FApply = runtime::TypedPackedFunc<TReturn(SearchTask, Schedule, BlockRV, TContextInfo)>;

  /*! \brief Name of the rule */
  String name;
  /*! \brief An std::function that applies the rule */
  FApply apply_;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("name", &name); }

  /*!
   * \brief Apply the rule with a schedule with contexts
   * \param task The search task
   * \param sch The schedule that the context info is attached to
   * \param block The block the rule applies on
   * \param info The information about the context the rule is in
   * \return A schedule-context mapping
   */
  TReturn Apply(const SearchTask& task, const Schedule& sch, const BlockRV& block,
                const TContextInfo& info) const;

  static constexpr const char* _type_key = "meta_schedule.SearchRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(SearchRuleNode, Object);
};

/*!
 * \brief Managed reference to SearchRuleNode
 * \sa SearchRuleNode
 */
class SearchRule : public ObjectRef {
 public:
  using TContextInfo = SearchRuleNode::TContextInfo;
  using TReturn = SearchRuleNode::TReturn;
  using FApply = SearchRuleNode::FApply;

  /*!
   * \brief Constructing with name and an std::function
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
  TVM_DLL static SearchRule Compose(const String& name, std::vector<SearchRule> rules);

  TVM_DEFINE_OBJECT_REF_METHODS(SearchRule, ObjectRef, SearchRuleNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SPACE_SEARCH_RULE_H_
