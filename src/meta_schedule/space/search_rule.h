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
  /*! \brief A packed function that applies the rule */
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
   * \brief Constructing with name and a packed function
   * \param name Name of the search rule
   * \param apply The application function
   */
  explicit SearchRule(String name, FApply apply);

  TVM_DEFINE_OBJECT_REF_METHODS(SearchRule, ObjectRef, SearchRuleNode);
};

/*!
 * \brief Composing search rules sequentially into a single rule
 * \param name Name of the new composite search rule
 * \param rules The rules provided sequentially
 * \return The composite rule
 */
TVM_DLL SearchRule SearchRuleCompose(const String& name, const Array<SearchRule>& rules);

/********** Built-in SearchRules **********/

/*!
 * \brief Create a rule that inlines all possible pure spatial block
 * \param strict_mode Requires the block to be strictly inlineable
 * \return The rule created
 */
TVM_DLL SearchRule InlinePureSpatial(bool strict_mode);

/*!
 * \brief Create a rule that does multi-level tiling if there is sufficient amount of data reuse.
 * Optionally add read cache and write cache, do fusion if possible
 * \param structure The tiling structure
 * \param must_cache_read Add cache_read before the multi-level tiling
 * \param can_cache_write Add cache_write after the multi-level tiling
 * \param must_cache_write Must add cache_write after the multi-level tiling
 * \param fusion_levels The possible tile levels that a single elementwise consumer is fused at
 * \param vector_load_max_len For cache_read, if vectorized load is used, the max length of the
 * vectorized load
 * \param tile_marks The marks to be used on each tile
 * \return The rule created
 */
TVM_DLL SearchRule MultiLevelTilingAndFusion(String structure, bool must_cache_read,
                                             String cache_read_scope, bool can_cache_write,
                                             bool must_cache_write, String cache_write_scope,
                                             Array<Integer> fusion_levels,
                                             Optional<Integer> vector_load_max_len,
                                             Optional<Array<String>> tile_marks);

/*!
 * \brief A rule that parallelizes the outer loops
 * \return The rule created
 */
TVM_DLL SearchRule MarkParallelizeOuter(int max_jobs_per_core);

/*!
 * \brief A rule that parallelizes the outer loops
 * \return The rule created
 */
TVM_DLL SearchRule MarkVectorizeInner(int max_extent);

/*!
 * \brief Rewrite block and its surrounding loops to match the tensor intrinsics if possible
 * \param tensor_intrins The tensor intrinsics to be matched
 * \return The rule created
 */
TVM_DLL SearchRule MarkTensorize(Array<tir::TensorIntrin> tensor_intrins);

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SPACE_SEARCH_RULE_H_
