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
 *     (task: SearchTask, sch: Schedule, block: BlockRV)
 *        -> Dict[Schedule, Any]
 *
 * \note The input schedule becomes invalid after calling SearchRule API, because
 * it is possible to be mutated.
 */
class SearchRuleNode : public Object {
 public:
  /*! \brief The SearchRule application function */
  using FApply = runtime::TypedPackedFunc<Array<Schedule>(SearchTask, Schedule, BlockRV)>;

  /*! \brief Name of the rule */
  String name;
  /*! \brief A packed function that applies the rule */
  FApply apply_;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("name", &name); }

  /*!
   * \brief Apply the rule with a schedule with contexts
   * \param task The search task
   * \param sch The schedule
   * \param block The block the rule applies on
   * \return A schedule-context mapping
   */
  Array<Schedule> Apply(const SearchTask& task, const Schedule& sch, const BlockRV& block) const;

  static constexpr const char* _type_key = "meta_schedule.SearchRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(SearchRuleNode, Object);
};

/*!
 * \brief Managed reference to SearchRuleNode
 * \sa SearchRuleNode
 */
class SearchRule : public ObjectRef {
 public:
  using FApply = SearchRuleNode::FApply;

  /*!
   * \brief Constructing with name and a packed function
   * \param name Name of the search rule
   * \param apply The application function
   */
  explicit SearchRule(String name, FApply apply);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SearchRule, ObjectRef, SearchRuleNode);
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
 * \param max_innermost_factor The maximum size of the innermost factor
 * \param must_cache_read Add cache_read before the multi-level tiling
 * \param can_cache_write Add cache_write after the multi-level tiling
 * \param must_cache_write Must add cache_write after the multi-level tiling
 * \param fusion_levels The possible tile levels that a single elementwise consumer is fused at
 * \param vector_load_max_len For cache_read, if vectorized load is used, the max length of the
 * vectorized load
 * \param tile_binds The marks to be used on each tile
 * \return The rule created
 */
TVM_DLL SearchRule MultiLevelTiling(String structure, int max_innermost_factor,
                                    bool must_cache_read, String cache_read_scope,
                                    bool can_cache_write, bool must_cache_write,
                                    String cache_write_scope, bool consumer_inline_strict,
                                    Array<Integer> fusion_levels,
                                    Optional<Integer> vector_load_max_len,
                                    Optional<Array<String>> tile_binds);

/*!
 * \brief Create a rule that does multi-level tiling if there is sufficient amount of data reuse.
 * Optionally add read cache and write cache, do fusion if possible
 * \param structure The tiling structure
 * \param max_innermost_factor The maximum size of the innermost factor
 * \param must_cache_read Add cache_read before the multi-level tiling
 * \param can_cache_write Add cache_write after the multi-level tiling
 * \param must_cache_write Must add cache_write after the multi-level tiling
 * \param fusion_levels The possible tile levels that a single elementwise consumer is fused at
 * \param compute_intrin The tensor intrinsinc for doing computation
 * \param vector_load_max_len For cache_read, if vectorized load is used, the max length of the
 * vectorized load
 * \param tile_binds The marks to be used on each tile
 * \return The rule created
 */
TVM_DLL SearchRule MultiLevelTilingWithTensorCore(
    String structure, int max_innermost_factor, bool must_cache_read, String cache_read_scope,
    bool can_cache_write, bool must_cache_write, String cache_write_scope,
    bool consumer_inline_strict, Array<Integer> fusion_levels, String compute_intrin,
    Optional<Integer> vector_load_max_len, Optional<Array<String>> tile_binds);

/*!
 * \brief A rule that randomly select a compute-at location for a free block
 * \return The rule created
 */
TVM_DLL SearchRule RandomComputeLocation();

/*!
 * \brief Mark parallelize, vectorize and unroll to each block correspondingly
 * \param max_jobs_per_core The maximum number of jobs to be launched per CPU core. It sets the
 * uplimit of CPU parallelism, i.e. `num_cores * max_jobs_per_core`. Use -1 to disable parallelism.
 * \param max_vectorize_extent The maximum extent to be vectorized. It sets the uplimit of the CPU
 * vectorization. Use -1 to disable vectorization.
 * \param unroll_max_steps The maximum number of unroll steps to be done. Use an empty array to
 * disable unroll
 * \param unroll_explicit Whether to explicitly unroll the loop, or just add a unroll pragma
 * \return The rule created
 */
TVM_DLL SearchRule ParallelizeVectorizeUnroll(int max_jobs_per_core, int max_vectorize_extent,
                                              Array<Integer> unroll_max_steps,
                                              bool unroll_explicit);

/*!
 * \brief Add rfactor to some blocks if needed
 * \return The rule created
 */
TVM_DLL SearchRule AddRFactor(int max_jobs_per_core, int max_innermost_factor);

/*!
 * \brief Handle special cases in Winograd transformation for GPU. We need to change the compute
 * location of the producers of compute ops that perform "fake reduction" with const tensors.
 * \return The rule created
 */
TVM_DLL SearchRule SpecialComputeLocationGPU();

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SPACE_SEARCH_RULE_H_
