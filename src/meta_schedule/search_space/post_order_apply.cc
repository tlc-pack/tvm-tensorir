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
#include <tvm/tir/stmt_functor.h>

#include "../measure.h"
#include "../search.h"

namespace tvm {
namespace meta_schedule {

/********** PostOrderApply **********/

/*! \brief Search space that is specified by applying rules in post-DFS order */
class PostOrderApplyNode : public SearchSpaceNode {
 public:
  /*! \brief The rule to be applied */
  SearchRule rule;
  /*! \brief The source of randomness */
  Sampler sampler_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("rule", &rule);
    // sampler_ is not visited
  }
  /*! \brief Default destructor */
  ~PostOrderApplyNode() = default;
  /*!
   * \brief Sample a schedule out of the search space
   * \param task The search task to be sampled from
   * \return The schedule sampled
   */
  Schedule SampleSchedule(const SearchTask& task) override;
  /*!
   * \brief Get support of the search space
   * \param task The search task to be sampled from
   * \return An array with a single element returned from SampleSchedule
   * \sa ScheduleFnNode::SampleSchedule
   */
  Array<Schedule> GetSupport(const SearchTask& task) override;
  /*!
   * \brief Apply the rule on the subtree rooted at the given block
   * \param block The block to be visited
   * \param sch The schedule snippet
   * \param is_root If the given block is the root of the PrimFunc
   * \return A list of schedules generated
   */
  Array<Schedule> VisitBlock(const tir::Block& block, const Schedule& sch, bool is_root);

  static constexpr const char* _type_key = "meta_schedule.PostOrderApply";
  TVM_DECLARE_FINAL_OBJECT_INFO(PostOrderApplyNode, SearchSpaceNode);
};

/*!
 * \brief Managed reference to PostOrderApplyNode
 * \sa PostOrderApplyNode
 */
class PostOrderApply : public SearchSpace {
 public:
  /*!
   * \brief Constructor
   * \param rule The rule to be applied
   */
  explicit PostOrderApply(SearchRule rule);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PostOrderApply, SearchSpace, PostOrderApplyNode);
};

/********** Constructor **********/

PostOrderApply::PostOrderApply(SearchRule rule) {
  ObjectPtr<PostOrderApplyNode> n = make_object<PostOrderApplyNode>();
  n->rule = std::move(rule);
  n->sampler_ = Sampler();
  data_ = std::move(n);
}

/********** Sampling **********/

Schedule PostOrderApplyNode::SampleSchedule(const SearchTask& task) {
  Array<Schedule> support = GetSupport(task);
  CHECK(!support.empty()) << "ValueError: Found null support";
  int i = sampler_.SampleInt(0, support.size());
  return support[i];
}

Array<Schedule> PostOrderApplyNode::GetSupport(const SearchTask& task) {
  const auto* block_realize = task->func->body.as<tir::BlockRealizeNode>();
  CHECK(block_realize != nullptr) << "TypeError: PrimFunc should root at BlockRealize, but gets: "
                                  << task->func->body->GetTypeKey();
  return VisitBlock(
      /*block=*/block_realize->block,
      /*sch=*/Schedule(task->func),
      /*is_root=*/true);
}

Array<Schedule> PostOrderApplyNode::VisitBlock(const tir::Block& block, const Schedule& sch,
                                               bool is_root) {
  Array<tir::Block> children;
  // Collect children
  tir::PreOrderVisit(block, [&children](const ObjectRef& node) {
    if (const auto* block_realize = node.as<tir::BlockRealizeNode>()) {
      children.push_back(block_realize->block);
      return false;
    }
    return true;
  });
  // Visit each children
  Array<Schedule> schedules{sch};
  for (const tir::Block& child : children) {
    Array<Schedule> new_schedules;
    for (const Schedule& sch : schedules) {
      // if this child still exists
      if (sch->sch->stmt2ref.count(child.get())) {
        Array<Schedule> result = VisitBlock(child, sch, false);
        new_schedules.insert(new_schedules.end(), result.begin(), result.end());
      } else {
        new_schedules.push_back(sch);
      }
    }
    schedules = new_schedules;
  }
  // Visit itself
  if (!is_root) {
    Array<Schedule> new_schedules;
    for (const Schedule& sch : schedules) {
      RulePackedArgs result = rule->Apply(sch, sch->GetBlock(block->tag));
      new_schedules.insert(new_schedules.end(), result->proceed.begin(), result->proceed.end());
      new_schedules.insert(new_schedules.end(), result->skipped.begin(), result->skipped.end());
    }
    schedules = new_schedules;
  }
  return schedules;
}

/********** FFI **********/

struct Internal {
  /*!
   * \brief Constructor of PostOrderApply
   * \param rule The rule to be applied
   * \return The PostOrderApply constructed
   * \sa PostOrderApply::PostOrderApply
   */
  static PostOrderApply New(SearchRule rule) { return PostOrderApply(rule); }
};

TVM_REGISTER_NODE_TYPE(PostOrderApplyNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostOrderApply").set_body_typed(Internal::New);

}  // namespace meta_schedule
}  // namespace tvm
