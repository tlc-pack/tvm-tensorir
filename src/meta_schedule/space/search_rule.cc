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
#include "./search_rule.h"  // NOLINT(build/include)

#include "../analysis.h"
#include "../search.h"
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

using TContextInfo = SearchRuleNode::TContextInfo;
using TReturn = SearchRuleNode::TReturn;
using FApply = SearchRuleNode::FApply;

/********** Constructors **********/

SearchRule::SearchRule(String name, SearchRuleNode::FApply apply) {
  ObjectPtr<SearchRuleNode> n = make_object<SearchRuleNode>();
  n->name = std::move(name);
  n->apply_ = std::move(apply);
  data_ = std::move(n);
}

/********** SearchRule **********/

TReturn SearchRuleNode::Apply(const SearchTask& task, const Schedule& sch, const BlockRV& block,
                              const TContextInfo& info) const {
  if (sch->Eval(block)->stmt == nullptr) {
    return {{sch, info}};
  }
  return apply_(task, sch, block, info);
}

SearchRule SearchRuleCompose(const String& name, const Array<SearchRule>& rules) {
  auto apply = [rules](SearchTask task, Schedule sch, BlockRV block, TContextInfo info) -> TReturn {
    using ItemType = std::pair<Schedule, TContextInfo>;
    std::vector<ItemType> curr{{sch, info}};
    std::vector<ItemType> next;
    for (const SearchRule& rule : rules) {
      for (const ItemType& sch_info : curr) {
        TReturn results = rule->Apply(task, sch_info.first, block, sch_info.second);
        for (ItemType kv : results) {
          next.emplace_back(std::move(kv.first), std::move(kv.second));
        }
      }
      curr.clear();
      curr.swap(next);
    }
    return {curr.begin(), curr.end()};
  };
  return SearchRule(name, SearchRuleNode::FApply(apply));
}

/********** Always-Inline **********/

/*! \brief A rule that inlines all possible blocks */
class RuleAlwaysInline {
 public:
  /*! \brief Default constructor */
  RuleAlwaysInline() = default;

  /*! \brief Rule application */
  TReturn Apply(const SearchTask& task, const Schedule& sch, const BlockRV& block_rv,
                const TContextInfo& info) {
    static const Op& op_exp = Op::Get("tir.exp");
    if (HasReduceBlockVar(sch, block_rv) || IsOutputBlock(sch, block_rv)) {
      return {{sch, info}};
    }
    if (HasBranch(sch, block_rv) || CountOp(sch, block_rv, op_exp)) {
      return {{sch, info}};
    }
    if (Optional<Array<Bool>> access = InspectLoadIndices(sch, block_rv)) {
      CHECK_EQ(access.value().size(), 3);
      bool injective = access.value()[1];
      bool order = access.value()[2];
      if (!order || !injective) {
        return {{sch, info}};
      }
    } else {
      return {{sch, info}};
    }
    sch->ComputeInline(block_rv);
    return {{sch, info}};
  }
};

SearchRule AlwaysInline() {
  auto f_apply = [](SearchTask task, Schedule sch, BlockRV block, TContextInfo info) -> TReturn {
    RuleAlwaysInline rule;
    return rule.Apply(task, sch, block, info);
  };
  return SearchRule("always_inline", f_apply);
}

/********** Add-Cache-Write **********/

/*! \brief A rule that adds a cache write stage after multi-level tiling */
class RuleAddCacheWrite {
 public:
  /*! \brief Default constructor */
  RuleAddCacheWrite() = default;

  /*! \brief Rule application */
  TReturn Apply(const SearchTask& task, const Schedule& sch, const BlockRV& block_rv,
                const TContextInfo& info) {
    if (!NeedsMultiLevelTiling(sch, block_rv)) {
      return {{sch, info}};
    }
    // The only consumer will not be fused
    if (Optional<BlockRV> opt_consumer_rv = sch->GetOnlyConsumer(block_rv)) {
      BlockRV consumer_rv = opt_consumer_rv.value();
      if (!HasReduceBlockVar(sch, block_rv) || !HasReduceBlockVar(sch, consumer_rv)) {
        if (IsElementWiseMatch(sch, block_rv, consumer_rv)) {
          return {{sch, info}};
        }
      }
    }
    // Add a cache write
    sch->CacheWrite(block_rv, "local");
    return {{sch, info}};
  }
};

SearchRule AddCacheWrite() {
  auto f_apply = [](SearchTask task, Schedule sch, BlockRV block, TContextInfo info) -> TReturn {
    RuleAddCacheWrite rule;
    return rule.Apply(task, sch, block, info);
  };
  return SearchRule("multi_level_tiling", f_apply);
}

/********** Multi-Level-Tiling-With-Fusion **********/

/*!
 * \brief A rule that does multi-level tiling and fusion together if there is sufficient
 * amount of data reuse
 */
class RuleMultiLevelTilingWithFusion {
 public:
  /*! \brief The structure of tiling, e.g. "SSRSRS" on CPU */
  String tiling_structure;

  /*!
   * \brief Constructor
   * \param tiling_structure The structure of tiling
   */
  explicit RuleMultiLevelTilingWithFusion(String tiling_structure)
      : tiling_structure(std::move(tiling_structure)) {}

  /*! \brief Rule application */
  TReturn Apply(const SearchTask& task, const Schedule& sch, const BlockRV& block_rv,
                const TContextInfo& info) {
    // Rule out the possibility that it does not need multi-level tiling
    if (!NeedsMultiLevelTiling(sch, block_rv)) {
      return {{sch, info}};
    }
    // Get the only consumer
    Optional<BlockRV> opt_consumer_rv = sch->GetOnlyConsumer(block_rv);
    if (!opt_consumer_rv.defined()) {
      return {{sch, info}};
    }
    // Check elementwise-match
    BlockRV consumer_rv = opt_consumer_rv.value();
    if (HasReduceBlockVar(sch, block_rv) && HasReduceBlockVar(sch, consumer_rv)) {
      return {{sch, info}};
    }
    if (!IsElementWiseMatch(sch, block_rv, consumer_rv)) {
      return {{sch, info}};
    }
    DoMultiLevelTiling(sch, block_rv, tiling_structure);
    LOG(INFO) << "We can do multi-level-tiling with fusion!";
    // TODO(@junrushao1994): add fusion
    return {{sch, info}};
  }
};

SearchRule MultiLevelTilingWithFusion(String tiling_structure) {
  auto f_apply = [tiling_structure{std::move(tiling_structure)}](
                     SearchTask task, Schedule sch, BlockRV block, TContextInfo info) -> TReturn {
    RuleMultiLevelTilingWithFusion rule(tiling_structure);
    return rule.Apply(task, sch, block, info);
  };
  return SearchRule("multi_level_tiling_with_fusion", f_apply);
}

/********** Multi-Level-Tiling **********/

/*! \brief A rule that does multi-level tiling if there is sufficient amount of data reuse */
class RuleMultiLevelTiling {
 public:
  /*! \brief The structure of tiling, e.g. "SSRSRS" on CPU */
  String tiling_structure;

  /*!
   * \brief Constructor
   * \param tiling_structure The structure of tiling
   */
  explicit RuleMultiLevelTiling(String tiling_structure)
      : tiling_structure(std::move(tiling_structure)) {}

  /*! \brief Rule application */
  TReturn Apply(const SearchTask& task, const Schedule& sch, const BlockRV& block_rv,
                const TContextInfo& info) {
    // Right now it only works with a leaf block with a single statement
    if (NeedsMultiLevelTiling(sch, block_rv)) {
      DoMultiLevelTiling(sch, block_rv, tiling_structure);
      return {{sch, info}};
    }
    return {{sch, info}};
  }
};

SearchRule MultiLevelTiling(String tiling_structure) {
  auto f_apply = [tiling_structure{std::move(tiling_structure)}](
                     SearchTask task, Schedule sch, BlockRV block, TContextInfo info) -> TReturn {
    RuleMultiLevelTiling rule(tiling_structure);
    return rule.Apply(task, sch, block, info);
  };
  return SearchRule("multi_level_tiling", f_apply);
}

/********** FFI **********/

struct Internal {
  /*!
   * \brief Constructor of SearchRule
   * \param name Name of the search rule
   * \param apply The application function
   * \return The SearchRule created
   * \sa SearchRule::SearchRule
   */
  static SearchRule SearchRuleNew(String name, PackedFunc apply) { return SearchRule(name, apply); }
  /*!
   * \brief Apply the rule with a single schedule
   * \param rule The search rule to be called
   * \param task The search task
   * \param sch The schedule that the context info is attached to
   * \param block The block the rule applies on
   * \param info The information about the context the rule is in
   * \return The result of rule application
   * \sa SearchRuleNode::Apply
   */
  static TReturn SearchRuleApply(SearchRule rule, SearchTask task, Schedule sch, BlockRV block,
                                 TContextInfo info) {
    return rule->Apply(task, sch, block, info);
  }
  /*!
   * \brief Composing search rules sequentially into a single rule
   * \param name Name of the new composite search rule
   * \param rules The rules provided sequentially
   * \return The composite rule
   * \sa SearchRule::Compose
   */
  static SearchRule Compose(String name, Array<SearchRule> rules) {
    return SearchRuleCompose(name, rules);
  }
};

TVM_REGISTER_NODE_TYPE(SearchRuleNode);
TVM_REGISTER_GLOBAL("meta_schedule.SearchRule").set_body_typed(Internal::SearchRuleNew);
TVM_REGISTER_GLOBAL("meta_schedule.SearchRuleApply").set_body_typed(Internal::SearchRuleApply);
TVM_REGISTER_GLOBAL("meta_schedule.SearchRuleCompose").set_body_typed(SearchRuleCompose);
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.AlwaysInline").set_body_typed(AlwaysInline);
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.AddCacheWrite").set_body_typed(AddCacheWrite);
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.MultiLevelTilingWithFusion")
    .set_body_typed(MultiLevelTilingWithFusion);
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.MultiLevelTiling").set_body_typed(MultiLevelTiling);

}  // namespace meta_schedule
}  // namespace tvm
