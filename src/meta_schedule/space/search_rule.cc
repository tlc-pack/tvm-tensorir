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

/********** Always-Inline **********/

/*! \brief Create a rule that inlines all possible blocks */
class AlwaysInline {
 public:
  /*! \brief Default constructor */
  AlwaysInline() = default;

  /*! \brief Rule application */
  RulePackedArgs operator()(Schedule sch, BlockRV block_rv) {
    static const Op& op_exp = Op::Get("tir.exp");
    if (HasReduceBlockVar(sch, block_rv) || IsOutputBlock(sch, block_rv)) {
      return RulePackedArgs(sch);
    }
    if (HasBranch(sch, block_rv) || CountOp(sch, block_rv, op_exp)) {
      return RulePackedArgs(sch);
    }
    if (Optional<Array<Bool>> access = InspectLoadIndices(sch, block_rv)) {
      CHECK_EQ(access.value().size(), 3);
      bool injective = access.value()[1];
      bool order = access.value()[2];
      if (!order || !injective) {
        return RulePackedArgs(sch);
      }
    } else {
      return RulePackedArgs(sch);
    }
    sch->ComputeInline(block_rv);
    return RulePackedArgs(/*proceed=*/{}, /*ignored=*/{sch});
  }

  /*! \brief Rule creator */
  static SearchRule MakeRule() {
    auto invoke = [](Schedule sch, BlockRV block) -> RulePackedArgs {
      AlwaysInline rule;
      return rule(sch, block);
    };
    return SearchRule("always_inline", invoke);
  }
};

/********** Add-Cache-Write **********/

/*! \brief Create a rule that adds a cache write stage after multi-level tiling */
class AddCacheWrite {
 public:
  /*! \brief Default constructor */
  AddCacheWrite() = default;

  /*! \brief Rule application */
  RulePackedArgs operator()(Schedule sch, BlockRV block_rv) {
    if (!NeedsMultiLevelTiling(sch, block_rv)) {
      return RulePackedArgs(sch);
    }
    // The only consumer will not be fused
    if (Optional<BlockRV> opt_consumer_rv = sch->GetOnlyConsumer(block_rv)) {
      BlockRV consumer_rv = opt_consumer_rv.value();
      if (!HasReduceBlockVar(sch, block_rv) || !HasReduceBlockVar(sch, consumer_rv)) {
        if (IsElementWiseMatch(sch, block_rv, consumer_rv)) {
          return RulePackedArgs(sch);
        }
      }
    }
    // Add a cache write
    sch->CacheWrite(block_rv, "local");
    return RulePackedArgs(/*proceed=*/{}, /*ignored=*/{sch});
  }

  /*! \brief Rule creator */
  static SearchRule MakeRule() {
    auto invoke = [](Schedule sch, BlockRV block) -> RulePackedArgs {
      AddCacheWrite rule;
      return rule(sch, block);
    };
    return SearchRule("multi_level_tiling", invoke);
  }
};

/********** Multi-Level-Tiling-With-Fusion **********/

/*!
 * \brief Create a rule that does multi-level tiling and fusion together if there is sufficient
 * amount of data reuse
 */
class MultiLevelTilingWithFusion {
 public:
  /*! \brief The structure of tiling, e.g. "SSRSRS" on CPU */
  String tiling_structure;

  /*!
   * \brief Constructor
   * \param tiling_structure The structure of tiling
   */
  explicit MultiLevelTilingWithFusion(String tiling_structure)
      : tiling_structure(std::move(tiling_structure)) {}

  /*! \brief Rule application */
  RulePackedArgs operator()(Schedule sch, BlockRV block_rv) {
    // Rule out the possibility that it does not need multi-level tiling
    if (!NeedsMultiLevelTiling(sch, block_rv)) {
      return RulePackedArgs(sch);
    }
    // Get the only consumer
    Optional<BlockRV> opt_consumer_rv = sch->GetOnlyConsumer(block_rv);
    if (!opt_consumer_rv.defined()) {
      return RulePackedArgs(sch);
    }
    // Check elementwise-match
    BlockRV consumer_rv = opt_consumer_rv.value();
    if (HasReduceBlockVar(sch, block_rv) && HasReduceBlockVar(sch, consumer_rv)) {
      return RulePackedArgs(sch);
    }
    if (!IsElementWiseMatch(sch, block_rv, consumer_rv)) {
      return RulePackedArgs(sch);
    }
    DoMultiLevelTiling(sch, block_rv, tiling_structure);
    LOG(INFO) << "We can do multi-level-tiling with fusion!";
    // TODO(@junrushao1994): add fusion
    return RulePackedArgs(/*proceed=*/{}, /*ignored=*/{sch});
  }

  /*! \brief Rule creator */
  static SearchRule MakeRule(String tiling_structure) {
    auto invoke = [tiling_structure](Schedule sch, BlockRV block) -> RulePackedArgs {
      MultiLevelTilingWithFusion rule(tiling_structure);
      return rule(sch, block);
    };
    return SearchRule("multi_level_tiling_with_fusion", invoke);
  }
};

/********** Multi-Level-Tiling **********/

/*! \brief Create a rule that does multi-level tiling if there is sufficient amount of data reuse */
class MultiLevelTiling {
 public:
  /*! \brief The structure of tiling, e.g. "SSRSRS" on CPU */
  String tiling_structure;

  /*!
   * \brief Constructor
   * \param tiling_structure The structure of tiling
   */
  explicit MultiLevelTiling(String tiling_structure)
      : tiling_structure(std::move(tiling_structure)) {}

  /*! \brief Rule application */
  RulePackedArgs operator()(Schedule sch, BlockRV block_rv) {
    // Right now it only works with a leaf block with a single statement
    if (NeedsMultiLevelTiling(sch, block_rv)) {
      DoMultiLevelTiling(sch, block_rv, tiling_structure);
      return RulePackedArgs(/*proceed=*/{}, /*ignored=*/{sch});
    }
    return RulePackedArgs(sch);
  }

  /*! \brief Rule creator */
  static SearchRule MakeRule(String tiling_structure) {
    auto invoke = [tiling_structure](Schedule sch, BlockRV block) -> RulePackedArgs {
      MultiLevelTiling rule(tiling_structure);
      return rule(sch, block);
    };
    return SearchRule("multi_level_tiling", invoke);
  }
};

/********** FFI **********/

struct Internal {
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
TVM_REGISTER_GLOBAL("meta_schedule.RulePackedArgs").set_body_typed(Internal::RulePackedArgsNew);
TVM_REGISTER_GLOBAL("meta_schedule.SearchRule").set_body_typed(Internal::SearchRuleNew);
TVM_REGISTER_GLOBAL("meta_schedule.SearchRuleCall").set_body_typed(Internal::SearchRuleCall);
TVM_REGISTER_GLOBAL("meta_schedule.SearchRuleCompose").set_body_typed(Internal::Compose);
TVM_REGISTER_GLOBAL("meta_schedule.rule.AlwaysInline").set_body_typed(AlwaysInline::MakeRule);
TVM_REGISTER_GLOBAL("meta_schedule.rule.AddCacheWrite").set_body_typed(AddCacheWrite::MakeRule);
TVM_REGISTER_GLOBAL("meta_schedule.rule.MultiLevelTilingWithFusion")
    .set_body_typed(MultiLevelTilingWithFusion::MakeRule);
TVM_REGISTER_GLOBAL("meta_schedule.rule.MultiLevelTiling")
    .set_body_typed(MultiLevelTiling::MakeRule);

}  // namespace meta_schedule
}  // namespace tvm
