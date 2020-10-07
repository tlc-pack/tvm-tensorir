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
    tir::StmtSRef block_sref = sch->Eval(block_rv);
    if (IsSpatial(sch->sch, block_sref) || IsOutputBlock(sch->sch, block_sref)) {
      return {{sch, info}};
    }
    if (IsStrictlyInlineable(sch->sch, block_sref)) {
      sch->ComputeInline(block_rv);
      return {{sch, info}};
    }
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
    tir::StmtSRef block = sch->Eval(block_rv);
    if (!NeedsMultiLevelTiling(sch->sch, block)) {
      return {{sch, info}};
    }
    // The only consumer will not be fused
    if (Optional<BlockRV> opt_consumer_rv = sch->GetOnlyConsumer(block_rv)) {
      BlockRV consumer_rv = opt_consumer_rv.value();
      if (!IsSpatial(sch->sch, block) && !IsSpatial(sch->sch, block)) {
        return {{sch, info}};
      }
      if (IsElementWiseMatch(sch->sch, block, sch->Eval(consumer_rv))) {
        // Add a cache write
        sch->CacheWrite(block_rv, "local");
        return {{sch, info}};
      }
    }
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

void DoMultiLevelTiling(Schedule sch, BlockRV block_rv, String tiling_structure) {
  // Do the multi-level tiling
  std::vector<int> s_idx = FindCharPos(tiling_structure, 'S');
  std::vector<int> r_idx = FindCharPos(tiling_structure, 'R');
  std::vector<std::vector<LoopRV>> order(tiling_structure.size());
  Array<LoopRV> axes = sch->GetAxes(block_rv);
  Array<Integer> iter_types = GetBlockVarTypes(sch->sch, sch->Eval(block_rv));
  CHECK_EQ(axes.size(), iter_types.size());
  int n = axes.size();
  for (int i = 0; i < n; ++i) {
    std::vector<int>* idx = nullptr;
    if (iter_types[i] == tir::IterVarType::kDataPar) {
      idx = &s_idx;
    } else if (iter_types[i] == tir::IterVarType::kCommReduce) {
      idx = &r_idx;
    } else {
      continue;
    }
    int n_tiles = idx->size();
    Array<tir::Var> factors = sch->SamplePerfectTile(/*n=*/n_tiles, /*loop=*/axes[i]);
    Array<LoopRV> splits =
        sch->Split(/*loop=*/axes[i], /*factors=*/{factors.begin(), factors.end()});
    for (int j = 0; j < n_tiles; ++j) {
      order[idx->at(j)].push_back(splits[j]);
    }
  }
  sch->Reorder(ConcatArray(order));
}

/*!
 * \brief A rule that does multi-level tiling and fusion together if there is sufficient
 * amount of data reuse
 */
class RuleMultiLevelTilingWithFusion {
 public:
  /*! \brief The structure of tiling, e.g. "SSRSRS" on CPU, or "SSSRRSRS" on GPU */
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
    tir::StmtSRef block_sref = sch->Eval(block_rv);
    if (!NeedsMultiLevelTiling(sch->sch, block_sref)) {
      return {{sch, info}};
    }
    // Get the only consumer
    Optional<BlockRV> opt_consumer_rv = sch->GetOnlyConsumer(block_rv);
    if (!opt_consumer_rv.defined()) {
      return {{sch, info}};
    }
    // Check elementwise-match
    BlockRV consumer_rv = opt_consumer_rv.value();
    if (!IsSpatial(sch->sch, block_sref) && !IsSpatial(sch->sch, block_sref)) {
      return {{sch, info}};
    }
    if (!IsElementWiseMatch(sch->sch, block_sref, sch->Eval(consumer_rv))) {
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
    if (NeedsMultiLevelTiling(sch->sch, sch->Eval(block_rv))) {
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

/********** Tensorize Rewrite **********/

class RuleTensorizeRewrite {
 public:
  tir::PrimFunc desc_func;

  explicit RuleTensorizeRewrite(tir::PrimFunc desc_func) : desc_func(std::move(desc_func)) {}

  TReturn Apply(const SearchTask& task, const Schedule& sch, const BlockRV& block_rv,
                const TContextInfo& info) {
    if (CanTensorizeRewrite(sch, block_rv, desc_func)) {
      DoTensorizeRewrite(sch, block_rv, desc_func);
      return {{sch, info}};
    }
    return {{sch, info}};
  }
};

/*! \brief Rule creator */
SearchRule TensorizeRewrite(tir::PrimFunc desc_func) {
  auto invoke = [&](SearchTask task, Schedule sch, BlockRV block, TContextInfo info) -> TReturn {
    RuleTensorizeRewrite rule(desc_func);
    return rule.Apply(task, sch, block, info);
  };
  return SearchRule("tensorize_rewrite", invoke);
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
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.TensorizeRewrite").set_body_typed(TensorizeRewrite);

}  // namespace meta_schedule
}  // namespace tvm
