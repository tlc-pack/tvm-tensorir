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
    if (!IsSpatial(sch->sch, block_sref)) {
      LOG(INFO) << "Not a spatial block:\n" << block_sref;
      return {{sch, info}};
    }
    if (IsOutputBlock(sch->sch, block_sref)) {
      LOG(INFO) << "Is aa output block:\n" << block_sref;
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

/*! \brief Necessary information shared across tiling stages */
class TilingContextObj : public Object {
 public:
  /*! \brief The producer block to be multi-level tiled*/
  BlockRV producer;
  /*! \brief The consumer block that is fused into the producer block  */
  Optional<BlockRV> consumer;
  /*! \brief If the consumer a cache_write stage, which fusion is a must */
  bool is_consumer_cache_write;
  /*! \brief Number of spatial loops in the producer */
  mutable int n_spatial = -1;
  /*! \brief The order of loops on top of the producer after multi-level tiling */
  mutable std::vector<LoopRV> after_axes;

  static Map<String, ObjectRef> New(BlockRV producer, Optional<BlockRV> consumer,
                                    bool is_consumer_cache_write) {
    ObjectPtr<TilingContextObj> n = make_object<TilingContextObj>();
    n->producer = std::move(producer);
    n->consumer = std::move(consumer);
    n->is_consumer_cache_write = is_consumer_cache_write;
    return {{"tiling", ObjectRef(n)}};
  }

  static constexpr const char* _type_key = "meta_schedule.TilingContextObj";
  TVM_DECLARE_FINAL_OBJECT_INFO(TilingContextObj, Object);
};

/*! \brief A rule that adds a cache write stage after multi-level tiling */
class RuleAddCacheWrite {
 public:
  /*! \brief Default constructor */
  RuleAddCacheWrite() = default;

  /*! \brief Rule application */
  TReturn Apply(const SearchTask& task, const Schedule& sch, const BlockRV& block_rv,
                const TContextInfo& info) {
    CHECK(!info.defined());
    tir::StmtSRef block_sref = sch->Eval(block_rv);
    // The block does not need multi-level-tiling, then do not add cache write after it
    if (!NeedsMultiLevelTiling(sch->sch, block_sref)) {
      return {{sch, NullOpt}};
    }
    // If the block already has an elementwise matched consumer, do not add cache write after it
    if (Optional<BlockRV> opt_consumer_rv = sch->GetOnlyConsumer(block_rv)) {
      BlockRV consumer_rv = opt_consumer_rv.value();
      tir::StmtSRef consumer_sref = sch->Eval(consumer_rv);
      if (IsSpatial(sch->sch, block_sref) || IsSpatial(sch->sch, consumer_sref)) {
        if (IsElementWiseMatch(sch->sch, block_sref, consumer_sref)) {
          // It has an elementwise only consumer
          return {{sch, TilingContextObj::New(block_rv, consumer_rv, false)}};
        }
      }
    }
    // Do nothing, or add a cache write
    Schedule new_sch = sch->Copy(sch->sampler.ForkSeed());
    BlockRV producer = new_sch->CacheWrite(block_rv, "local");
    return {// Case 0: do nothing
            {sch, TilingContextObj::New(block_rv, NullOpt, false)},
            // Case 1: add cache write
            {new_sch, TilingContextObj::New(producer, block_rv, true)}};
  }
};

SearchRule AddCacheWrite() {
  auto f_apply = [](SearchTask task, Schedule sch, BlockRV block, TContextInfo info) -> TReturn {
    RuleAddCacheWrite rule;
    return rule.Apply(task, sch, block, info);
  };
  return SearchRule("multi_level_tiling", f_apply);
}

/********** Multi-Level-Tiling **********/

/*! \brief A rule that does multi-level tiling if there is sufficient amount of data reuse */
class RuleMultiLevelTiling {
 public:
  /*! \brief The structure of tiling, e.g. "SSRSRS" on CPU */
  String structure;

  /*!
   * \brief Constructor
   * \param structure The structure of tiling
   */
  explicit RuleMultiLevelTiling(String structure) : structure(std::move(structure)) {}

  /*! \brief Rule application */
  TReturn Apply(const SearchTask& task, const Schedule& sch, const BlockRV& _block_rv,
                const TContextInfo& info) {
    if (!info.defined()) {
      return {{sch, NullOpt}};
    }
    const auto* context = info.value().at("tiling").as<TilingContextObj>();
    CHECK(context);
    // Find 'S' in the format, i.e. spatial levels
    std::vector<int> s_idx = FindCharPos(structure, 'S');
    // Find 'R' in the format, i.e. reduction levels
    std::vector<int> r_idx = FindCharPos(structure, 'R');
    // Concat of `order` is the reordering order
    std::vector<std::vector<LoopRV>> order(structure.size());
    // Get block vars and loop axes
    Array<LoopRV> axes = sch->GetAxes(context->producer);
    Array<Integer> iter_types = GetBlockVarTypes(sch->sch, sch->Eval(context->producer));
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
    context->n_spatial = s_idx.size();
    context->after_axes = ConcatArray(order);
    sch->Reorder(context->after_axes);
    return {{sch, info}};
  }
};

SearchRule MultiLevelTiling(String structure) {
  auto f_apply = [structure{std::move(structure)}](SearchTask task, Schedule sch, BlockRV block,
                                                   TContextInfo info) -> TReturn {
    RuleMultiLevelTiling rule(structure);
    return rule.Apply(task, sch, block, info);
  };
  return SearchRule("multi_level_tiling", f_apply);
}

/********** Fusion **********/

/*! \brief A rule that does fusion after multi-level tiling */
class RuleFusion {
 public:
  /*! \brief Candidate levels of fusion */
  Array<Integer> levels;

  /*!
   * \brief Constructor
   * \param levels Candidate levels of fusion
   */
  explicit RuleFusion(Array<Integer> levels) : levels(std::move(levels)) {}

  /*! \brief Rule application */
  TReturn Apply(const SearchTask& task, const Schedule& sch, const BlockRV& _block_rv,
                const TContextInfo& info) {
    if (!info.defined()) {
      return {{sch, NullOpt}};
    }
    const auto* context = info.value().at("tiling").as<TilingContextObj>();
    CHECK(context);
    if (!context->consumer.defined()) {
      return {{sch, NullOpt}};
    }
    TReturn ret;
    if (!context->is_consumer_cache_write) {
      ret.Set(sch, NullOpt);
    }
    for (const Integer& _level : levels) {
      int level = _level;
      int loop_idx = level * context->n_spatial;
      Schedule new_sch = sch->Copy(sch->sampler.ForkSeed());
      new_sch->ReverseComputeAt(context->consumer.value(), context->after_axes[loop_idx - 1]);
      ret.Set(new_sch, NullOpt);
    }
    return ret;
  }
};

SearchRule Fusion(Array<Integer> levels) {
  auto f_apply = [levels{std::move(levels)}](SearchTask task, Schedule sch, BlockRV block,
                                             TContextInfo info) -> TReturn {
    RuleFusion rule(levels);
    return rule.Apply(task, sch, block, info);
  };
  return SearchRule("fusion", f_apply);
}

/********** ParallelizeOuter **********/

/*! \brief A rule that parallelizes the outer loops */
class RuleParallelizeOuter {
 public:
  /*! \brief The maximum extent of loops to be parallelized together */
  int max_extent;

  explicit RuleParallelizeOuter(int max_extent) : max_extent(max_extent) {}

  /*! \brief Rule application */
  TReturn Apply(const SearchTask& task, const Schedule& sch, const BlockRV& block_rv,
                const TContextInfo& info) {
    tir::StmtSRef block_sref = sch->Eval(block_rv);
    if (!IsSubrootBlock(sch->sch, block_sref)) {
      return {{sch, info}};
    }
    Array<LoopRV> loop_rvs = sch->GetAxes(block_rv);
    Array<tir::StmtSRef> loop_srefs;
    loop_srefs.reserve(loop_rvs.size());
    for (const LoopRV& loop_rv : loop_rvs) {
      loop_srefs.push_back(sch->Eval(loop_rv));
    }
    Array<Integer> loop_types = GetLoopType(sch->sch, block_sref, loop_srefs);
    tir::Var n_fusible_rv =
        sch->SampleFusibleLoops(loop_rvs, loop_types, max_extent, /*include_overflow_loop=*/true,
                                ScheduleNode::Order::outer_to_inner, ScheduleNode::Mode::max);
    PrimExpr min = Integer(0);
    PrimExpr extent = n_fusible_rv;
    sch->MarkParallel(loop_rvs, Range::FromMinExtent(min, extent));
    return {{sch, info}};
  }
};

SearchRule ParallelizeOuter(int max_extent) {
  auto f_apply = [max_extent](SearchTask task, Schedule sch, BlockRV block,
                              TContextInfo info) -> TReturn {
    RuleParallelizeOuter rule(max_extent);
    return rule.Apply(task, sch, block, info);
  };
  return SearchRule("parallelize_outer", f_apply);
}

/********** VectorizeInner **********/

/*! \brief A rule that parallelizes the outer loops */
class RuleVectorizeInner {
 public:
  /*! \brief The maximum extent of loops to be parallelized together */
  int max_extent;

  explicit RuleVectorizeInner(int max_extent) : max_extent(max_extent) {}

  /*! \brief Rule application */
  TReturn Apply(const SearchTask& task, const Schedule& sch, const BlockRV& block_rv,
                const TContextInfo& info) {
    tir::StmtSRef block_sref = sch->Eval(block_rv);
    if (!IsLeafBlock(sch->sch, block_sref)) {
      return {{sch, info}};
    }
    Array<LoopRV> loop_rvs = sch->GetAxes(block_rv);
    Array<tir::StmtSRef> loop_srefs;
    loop_srefs.reserve(loop_rvs.size());
    for (const LoopRV& loop_rv : loop_rvs) {
      loop_srefs.push_back(sch->Eval(loop_rv));
    }
    Array<Integer> loop_types = GetLoopType(sch->sch, block_sref, loop_srefs);
    tir::Var n_fusible_rv =
        sch->SampleFusibleLoops(loop_rvs, loop_types, max_extent, /*include_overflow_loop=*/false,
                                ScheduleNode::Order::inner_to_order, ScheduleNode::Mode::max);
    int n_loops = loop_rvs.size();
    PrimExpr min = Integer(n_loops) - n_fusible_rv;
    PrimExpr extent = n_fusible_rv;
    sch->MarkVectorize(loop_rvs, Range::FromMinExtent(min, extent));
    return {{sch, info}};
  }
};

SearchRule VectorizeInner(int max_extent) {
  auto f_apply = [max_extent](SearchTask task, Schedule sch, BlockRV block,
                              TContextInfo info) -> TReturn {
    RuleVectorizeInner rule(max_extent);
    return rule.Apply(task, sch, block, info);
  };
  return SearchRule("vectorize_inner", f_apply);
}

/********** Tensorize Rewrite **********/

class RuleTensorizeRewrite {
 public:
  tir::PrimFunc desc_func;

  explicit RuleTensorizeRewrite(tir::PrimFunc desc_func) : desc_func(std::move(desc_func)) {}

  TReturn Apply(const SearchTask& task, const Schedule& sch, const BlockRV& block_rv,
                const TContextInfo& info) {
    tir::StmtSRef block = sch->Eval(block_rv);
    if (CanTensorizeRewrite(sch->sch, block, desc_func)) {
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
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.MultiLevelTiling").set_body_typed(MultiLevelTiling);
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.Fusion").set_body_typed(Fusion);
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.ParallelizeOuter").set_body_typed(ParallelizeOuter);
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.VectorizeInner").set_body_typed(VectorizeInner);
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.TensorizeRewrite").set_body_typed(TensorizeRewrite);

}  // namespace meta_schedule
}  // namespace tvm
