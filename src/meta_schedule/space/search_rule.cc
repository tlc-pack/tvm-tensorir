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
      return {{sch, info}};
    }
    if (IsOutputBlock(sch->sch, block_sref)) {
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

/********** Multi-Level-Tiling-And-Fusion **********/

class RuleMultiLevelTilingAndFusion {
 public:
  String structure;
  bool add_read_cache;
  bool add_write_cache;
  Array<Integer> fusion_levels;
  std::vector<int> s_idx;
  std::vector<int> r_idx;

  explicit RuleMultiLevelTilingAndFusion(String structure, bool add_read_cache,
                                         bool add_write_cache, Array<Integer> fusion_levels)
      : structure(structure),
        add_read_cache(add_read_cache),
        add_write_cache(add_write_cache),
        fusion_levels(fusion_levels),
        s_idx(),
        r_idx() {
    // Process `structure` and set `s_idx` and `r_idx` properly
    int structure_len = structure.length();
    int num_s_in_prefix = structure_len;
    for (int i = 0; i < structure_len; ++i) {
      char c = structure->data[i];
      if (c == 'S') {
        s_idx.push_back(i);
      } else if (c == 'R') {
        if (r_idx.empty()) {
          num_s_in_prefix = i;
        }
        r_idx.push_back(i);
      } else {
        LOG(FATAL) << "ValueError: Invalid tiling structure, only accepts string of 'S's and 'R's, "
                      "but gets: "
                   << structure;
      }
    }
    // Process `fusion_levels`
    std::unordered_set<int> used_levels;
    for (const Integer& _level : fusion_levels) {
      int level = _level;
      CHECK_GE(level, 1) << "ValueError: The fusion level must be >= 1, but gets " << level;
      CHECK_LE(level, num_s_in_prefix)
          << "ValueError: The fusion level must be <= "
             "the number of prefix spatial tiles, but gets fusion_level "
          << level << " and number of prefix spatial tiles " << num_s_in_prefix;
      CHECK(!used_levels.count(level))
          << "ValueError: Duplicate fusion levels are not allowed, but gets multiple " << level;
      used_levels.insert(level);
    }
  }

  struct State {
    Schedule sch;
    BlockRV block_rv;
    Optional<BlockRV> only_consumer;
    bool only_consumer_is_cache_write;
    Array<Array<LoopRV>> tiles;

    explicit State(Schedule sch, BlockRV block_rv, Optional<BlockRV> only_consumer = NullOpt,
                   bool only_consumer_is_cache_write = false, Array<Array<LoopRV>> tiles = {})
        : sch(std::move(sch)),
          block_rv(std::move(block_rv)),
          only_consumer(std::move(only_consumer)),
          only_consumer_is_cache_write(only_consumer_is_cache_write),
          tiles(std::move(tiles)) {}
  };

  std::vector<State> AddWriteCache(State state) const {
    Schedule sch = state.sch;
    BlockRV block_rv = state.block_rv;
    tir::StmtSRef block_sref = sch->Eval(block_rv);
    // Find the only-consumer for the block
    // If the only-consumer can be fused, then do not add any write cache
    if (Optional<BlockRV> opt_consumer_rv = sch->GetOnlyConsumer(state.block_rv)) {
      BlockRV consumer_rv = opt_consumer_rv.value();
      tir::StmtSRef consumer_sref = sch->Eval(consumer_rv);
      // Check if it can be directly fused
      if ((IsSpatial(sch->sch, block_sref) || IsSpatial(sch->sch, consumer_sref)) &&
          IsElementWiseMatch(sch->sch, block_sref, consumer_sref)) {
        state.only_consumer = consumer_rv;
        state.only_consumer_is_cache_write = false;
        return {state};
      }
    }
    std::vector<State> result;
    // Case 0. Do not add write cache, then fusion won't happen later either
    result.push_back(state);
    // Case 1. Add a write cache
    if (add_write_cache) {
      // Fork a new schedule
      state.sch = sch->Copy(sch->sampler.ForkSeed());
      // The original block to tiled
      state.block_rv = state.sch->CacheWrite(block_rv, "local");
      // The cache write block
      state.only_consumer = block_rv;
      state.only_consumer_is_cache_write = true;
      result.push_back(std::move(state));
    }
    return result;
  }

  State DoTiling(State state) const {
    Schedule& sch = state.sch;
    BlockRV& block_rv = state.block_rv;
    // Concat of `tiles` is the reordering order
    std::vector<Array<LoopRV>> tiles(structure.size());
    // Get block vars and loop axes
    Array<Integer> iter_types = GetBlockVarTypes(sch->sch, sch->Eval(block_rv));
    Array<LoopRV> axes = sch->GetAxes(block_rv);
    CHECK_EQ(axes.size(), iter_types.size());
    // For each loop axis, tile it
    for (int i = 0, n = axes.size(); i < n; ++i) {
      const std::vector<int>* idx = nullptr;
      if (iter_types[i] == tir::IterVarType::kDataPar) {
        idx = &s_idx;
      } else if (iter_types[i] == tir::IterVarType::kCommReduce) {
        idx = &r_idx;
      } else {
        continue;
      }
      // Number of splits to be made
      int n_tiles = idx->size();
      // Do the split
      Array<tir::Var> factors = sch->SamplePerfectTile(/*n=*/n_tiles, /*loop=*/axes[i]);
      Array<LoopRV> splits =
          sch->Split(/*loop=*/axes[i], /*factors=*/{factors.begin(), factors.end()});
      // Put every tile to its slot
      for (int j = 0; j < n_tiles; ++j) {
        tiles[idx->at(j)].push_back(splits[j]);
      }
    }
    sch->Reorder(ConcatArray(tiles));
    state.tiles = Array<Array<LoopRV>>{tiles.begin(), tiles.end()};
    return state;
  }

  std::vector<State> FuseWithElementwiseConsumer(State state) const {
    // If the only-consumer does not exist, or is not elementwise, then do not do fusion
    if (!state.only_consumer.defined()) {
      return {state};
    }
    std::vector<State> result;
    // Special case.
    // `cache_write` must be fused at some level, otherwise it has no benefit
    // On the other hand, If the only consumer is not cache_write, then we may choose not to fuse
    if (!state.only_consumer_is_cache_write) {
      result.push_back(state);
    }
    Schedule sch = state.sch;
    BlockRV consumer = state.only_consumer.value();
    for (const Integer& _level : fusion_levels) {
      // Enumerate the level of tile to be fused at
      int level = _level;
      const LoopRV& loop = state.tiles[level - 1].back();
      State new_state = state;
      new_state.sch = state.sch->Copy(sch->sampler.ForkSeed());
      new_state.sch->ReverseComputeAt(consumer, loop);
      result.push_back(new_state);
    }
    return result;
  }

  TReturn Apply(const SearchTask& task, const Schedule& sch, BlockRV block_rv,
                const TContextInfo& _info) const {
    // If multi-level-tiling is not required
    if (!NeedsMultiLevelTiling(sch->sch, sch->Eval(block_rv))) {
      return {{sch, NullOpt}};
    }
    // States
    std::vector<State> states{State(sch, block_rv)};
    // Add write cache
    {
      std::vector<State> next_states;
      for (State& state : states) {
        std::vector<State> news = AddWriteCache(std::move(state));
        next_states.insert(next_states.end(), std::make_move_iterator(news.begin()),
                           std::make_move_iterator(news.end()));
      }
      states.swap(next_states);
    }
    // Do the multi-level tiling
    {
      std::vector<State> next_states;
      for (State& state : states) {
        next_states.push_back(DoTiling(std::move(state)));
      }
      states.swap(next_states);
    }
    // Fuse with elementwise consumer
    {
      std::vector<State> next_states;
      for (State& state : states) {
        std::vector<State> news = FuseWithElementwiseConsumer(std::move(state));
        next_states.insert(next_states.end(), std::make_move_iterator(news.begin()),
                           std::make_move_iterator(news.end()));
      }
      states.swap(next_states);
    }
    TReturn ret;
    for (const State& state : states) {
      ret.Set(state.sch, NullOpt);
    }
    return ret;
  }
};

SearchRule MultiLevelTilingAndFusion(String structure, bool add_read_cache, bool add_write_cache,
                                     Array<Integer> fusion_levels) {
  RuleMultiLevelTilingAndFusion rule(structure, add_read_cache, add_write_cache, fusion_levels);
  auto f_apply = [rule{std::move(rule)}](SearchTask task, Schedule sch, BlockRV block,
                                         TContextInfo info) -> TReturn {
    return rule.Apply(task, sch, block, info);
  };
  return SearchRule("always_inline", f_apply);
}

/********** MarkParallelizeOuter **********/

/*! \brief A rule that parallelizes the outer loops */
class RuleMarkParallelizeOuter {
 public:
  /*! \brief The maximum extent of loops to be parallelized together */
  int max_extent;

  explicit RuleMarkParallelizeOuter(int max_extent) : max_extent(max_extent) {}

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

SearchRule MarkParallelizeOuter(int max_extent) {
  auto f_apply = [max_extent](SearchTask task, Schedule sch, BlockRV block,
                              TContextInfo info) -> TReturn {
    RuleMarkParallelizeOuter rule(max_extent);
    return rule.Apply(task, sch, block, info);
  };
  return SearchRule("mark_parallelize_outer", f_apply);
}

/********** MarkVectorizeInner **********/

/*! \brief A rule that parallelizes the outer loops */
class RuleMarkVectorizeInner {
 public:
  /*! \brief The maximum extent of loops to be parallelized together */
  int max_extent;

  explicit RuleMarkVectorizeInner(int max_extent) : max_extent(max_extent) {}

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

SearchRule MarkVectorizeInner(int max_extent) {
  auto f_apply = [max_extent](SearchTask task, Schedule sch, BlockRV block,
                              TContextInfo info) -> TReturn {
    RuleMarkVectorizeInner rule(max_extent);
    return rule.Apply(task, sch, block, info);
  };
  return SearchRule("vectorize_inner", f_apply);
}

/********** MarkTensorize **********/

class RuleMarkTensorize {
 public:
  /*! \brief The tensor intrinsics to be used */
  Array<tir::TensorIntrin> tensor_intrins;

  explicit RuleMarkTensorize(Array<tir::TensorIntrin> tensor_intrins)
      : tensor_intrins(tensor_intrins) {}

  void BlockizeAndMark(const Schedule& sch, const BlockRV& block_rv, const tir::PrimFunc& desc_func,
                       const TensorizeInfoNode* info) {
    // Construct a mapping from tir loops back to LoopRVs
    Map<tir::StmtSRef, LoopRV> loop2rv;
    {
      Array<LoopRV> loop_rvs = sch->GetAxes(block_rv);
      for (const LoopRV& loop_rv : loop_rvs) {
        loop2rv.Set(sch->Eval(loop_rv), loop_rv);
      }
    }
    // Split the loops
    arith::Analyzer analyzer;
    std::unordered_set<const tir::StmtSRefNode*> inner_loops;
    std::vector<LoopRV> reorder_suffix;
    reorder_suffix.resize(info->loop_map.size());
    for (const auto& kv : info->loop_map) {
      // Extract mapping (block_loop => desc_loop)
      const tir::StmtSRef& block_loop_sref = kv.first;
      const tir::LoopNode* block_loop = block_loop_sref->GetStmt<tir::LoopNode>();
      const tir::LoopNode* desc_loop = kv.second.get();
      CHECK(block_loop != nullptr && desc_loop != nullptr);
      // Extract the loop extent
      PrimExpr block_extent = analyzer.Simplify(block_loop->extent);
      PrimExpr desc_extent = analyzer.Simplify(desc_loop->extent);
      const auto* int_block_extent = block_extent.as<IntImmNode>();
      const auto* int_desc_extent = desc_extent.as<IntImmNode>();
      CHECK(int_block_extent != nullptr && int_desc_extent != nullptr);
      // Check divisibility
      int64_t total = int_block_extent->value;
      int64_t inner = int_desc_extent->value;
      CHECK_EQ(total % inner, 0);
      int64_t outer = int_block_extent->value / int_desc_extent->value;
      // Do the split
      Array<LoopRV> split =
          sch->Split(loop2rv.at(block_loop_sref), {Integer(outer), Integer(inner)});
      CHECK_EQ(split.size(), 2);
      inner_loops.insert(sch->Eval(split[1]).operator->());
      // The inner split will be reordered to the loop domain that is tensorized
      int desc_loop_index = info->desc_loop_indexer.at(GetRef<tir::Loop>(desc_loop));
      reorder_suffix[desc_loop_index] = split[1];
    }
    // Reorder the loops
    std::vector<LoopRV> reorder_list;
    bool meet = false;
    Array<LoopRV> all_loops = sch->GetAxes(block_rv);
    for (const LoopRV& loop : all_loops) {
      if (inner_loops.count(sch->Eval(loop).operator->())) {
        meet = true;
      } else if (meet) {
        reorder_list.push_back(loop);
      }
    }
    reorder_list.insert(reorder_list.end(), reorder_suffix.begin(), reorder_suffix.end());
    sch->Reorder(reorder_list);
    // Do blockize
    if (!reorder_suffix.empty()) {
      sch->Blockize(reorder_suffix[0], "");
    }
    // Annotate the block
    {
      tir::StmtSRef last_loop_sref = sch->Eval(reorder_list.back());
      const auto* last_loop = last_loop_sref->GetStmt<tir::LoopNode>();
      CHECK(last_loop) << "TypeError: Expects Loop, but gets: "
                       << last_loop_sref->stmt->GetTypeKey();
      const auto* realize = last_loop->body.as<tir::BlockRealizeNode>();
      CHECK(realize) << "TypeError: Expects BlockRealize, but gets: "
                     << last_loop->body->GetTypeKey();
      AnnotateBlockType(sch->sch, sch->sch->stmt2ref.at(realize->block.get()), "lazy_tensorize");
    }
  }

  /*! \brief Rule application */
  TReturn Apply(const SearchTask& task, const Schedule& sch, const BlockRV& block_rv,
                const TContextInfo& info) {
    tir::StmtSRef block_sref = sch->Eval(block_rv);
    TReturn result{{sch, info}};
    Optional<Schedule> next_sch = NullOpt;
    for (const tir::TensorIntrin& intrin : tensor_intrins) {
      if (!next_sch.defined()) {
        next_sch = sch->Copy(sch->sampler.ForkSeed());
      }
      Schedule cur_sch = next_sch.value();
      if (Optional<TensorizeInfo> opt_tensorize_info =
              GetTensorizeLoopMapping(cur_sch->sch, block_sref, intrin->description)) {
        BlockizeAndMark(cur_sch, block_rv, intrin->description, opt_tensorize_info.value().get());
        result.Set(cur_sch, {});
        next_sch = NullOpt;
      }
    }
    return result;
  }
};

SearchRule MarkTensorize(Array<tir::TensorIntrin> tensor_intrins) {
  auto f_apply = [tensor_intrins{std::move(tensor_intrins)}](
                     SearchTask task, Schedule sch, BlockRV block, TContextInfo info) -> TReturn {
    RuleMarkTensorize rule(tensor_intrins);
    return rule.Apply(task, sch, block, info);
  };
  return SearchRule("mark_tensorize", f_apply);
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
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.MultiLevelTilingAndFusion")
    .set_body_typed(MultiLevelTilingAndFusion);
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.MarkParallelizeOuter")
    .set_body_typed(MarkParallelizeOuter);
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.MarkVectorizeInner")
    .set_body_typed(MarkVectorizeInner);
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.MarkTensorize").set_body_typed(MarkTensorize);

}  // namespace meta_schedule
}  // namespace tvm
