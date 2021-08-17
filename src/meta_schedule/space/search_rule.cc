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

#include <tvm/auto_scheduler/search_policy.h>

#include "../analysis.h"
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

/**************** TIR Nodes ****************/
using tir::BlockNode;
using tir::ForNode;

/********** Constructors **********/

SearchRule::SearchRule(String name, SearchRuleNode::FApply apply) {
  ObjectPtr<SearchRuleNode> n = make_object<SearchRuleNode>();
  n->name = std::move(name);
  n->apply_ = std::move(apply);
  data_ = std::move(n);
}

/********** SearchRule **********/

Array<Schedule> SearchRuleNode::Apply(const SearchTask& task, const Schedule& sch,
                                      const BlockRV& block) const {
  return apply_(task, sch, block);
}

SearchRule SearchRuleCompose(const String& name, const Array<SearchRule>& rules) {
  auto apply = [rules](SearchTask task, Schedule sch, BlockRV block) -> Array<Schedule> {
    std::vector<Schedule> curr{sch};
    std::vector<Schedule> next;
    for (const SearchRule& rule : rules) {
      for (const Schedule& curr_sch : curr) {
        Array<Schedule> results = rule->Apply(task, curr_sch, block);
        next.reserve(next.size() + results.size());
        for (const Schedule& result_sch : results) {
          next.push_back(result_sch);
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
class RuleInlinePureSpatial {
 public:
  /*! \brief Requires the pure spatial block to be strictly-inlineable */
  bool strict_mode;

  /*! \brief Default constructor */
  explicit RuleInlinePureSpatial(bool strict_mode) : strict_mode(strict_mode) {}

  static bool NeedsInline(const tir::Schedule& sch, const tir::StmtSRef& block_sref,
                          bool strict_mode) {
    if (!IsSpatial(sch->state(), block_sref)) {
      return false;
    }
    if (IsOutputBlock(sch->state(), block_sref)) {
      return false;
    }
    if (strict_mode && !IsStrictlyInlineable(sch->state(), block_sref)) {
      return false;
    }
    Array<tir::StmtSRef> loop_srefs = tir::GetLoops(block_sref);
    for (const tir::StmtSRef& loop_sref : loop_srefs) {
      if (!HasSingleChild(loop_sref)) {
        return false;
      }
    }
    return true;
  }

  /*! \brief Rule application */
  Array<Schedule> Apply(const SearchTask& task, const Schedule& sch,
                        const BlockRV& block_rv) const {
    tir::StmtSRef block_sref = sch->GetSRef(block_rv);
    if (IsSubrootBlock(sch->state(), block_sref) && NeedsInline(sch, block_sref, strict_mode)) {
      sch->ComputeInline(block_rv);
    }
    return {sch};
  }
};

SearchRule InlinePureSpatial(bool strict_mode) {
  auto f_apply = [strict_mode](SearchTask task, Schedule sch, BlockRV block) -> Array<Schedule> {
    RuleInlinePureSpatial rule(strict_mode);
    return rule.Apply(task, sch, block);
  };
  return SearchRule("inline_pure_spatial", f_apply);
}

/********** Multi-Level-Tiling-And-Fusion **********/

/*! \brief A rule that does multi-level tiling, and possibly cache_read/write */
class RuleMultiLevelTiling {
 public:
  /*!
   * \brief Structure of the tiling.
   * Recommendation:
   * - 'SSRSRS' on CPU
   * - 'SSSRRSRS' on GPU
   */
  String structure;
  /*! \brief The maximum size of the innermost factor */
  int max_innermost_factor;
  /*! \brief Whether we must cache_read the inputs */
  bool must_cache_read;
  /*! \brief The storage scope of cache_read */
  String cache_read_scope;
  /*! \brief Whether we could cache_write the inputs */
  bool can_cache_write;
  /*! \brief Whether we must cache_write the inputs */
  bool must_cache_write;
  /*! \brief The storage scope of cache_write */
  String cache_write_scope;
  /*! \brief Whether to use strict mode when inlining consumers */
  bool consumer_inline_strict;
  /*! \brief Which levels of tiles the consumer is fused into */
  std::vector<int> fusion_levels;
  /*! \brief The length of vectorized cooperative fetching */
  Optional<Integer> vector_load_max_len;
  /*! \brief Which thread axis each level of tile should be bound to */
  Array<String> tile_binds;
  /*! \brief The indices of spatial tiles in the structure string */
  std::vector<int> s_idx;
  /*! \brief The indices of reduction tiles in the structure string */
  std::vector<int> r_idx;

  /*!
   * \brief Parse the structure string, extract the indices of each spatial/reduction tile,
   * and return the number of contiguous spatial tiles in the prefix of the structure string.
   * \param s_idx The indices of the spatial tiles
   * \param r_idx The indices of the reduction tiles
   * \return The number of contiguous spatial tiles in the prefix of the structure string
   */
  static int ParseStructure(const String& structure, std::vector<int>* s_idx,
                            std::vector<int>* r_idx) {
    int structure_len = structure.length();
    int num_s_in_prefix = structure_len;
    for (int i = 0; i < structure_len; ++i) {
      char c = structure->data[i];
      if (c == 'S') {
        s_idx->push_back(i);
      } else if (c == 'R') {
        if (r_idx->empty()) {
          num_s_in_prefix = i;
        }
        r_idx->push_back(i);
      } else {
        LOG(FATAL) << "ValueError: Invalid tiling structure, only accepts string of 'S's and 'R's, "
                      "but gets: "
                   << structure;
      }
    }
    ICHECK(!s_idx->empty())
        << "ValueError: Invalid tiling structure, cannot find any 'S' in the format";
    ICHECK(!r_idx->empty())
        << "ValueError: Invalid tiling structure, cannot find any 'R' in the format";
    return num_s_in_prefix;
  }

  /*!
   * \brief Extract the buffer indices of the read buffers, excluding those update buffers
   * \param block_sref The block to be extracted
   * \return A vector of integers, the indices of the read buffers
   */
  static std::vector<int> GetReadBufferIndices(const tir::StmtSRef& block_sref) {
    const auto* block = block_sref->StmtAs<tir::BlockNode>();
    ICHECK(block) << "TypeError: Expects 'Block', but gets: " << block_sref->stmt->GetTypeKey();
    std::vector<int> result;
    int n_reads = block->reads.size();
    int n_writes = block->writes.size();
    for (int i = 0; i < n_reads; ++i) {
      const tir::Buffer& read_buffer = block->reads[i]->buffer;
      bool found = false;
      for (int j = 0; j < n_writes; ++j) {
        const tir::Buffer& write_buffer = block->writes[j]->buffer;
        if (read_buffer.same_as(write_buffer)) {
          // Exclude update buffers
          found = true;
          break;
        }
      }
      if (!found) {
        result.push_back(i);
      }
    }
    std::reverse(result.begin(), result.end());
    return result;
  }

  /*!
   * \brief Fuse the axes of a cache_read block generated by reading a buffer
   * \param sch The schedule
   * \param block_rv The block generated by cache_read
   * \param buffer The buffer that is cache_read
   * \return The fused loop
   */
  static LoopRV FuseBufferAxes(const Schedule& sch, const BlockRV& block_rv, int buffer_ndim) {
    Array<LoopRV> to_fuse;
    Array<LoopRV> axes = sch->GetLoops(block_rv);
    int n_axes = axes.size();
    for (int i = n_axes - buffer_ndim; i < n_axes; ++i) {
      to_fuse.push_back(axes[i]);
    }
    return sch->Fuse(to_fuse);
  }

  explicit RuleMultiLevelTiling(String structure, int max_innermost_factor, bool must_cache_read,
                                String cache_read_scope, bool can_cache_write,
                                bool must_cache_write, String cache_write_scope,
                                bool consumer_inline_strict, Array<Integer> fusion_levels_,
                                Optional<Integer> vector_load_max_len,
                                Optional<Array<String>> tile_binds)
      : structure(structure),
        max_innermost_factor(max_innermost_factor),
        must_cache_read(must_cache_read),
        cache_read_scope(cache_read_scope),
        can_cache_write(can_cache_write),
        must_cache_write(must_cache_write),
        cache_write_scope(cache_write_scope),
        consumer_inline_strict(consumer_inline_strict),
        fusion_levels(AsVector<Integer, int>(fusion_levels_)),
        vector_load_max_len(vector_load_max_len),
        tile_binds(tile_binds.value_or(Array<String>{})),
        s_idx(),
        r_idx() {
    // Process `structure` and set `s_idx` and `r_idx` properly
    int num_s_in_prefix = ParseStructure(structure, &s_idx, &r_idx);
    // Process `fusion_levels`
    if (!fusion_levels.empty()) {
      std::sort(fusion_levels.begin(), fusion_levels.end());
      CHECK_GE(fusion_levels.front(), 1)
          << "ValueError: The fusion level must be >= 1, but gets " << fusion_levels_;
      CHECK_LE(fusion_levels.back(), num_s_in_prefix)
          << "ValueError: The fusion level must be <= "
             "the number of prefix spatial tiles, but gets fusion_level "
          << fusion_levels_ << ", and number of prefix spatial tiles " << num_s_in_prefix;
    }
  }

  /*! \brief The internal state */
  struct State {
    /*! \brief The schedule */
    Schedule sch;
    /*! \brief The block random variable to focus on */
    BlockRV block_rv;
    /*! \brief The write cache */
    Optional<BlockRV> write_cache;
    /*! \brief Whether the write cache is added in the rule */
    bool write_cache_is_added;
    /*! \brief The tiles according to the tile structure */
    Array<Array<LoopRV>> tiles;

    explicit State(Schedule sch, BlockRV block_rv, Optional<BlockRV> write_cache = NullOpt,
                   bool write_cache_is_added = false, Array<Array<LoopRV>> tiles = {})
        : sch(std::move(sch)),
          block_rv(std::move(block_rv)),
          write_cache(std::move(write_cache)),
          write_cache_is_added(write_cache_is_added),
          tiles(std::move(tiles)) {}
  };

  std::vector<State> AddWriteCache(State state) const {
    if (!can_cache_write) {
      return {std::move(state)};
    }
    std::vector<State> result;
    // Case 1. Do not add write cache, then fusion won't happen later either
    if (!must_cache_write) {
      // Check the only consumer of the block can be treated as write cache
      // If so, check if we can continuously inline a chain of only consumers for better locality
      Schedule sch = state.sch;
      BlockRV block_rv = state.block_rv;
      Array<BlockRV> consumer_chain = GetInlineableConsumerChain(sch, block_rv);
      if (!consumer_chain.empty()) {
        // inline all but the last consumer
        for (size_t i = 0; i + 1 < consumer_chain.size(); i++) {
          sch->ComputeInline(consumer_chain[i]);
        }
        // let the last consumer act as a write cache
        state.write_cache = consumer_chain.back();
        state.write_cache_is_added = false;
        return {std::move(state)};
      }
      // In this case, it doesn't have an elementwise-matched consumer
      result.push_back(state);
    }
    // Case 2. Add a write cache
    // If the program comes to this point, then there are two possibilities
    // 1) `must_cache_write = True`
    // 2) The elementwise-matched consumer doesn't exist
    // Fork a new schedule
    state.sch = Downcast<Schedule>(state.sch->Copy(state.sch->ForkSeed()));
    // the copy block after calling cache write
    BlockRV write_cache = state.block_rv;
    // The original block to tiled
    state.block_rv = state.sch->CacheWrite(state.block_rv, 0, cache_write_scope);
    Array<BlockRV> consumer_chain = GetInlineableConsumerChain(state.sch, write_cache);
    if (!consumer_chain.empty()) {
      // inline the cache write copy stage and all but the last consumer
      state.sch->ComputeInline(write_cache);
      for (size_t i = 0; i + 1 < consumer_chain.size(); i++) {
        state.sch->ComputeInline(consumer_chain[i]);
      }
      // let the last consumer act as a write cache
      state.write_cache = consumer_chain.back();
    } else {
      state.write_cache = write_cache;
    }

    state.write_cache_is_added = true;
    result.push_back(std::move(state));
    return result;
  }

  /*!
   * \brief Get a chain of elementwise-matched consumers of the producer block, such that the
   * producer and all but the last consumer can be inlined into one stage and the last consumer can
   * act as a write cache.
   * \param sch The schedule
   * \param producer_block_rv The producer block
   * \return The chain of elementwise-matched consumers.
   */
  Array<BlockRV> GetInlineableConsumerChain(Schedule sch, const BlockRV& producer_block_rv) const {
    if (!RuleInlinePureSpatial::NeedsInline(sch, sch->GetSRef(producer_block_rv),
                                            this->consumer_inline_strict)) {
      return {};
    }
    Array<BlockRV> result;
    BlockRV current_block_rv = producer_block_rv;  // current producer block
    while (true) {
      Array<BlockRV> consumers = sch->GetConsumers(current_block_rv);
      if (consumers.size() != 1) {
        break;
      }
      BlockRV consumer_rv = consumers[0];
      tir::StmtSRef consumer_sref = sch->GetSRef(consumer_rv);
      if (!IsSpatial(sch->state(), consumer_sref)) {
        break;
      }
      if (!IsElementWiseMatch(sch->state(), sch->GetSRef(current_block_rv), consumer_sref)) {
        break;
      }
      // Then `consumer_rv` must be an elementwise-matched consumer of `block_rv`
      if (!RuleInlinePureSpatial::NeedsInline(sch, consumer_sref, this->consumer_inline_strict)) {
        if (IsOutputBlock(sch->state(), consumer_sref)) {
          result.push_back(consumer_rv);
        }
        break;
      }

      result.push_back(consumer_rv);
      current_block_rv = consumer_rv;  // check the next consumer
    }
    return result;
  }

  std::vector<State> AddReadCache(State state) const {
    if (!must_cache_read) {
      return {state};
    }
    // Extract the block to be worked on
    Schedule& sch = state.sch;
    BlockRV& block_rv = state.block_rv;
    tir::StmtSRef block_sref = sch->GetSRef(block_rv);
    // Find all indices of the read buffers
    std::vector<int> read_buffer_indices = GetReadBufferIndices(block_sref);
    // Enumerate all buffers that are read but not written
    for (int i : read_buffer_indices) {
      tir::Buffer buffer = block_sref->StmtAs<tir::BlockNode>()->reads[i]->buffer;
      int buffer_ndim = buffer->shape.size();
      // Do cache_read
      BlockRV cache_read_block = sch->CacheRead(block_rv, i, cache_read_scope);
      // Insert cache_read block to the proper place
      const Array<LoopRV>& r_tiles = state.tiles[r_idx.front()];
      ICHECK(!r_tiles.empty()) << "ValueError: Cannot find any reduction loop in the block";
      sch->ComputeAt(cache_read_block, r_tiles.back(), true);
      // Fuse the iterators of the cache_read
      LoopRV fused = FuseBufferAxes(sch, cache_read_block, buffer_ndim);
      // Do cooperative fetching
      if (vector_load_max_len.defined()) {
        int max_vec_len = vector_load_max_len.value();
        // cooperative fetch + vectorized loading
        // Split into inner and outer
        Array<ExprRV> factors = sch->SamplePerfectTile(fused, 2, max_vec_len);
        ICHECK_EQ(factors.size(), 2);
        Array<LoopRV> splits = sch->Split(fused, {factors[0], factors[1]});
        ICHECK_EQ(splits.size(), 2);
        // Vectorize the inner loop
        sch->Vectorize(splits[1]);
        fused = splits[0];
      }
      // Add cooperative fetching
      sch->MarkLoop(fused, tir::attr::loop_type, String("lazy_cooperative_fetch"));
    }
    return {state};
  }

  std::vector<State> DoTiling(State state) const {
    Schedule& sch = state.sch;
    BlockRV& block_rv = state.block_rv;
    // Concat of `tiles` is the reordering order
    std::vector<Array<LoopRV>> tiles(structure.size());
    // Get block vars and loop axes
    // TODO: fix
    Array<Integer> iter_types = GetBlockVarTypes(sch->state(), sch->GetSRef(block_rv));
    Array<LoopRV> axes = sch->GetLoops(block_rv);
    ICHECK_EQ(axes.size(), iter_types.size());
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
      Array<ExprRV> factors = sch->SamplePerfectTile(
          /*loop=*/axes[i], /*n=*/n_tiles, /*max_innermost_factor=*/max_innermost_factor);
      Array<LoopRV> splits =
          sch->Split(/*loop=*/axes[i], /*factors=*/{factors.begin(), factors.end()});
      // Put every tile to its slot
      for (int j = 0; j < n_tiles; ++j) {
        tiles[idx->at(j)].push_back(splits[j]);
      }
    }
    sch->Reorder(ConcatArray(tiles));
    // Bind the tiles
    int n_binds = std::min(tile_binds.size(), tiles.size());
    for (int i = 0; i < n_binds; ++i) {
      LoopRV fused = sch->Fuse(tiles[i]);
      sch->Bind(fused, tile_binds[i]);
      tiles[i] = {fused};
    }
    state.tiles = Array<Array<LoopRV>>{tiles.begin(), tiles.end()};
    return {state};
  }

  std::vector<State> FuseWriteCache(State state) const {
    // If the only-consumer does not exist, or is not elementwise, then do not do fusion
    if (!state.write_cache.defined()) {
      return {state};
    }
    std::vector<State> result;
    // Special case.
    //    Stages added by `cache_write` must be fused at some level, otherwise it has no benefit.
    //    On the other hand, If the consumer stage is not added by  `cache_write`,
    //    we may choose not to fuse by setting `must_cache_write = False`
    if (!state.write_cache_is_added && !must_cache_write) {
      result.push_back(state);
    }
    Schedule sch = state.sch;
    BlockRV consumer = state.write_cache.value();
    // Enumerate the level of tile to be fused at
    for (int level : fusion_levels) {
      const LoopRV& loop = state.tiles[level - 1].back();
      State new_state = state;
      new_state.sch = Downcast<Schedule>(state.sch->Copy(sch->ForkSeed()));
      new_state.sch->ReverseComputeAt(consumer, loop, true);
      result.push_back(new_state);
    }
    return result;
  }

#define TVM_SEARCH_RULE_APPLY_SUB_RULE(SrcStates, FSubRule)                          \
  {                                                                                  \
    std::vector<State> next_states;                                                  \
    for (const State& state : SrcStates) {                                           \
      std::vector<State> result = FSubRule(state);                                   \
      next_states.insert(next_states.end(), std::make_move_iterator(result.begin()), \
                         std::make_move_iterator(result.end()));                     \
    }                                                                                \
    SrcStates.swap(next_states);                                                     \
  }

  Array<Schedule> Apply(const SearchTask& task, const Schedule& sch,
                        const BlockRV& block_rv) const {
    tir::StmtSRef block_sref = sch->GetSRef(block_rv);
    if (!NeedsMultiLevelTiling(sch->state(), block_sref)) {
      return {sch};
    }
    // States
    std::vector<State> states{State(sch, block_rv)};
    // Add write cache
    TVM_SEARCH_RULE_APPLY_SUB_RULE(states, AddWriteCache);
    // Do the multi-level tiling
    TVM_SEARCH_RULE_APPLY_SUB_RULE(states, DoTiling);
    // Add read cache
    TVM_SEARCH_RULE_APPLY_SUB_RULE(states, AddReadCache);
    // Fuse with write cache
    TVM_SEARCH_RULE_APPLY_SUB_RULE(states, FuseWriteCache);
    Array<Schedule> ret;
    ret.reserve(states.size());
    for (const State& state : states) {
      ret.push_back(state.sch);
    }
    return ret;
  }

#undef TVM_SEARCH_RULE_APPLY_SUBRULE
};

SearchRule MultiLevelTiling(String structure, int max_innermost_factor, bool must_cache_read,
                            String cache_read_scope, bool can_cache_write, bool must_cache_write,
                            String cache_write_scope, bool consumer_inline_strict,
                            Array<Integer> fusion_levels, Optional<Integer> vector_load_max_len,
                            Optional<Array<String>> tile_binds) {
  if (!can_cache_write && must_cache_write) {
    LOG(FATAL) << "ValueError: Conflict options, cannot have can_cache_write = false, and "
                  "must_cache_write = true at the same time";
  }
  RuleMultiLevelTiling rule(structure, max_innermost_factor, must_cache_read, cache_read_scope,
                            can_cache_write, must_cache_write, cache_write_scope,
                            consumer_inline_strict, fusion_levels, vector_load_max_len, tile_binds);
  auto f_apply = [rule{std::move(rule)}](SearchTask task, Schedule sch,
                                         BlockRV block) -> Array<Schedule> {
    return rule.Apply(task, sch, block);
  };
  return SearchRule("multi_level_tiling", f_apply);
}

/********** RandomComputeLocation **********/

class RuleRandomComputeLocation {
 public:
  bool IsFreeBlock(const tir::Schedule sch, const tir::StmtSRef& block_sref) const {
    if (!IsSubrootBlock(sch->state(), block_sref)) {
      return false;
    }
    tir::ScheduleState state = sch->state();
    if (!IsCompleteBlock(state, block_sref,
                         GetScopeRoot(state, block_sref, /*require_stage_pipeline=*/false))) {
      return false;
    }
    Array<tir::StmtSRef> loop_srefs = tir::GetLoops(block_sref);
    for (const tir::StmtSRef& loop_sref : loop_srefs) {
      if (!HasSingleChild(loop_sref)) {
        return false;
      }
    }
    Array<PrimExpr> binds = tir::GetBlockRealize(block_sref)->iter_values;
    for (const PrimExpr& bind : binds) {
      if (!bind->IsInstance<IntImmNode>() && !bind->IsInstance<tir::VarNode>()) {
        return false;
      }
    }
    return true;
  }

  Array<Schedule> Apply(const SearchTask& task, const Schedule& sch,
                        const BlockRV& block_rv) const {
    tir::StmtSRef block_sref = sch->GetSRef(block_rv);
    if (!IsFreeBlock(sch, block_sref)) {
      return {sch};
    }
    Array<BlockRV> consumers = sch->GetConsumers(block_rv);
    if (consumers.size() != 1) {
      return {sch};
    }
    BlockRV consumer = consumers[0];
    // Try to compute `block_rv` at `consumer`
    for (;;) {
      LoopRV compute_at_loc = sch->SampleComputeLocation(consumer);
      try {
        sch->ComputeAt(block_rv, compute_at_loc, true);
      } catch (const dmlc::Error& e) {
        // ComputeAt fails, cleanup the following before re-try:
        // 1) trace: instruction & decisions
        // 2) sym_tab
        sch->trace().value()->Pop();
        sch->RemoveRV(compute_at_loc);
        continue;
      }
      break;
    }
    return {sch};
  }
};

SearchRule RandomComputeLocation() {
  auto f_apply = [](SearchTask task, Schedule sch, BlockRV block) -> Array<Schedule> {
    return RuleRandomComputeLocation().Apply(task, sch, block);
  };
  return SearchRule("random_compute_location", f_apply);
}

/********** ParallelizeVectorizeUnroll **********/

class RuleParallelizeVectorizeUnroll {
 public:
  int max_jobs_per_core;
  int max_vectorize_extent;
  Array<Integer> unroll_max_steps;
  bool unroll_explicit;

  mutable std::atomic<int> warned_num_cores_missing;

  explicit RuleParallelizeVectorizeUnroll(int max_jobs_per_core, int max_vectorize_extent,
                                          const Array<Integer>& unroll_max_steps,
                                          bool unroll_explicit)
      : max_jobs_per_core(max_jobs_per_core),
        max_vectorize_extent(max_vectorize_extent),
        unroll_max_steps(unroll_max_steps),
        unroll_explicit(unroll_explicit),
        warned_num_cores_missing(0) {}

  RuleParallelizeVectorizeUnroll(const RuleParallelizeVectorizeUnroll& other) noexcept
      : max_jobs_per_core(other.max_jobs_per_core),
        max_vectorize_extent(other.max_vectorize_extent),
        unroll_max_steps(other.unroll_max_steps),
        unroll_explicit(other.unroll_explicit),
        warned_num_cores_missing(static_cast<int>(other.warned_num_cores_missing)) {}

  Array<Schedule> Apply(const SearchTask& task, const Schedule& sch,
                        const BlockRV& block_rv) const {
    BlockRV root_rv = sch->GetBlock("root");
    if (HasAnyAnn(sch->GetSRef(root_rv))) {
      return {sch};
    }
    // Parallelization
    if (max_jobs_per_core != -1) {
      int max_extent =
          GetTargetNumCores(task->target, &warned_num_cores_missing) * max_jobs_per_core;
      sch->MarkBlock(root_rv, tir::attr::auto_parallel_extent, Integer(max_extent));
    }
    // Vectorization
    if (max_vectorize_extent != -1) {
      sch->MarkBlock(root_rv, tir::attr::auto_vectorize_extent, Integer(max_vectorize_extent));
    }
    // Unroll
    if (!unroll_max_steps.empty()) {
      int n = unroll_max_steps.size();
      double prob = 1.0 / n;
      Array<FloatImm> probs(n, FloatImm(DataType::Float(64), prob));
      PrimExpr max_step = sch->SampleCategorical(unroll_max_steps, probs);
      if (unroll_explicit) {
        sch->MarkBlock(root_rv, tir::attr::auto_unroll_explicit, max_step);
      } else {
        sch->MarkBlock(root_rv, tir::attr::auto_unroll_implicit, max_step);
      }
    }
    return {sch};
  }
};

SearchRule ParallelizeVectorizeUnroll(int max_jobs_per_core, int max_vectorize_extent,
                                      Array<Integer> unroll_max_steps, bool unroll_explicit) {
  RuleParallelizeVectorizeUnroll rule(max_jobs_per_core, max_vectorize_extent, unroll_max_steps,
                                      unroll_explicit);
  auto f_apply = [rule](SearchTask task, Schedule sch, BlockRV block) -> Array<Schedule> {
    return rule.Apply(task, sch, block);
  };
  return SearchRule("parallelize_vectorize_unroll", f_apply);
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
      Array<LoopRV> loop_rvs = sch->GetLoops(block_rv);
      for (const LoopRV& loop_rv : loop_rvs) {
        loop2rv.Set(sch->GetSRef(loop_rv), loop_rv);
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
      const tir::ForNode* block_loop = block_loop_sref->StmtAs<tir::ForNode>();
      const tir::ForNode* desc_loop = kv.second.get();
      ICHECK(block_loop != nullptr && desc_loop != nullptr);
      // Extract the loop extent
      PrimExpr block_extent = analyzer.Simplify(block_loop->extent);
      PrimExpr desc_extent = analyzer.Simplify(desc_loop->extent);
      const auto* int_block_extent = block_extent.as<IntImmNode>();
      const auto* int_desc_extent = desc_extent.as<IntImmNode>();
      ICHECK(int_block_extent != nullptr && int_desc_extent != nullptr);
      // Check divisibility
      int64_t total = int_block_extent->value;
      int64_t inner = int_desc_extent->value;
      ICHECK_EQ(total % inner, 0);
      int64_t outer = int_block_extent->value / int_desc_extent->value;
      // Do the split
      Array<LoopRV> split =
          sch->Split(loop2rv.at(block_loop_sref), {Integer(outer), Integer(inner)});
      ICHECK_EQ(split.size(), 2);
      inner_loops.insert(sch->GetSRef(split[1]).operator->());
      // The inner split will be reordered to the loop domain that is tensorized
      int desc_loop_index = info->desc_loop_indexer.at(GetRef<tir::For>(desc_loop));
      reorder_suffix[desc_loop_index] = split[1];
    }
    // Reorder the loops
    std::vector<LoopRV> reorder_list;
    bool meet = false;
    Array<LoopRV> all_loops = sch->GetLoops(block_rv);
    for (const LoopRV& loop : all_loops) {
      if (inner_loops.count(sch->GetSRef(loop).operator->())) {
        meet = true;
      } else if (meet) {
        reorder_list.push_back(loop);
      }
    }
    reorder_list.insert(reorder_list.end(), reorder_suffix.begin(), reorder_suffix.end());
    sch->Reorder(reorder_list);
    // Do blockize
    if (!reorder_suffix.empty()) {
      sch->Blockize(reorder_suffix[0]);
    }
    // Annotate the block
    sch->MarkBlock(block_rv, tir::attr::auto_tensorize, Integer(1));
  }

  /*! \brief Rule application */
  Array<Schedule> Apply(const SearchTask& task, const Schedule& sch, const BlockRV& block_rv) {
    tir::StmtSRef block_sref = sch->GetSRef(block_rv);
    Array<Schedule> result{sch};
    Optional<Schedule> next_sch = NullOpt;
    for (const tir::TensorIntrin& intrin : tensor_intrins) {
      if (!next_sch.defined()) {
        next_sch = Downcast<Schedule>(sch->Copy(sch->ForkSeed()));
      }
      Schedule cur_sch = next_sch.value();
      if (Optional<TensorizeInfo> opt_tensorize_info =
              GetTensorizeLoopMapping(cur_sch->state(), block_sref, intrin->description)) {
        BlockizeAndMark(cur_sch, block_rv, intrin->description, opt_tensorize_info.value().get());
        result.push_back(cur_sch);
        next_sch = NullOpt;
      }
    }
    return result;
  }
};

SearchRule MarkTensorize(Array<tir::TensorIntrin> tensor_intrins) {
  auto f_apply = [tensor_intrins{std::move(tensor_intrins)}](SearchTask task, Schedule sch,
                                                             BlockRV block) -> Array<Schedule> {
    RuleMarkTensorize rule(tensor_intrins);
    return rule.Apply(task, sch, block);
  };
  return SearchRule("mark_tensorize", f_apply);
}

/********** SimplifyComputeWithConstTensor **********/
class RuleSimplifyComputeWithConstTensor {
 public:
  /*! \brief The maximum size of the innermost factor */
  int max_innermost_factor;

  explicit RuleSimplifyComputeWithConstTensor(int max_innermost_factor)
      : max_innermost_factor(max_innermost_factor) {}

  Array<Schedule> Apply(const SearchTask& task, const Schedule& sch, const BlockRV& block_rv) {
    tir::StmtSRef block_sref = sch->GetSRef(block_rv);
    const tir::BlockRealize& block_realize = tir::GetBlockRealize(block_sref);
    const tir::Block& block = block_realize->block;
    auto it = block->annotations.find(
        tvm::auto_scheduler::SearchPolicyKey::simplify_const_tensor_indices);
    if (it == block->annotations.end()) {
      return {sch};
    }

    // indices of the const tensor
    Array<String> const_indices = Downcast<Array<String>>((*it).second);
    // find the corresponding loops
    std::unordered_set<const tir::VarNode*> unrolled_loop_vars;
    for (size_t i = 0; i < block->iter_vars.size(); i++) {
      const auto& var_name = block->iter_vars[i]->var->name_hint;
      // only consider simple bindings
      if (std::find(const_indices.begin(), const_indices.end(), var_name) != const_indices.end() &&
          block_realize->iter_values[i].as<tir::VarNode>()) {
        unrolled_loop_vars.insert(block_realize->iter_values[i].as<tir::VarNode>());
      }
    }

    Array<LoopRV> axes = sch->GetLoops(block_rv);
    Array<LoopRV> unrolled_inner_iters;
    Array<LoopRV> outer_iters;

    size_t tile_level = 2;

    // unroll the loops of the const tensor indices
    for (const LoopRV& ax : axes) {
      tir::StmtSRef loop_sref = sch->GetSRef(ax);
      const auto* for_node = loop_sref->StmtAs<tir::ForNode>();
      if (unrolled_loop_vars.count(for_node->loop_var.get())) {
        sch->Unroll(ax);
        unrolled_inner_iters.push_back(ax);
      } else {
        outer_iters.push_back(ax);
      }
    }

    Array<Array<LoopRV>> tiled_outer_iters;
    // tile spatial axes
    for (const LoopRV& ax : outer_iters) {
      Array<PrimExpr> factors = sch->SamplePerfectTile(ax, tile_level, max_innermost_factor);
      tiled_outer_iters.push_back(sch->Split(ax, AsOptArray<PrimExpr, PrimExpr>(factors)));
    }
    Array<LoopRV> new_loop_order;
    new_loop_order.reserve(tiled_outer_iters.size() * tile_level + unrolled_inner_iters.size());
    for (size_t i = 0; i < tile_level; i++) {
      for (size_t j = 0; j < tiled_outer_iters.size(); j++) {
        new_loop_order.push_back(tiled_outer_iters[j][i]);
      }
    }
    std::copy(unrolled_inner_iters.begin(), unrolled_inner_iters.end(),
              std::back_inserter(new_loop_order));
    sch->Reorder(new_loop_order);
    return {sch};
  }
};

SearchRule SimplifyComputeWithConstTensor(int max_innermost_factor) {
  auto f_apply = [max_innermost_factor](SearchTask task, Schedule sch,
                                        BlockRV block) -> Array<Schedule> {
    return RuleSimplifyComputeWithConstTensor(max_innermost_factor).Apply(task, sch, block);
  };
  return SearchRule("simplify_compute_with_const_tensor", f_apply);
}

/********** Helper Functions for RuleAddRFactor and RuleCrossThreadReduction **********/

/*!
 * \brief Reorder the reduction loops to innermost positions if needed.
 * \param sch The Meta-Schedule schedule
 * \param block_rv The block where to apply the reorder
 * \param fused_reduce_loop The fusion-generated loop to return.
 * \param num_spatial_loops The number of spatial loops to return.
 * \note Before invoking this helper function, make sure that the block has only spatial and
 *       reduction loop axes.
 */
void ReorderAndFuseReductionLoops(const Schedule& sch, const BlockRV& block_rv,
                                  LoopRV* fused_reduce_loop, int* num_spatial_loops) {
  Array<LoopRV> loops = sch->GetLoops(block_rv);
  Array<tir::StmtSRef> loop_srefs;
  for (const LoopRV& loop_rv : loops) {
    loop_srefs.push_back(sch->GetSRef(loop_rv));
  }

  Array<LoopRV> new_order;
  // Step 1. Add spatial loops.
  *num_spatial_loops = 0;
  for (int i = 0; i < static_cast<int>(loops.size()); ++i) {
    if (GetLoopIterType(sch->state(), loop_srefs[i]) == tir::kDataPar) {
      new_order.push_back(loops[i]);
      (*num_spatial_loops)++;
    }
  }
  // Step 2. Add reduction loops.
  Array<LoopRV> reduction_loops;
  for (int i = 0; i < static_cast<int>(loops.size()); ++i) {
    if (GetLoopIterType(sch->state(), loop_srefs[i]) == tir::kCommReduce) {
      new_order.push_back(loops[i]);
      reduction_loops.push_back(loops[i]);
    }
  }
  // Step 3. Apply reordering if new_order differs from the original order.
  CHECK_EQ(new_order.size(), loops.size());
  bool need_reorder = false;
  for (int i = 0; i < static_cast<int>(loops.size()); ++i) {
    if (!new_order[i].same_as(loops[i])) {
      need_reorder = true;
      break;
    }
  }
  if (need_reorder) {
    sch->Reorder(new_order);
  }

  // Step 4. Fuse all the reduction loops if there are multiple reduction loops.
  CHECK(!reduction_loops.empty()) << "ValueError: There should be at least one reduction loop";
  if (reduction_loops.size() > 1) {
    *fused_reduce_loop = sch->Fuse(reduction_loops);
  } else {
    *fused_reduce_loop = reduction_loops[0];
  }
}

/*!
 * \brief Check whether the given `block` is in thread scope, i.e., some of its outer loop is bound
 *        to threadIdx.
 * \param sch The Meta-Schedule schedule
 * \param block The block to be checked
 * \return A boolean flag indicating whether the block is in thread scope.
 */
bool InThreadScope(const Schedule& sch, const BlockRV& block) {
  const Array<LoopRV>& axes = sch->GetLoops(block);
  for (const LoopRV& loop_rv : axes) {
    const tir::For& loop = sch->Get(loop_rv);
    if (!loop->thread_binding.defined()) {
      continue;
    }
    if (std::string(loop->thread_binding.value()->thread_tag).substr(0, 9) == "threadIdx") {
      return true;
    }
  }
  return false;
}

/********** AddRFactor **********/

class RuleAddRFactor {
 public:
  int max_innermost_factor;
  int max_jobs_per_core;
  mutable std::atomic<int> warned_num_cores_missing;

  explicit RuleAddRFactor(int max_jobs_per_core, int max_innermost_factor)
      : max_innermost_factor(max_innermost_factor),
        max_jobs_per_core(max_jobs_per_core),
        warned_num_cores_missing(0) {}

  RuleAddRFactor(const RuleAddRFactor& other) noexcept
      : max_innermost_factor(other.max_innermost_factor),
        max_jobs_per_core(other.max_jobs_per_core),
        warned_num_cores_missing(other.warned_num_cores_missing.load()) {}

  /*! \brief Rule application */
  Array<Schedule> Apply(const SearchTask& task, const Schedule& sch,
                        const BlockRV& block_rv) const {
    // Check the conditions of the rule.
    tir::StmtSRef block_sref = sch->GetSRef(block_rv);
    const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
    if (HasAnyAnn(block_sref)) {
      return {sch};
    }
    if (block->writes.size() != 1) {
      return {sch};
    }
    int target_num_cores = GetTargetNumCores(task->target, &warned_num_cores_missing);
    if (!NeedsRFactorOrCrossThreadReduction(
            sch->state(), block_sref, target_num_cores * max_jobs_per_core, target_num_cores) ||
        HasCacheWriteBlock(sch, block_rv, 0)) {
      return {sch};
    }

    // Make a copy of the original schedule.
    Schedule ori_sch = Downcast<Schedule>(sch->Copy(sch->ForkSeed()));

    // Reorder the loop axes if reduction loops are not innermost.
    // After the reordering, fuse all the reduction loops.
    int num_spatial_loops;
    LoopRV fused_reduce_loop;
    ReorderAndFuseReductionLoops(sch, block_rv, &fused_reduce_loop, &num_spatial_loops);

    Array<ExprRV> factors = sch->SamplePerfectTile(fused_reduce_loop, 2, max_innermost_factor);
    const Array<LoopRV>& split_res =
        sch->Split(fused_reduce_loop, AsOptArray<ExprRV, PrimExpr>(factors));
    Array<Schedule> res;
    for (const LoopRV& split_loop : split_res) {
      Schedule sch_tmp = Downcast<Schedule>(sch->Copy(sch->ForkSeed()));
      const BlockRV& block_rf = sch_tmp->RFactor(split_loop, num_spatial_loops);
      Array<LoopRV> axes = sch_tmp->GetLoops(block_rf);
      CHECK_GT(static_cast<int>(axes.size()), num_spatial_loops);

      if (split_loop.same_as(split_res[1])) {
        Array<LoopRV> new_order;
        for (int i = 0; i < static_cast<int>(axes.size()); ++i) {
          if (i != num_spatial_loops) {
            new_order.push_back(axes[i]);
          }
        }
        new_order.push_back(axes[num_spatial_loops]);
        sch_tmp->Reorder(new_order);
      }
      res.push_back(sch_tmp);
    }

    res.push_back(ori_sch);
    return res;
  }
};

SearchRule AddRFactor(int max_jobs_per_core, int max_innermost_factor) {
  RuleAddRFactor rule(max_jobs_per_core, max_innermost_factor);
  auto f_apply = [rule](SearchTask task, Schedule sch, BlockRV block) -> Array<Schedule> {
    return rule.Apply(task, sch, block);
  };
  return SearchRule("add_rfactor", f_apply);
}

/********** CrossThreadReduction **********/

class RuleCrossThreadReduction {
 public:
  /*! \brief Rule application */
  Array<Schedule> Apply(const SearchTask& task, const Schedule& sch,
                        const BlockRV& block_rv) const {
    // Check target. And extract `max_threads_per_block` and `warp_size` from target.
    CHECK(task->target->kind->name == "cuda")
        << "TargetError: Search rule \"CrossThreadReduction\" can only run on CUDA";
    Optional<Integer> opt_max_threads_per_block =
        task->target->GetAttr<Integer>("max_threads_per_block");
    Optional<Integer> opt_warp_size = task->target->GetAttr<Integer>("thread_warp_size");
    CHECK(opt_max_threads_per_block.defined());
    CHECK(opt_warp_size.defined());
    int64_t max_threads_per_block = opt_max_threads_per_block.value()->value;
    int64_t warp_size = opt_warp_size.value()->value;

    // Check the conditions of the rule.
    const tir::StmtSRef& block_sref = sch->GetSRef(block_rv);
    const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
    if (HasAnyAnn(block_sref)) {
      return {sch};
    }
    if (block->writes.size() != 1) {
      return {sch};
    }
    if (!NeedsRFactorOrCrossThreadReduction(sch->state(), block_sref, max_threads_per_block,
                                            warp_size) ||
        HasCacheWriteBlock(sch, block_rv, 0)) {
      return {sch};
    }

    // Make a copy of the original schedule. The new copy is used for scheduling.
    Schedule tmp_sch = Downcast<Schedule>(sch->Copy(sch->ForkSeed()));
    // Check the opportunity for kernel fusion. We say "fusible", if we can compute_at the block to
    // its consumer. We want to fuse as much as possible because it results in significantly faster
    // schedule.
    bool fusible = false;
    // `target_block` is the consumer block that we want to compute_at the input block to.
    Optional<BlockRV> target_block = NullOpt;
    Array<BlockRV> consumers = tmp_sch->GetConsumers(block_rv);
    if (!consumers.empty()) {
      // Calculate the lowest common ancestor of all the consumers.
      std::vector<tir::StmtSRef> consumers_sref;
      consumers_sref.reserve(consumers.size());
      for (const BlockRV& consumer : consumers) {
        consumers_sref.push_back(tmp_sch->GetSRef(consumer));
      }
      const tir::StmtSRef& root_sref = tir::GetSRefTreeRoot(tmp_sch->GetSRef(block_rv));
      const tir::StmtSRef& lca_sref = consumers.size() == 1
                                          ? consumers_sref[0]
                                          : tir::LowestCommonAncestor(consumers_sref, root_sref);

      // * If the LCA is a block, we check whether the LCA appears in the consumers. If so, we let
      //   `target_block` be the LCA block.
      // * If the LCA is a loop, we check whether the loop was bound to blockIdx.x. If so, it means
      //   that we once applied this rule to the blocks under this loop. In this case we let
      //   `target_block` be the first consumer to fuse further.
      if (lca_sref->StmtAs<tir::BlockNode>() != nullptr) {
        // LCA is a block.
        for (int i = 0; i < static_cast<int>(consumers.size()); ++i) {
          if (consumers_sref[i]->stmt == lca_sref->stmt) {
            ICHECK(!target_block.defined());
            target_block = consumers[i];
          }
        }
        // Simple check: if there's only one consumer, then `target_block` shouldn't be NullOpt.
        if (consumers.size() == 1) {
          ICHECK(target_block.defined());
        }
      } else {
        // LCA is a for loop.
        target_block = consumers[0];
      }
    }

    // To perform the fusion, we need to calculate the number of common dimensions that the input
    // block and `target_block` have. It indicates the target loop used for compute_at.
    int num_common_axes = -1;
    if (target_block.defined()) {
      num_common_axes =
          GetNumberCommonOutputDims(tmp_sch->Get(block_rv), tmp_sch->Get(target_block.value()));
      if (num_common_axes > 0 &&
          !NeedsMultiLevelTiling(tmp_sch->state(), tmp_sch->GetSRef(target_block.value()))) {
        fusible = true;
      }
    }

    if (fusible) {
      ICHECK(target_block.defined());
      const BlockRV& target_block_rv = target_block.value();
      Array<LoopRV> target_block_loops = tmp_sch->GetLoops(target_block_rv);

      // Step 1. If the outer loops of `target_block` haven't been bound to threadIdx, we should
      // first bound the innermost outer loop of `target_block` to threadIdx. Possibly we need to
      // split the loop before binding.
      if (!InThreadScope(tmp_sch, target_block_rv)) {
        const tir::StmtSRef& target_block_sref = tmp_sch->GetSRef(target_block_rv);
        CHECK(!IsReductionBlock(
            tmp_sch->state(), target_block_sref,
            GetScopeRoot(tmp_sch->state(), target_block_sref, /*require_stage_pipeline=*/false)))
            << "ValueError: In this case the target block is expected to be a complete block.";
        const LoopRV& loop_to_split = target_block_loops.back();
        Array<LoopRV> split_res = tmp_sch->Split(loop_to_split, {NullOpt, Integer(warp_size)});
        ICHECK_EQ(split_res.size(), 2);
        tmp_sch->Bind(split_res[1], "threadIdx.x");
        target_block_loops.pop_back();
        target_block_loops.push_back(split_res[0]);
        target_block_loops.push_back(split_res[1]);
      }

      // Step 2. Get the compute_at target loop, and do the compute_at.
      ICHECK_GT(target_block_loops.size(), num_common_axes);
      const LoopRV& target_loop = target_block_loops[num_common_axes - 1];
      tmp_sch->ComputeAt(block_rv, target_loop, true);
      // Step 3. Set the storage scope of the output buffer.
      tmp_sch->SetScope(block_rv, 0, "shared");
      // Step 4. Reorder the loop axes if reduction loops are not innermost. After the reordering,
      // fuse all the reduction loops.
      int num_spatial_loops;
      LoopRV fused_reduce_loop;
      ReorderAndFuseReductionLoops(tmp_sch, block_rv, &fused_reduce_loop, &num_spatial_loops);
      // Step 5. Split the fused reduction loop and bind the inner one to threadIdx, bind the
      // target_loop to blockIdx. Note that the target loop might have been bound to blockIdx
      // before.
      Array<LoopRV> split_res = tmp_sch->Split(fused_reduce_loop, {NullOpt, Integer(warp_size)});
      ICHECK_EQ(split_res.size(), 2);
      tmp_sch->Bind(split_res[1], "threadIdx.x");
    } else {
      // If we cannot do the fusion, just fuse all the reduction loops, split the fused loop and
      // bound the inner one to threadIdx.
      int num_spatial_loops;
      LoopRV fused_reduce_loop;
      ReorderAndFuseReductionLoops(tmp_sch, block_rv, &fused_reduce_loop, &num_spatial_loops);
      Array<LoopRV> split_res = tmp_sch->Split(fused_reduce_loop, {NullOpt, Integer(warp_size)});
      tmp_sch->Bind(split_res[1], "threadIdx.x");
    }

    return {tmp_sch, sch};
  }
};

SearchRule CrossThreadReduction() {
  RuleCrossThreadReduction rule;
  auto f_apply = [rule](SearchTask task, Schedule sch, BlockRV block) -> Array<Schedule> {
    return rule.Apply(task, sch, block);
  };
  return SearchRule("cross_thread_reduction", f_apply);
}

/********** SpecialComputeLocationGPU **********/
class RuleSpecialComputeLocationGPU {
 public:
  Array<Schedule> Apply(const SearchTask& task, const Schedule& sch,
                        const BlockRV& block_rv) const {
    if (sch->GetProducers(block_rv).empty()) {
      return {sch};
    }
    tir::StmtSRef block_sref = sch->GetSRef(block_rv);
    if (!RuleInlinePureSpatial::NeedsInline(sch, block_sref, false)) {
      return {sch};
    }
    Array<BlockRV> consumers = sch->GetConsumers(block_rv);
    tir::Block block = sch->Get(block_rv);
    if (consumers.size() != 1 ||
        !sch->Get(consumers[0])
             ->annotations.count(
                 tvm::auto_scheduler::SearchPolicyKey::simplify_const_tensor_indices)) {
      return {sch};
    }
    Array<tir::LoopRV> consumer_loops = sch->GetLoops(consumers[0]);
    for (size_t i = 0; i < consumer_loops.size(); i++) {
      tir::StmtSRef loop_sref = sch->GetSRef(consumer_loops[i]);
      if (tir::GetLoopIterType(sch->state(), loop_sref) == tir::kUnrolled) {
        sch->ComputeAt(block_rv, consumer_loops[i - 1], true);
        sch->SetScope(block_rv, 0, "local");
        break;
      }
    }
    return {sch};
  }
};

SearchRule SpecialComputeLocationGPU() {
  auto f_apply = [](SearchTask task, Schedule sch, BlockRV block) -> Array<Schedule> {
    return RuleSpecialComputeLocationGPU().Apply(task, sch, block);
  };
  return SearchRule("special_compute_location_gpu", f_apply);
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
   * \return The result of rule application
   * \sa SearchRuleNode::Apply
   */
  static Array<Schedule> SearchRuleApply(SearchRule rule, SearchTask task, Schedule sch,
                                         BlockRV block) {
    return rule->Apply(task, sch, block);
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
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.InlinePureSpatial")
    .set_body_typed(InlinePureSpatial);
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.MultiLevelTiling").set_body_typed(MultiLevelTiling);
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.RandomComputeLocation")
    .set_body_typed(RandomComputeLocation);
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.ParallelizeVectorizeUnroll")
    .set_body_typed(ParallelizeVectorizeUnroll);
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.MarkTensorize").set_body_typed(MarkTensorize);
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.SimplifyComputeWithConstTensor")
    .set_body_typed(SimplifyComputeWithConstTensor);
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.AddRFactor").set_body_typed(AddRFactor);
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.CrossThreadReduction")
    .set_body_typed(CrossThreadReduction);
TVM_REGISTER_GLOBAL("meta_schedule.search_rule.SpecialComputeLocationGPU")
    .set_body_typed(SpecialComputeLocationGPU);

}  // namespace meta_schedule
}  // namespace tvm
