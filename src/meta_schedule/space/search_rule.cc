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

#include "../../tir/schedule/schedule_common.h"
#include "../analysis.h"
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

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

  /*! \brief Rule application */
  Array<Schedule> Apply(const SearchTask& task, const Schedule& sch,
                        const BlockRV& block_rv) const {
    tir::StmtSRef block_sref = sch->Eval(block_rv);
    if (!IsSpatial(sch->sch, block_sref)) {
      return {sch};
    }
    if (IsOutputBlock(sch->sch, block_sref)) {
      return {sch};
    }
    if (!strict_mode || IsStrictlyInlineable(sch->sch, block_sref)) {
      sch->ComputeInline(block_rv);
      return {sch};
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
    CHECK(!s_idx->empty())
        << "ValueError: Invalid tiling structure, cannot find any 'S' in the format";
    CHECK(!r_idx->empty())
        << "ValueError: Invalid tiling structure, cannot find any 'R' in the format";
    return num_s_in_prefix;
  }

  /*!
   * \brief Extract the buffer indices of the read buffers, excluding those update buffers
   * \param block_sref The block to be extracted
   * \return A vector of integers, the indices of the read buffers
   */
  static std::vector<int> GetReadBufferIndices(const tir::StmtSRef& block_sref) {
    const auto* block = block_sref->GetStmt<tir::BlockNode>();
    CHECK(block) << "TypeError: Expects 'Block', but gets: " << block_sref->stmt->GetTypeKey();
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
  static LoopRV FuseBufferAxes(const Schedule& sch, const BlockRV& block_rv,
                               const tir::Buffer& buffer) {
    Array<LoopRV> to_fuse;
    Array<LoopRV> axes = sch->GetAxes(block_rv);
    int n_axes = axes.size();
    int ndim = buffer->shape.size();
    for (int i = n_axes - ndim; i < n_axes; ++i) {
      to_fuse.push_back(axes[i]);
    }
    return sch->Fuse(to_fuse);
  }

  explicit RuleMultiLevelTiling(String structure, bool must_cache_read, String cache_read_scope,
                                bool can_cache_write, bool must_cache_write,
                                String cache_write_scope, Array<Integer> fusion_levels_,
                                Optional<Integer> vector_load_max_len,
                                Optional<Array<String>> tile_binds)
      : structure(structure),
        must_cache_read(must_cache_read),
        cache_read_scope(cache_read_scope),
        can_cache_write(can_cache_write),
        must_cache_write(must_cache_write),
        cache_write_scope(cache_write_scope),
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
    Schedule sch = state.sch;
    BlockRV block_rv = state.block_rv;
    tir::StmtSRef block_sref = sch->Eval(block_rv);
    // Find the only-consumer for the block
    // If the only-consumer can be fused, then do not add any write cache
    Array<BlockRV> consumers = sch->GetConsumers(state.block_rv);
    if (consumers.size() == 1) {
      BlockRV consumer_rv = consumers[0];
      tir::StmtSRef consumer_sref = sch->Eval(consumer_rv);
      // Check if it can be directly fused
      if ((IsSpatial(sch->sch, block_sref) || IsSpatial(sch->sch, consumer_sref)) &&
          IsElementWiseMatch(sch->sch, block_sref, consumer_sref)) {
        state.write_cache = consumer_rv;
        state.write_cache_is_added = false;
        return {state};
      }
    }
    std::vector<State> result;
    // Case 0. Do not add write cache, then fusion won't happen later either
    if (!must_cache_write) {
      result.push_back(state);
    }
    // Case 1. Add a write cache
    if (can_cache_write) {
      // Fork a new schedule
      state.sch = sch->Copy(sch->sampler.ForkSeed());
      // The original block to tiled
      state.block_rv = state.sch->CacheWrite(block_rv, 0, cache_write_scope);
      // The cache write block
      state.write_cache = block_rv;
      state.write_cache_is_added = true;
      result.push_back(std::move(state));
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
    tir::StmtSRef block_sref = sch->Eval(block_rv);
    // Find all indices of the read buffers
    std::vector<int> read_buffer_indices = GetReadBufferIndices(block_sref);
    // Enumerate all buffers that are read but not written
    for (int i : read_buffer_indices) {
      const auto* block = block_sref->GetStmt<tir::BlockNode>();
      const tir::Buffer& buffer = block->reads[i]->buffer;
      // Do cache_read
      BlockRV cache_read_block = sch->CacheRead(block_rv, i, cache_read_scope);
      // Insert cache_read block to the proper place
      const Array<LoopRV>& r_tiles = state.tiles[r_idx.front()];
      CHECK(!r_tiles.empty()) << "ValueError: Cannot find any reduction loop in the block";
      sch->ComputeAt(cache_read_block, r_tiles.back());
      // Fuse the iterators of the cache_read
      LoopRV fused = FuseBufferAxes(sch, cache_read_block, buffer);
      // Do cooperative fetching
      if (vector_load_max_len.defined()) {
        int max_vec_len = vector_load_max_len.value();
        // cooperative fetch + vectorized loading
        // Split into inner and outer
        Array<tir::Var> factors = sch->SamplePerfectTile(2, fused, max_vec_len);
        CHECK_EQ(factors.size(), 2);
        Array<LoopRV> splits = sch->Split(fused, {factors[0], factors[1]});
        CHECK_EQ(splits.size(), 2);
        // Vectorize the inner loop
        sch->Vectorize(splits[1]);
        fused = splits[0];
      }
      // Add cooperative fetching
      sch->MarkLoop(fused, tir::attr::loop_type, tir::StringImm("lazy_cooperative_fetch"));
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
    // `cache_write` must be fused at some level, otherwise it has no benefit
    // On the other hand, If the only consumer is not cache_write, then we may choose not to fuse
    if (!state.write_cache_is_added) {
      result.push_back(state);
    }
    Schedule sch = state.sch;
    BlockRV consumer = state.write_cache.value();
    // Enumerate the level of tile to be fused at
    for (int level : fusion_levels) {
      const LoopRV& loop = state.tiles[level - 1].back();
      State new_state = state;
      new_state.sch = state.sch->Copy(sch->sampler.ForkSeed());
      new_state.sch->ReverseComputeAt(consumer, loop);
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
    tir::StmtSRef block_sref = sch->Eval(block_rv);
    if (HasAnyAnn(block_sref)) {
      return {sch};
    }
    if (!NeedsMultiLevelTiling(sch->sch, block_sref)) {
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

SearchRule MultiLevelTiling(String structure, bool must_cache_read, String cache_read_scope,
                            bool can_cache_write, bool must_cache_write, String cache_write_scope,
                            Array<Integer> fusion_levels, Optional<Integer> vector_load_max_len,
                            Optional<Array<String>> tile_binds) {
  if (!can_cache_write && must_cache_write) {
    LOG(FATAL) << "ValueError: Conflict options, cannot have can_cache_write = false, and "
                  "must_cache_write = true at the same time";
  }
  RuleMultiLevelTiling rule(structure, must_cache_read, cache_read_scope, can_cache_write,
                            must_cache_write, cache_write_scope, fusion_levels, vector_load_max_len,
                            tile_binds);
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
    if (!IsSubrootBlock(sch, block_sref)) {
      return false;
    }
    Array<tir::StmtSRef> loop_srefs = sch->GetLoopsInScope(block_sref);
    for (const tir::StmtSRef& loop_sref : loop_srefs) {
      const auto* loop = loop_sref->GetStmt<tir::LoopNode>();
      CHECK(loop) << "TypeError: Expects Loop, but gets: " << loop_sref->stmt->GetTypeKey();
      if (loop->body->IsInstance<tir::SeqStmtNode>()) {
        return false;
      }
    }
    Array<PrimExpr> binds = tir::GetBlockRealize(block_sref)->binding_values;
    for (const PrimExpr& bind : binds) {
      if (!bind->IsInstance<IntImmNode>() && !bind->IsInstance<tir::VarNode>()) {
        return false;
      }
    }
    return true;
  }

  Array<Schedule> Apply(const SearchTask& task, const Schedule& sch,
                        const BlockRV& block_rv) const {
    tir::StmtSRef block_sref = sch->Eval(block_rv);
    if (!IsFreeBlock(sch->sch, block_sref)) {
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
        sch->ComputeAt(block_rv, compute_at_loc);
      } catch (const dmlc::Error& e) {
        // ComputeAt fails, cleanup the following before re-try:
        // 1) sym_tab
        // 2) decisions
        // 3) trace
        sch->trace->Pop();
        sch->sym_tab.erase(compute_at_loc);
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

  mutable bool warned_num_cores_missing;

  explicit RuleParallelizeVectorizeUnroll(int max_jobs_per_core, int max_vectorize_extent,
                                          const Array<Integer>& unroll_max_steps,
                                          bool unroll_explicit)
      : max_jobs_per_core(max_jobs_per_core),
        max_vectorize_extent(max_vectorize_extent),
        unroll_max_steps(unroll_max_steps),
        unroll_explicit(unroll_explicit),
        warned_num_cores_missing(false) {}

  int GetMaxParallelExtent(const Target& target) const {
    int num_cores = target->GetAttr<Integer>("num_cores").value_or(-1);
    if (num_cores == -1) {
      static const auto* f_cpu_count = runtime::Registry::Get("meta_schedule._cpu_count");
      CHECK(f_cpu_count)
          << "ValueError: Cannot find the packed function \"meta_schedule._cpu_count\"";
      num_cores = (*f_cpu_count)(false);
      if (!warned_num_cores_missing) {
        LOG(WARNING) << "Warning: Target does not have attribute \"num_cores\", falling back the "
                        "number of CPU cores on the local machine. The inaccuracy in number of "
                        "cores may lead to dramatically inferior performance. Falling back to "
                        "assuming "
                     << num_cores << " CPU core(s)";
        warned_num_cores_missing = true;
      }
    }
    return num_cores * max_jobs_per_core;
  }

  static bool IsLeftmostSubroot(const tir::Schedule& sch, tir::StmtSRef block_sref) {
    if (!IsSubrootBlock(sch, block_sref)) {
      return false;
    }
    tir::StmtSRefNode* child_sref = block_sref.operator->();
    for (tir::StmtSRefNode* parent_sref = child_sref->parent;;
         child_sref = parent_sref, parent_sref = child_sref->parent) {
      const auto* parent_loop = parent_sref->GetStmt<tir::LoopNode>();
      if (parent_loop == nullptr) {
        return true;
      }
      const auto* seq_stmt = parent_loop->body.as<tir::SeqStmtNode>();
      if (seq_stmt == nullptr) {
        continue;
      }
      const tir::Stmt& first_child = seq_stmt->seq[0];
      if (first_child.get() != child_sref->stmt) {
        return false;
      }
    }
    return true;
  }

  Array<Schedule> Apply(const SearchTask& task, const Schedule& sch,
                        const BlockRV& block_rv) const {
    // Extract basic information
    Array<LoopRV> loop_rvs = sch->GetAxes(block_rv);
    tir::StmtSRef block_sref = sch->Eval(block_rv);
    // Check if the block is root and leaf
    bool is_leftmost_root = IsLeftmostSubroot(sch->sch, block_sref);
    bool is_leaf = IsLeafBlock(sch->sch, block_sref);
    // Parallelization
    if (max_jobs_per_core != -1 && is_leftmost_root) {
      int max_extent = GetMaxParallelExtent(task->target);
      sch->MarkBlock(block_rv, tir::attr::auto_parallel_extent, max_extent);
    }
    // Vectorization
    if (max_vectorize_extent != -1 && is_leaf) {
      sch->MarkBlock(block_rv, tir::attr::auto_vectorize_extent, max_vectorize_extent);
    }
    // Unroll
    if (!unroll_max_steps.empty() && is_leftmost_root) {
      int n = unroll_max_steps.size();
      double prob = 1.0 / n;
      Array<FloatImm> probs(n, FloatImm(DataType::Float(64), prob));
      tir::Var max_step = sch->SampleCategorical(unroll_max_steps, probs);
      if (unroll_explicit) {
        sch->MarkBlock(block_rv, tir::attr::auto_unroll_explicit, max_step);
      } else {
        sch->MarkBlock(block_rv, tir::attr::auto_unroll_implicit, max_step);
      }
    }
    return {sch};
  }
};

SearchRule ParallelizeVectorizeUnroll(int max_jobs_per_core, int max_vectorize_extent,
                                      Array<Integer> unroll_max_steps, bool unroll_explicit) {
  RuleParallelizeVectorizeUnroll rule(max_jobs_per_core, max_vectorize_extent, unroll_max_steps,
                                      unroll_explicit);
  auto f_apply = [rule{std::move(rule)}](SearchTask task, Schedule sch,
                                         BlockRV block) -> Array<Schedule> {
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
    sch->MarkBlock(block_rv, tir::attr::auto_tensorize, Integer(1));
  }

  /*! \brief Rule application */
  Array<Schedule> Apply(const SearchTask& task, const Schedule& sch, const BlockRV& block_rv) {
    tir::StmtSRef block_sref = sch->Eval(block_rv);
    Array<Schedule> result{sch};
    Optional<Schedule> next_sch = NullOpt;
    for (const tir::TensorIntrin& intrin : tensor_intrins) {
      if (!next_sch.defined()) {
        next_sch = sch->Copy(sch->sampler.ForkSeed());
      }
      Schedule cur_sch = next_sch.value();
      if (Optional<TensorizeInfo> opt_tensorize_info =
              GetTensorizeLoopMapping(cur_sch->sch, block_sref, intrin->description)) {
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

}  // namespace meta_schedule
}  // namespace tvm
