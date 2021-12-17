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
#include <unordered_map>

#include "../utils.h"

namespace tvm {
namespace tir {
/*!
 * \brief Get the buffer dimensions for all the read buffers of a block, but marks the reduction
 * buffers' dimensions as -1
 * \param block_sref The block to be processed
 * \return The buffer dimensions for all the read buffers of a block, except for reduction buffers
 * \note The method is not designed for generic analysis and relies on assumptions in the scenario
 * of multi-level tiling, so it's intentionally kept inside this file not in the analysis header
 */
std::vector<int> GetReadBufferNDims(const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  const BufferNode* write_buffer = block->writes[0]->buffer.get();
  int n = block->reads.size();
  std::vector<int> results(n, -1);
  for (int i = 0; i < n; ++i) {
    const BufferNode* read_buffer = block->reads[i]->buffer.get();
    if (read_buffer != write_buffer) {
      results[i] = read_buffer->shape.size();
    }
  }
  return results;
}

Optional<LoopRV> TilingwithTensorIntrin(const tir::Schedule& sch, const tir::BlockRV& block_rv,
                                        const String& intrin_name) {
  Optional<tir::TensorizeInfo> opt_tensorize_info = GetTensorizeLoopMapping(
      sch->state(), sch->GetSRef(block_rv), tir::TensorIntrin::Get(intrin_name)->description);
  if (!opt_tensorize_info) return NullOpt;
  const tir::TensorizeInfoNode* info = opt_tensorize_info.value().get();
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
    Array<LoopRV> split = sch->Split(loop2rv.at(block_loop_sref), {Integer(outer), Integer(inner)});
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
  ICHECK(!reorder_suffix.empty());
  return reorder_suffix[0];
}

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

using tir::BlockRV;
using tir::ExprRV;
using tir::IterVarType;
using tir::LoopRV;
using tir::Schedule;

/*!
 * \brief Configuration of data reuse type:
 * 0) kNoReuse: no reuse is allowed, then no cache_read/write is performed.
 * 1) kMayReuse: reuse is allowed, but no reuse is explored.
 * 2) kMustReuse: reuse is allowed and no reuse is not explored.
 */
enum class ReuseType : int32_t {
  kNoReuse = 0,
  kMayReuse = 1,
  kMustReuse = 2,
};

/*!
 * \brief Converts a string to ReuseType.
 * \param str The string to be converted.
 * \return The converted ReuseType.
 */
ReuseType Str2ReuseType(const String& str) {
  if (str == "no") {
    return ReuseType::kNoReuse;
  } else if (str == "may") {
    return ReuseType::kMayReuse;
  } else if (str == "must") {
    return ReuseType::kMustReuse;
  } else {
    LOG(FATAL) << "ValueError: Unknown ReuseType: " << str;
    throw;
  }
}

/*! \brief Configuration of data reuse patterns */
struct ReuseConfig {
  /*! \brief Type of data reuse: no-reuse, may-reuse or must-reuse */
  ReuseType req;
  /*! \brief Which levels are caching stage inserted at */
  std::vector<int> levels;
  /*! \brief The storage scope */
  String scope;

  /*! \brief Default constructor: no data reuse */
  ReuseConfig() : req(ReuseType::kNoReuse) {}

  /*! \brief Construct from a configuration dictionary */
  explicit ReuseConfig(const Map<String, ObjectRef>& config)
      : req(Str2ReuseType(Downcast<String>(config.at("req")))),
        levels(support::AsVector<Integer, int>(Downcast<Array<Integer>>(config.at("levels")))),
        scope(Downcast<String>(config.at("scope"))) {
    ICHECK_EQ(config.size(), 3);
  }
};

/*! \brief The state of auto scheduling for the multi-level tiling rule */
struct State {
  /*! \brief The schedule to date */
  Schedule sch;
  /*! \brief The block to be tiled */
  BlockRV block_rv;
  /*! \brief The write cache */
  Optional<BlockRV> write_cache;
  /*! \brief Indicating if the write cache is generated by cache_write */
  bool write_cache_is_added;
  /*! \brief The loop tiles */
  Array<Array<LoopRV>> tiles;
  /*! \brief Whether Tensor Core is used for the inner computation */
  bool tensor_core_is_used;
  /*! \brief The Tensor Core cache read block A for Tensor Core computation */
  Optional<BlockRV> tensor_core_load_A;
  /*! \brief The Tensor Core cache read block B for Tensor Core computation */
  Optional<BlockRV> tensor_core_load_B;
  /*! \brief The Tensor Core cache write block for Tensor Core computation */
  Optional<BlockRV> tensor_core_store;

  /*! \brief Default constructor */
  explicit State(Schedule sch, BlockRV block_rv, Optional<BlockRV> write_cache = NullOpt,
                 bool write_cache_is_added = false, Array<Array<LoopRV>> tiles = {},
                 bool tensor_core_is_used = false)
      : sch(sch),
        block_rv(block_rv),
        write_cache(write_cache),
        write_cache_is_added(write_cache_is_added),
        tiles(tiles),
        tensor_core_is_used(tensor_core_is_used) {}
};

/*!
 * \brief Helper to apply a sub-rule to a list of auto scheduling states
 * \tparam FLambda The type of the sub-rule functor
 * \param states The list of states to be applied
 * \return The list of states after applying the sub-rule
 */
template <class FLambda>
std::vector<State> SubRule(std::vector<State> states, FLambda sub_rule) {
  std::vector<State> results;
  for (auto&& state : states) {
    std::vector<State> next = sub_rule(std::move(state));
    results.insert(results.end(),                          //
                   std::make_move_iterator(next.begin()),  //
                   std::make_move_iterator(next.end()));
  }
  return results;
}

/*!
 * \brief The mega rule: multi-level tiling with data reuse
 */
class MultiLevelTilingNode : public ScheduleRuleNode {
 public:
  // SubRule 0. detect compute intrin
  inline std::vector<State> DetectTensorCore(State state) const;
  // SubRule 1. add write cache
  inline std::vector<State> AddWriteReuse(State state) const;
  // SubRule 2. tile the loop nest
  inline std::vector<State> TileLoopNest(State state) const;
  // SubRule 3. add read cache
  inline std::vector<State> AddReadReuse(State state) const;
  // SubRule 4. fuse write cache
  inline std::vector<State> FuseWriteReuse(State state) const;

  State TensorCoreLoad(State state) const {
    // Add the cache read stage for Tensor Core
    state.tensor_core_load_A = state.sch->CacheRead(state.block_rv, 1, "wmma.matrix_a");
    state.tensor_core_load_B = state.sch->CacheRead(state.block_rv, 2, "wmma.matrix_b");
    const Array<LoopRV>& r_tiles = state.tiles[r_indices_.back()];
    // Insert cache_read block to the proper place
    ICHECK(!r_tiles.empty()) << "ValueError: Cannot find any reduction loop in the block";
    state.sch->ComputeAt(state.tensor_core_load_A.value(), r_tiles.back(), true);
    state.sch->ComputeAt(state.tensor_core_load_B.value(), r_tiles.back(), true);
    // Annotate the block
    state.sch->Annotate(state.tensor_core_load_A.value(), tir::attr::meta_schedule_auto_tensorize,
                        String("wmma_load_a"));
    state.sch->Annotate(state.tensor_core_load_B.value(), tir::attr::meta_schedule_auto_tensorize,
                        String("wmma_load_b"));
    return state;
  }

  State TensorCoreStore(State state) const {
    // Add the cache read stage for Tensor Core
    state.tensor_core_store = state.sch->CacheWrite(state.block_rv, 0, "wmma.accumulator");
    // Annotate the block
    state.sch->Annotate(state.tensor_core_store.value(), tir::attr::meta_schedule_auto_tensorize,
                        String("wmma_store"));
    return state;
  }

  State TensorCoreStoreFusion(State state, int level) const {
    const LoopRV& loop = state.tiles[level].back();
    state.sch->ReverseComputeAt(state.tensor_core_store.value(), loop, true);
    return state;
  }

  BlockRV GetRootBlockRV(const Schedule& sch, BlockRV block_rv) const {
    const tir::StmtSRefNode* block = sch->GetSRef(block_rv).get();
    for (; block->parent != nullptr; block = block->parent)
      ;
    for (const auto& kv : sch->mod()->functions) {
      const GlobalVar& gv = kv.first;
      const BaseFunc& base_func = kv.second;
      if (const auto* func = base_func.as<tir::PrimFuncNode>()) {
        const tir::BlockNode* root = func->body.as<tir::BlockRealizeNode>()->block.get();
        if (root == block->StmtAs<tir::BlockNode>()) {
          BlockRV root_rv = sch->GetBlock(root->name_hint, gv->name_hint);
          return root_rv;
        }
      }
    }
    ICHECK(false) << "Ill schedule data structure";
    throw;
  }

  // Do nothing; Inherited from ScheduleRuleNode
  void InitializeWithTuneContext(const TuneContext& context) final {}
  // Entry of the mega rule; Inherited from ScheduleRuleNode
  Array<Schedule> Apply(const Schedule& sch, const BlockRV& block_rv) final {
    if (!NeedsMultiLevelTiling(sch->state(), sch->GetSRef(block_rv))) {
      return {sch};
    }
    std::vector<State> states{State(sch, block_rv)};
    states = SubRule(std::move(states), [&](State state) { return DetectTensorCore(state); });
    states = SubRule(std::move(states), [&](State state) { return AddWriteReuse(state); });
    states = SubRule(std::move(states), [&](State state) { return TileLoopNest(state); });
    states = SubRule(std::move(states), [&](State state) { return AddReadReuse(state); });
    states = SubRule(std::move(states), [&](State state) { return FuseWriteReuse(state); });
    Array<Schedule> results;
    for (auto&& state : states) {
      results.push_back(std::move(state.sch));
    }
    return results;
  }

 public:
  /*!
   * \brief The tiling structure. Recommended:
   * - 'SSRSRS' on CPU
   * - 'SSSRRSRS' on GPU
   */
  String structure;
  /*! \brief For each level of tiles, which thread axis it is bound to */
  Array<String> tile_binds;
  /*! \brief Whether to use Tensor Core */
  bool use_tensor_core;
  /*! \brief The maximum size of the innermost factor */
  int max_innermost_factor;
  /*! \brief The length of vector lane in vectorized cooperative fetching */
  int vector_load_max_len;
  /*! \brief Data reuse configuration for reading */
  ReuseConfig reuse_read_;
  /*! \brief Data reuse configuration for writing */
  ReuseConfig reuse_write_;
  /*! \brief The indices of spatial tiles in `structure` */
  std::vector<int> s_indices_;
  /*! \brief The indices of reduction tiles in `structure` */
  std::vector<int> r_indices_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("structure", &structure);
    v->Visit("tile_binds", &tile_binds);
    v->Visit("use_tensor_core", &use_tensor_core);
    v->Visit("max_innermost_factor", &max_innermost_factor);
    v->Visit("vector_load_max_len", &vector_load_max_len);
    // `reuse_read_` is not visited
    // `reuse_write_` is not visited
    // `s_indices_` is not visited
    // `r_indices_` is not visited
  }

  static constexpr const char* _type_key = "meta_schedule.MultiLevelTiling";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiLevelTilingNode, ScheduleRuleNode);
};

inline std::vector<State> MultiLevelTilingNode::DetectTensorCore(State state) const {
  std::vector<State> result;
  // If Tensor Core is not allowed, we skip this subrule
  if (!use_tensor_core) return {state};
  // Do tiling to match Tensor Core wmma sync intrin
  BlockRV block_rv = state.block_rv;
  Optional<LoopRV> tiled_loop_rv = TilingwithTensorIntrin(state.sch, block_rv, "wmma_sync");
  if (!tiled_loop_rv.defined()) return {state};
  // Do blockize
  state.block_rv = state.sch->Blockize(tiled_loop_rv.value());
  // Annotate the block
  state.sch->Annotate(block_rv, tir::attr::meta_schedule_auto_tensorize, String("wmma_sync"));
  state.sch->Annotate(state.block_rv, tir::attr::meta_schedule_auto_tensorize, String("wmma_fill"));
  state.tensor_core_is_used = true;
  // Annotate the root block to notify the following postprocessors
  state.sch->Annotate(GetRootBlockRV(state.sch, state.block_rv),
                      tir::attr::meta_schedule_tensor_core_enabled, String("1"));
  result.push_back(state);
  return result;
}

inline std::vector<State> MultiLevelTilingNode::AddWriteReuse(State state) const {
  const ReuseConfig& config = this->reuse_write_;
  if (config.req == ReuseType::kNoReuse) {
    if (state.tensor_core_is_used) state = TensorCoreStore(state);
    return {std::move(state)};
  }
  // Case 1. If the write cache is already there, we don't need to add another.
  if (config.req == ReuseType::kMayReuse) {
    Array<BlockRV> consumer_rvs = state.sch->GetConsumers(state.block_rv);
    if (consumer_rvs.size() == 1 && IsWriteCache(state.sch->GetSRef(consumer_rvs[0]))) {
      state.write_cache = consumer_rvs[0];
      state.write_cache_is_added = false;
      if (state.tensor_core_is_used) state = TensorCoreStore(state);
      return {std::move(state)};
    }
  }
  std::vector<State> results;
  results.reserve(2);
  // Case 2. No write cache is added
  if (config.req == ReuseType::kMayReuse) {
    State new_state(/*sch=*/state.sch->Copy(), /*block_rv=*/state.block_rv,
                    /*write_cache=*/NullOpt,
                    /*write_cache_is_added=*/false);
    new_state.sch->Seed(state.sch->ForkSeed());
    if (new_state.tensor_core_is_used) new_state = TensorCoreStore(new_state);
    results.emplace_back(std::move(new_state));
  }
  // Case 3. Add one write cache
  BlockRV write_cache = state.sch->CacheWrite(/*block_rv=*/state.block_rv, /*read_buffer_index=*/0,
                                              /*storage_scope=*/config.scope);
  state.write_cache = write_cache;
  {
    tir::Annotate(state.sch->state(), state.sch->GetSRef(write_cache),  //
                  tir::attr::meta_schedule_cache_type,                  //
                  Integer(tir::attr::meta_schedule_cache_type_write));
  }
  state.write_cache_is_added = true;
  if (state.tensor_core_is_used) state = TensorCoreStore(state);
  results.emplace_back(std::move(state));
  return results;
}

inline std::vector<State> MultiLevelTilingNode::TileLoopNest(State state) const {
  Schedule& sch = state.sch;
  const BlockRV& block_rv = state.block_rv;
  // Step 1. Assuming trivial binding, pair the loops and their iter-var-types
  Array<LoopRV> loops = sch->GetLoops(block_rv);
  std::vector<IterVarType> iter_types = GetBlockVarTypes(sch->GetSRef(state.block_rv));
  ICHECK_EQ(loops.size(), iter_types.size());
  // Step 2. For each loop axis, tile it
  std::vector<Array<LoopRV>> tiles(s_indices_.size() + r_indices_.size());
  for (int i = 0, n = loops.size(); i < n; ++i) {
    const std::vector<int>* idx = nullptr;
    if (iter_types[i] == IterVarType::kDataPar) {
      idx = &s_indices_;
    } else if (iter_types[i] == IterVarType::kCommReduce) {
      idx = &r_indices_;
    } else {
      continue;
    }
    // Do the split
    int n_tiles = idx->size();
    LoopRV loop = loops[i];
    Array<ExprRV> factors = sch->SamplePerfectTile(
        /*loop=*/loop,
        /*n=*/n_tiles,
        /*max_innermost_factor=*/max_innermost_factor);
    Array<LoopRV> splits = sch->Split(/*loop=*/loop,
                                      /*factors=*/{factors.begin(), factors.end()});
    // Put every tile to its slot
    for (int j = 0; j < n_tiles; ++j) {
      tiles[idx->at(j)].push_back(splits[j]);
    }
  }
  // Step 3. Reorder to organize the tiles
  sch->Reorder(support::ConcatArrayList<LoopRV>(tiles.begin(), tiles.end()));
  // Step 4. Bind the tiles to threads
  int n_binds = std::min(tile_binds.size(), tiles.size());
  for (int i = 0; i < n_binds; ++i) {
    LoopRV fused = sch->Fuse(tiles[i]);
    sch->Bind(fused, tile_binds[i]);
    tiles[i] = {fused};
  }
  state.tiles = Array<Array<LoopRV>>{tiles.begin(), tiles.end()};
  return {state};
}

inline std::vector<State> MultiLevelTilingNode::AddReadReuse(State state) const {
  const ReuseConfig& config = this->reuse_read_;
  if (config.req == ReuseType::kNoReuse) {
    if (state.tensor_core_is_used) state = TensorCoreLoad(state);
    return {std::move(state)};
  }
  ICHECK(config.req != ReuseType::kMayReuse);
  const BlockRV& block_rv = state.block_rv;
  std::vector<State> results;
  results.reserve(config.levels.size());
  for (int level : config.levels) {
    Schedule sch = state.sch->Copy();
    sch->Seed(state.sch->ForkSeed());
    const LoopRV& loop_rv = state.tiles[level - 1].back();
    // Enumerate all buffers that are read but not written
    std::vector<int> read_buffer_ndims = tir::GetReadBufferNDims(sch->GetSRef(block_rv));
    for (int i = 0, n_reads = read_buffer_ndims.size(); i < n_reads; ++i) {
      int buffer_ndim = read_buffer_ndims[i];
      if (buffer_ndim == -1) {
        continue;
      }
      // Do cache_read
      BlockRV cache_read_block = sch->CacheRead(block_rv, i, config.scope);
      {
        tir::Annotate(sch->state(), sch->GetSRef(cache_read_block),  //
                      tir::attr::meta_schedule_cache_type,
                      Integer(tir::attr::meta_schedule_cache_type_read));
      }
      // Insert cache_read block to the proper place
      sch->ComputeAt(cache_read_block, loop_rv, true);
      // Fuse the iterators of the cache_read
      Array<LoopRV> buffer_loops = sch->GetLoops(cache_read_block);
      LoopRV fused = sch->Fuse(Array<LoopRV>{buffer_loops.end() - buffer_ndim,  //
                                             buffer_loops.end()});
      // Annotate cooperative fetching
      if (vector_load_max_len > 0) {
        // cooperative fetch + vectorized loading
        // Split into inner and outer, vectorize the inner loop
        Array<ExprRV> factors = sch->SamplePerfectTile(fused, 2, vector_load_max_len);
        // Add cooperative fetching
        sch->Annotate(cache_read_block, tir::attr::meta_schedule_cooperative_fetch, factors[1]);
      }
    }
    State new_state = state;
    new_state.sch = sch;
    if (new_state.tensor_core_is_used) new_state = TensorCoreLoad(new_state);
    results.push_back(std::move(new_state));
  }
  return results;
}

inline std::vector<State> MultiLevelTilingNode::FuseWriteReuse(State state) const {
  const ReuseConfig& config = this->reuse_write_;
  if (config.req == ReuseType::kNoReuse) {
    if (state.tensor_core_is_used) state = TensorCoreStoreFusion(state, r_indices_.front() - 1);
    return {std::move(state)};
  }
  // If the only-consumer does not exist, or is not elementwise, then do not do fusion
  if (!state.write_cache.defined()) {
    if (state.tensor_core_is_used) state = TensorCoreStoreFusion(state, r_indices_.front() - 1);
    return {std::move(state)};
  }
  std::vector<State> results;
  // Special case.
  //    Stages added by `cache_write` must be fused at some level, otherwise it has no benefit.
  //    On the other hand, If the consumer stage is not added by  `cache_write`,
  //    we may choose not to fuse by setting `must_cache_write = False`
  if (!state.write_cache_is_added && config.req != ReuseType::kMustReuse) {
    results.push_back(state);
  }
  BlockRV consumer = state.write_cache.value();
  // Enumerate the level of tile to be fused at
  for (int level : config.levels) {
    State new_state = state;
    new_state.sch = state.sch->Copy();
    new_state.sch->Seed(state.sch->ForkSeed());
    const LoopRV& loop_rv = new_state.tiles[level - 1].back();
    if (new_state.tensor_core_is_used) new_state = TensorCoreStoreFusion(new_state, level - 1);
    new_state.sch->ReverseComputeAt(consumer, loop_rv, true);
    results.push_back(std::move(new_state));
  }
  return results;
}

// Constructor

ScheduleRule ScheduleRule::MultiLevelTiling(String structure, Optional<Array<String>> tile_binds,
                                            bool use_tensor_core,
                                            Optional<Integer> max_innermost_factor,
                                            Optional<Integer> vector_load_max_len,
                                            Optional<Map<String, ObjectRef>> reuse_read,
                                            Optional<Map<String, ObjectRef>> reuse_write) {
  ObjectPtr<MultiLevelTilingNode> n = make_object<MultiLevelTilingNode>();
  n->structure = structure;
  n->tile_binds = tile_binds.value_or({});
  n->use_tensor_core = use_tensor_core;
  if (use_tensor_core) {
    // Check whether corresponding wmma intrinsics are registered
    tir::TensorIntrin::Get("wmma_sync");
    tir::TensorIntrin::Get("wmma_load_a");
    tir::TensorIntrin::Get("wmma_load_b");
    tir::TensorIntrin::Get("wmma_store");
    tir::TensorIntrin::Get("wmma_fill");
  }
  n->max_innermost_factor = max_innermost_factor.value_or(Integer(-1))->value;
  n->vector_load_max_len = vector_load_max_len.value_or(Integer(-1))->value;
  n->reuse_read_ = reuse_read.defined() ? ReuseConfig(reuse_read.value()) : ReuseConfig();
  n->reuse_write_ = reuse_write.defined() ? ReuseConfig(reuse_write.value()) : ReuseConfig();
  for (int i = 0, len = structure.size(); i < len; ++i) {
    char c = structure.data()[i];
    if (c == 'S') {
      n->s_indices_.push_back(i);
    } else if (c == 'R') {
      n->r_indices_.push_back(i);
    } else {
      LOG(FATAL) << "ValueError: Invalid tiling structure: " << structure;
    }
  }
  return ScheduleRule(n);
}

TVM_REGISTER_NODE_TYPE(MultiLevelTilingNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleMultiLevelTiling")
    .set_body_typed(ScheduleRule::MultiLevelTiling);

}  // namespace meta_schedule
}  // namespace tvm
