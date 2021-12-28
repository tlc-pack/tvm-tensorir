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
#ifndef TVM_META_SCHEDULE_SCHEDULE_RULE_ANALYSIS_H_
#define TVM_META_SCHEDULE_SCHEDULE_RULE_ANALYSIS_H_

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
std::vector<int> GetReadBufferNDims(const StmtSRef& block_sref);

Optional<LoopRV> TilingwithTensorIntrin(const tir::Schedule& sch, const tir::BlockRV& block_rv,
                                        const String& intrin_name);

bool IsCacheReadSharedPattern(const For& loop);

void FallbackRule(const For& loop, Array<Integer>* stage, Array<Integer>* order);

} // namespace tir
} // namespace tvm

namespace tvm{
namespace meta_schedule{
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
ReuseType Str2ReuseType(const String& str);

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

using tir::BlockRV;
using tir::ExprRV;
using tir::IterVarType;
using tir::LoopRV;
using tir::Schedule;

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

} // namespace meta_schedule
} // namespace tvm
#endif  // TVM_META_SCHEDULE_SCHEDULE_RULE_ANALYSIS_H_
