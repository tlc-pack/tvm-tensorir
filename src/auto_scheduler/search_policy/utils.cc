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

/*!
 * \file auto_scheduler/search_policy/utils.cc
 * \brief Common utilities
 */

#include "utils.h"

#include <algorithm>

namespace tvm {
namespace auto_scheduler {

Array<Integer> GetSpatialSplitStepIds(const State& s, int stage_id) {
  const auto& stage = s->stages[stage_id];
  const auto& pop = s->stages[stage_id]->op.as<te::ComputeOpNode>();
  ICHECK(pop != nullptr);
  const std::set<std::string>& no_split_at_inner_name_set =
      stage->op->attrs.count(SearchPolicyKey::no_split_at_inner)
          ? GetIterNameSetParam(stage->op->attrs, SearchPolicyKey::no_split_at_inner)
          : std::set<std::string>();
  size_t reduce_count = 0;
  for (const auto axis : pop->reduce_axis) {
    if (!no_split_at_inner_name_set.count(axis->var->name_hint)) {
      reduce_count++;
    }
  }

  Array<Integer> spatial_split_step_ids;
  for (int i = s->transform_steps.size() - 1; i >= 0; --i) {
    if (IsStageNumberChangingStep(s->transform_steps[i])) {
      if (stage_id > s->transform_steps[i]->stage_id) {
        stage_id--;
      }
    } else if (auto ps = s->transform_steps[i].as<SplitStepNode>()) {
      if (stage_id == ps->stage_id) {
        // Assume SplitStep on reduction axes are always after SplitStep on spatial axes.
        if (reduce_count) {
          reduce_count--;
        } else {
          spatial_split_step_ids.push_back(i);
        }
      }
    }
  }

  return spatial_split_step_ids;
}

std::vector<std::pair<int, int>> GetComputeLocationCandidates(const SearchTask& task,
                                                              const State& state, int stage_id) {
  int target_stage_id = GetSingleConsumerId(task, state, stage_id);
  if (target_stage_id < 0) {
    return {};
  }
  const Stage& target_stage = state->stages[target_stage_id];

  std::vector<std::pair<int, int>> candidates;
  bool target_compute_at_other = target_stage->compute_at == ComputeAtKind::kIter;
  bool target_is_tiled = IsTiled(target_stage);

  bool visited_reduce = false;
  // Enumerate compute_at location at target_stage
  // TODO(merrymercy): More analysis here to make smarter choices
  for (size_t i = 0; i < target_stage->iters.size(); ++i) {
    const Iterator& target_iter = target_stage->iters[i];
    if (target_iter->iter_kind == IteratorKind::kReduction) {
      visited_reduce = true;
      if (!target_is_tiled) {  // Do not go into reduce iter
        break;
      }
    } else if (target_iter->iter_kind == IteratorKind::kSpatial) {
      if (visited_reduce) {  // Do not go into inner tile
        break;
      }
    }

    if (target_iter->annotation == IteratorAnnotation::kUnroll) {
      // Do not go into the unroll region of const tensor indices
      break;
    }

    if (GetExtent(target_iter) == 1) {
      // Skip iterators with length of 1
      continue;
    }
    if (target_compute_at_other && target_iter->iter_kind == IteratorKind::kSpatial &&
        StrEndsWith(target_iter->name, ".0")) {
      // Skip the first level iterators if target stage compute_at another stage
      // In this case, the lengths of first level iterators are always one
      continue;
    }
    candidates.emplace_back(target_stage_id, i);

    if (state->attach_map->iter_to_attached_stages.count(std::make_pair(target_stage_id, i))) {
      break;
    }
  }

  // if the target_stage is already compute_at another stage X, try also compute_at X
  // We call stage X as `target_target_stage`
  if (target_compute_at_other) {
    int target_target_stage_id;
    target_target_stage_id = state->attach_map->stage_to_attach_iter.at(target_stage_id).first;
    const Stage& target_target_stage = state->stages[target_target_stage_id];

    for (size_t i = 0; i < target_target_stage->iters.size(); ++i) {
      const Iterator& target_target_iter = target_target_stage->iters[i];
      if (target_target_iter->iter_kind == IteratorKind::kReduction ||
          state->attach_map->iter_to_attached_stages.count(
              std::make_pair(target_target_stage_id, i))) {
        break;
      }

      if (target_target_iter->annotation == IteratorAnnotation::kUnroll) {
        // Do not go into the unroll region of const tensor indices
        break;
      }

      if (GetExtent(target_target_iter) == 1) {  // skip iterators with length of 1
        continue;
      }

      candidates.emplace_back(target_target_stage_id, i);
    }
  }

  return candidates;
}

State DoMultiLevelTiling(const State& state, int stage_id, const std::string& format,
                         std::vector<int>* spatial_split_step_ids
                         
                         // <bojian/DietCode>
                         // , bool simplify_tiling_structure
                         
                         ) {
  // Temporal object to be used if the input pointer is nullptr
  std::vector<int> temp_split_step_ids;
  if (spatial_split_step_ids == nullptr) {
    spatial_split_step_ids = &temp_split_step_ids;
  }
  std::vector<std::vector<Iterator>> space_levels;
  std::vector<std::vector<Iterator>> reduce_levels;
  std::vector<Iterator> space_outer, space_inner, reduce_outer, reduce_inner;
  Array<Iterator> split_res;

  for (const auto c : format) {
    if (tolower(c) == 's') {
      space_levels.emplace_back();
    } else if (tolower(c) == 'r') {
      reduce_levels.emplace_back();
    } else {
      LOG(FATAL) << "Invalid multi-level tiling format: " << format;
    }
  }
  size_t n_space = space_levels.size();
  size_t n_reduce = reduce_levels.size();

  spatial_split_step_ids->clear();

  State tmp_s = state;
  const Stage& stage = state->stages[stage_id];
  const std::set<std::string>& no_split_at_inner_name_set =
      stage->op->attrs.count(SearchPolicyKey::no_split_at_inner)
          ? GetIterNameSetParam(stage->op->attrs, SearchPolicyKey::no_split_at_inner)
          : std::set<std::string>();
  
  // <bojian/DietCode>
  Iterator final_spatial_iter;
  for (const auto& iter : state->stages[stage_id]->iters) {
    if (iter->iter_kind == IteratorKind::kSpatial) {
      final_spatial_iter = iter;
    }
  }


  for (const auto& iter : state->stages[stage_id]->iters) {

    if (!no_split_at_inner_name_set.count(iter->name)) {
      if (iter->iter_kind == IteratorKind::kSpatial) {
        ICHECK_GE(n_space, 1);

        if (n_space == 1) {
          space_levels[0].push_back(iter);
        } else {

          // <bojian/DietCode> Add the simplication for multi-level tiling.
          LOG(INFO) << "iter.range=" << iter->range;
          split_res = tmp_s.split(stage_id, iter, Array<Optional<Integer>>(n_space - 1, NullOpt));
          // Array<Optional<Integer>> split_steps(n_space - 1, NullOpt);
          // if (simplify_tiling_structure) {
          //   if (iter == final_spatial_iter) {
          //     LOG(WARNING) << "Simplifying tiling structure to (x, 1, 1, x, x)";
          //     split_steps.Set(0, Integer(1));
          //     split_steps.Set(1, Integer(1));
          //   } else {
          //     LOG(WARNING) << "Simplifying tiling structure to (x, x, 1, x, 1)";
          //     split_steps.Set(1, Integer(1));
          //     split_steps.Set(3, Integer(3));
          //   }
          // }
          // split_res = tmp_s.split(stage_id, iter, split_steps);

          for (size_t i = 0; i < n_space; i++) {
            space_levels[i].push_back(split_res[i]);
          }
          spatial_split_step_ids->push_back(tmp_s->transform_steps.size() - 1);
        }
      } else if (iter->iter_kind == IteratorKind::kReduction) {
        ICHECK_GE(n_reduce, 1);

        if (n_reduce == 1) {
          reduce_levels[0].push_back(iter);
        } else {

          // <bojian/DietCode> Ditto, but for reduction axes.
          LOG(INFO) << "iter.range=" << iter->range;
          split_res = tmp_s.split(stage_id, iter, Array<Optional<Integer>>(n_reduce - 1, NullOpt));
          // Array<Optional<Integer>> split_steps(n_reduce - 1, NullOpt);
          // if (simplify_tiling_structure) {
          //   LOG(WARNING) << "Simplifying tiling structure to (x, 1, x)";
          //   split_steps.Set(0, Integer(1));
          // }
          // split_res = tmp_s.split(stage_id, iter, split_steps);

          for (size_t i = 0; i < n_reduce; i++) {
            reduce_levels[i].push_back(split_res[i]);
          }
        }
      } else {
        LOG(FATAL) << "Invalid iter type: " << int(iter->iter_kind);
      }
    } else {
      if (iter->iter_kind == IteratorKind::kSpatial) {
        space_inner.push_back(iter);
      } else if (iter->iter_kind == IteratorKind::kReduction) {
        reduce_inner.push_back(iter);
      } else {
        LOG(FATAL) << "Invalid iter type: " << int(iter->iter_kind);
      }
    }
  }

  if (!space_outer.empty()) {
    ICHECK(!space_levels.empty());
    space_levels.front().insert(space_levels.front().begin(),
                                std::make_move_iterator(space_outer.begin()),
                                std::make_move_iterator(space_outer.end()));
  }
  if (!space_inner.empty()) {
    ICHECK(!space_levels.empty());
    space_levels.back().insert(space_levels.back().begin(),
                               std::make_move_iterator(space_inner.begin()),
                               std::make_move_iterator(space_inner.end()));
  }

  if (!reduce_outer.empty()) {
    ICHECK(!reduce_levels.empty());
    reduce_levels.front().insert(reduce_levels.front().begin(),
                                 std::make_move_iterator(reduce_outer.begin()),
                                 std::make_move_iterator(reduce_outer.end()));
  }
  if (!reduce_inner.empty()) {
    ICHECK(!reduce_levels.empty());
    reduce_levels.back().insert(reduce_levels.back().begin(),
                                std::make_move_iterator(reduce_inner.begin()),
                                std::make_move_iterator(reduce_inner.end()));
  }

  Array<Iterator> order;
  int space_ct = 0, reduce_ct = 0;
  for (const auto c : format) {
    if (tolower(c) == 's') {
      order.insert(order.end(), std::make_move_iterator(space_levels[space_ct].begin()),
                   std::make_move_iterator(space_levels[space_ct].end()));
      space_ct++;
    } else if (tolower(c) == 'r') {
      order.insert(order.end(), std::make_move_iterator(reduce_levels[reduce_ct].begin()),
                   std::make_move_iterator(reduce_levels[reduce_ct].end()));
      reduce_ct++;
    } else {
      LOG(FATAL) << "Invalid multi level tiling format: " << format;
    }
  }

  tmp_s.reorder(stage_id, order);
  return tmp_s;
}

State FollowTiling(const State& state, int stage_id, const std::vector<int>& split_step_ids,
                   int n_split) {
  if (n_split < 1 || n_split > 3) {
    LOG(FATAL) << "Invalid split parts, currently only support 1, 2 and 3";
  }
  // Apply up to three-level tiling structure:  space_L0, space_L1, space_L2
  std::vector<Iterator> space_0, space_1, space_2, space_3, tmp_order;
  Array<Iterator> split_res;

  auto pop = state->stages[stage_id]->op.as<te::ComputeOpNode>();
  ICHECK(pop != nullptr);
  const Stage& stage = state->stages[stage_id];
  const std::set<std::string>& no_split_at_inner_name_set =
      stage->op->attrs.count(SearchPolicyKey::no_split_at_inner)
          ? GetIterNameSetParam(stage->op->attrs, SearchPolicyKey::no_split_at_inner)
          : std::set<std::string>();
  int no_split_at_inner_name_in_stage_cnt = 0;
  for (const auto& iter : state->stages[stage_id]->iters) {
    no_split_at_inner_name_in_stage_cnt += no_split_at_inner_name_set.count(iter->name);
  }

  ICHECK_EQ(state->stages[stage_id]->iters.size() - no_split_at_inner_name_in_stage_cnt,
            split_step_ids.size());

  State tmp_s = state;
  int ct = 0;
  for (const auto& iter : state->stages[stage_id]->iters) {
    if (iter->iter_kind == IteratorKind::kSpatial) {
      // For spatial iterator, split it into multi iterators
      if (!no_split_at_inner_name_set.count(iter->name)) {
        IteratorAnnotation ann_type = iter->annotation;
        split_res = tmp_s.follow_split(stage_id, iter, split_step_ids[ct], n_split);
        // Restore annotation. Move unroll and vectorize to inner, move parallel
        // to outer
        switch (ann_type) {
          case IteratorAnnotation::kUnroll:
            split_res.Set(n_split, tmp_s.unroll(stage_id, split_res[n_split]));
            break;
          case IteratorAnnotation::kVectorize:
            split_res.Set(n_split, tmp_s.vectorize(stage_id, split_res[n_split]));
            break;
          case IteratorAnnotation::kParallel:
            split_res.Set(0, tmp_s.parallel(stage_id, split_res[0]));
            break;
          default:
            break;
        }

        space_0.push_back(split_res[0]);
        space_1.push_back(split_res[1]);
        if (n_split >= 2) {
          space_2.push_back(split_res[2]);
          if (n_split == 3) {
            space_3.push_back(split_res[3]);
          }
        }
        ct++;
      } else {
        if (no_split_at_inner_name_set.count(iter->name)) {
          if (n_split == 1) {
            space_1.push_back(iter);
          } else if (n_split == 2) {
            space_2.push_back(iter);
          } else {
            ICHECK_EQ(n_split, 3);
            space_3.push_back(iter);
          }
        }
      }
    } else {
      LOG(FATAL) << "Invalid iter type: " << int(iter->iter_kind);
    }
  }

  if (n_split == 3) {
    ConcatenateMove(&tmp_order, &space_0, &space_1, &space_2, &space_3);
  } else if (n_split == 2) {
    ConcatenateMove(&tmp_order, &space_0, &space_1, &space_2);
  } else {
    ConcatenateMove(&tmp_order, &space_0, &space_1);
  }
  tmp_s.reorder(stage_id, tmp_order);
  return tmp_s;
}

// Return whether a state has nested parallel, which is invalid on CPUs
bool HasNestedParallel(const State& state) {
  std::function<void(int stage_id, size_t*)> count_parallel_ct;

  count_parallel_ct = [&state, &count_parallel_ct](int stage_id, size_t* parallel_ct) {
    const Stage& stage = state->stages[stage_id];

    if (stage->compute_at == ComputeAtKind::kInlined) {
      return;
    }

    for (size_t i = 0; i < stage->iters.size(); ++i) {
      if (stage->iters[i]->annotation == IteratorAnnotation::kParallel) {
        (*parallel_ct)++;
      }

      IterKey iter_key(stage_id, i);
      auto pair = state->attach_map->iter_to_attached_stages.find(iter_key);
      if (pair != state->attach_map->iter_to_attached_stages.end()) {
        for (const auto& attach_stage_id : pair->second) {
          count_parallel_ct(attach_stage_id, parallel_ct);
        }
      }
    }
  };

  for (size_t stage_id = 0; stage_id < state->stages.size(); ++stage_id) {
    size_t parallel_ct = 0;

    if (state->stages[stage_id]->compute_at == ComputeAtKind::kRoot) {
      count_parallel_ct(stage_id, &parallel_ct);
      if (parallel_ct >= 2) {
        return true;
      }
    }
  }

  return false;
}

void PruneInvalidState(const SearchTask& task, Array<State>* states) {
  size_t pt = 0;
  for (size_t i = 0; i < states->size(); ++i) {
    if (!(*states)[i].defined()) {
      continue;
    }
    if (!IsGPUTask(task) && HasNestedParallel((*states)[i])) {
      continue;
    }

    if (i != pt) {
      states->Set(pt, (*states)[i]);
    }
    pt++;
  }

  if (pt == 0) {
    LOG(FATAL) << "Internal error: All states are invalid.";
  } else {
    states->resize(pt);
  }
}

/********** SplitFactorizationMemo **********/

extern bool is_sample_init_population_1st_iter;

const Array<Array<Integer>>& SplitFactorizationMemo::GetFactorizationSchemes(
    int extent, int n_lengths
    // , int max_innermost_factor
    ) {
  QueryKey key = // std::make_tuple(extent, n_lengths, max_innermost_factor);
                 std::make_pair(extent, n_lengths);
  const auto& it = memory_.find(key);
  if (it != memory_.end()) {
    return it->second;
  }
  // if (!is_sample_init_population_1st_iter) {
  //   LOG(FATAL) << "(extent=" << extent << ", n_lengths=" << n_lengths <<") has "
  //                 "not been found in SplitFactorizationMemo";
  // }

  tmp_stack_ = Array<Integer>(n_lengths, Integer());
  results_ = &memory_[key];
  n_lengths_ = n_lengths;

  DfsEnumerate(0, extent
               // , max_innermost_factor
               );

  return *results_;
}

void SplitFactorizationMemo::DfsEnumerate(int now, int remaining_length
                                          // , int max_innermost_factor
                                          ) {
  if (now == n_lengths_) {
    if (tmp_stack_.back().as<IntImmNode>()->value <=
        // max_innermost_factor
        max_innermost_factor_
        // in case the maximum innermost factor is not defined
        || max_innermost_factor_ == 0
        ) {
      results_->push_back(tmp_stack_);
    }
  } else {
    for (const auto& f : GetFactors(remaining_length)) {
      tmp_stack_.Set(now, Integer(f));
      DfsEnumerate(now + 1, remaining_length / f
                   // , max_innermost_factor
                   );
    }
  }
}

const std::vector<int>& SplitFactorizationMemo::GetFactors(int n) {
  auto it = factor_memory_.find(n);
  if (it != factor_memory_.end()) {
    return it->second;
  }

  std::vector<int>& res = factor_memory_[n];
  int step = n % 2 == 0 ? 1 : 2;
  for (size_t i = 1; i < static_cast<size_t>(std::sqrt(n)) + 1; i += step) {
    if (n % i == 0) {
      res.push_back(i);
      if (n / i != i) {
        res.push_back(n / i);
      }
    }
  }
  std::sort(res.begin(), res.end());
  return res;
}


std::vector<SplitStepInfo> FactorizationScheme::split_steps_info;
bool FactorizationScheme::simplify_schedule;
size_t FactorizationScheme::last_spatial_iter_id;
std::vector<std::vector<int>> FactorizationScheme::factor_indices_to_incr;


void FactorizationScheme::RandomSample(const HardwareParams& hardware_params,
                                       const size_t max_innermost_factor,
                                       std::mt19937* const rng) {
  // ===========================================================================
  // factor[1] (threadIdx.x)
  // ===========================================================================
  std::uniform_int_distribution<> num_warps_dist(
      1, hardware_params->max_threads_per_block / hardware_params->warp_size);
  size_t num_threads_per_block = num_warps_dist(*rng) * hardware_params->warp_size;
  // find all the possible factors of the number of threads per block
  if (is_sample_init_population_1st_iter) {
    LOG(INFO) << "num_threads_per_block=" << num_threads_per_block;
  }
  size_t num_spatial_axes = 0;
  for (const SplitStepInfo& info : split_steps_info) {
    if (info.is_spatial) {
      ++num_spatial_axes;
    }
  }
  if (is_sample_init_population_1st_iter) {
    LOG(INFO) << "num_spatial_axes=" << num_spatial_axes;
  }
  SplitFactorizationMemo memo;
  const Array<Array<Integer>>& num_threads_factor_schemes =
      memo.GetFactorizationSchemes(num_threads_per_block, num_spatial_axes - 1);

  if (is_sample_init_population_1st_iter) {
    LOG(INFO) << "Sampled the factors out of num_threads_factor_schemes.size()="
              << num_threads_factor_schemes.size();
  }
  CHECK(num_threads_factor_schemes.size() != 0);
  std::uniform_int_distribution<> num_threads_factor_schemes_dist(
      0, num_threads_factor_schemes.size() - 1);

  bool all_below_max_extents;
  Array<Integer> num_threads_factor_scheme;
  do {
    all_below_max_extents = true;

    num_threads_factor_scheme = num_threads_factor_schemes[num_threads_factor_schemes_dist(*rng)];
    // if (is_sample_init_population_1st_iter) {
    //   LOG(INFO) << "Factorization Scheme: " << ArrayToString(num_threads_factor_scheme);
    // }
    int64_t factor_prod = 1;
    for (const Integer& factor : num_threads_factor_scheme) {
      factor_prod *= factor;
    }
    num_threads_factor_scheme.push_back(num_threads_per_block / factor_prod);
    if (is_sample_init_population_1st_iter) {
      LOG(INFO) << "Factorization Scheme: " << ArrayToString(num_threads_factor_scheme);
    }
    for (size_t iter_id = 0, spatial_iter_id = 0;
         iter_id < split_steps_info.size(); ++iter_id) {
      if (split_steps_info[iter_id].is_spatial) {
        if (static_cast<size_t>(
              num_threads_factor_scheme[spatial_iter_id]->value) >
            split_steps_info[iter_id].max_extent) {
          all_below_max_extents = false;
        }
        ++spatial_iter_id;
      }
    }  // for (iter_id ∈ [0, split_steps_info.size()))
  } while (!all_below_max_extents);
  // do the looping again and assign the factors
  for (size_t iter_id = 0, spatial_iter_id = 0;
       iter_id < split_steps_info.size(); ++iter_id) {
    if (split_steps_info[iter_id].is_spatial) {
      split_factors[iter_id][1] =
          num_threads_factor_scheme[spatial_iter_id]->value;
      ++spatial_iter_id;
    }
  }
  // ===========================================================================
  // factor[0] (vthread)
  // ===========================================================================
  size_t reg_usage = num_threads_per_block;
  std::vector<size_t> factors_to_assign;
  std::vector<size_t>::iterator factors_to_assign_it;

  auto sample_factors = [&](std::function<  bool(const size_t)>  continue_predicate,
                            std::function<size_t(const size_t)>  max_extent,
                            std::function<  int&(const size_t)>  factor_to_assign) {
        factors_to_assign.clear();
        for (size_t iter_id = 0; iter_id < split_steps_info.size(); ++iter_id) {
          if (continue_predicate(iter_id)) {
            continue;
          }
          std::uniform_int_distribution<> dist(1, max_extent(iter_id));
          factors_to_assign.push_back(dist(*rng));
          reg_usage *= factors_to_assign.back();
        }
        std::shuffle(factors_to_assign.begin(), factors_to_assign.end(), *rng);
        factors_to_assign_it = factors_to_assign.begin();
        for (size_t iter_id = 0; iter_id < split_steps_info.size(); ++iter_id) {
          if (continue_predicate(iter_id)) {
            continue;
          }
          factor_to_assign(iter_id) = *factors_to_assign_it;
          ++factors_to_assign_it;
        }
      };

  sample_factors(
      [&](const size_t iter_id) -> bool {
        return (!split_steps_info[iter_id].is_spatial) || 
               (simplify_schedule && (iter_id != last_spatial_iter_id));
      },
      [&](const size_t iter_id) -> size_t {
        size_t max_vthread_extent =
            std::min(static_cast<size_t>(hardware_params->max_vthread_extent),
                     split_steps_info[iter_id].max_extent /
                     split_factors[iter_id][1]); 
        max_vthread_extent =
            std::min(max_vthread_extent,
                     hardware_params->max_local_memory_per_block / reg_usage);
        return max_vthread_extent;
      },
      [&](const size_t iter_id) -> int& {
        return split_factors[iter_id][0];
      }
      );

  if (is_sample_init_population_1st_iter) {
    LOG(INFO) << "Finished sampling the vthread";
  }
  // ===========================================================================
  // factor[3] (innermost)
  // ===========================================================================
  sample_factors(
      [&](const size_t iter_id) -> bool {
        return (!split_steps_info[iter_id].is_spatial) || 
               (iter_id == last_spatial_iter_id);
               // always make sure that the final spatial axis has a stride of 1
      },
      [&](const size_t iter_id) -> size_t {
        size_t max_innermost_extent =
            std::min(max_innermost_factor,
                     split_steps_info[iter_id].max_extent /
                     split_factors[iter_id][0] / 
                     split_factors[iter_id][1]);
        max_innermost_extent =
            std::min(max_innermost_extent,
                     hardware_params->max_local_memory_per_block / reg_usage);
        return max_innermost_extent;
      },
      [&](const size_t iter_id) -> int& {
        return split_factors[iter_id][3];
      }
      );
  if (is_sample_init_population_1st_iter) {
    LOG(INFO) << "Finished sampling the innermost loop extent";
  }
  // ===========================================================================
  // factor[2]
  // ===========================================================================
  if (reg_usage > static_cast<size_t>(hardware_params->max_local_memory_per_block)) {
      LOG(FATAL) << "reg_usage=" << reg_usage << " is already greater than the allowable size "
                 << hardware_params->max_local_memory_per_block
                 << ", current scheme=" << toString();
  }
  sample_factors(
      [&](const size_t iter_id) -> bool {
        return (!split_steps_info[iter_id].is_spatial) || simplify_schedule;
      },
      [&](const size_t iter_id) -> size_t {
        size_t max_2nd_innermost_extent =
            std::min(split_steps_info[iter_id].max_extent /
                     split_factors[iter_id][0] /
                     split_factors[iter_id][1] /
                     split_factors[iter_id][3],
                     hardware_params->max_local_memory_per_block / reg_usage);
        return max_2nd_innermost_extent;
      },
      [&](const size_t iter_id) -> int& {
        return split_factors[iter_id][2];
      }
      );
  if (is_sample_init_population_1st_iter) {
    LOG(INFO) << "Finished sampling the 2nd innermost loop extent";
  }

  size_t shmem_usage = 0;
  for (size_t iter_id = 0; iter_id < split_steps_info.size(); ++iter_id) {
    if (split_steps_info[iter_id].is_spatial) {
      shmem_usage += split_factors[iter_id][0] * split_factors[iter_id][1] *
                     split_factors[iter_id][2] * split_factors[iter_id][3];
    }
  }
  // repeat similar procedure for reduction axes
  // ===========================================================================
  // rfactor[1] (innermost)
  // ===========================================================================
  sample_factors(
      [&](const size_t iter_id) -> bool {
        return split_steps_info[iter_id].is_spatial;
      },
      [&](const size_t iter_id) -> size_t {
        size_t max_innermost_extent =
            std::min(max_innermost_factor,
                     split_steps_info[iter_id].max_extent);
        max_innermost_extent =
            std::min(max_innermost_extent,
                     hardware_params->max_shared_memory_per_block / sizeof(float) / shmem_usage);
        return max_innermost_extent;
      },
      [&](const size_t iter_id) -> int& {
        return split_factors[iter_id][1];
      }
      );
  if (is_sample_init_population_1st_iter) {
    LOG(INFO) << "Finished sampling the innermost loop extent (reduction axis)";
  }
  // ===========================================================================
  // rfactor[0]
  // ===========================================================================
  if (shmem_usage > static_cast<size_t>(hardware_params->max_shared_memory_per_block)) {
      LOG(FATAL) << "shmem_usage=" << shmem_usage << " is already greater than the allowable size "
                   << hardware_params->max_shared_memory_per_block
                   << ", current scheme=" << toString();
  }

  sample_factors(
      [&](const size_t iter_id) -> bool {
        return split_steps_info[iter_id].is_spatial || simplify_schedule;
      },
      [&](const size_t iter_id) -> size_t {
        size_t max_2nd_innermost_extent =
            std::min(split_steps_info[iter_id].max_extent / 
                     split_factors[iter_id][1],
                     hardware_params->max_shared_memory_per_block / sizeof(float) / shmem_usage);
        return max_2nd_innermost_extent;
      },
      [&](const size_t iter_id) -> int& {
        return split_factors[iter_id][0];
      }
      );
  if (is_sample_init_population_1st_iter) {
    LOG(INFO) << "Finished sampling the 2nd innermost loop extent (reduction axis)";
  }
  // If the sampled factorization scheme violates the hardware constraints, it
  // will just be discarded.
}


FactorizationSchemeCheckRetType
DietCodeSplitFactorizationMemo::IsLegit(const FactorizationScheme& scheme) {
  int num_threads = 1;
  size_t reg_usage = 1, acc_spatial = 0, acc_reduction = 1;

  for (size_t i = 0; i < FactorizationScheme::split_steps_info.size(); ++i) {
    const SplitStepInfo& split_step_info =
        FactorizationScheme::split_steps_info[i];
    const std::vector<int>& factors = scheme.split_factors[i];
    if (split_step_info.is_spatial) {
      CHECK(factors.size() == 4);
      num_threads *= factors[1];
      if (factors[0] > hardware_params_->max_vthread_extent) {
        return kOOB;
      }
      if (factors[3] > max_innermost_factor_) {
        return kOOB;
      }
      size_t split_factors_prod = factors[0] * factors[1] * factors[2] * factors[3];
      if (split_factors_prod > split_step_info.max_extent) {
        return kOOB;
      }
      reg_usage *= split_factors_prod;
      acc_spatial += split_factors_prod;
    } else {
      CHECK(factors.size() == 2);
      if (factors[1] > max_innermost_factor_) {
        return kOOB;
      }
      size_t split_factors_prod = factors[0] * factors[1];
      if (split_factors_prod > split_step_info.max_extent) {
        return kOOB;
      }
      acc_reduction *= split_factors_prod;
    }
  }
  // check the shared memory capacity (with the assumption that all data types
  // are 32-bit float)
  size_t shmem_usage = acc_spatial * acc_reduction * sizeof(float);
  if (shmem_usage >
      static_cast<size_t>(hardware_params_->max_shared_memory_per_block)) {
    return kOOB;
  }
  // check the number of threads per block
  if (num_threads % hardware_params_->warp_size != 0) {
    return kInvalid;
  }
  // LOG(INFO) << "scheme=" << scheme.toString();
  // LOG(INFO) << "num_threads=" << num_threads << ", warp_size="
  //           << hardware_params_->warp_size;
  if (num_threads > hardware_params_->max_threads_per_block) {
    return kOOB;
  }
  // check the number of registers used
  if (reg_usage >
      static_cast<size_t>(hardware_params_->max_local_memory_per_block)) {
    return kOOB;
  }
  return kValid;
}


void DietCodeSplitFactorizationMemo::BfsEnumerate() {
  while (!workstack_.empty()) {
    FactorizationScheme& workitem = workstack_.front();
    FactorizationSchemeCheckRetType ret = IsLegit(workitem);

    if (ret == kValid) {
      cache_.push_back(std::move(workitem));

      if (cache_.size() % 10000 == 0) {
        LOG(INFO) << "Number of valid schedules found so far: " << cache_.size();
        LOG(INFO) << "Latest examined scheme=" << cache_.back().toString();
      }
    }
    if (ret != kOOB) {
      // consider expanding the factorization scheme
      std::vector<FactorizationScheme> adj_schemes = workitem.neighbors();
      for (FactorizationScheme& scheme : adj_schemes) {
        if (examined_schemes_.find(scheme.toString()) ==
            examined_schemes_.end()) {
          examined_schemes_.insert(scheme.toString());
          workstack_.push(std::move(scheme));

          if (examined_schemes_.size() % 10000 == 0) {
            LOG(INFO) << "Number of schemes examined so far: " << examined_schemes_.size();
            LOG(INFO) << "Current workitem=" << workitem.toString();
          }
        }
      }
    }
    workstack_.pop();
  }
}


const std::vector<FactorizationScheme>&
DietCodeSplitFactorizationMemo::GetAllFactorizationSchemes(
    const std::vector<SplitStepInfo>& split_steps_info,
    const bool simplify_schedule) {
  if (!is_sample_init_population_1st_iter) {
    CHECK(FactorizationScheme::split_steps_info == split_steps_info);
    CHECK(FactorizationScheme::simplify_schedule == simplify_schedule);
    // return the cached factorization scheme
    return cache_;
  }
  workstack_.emplace(split_steps_info, simplify_schedule,
                     is_sample_init_population_1st_iter
                     /* initialize the static members */);
  examined_schemes_.insert(workstack_.front().toString());
  BfsEnumerate();
  LOG(INFO) << "Total number of possible factorization schemes w/ the "
               "dynamic workload: " << cache_.size();
  return cache_;
}


FactorizationScheme
DietCodeSplitFactorizationMemo::SampleFactorizationSchemes(
    const std::vector<SplitStepInfo>& split_steps_info,
    std::mt19937* const rng,
    const bool simplify_schedule) {
  FactorizationScheme scheme(split_steps_info, simplify_schedule,
                             is_sample_init_population_1st_iter);
  scheme.RandomSample(hardware_params_, max_innermost_factor_, rng);
  
  if (is_sample_init_population_1st_iter) {
    LOG(INFO) << "Randomly sampled factorization scheme=" << scheme.toString();
  }
  return scheme;
}


/********** Utils interface API for ffi **********/

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsGetConsumers")
    .set_body_typed([](const SearchTask& task, const State& state, int stage_id) {
      const std::set<int>& consumers = GetConsumers(task, state, stage_id);
      tvm::Map<IntImm, IntImm> ret;
      for (const auto& i : consumers) {
        ret.Set(Integer(i), Integer(i));
      }
      return ret;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsIsElementwiseMatch")
    .set_body_typed([](const SearchTask& task, const State& state, int stage_id,
                       int target_stage_id) {
      return ElementwiseMatch(task, state, stage_id, target_stage_id);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsIsTiled")
    .set_body_typed([](const Stage& stage) { return IsTiled(stage); });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsHasCacheReadStage")
    .set_body_typed([](const State& s, int stage_id) { return HasCacheReadStage(s, stage_id); });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsHasCacheWriteStage")
    .set_body_typed([](const State& s, int stage_id) { return HasCacheWriteStage(s, stage_id); });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsHasRfactorStage")
    .set_body_typed([](const State& s, int stage_id) { return HasRfactorStage(s, stage_id); });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsHasCrossThreadReduction")
    .set_body_typed([](const State& s, int stage_id) {
      return HasCrossThreadReduction(s, stage_id);
    });

}  // namespace auto_scheduler
}  // namespace tvm
