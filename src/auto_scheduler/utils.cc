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
 * \file auto_scheduler/utils.cc
 * \brief Common utilities.
 */

#include "utils.h"

// <bojian/DietCode>
#include <algorithm>
#include <functional>


namespace tvm {
namespace auto_scheduler {

NullStream& NullStream::Global() {
  static NullStream stream;
  return stream;
}


extern bool enable_verbose_logging;

std::unordered_map<size_t, size_t>
TopKDispatcher::dispatch(const std::vector<float>& scores,
                         const size_t num_states) {
  const size_t num_instances = scores.size() / num_states;
  std::unordered_map<size_t, size_t> disp_map_to_ret;

  size_t k = 0;
  // instance dispatch initial status
  std::unordered_set<size_t> inst_disp_init_remainder;

  for (size_t inst_id = 0; inst_id < num_instances; ++inst_id) {
    inst_disp_init_remainder.insert(inst_id);
  }

  struct ScoreboardItem {
    size_t state_id;
    float  score;
  };

  auto scoreboard_item_gt_cmp = [](const ScoreboardItem& LHS,
                                   const ScoreboardItem& RHS) {
    return LHS.score > RHS.score;
  };

  while (true) {
    ++k;  // increment the number of candidates selected
    disp_map_to_ret.clear();

    typedef std::vector<ScoreboardItem> Scoreboard;
    std::unordered_map<size_t, Scoreboard> inst_topK_candidates;

    for (size_t inst_id = 0; inst_id < num_instances; ++inst_id) {
      // ∀inst, pick its k most-preferred states
      Scoreboard& topK_candidates = inst_topK_candidates[inst_id];
      for (size_t state_id = 0; state_id < num_states; ++state_id) {
        const float score = scores[inst_id * num_states + state_id];
        if (topK_candidates.size() < k) {
          topK_candidates.push_back(ScoreboardItem{state_id, score});
          // no need to invoke push_heap here because the scoreboard is not yet full
          // std::push_heap(topK_candidates.begin(), topK_candidates.end(),
          //                scoreboard_item_gt_cmp);
        } else {
          if (score > topK_candidates.front().score) {
            if (enable_verbose_logging) {
              LOG(INFO) << "min score before popping="
                        << topK_candidates.front().score << " w/ "
                           "state_id=" << topK_candidates.front().state_id;
            }
            std::pop_heap(topK_candidates.begin(), topK_candidates.end(),
                          scoreboard_item_gt_cmp);
            topK_candidates.pop_back();
            topK_candidates.push_back({state_id, score});
            std::push_heap(topK_candidates.begin(), topK_candidates.end(),
                           scoreboard_item_gt_cmp);
            if (enable_verbose_logging) {
              LOG(INFO) << "min score after popping="
                        << topK_candidates.front().score << " w/ "
                           "state_id=" << topK_candidates.front().state_id;
            }
          }  // if (score > topK_candidates.front().score)
        }    // if (topK_candidates.size() < k)
      }      // for (state_id ∈ range(num_states))
    }        // for (inst_id ∈ range(num_instances))
    // Now that all the instances have their most-preferred states ready, we now
    // choose the minimum set that could cover all the candidates.
    
    // make a copy of the dispatch status
    std::unordered_set<size_t> inst_disp_remainder(inst_disp_init_remainder);
    std::unordered_set<size_t> selected_states;

    // keep iterating until all the iterators have been dispatched
    while (!inst_disp_remainder.empty()) {
      // count the number of votes per state [state_id → vote_cnt]
      std::unordered_map<size_t, float> votes;

      for (const size_t inst_id : inst_disp_remainder) {
        for (const ScoreboardItem& cand : inst_topK_candidates[inst_id]) {
          auto votes_it = votes.find(cand.state_id);
          if (votes_it == votes.end()) {
            votes[cand.state_id] = 0.;
          }
          votes[cand.state_id] += scores[inst_id * num_states + cand.state_id];
        }  // for (cand ∈ inst_topK_candidates[inst_id])
      }    // for (inst_id ∈ inst_disp_remainder)

      // pick the state_id with the maximum accumulated score
      const auto& votes_max_it = 
          std::max_element(votes.begin(), votes.end(),
                           [](const std::pair<size_t, float>& LHS,
                              const std::pair<size_t, float>& RHS)
                             -> bool {
                             return LHS.second < RHS.second;
                           }
                           );
      selected_states.insert(votes_max_it->first);

      std::unordered_set<size_t> inst_disp_remainder_copy = inst_disp_remainder;

      for (const size_t inst_id : inst_disp_remainder) {
        Scoreboard& topK_candidates = inst_topK_candidates[inst_id];

        Scoreboard::iterator topK_candidates_it;
        for (topK_candidates_it  = topK_candidates.begin();
             topK_candidates_it != topK_candidates.end();
             ++topK_candidates_it) {
          if (topK_candidates_it->state_id == votes_max_it->first) {
            break;
          }
        }
        if (topK_candidates_it != topK_candidates.end()) {
          inst_disp_remainder_copy.erase(inst_id);
          disp_map_to_ret[inst_id] = votes_max_it->first;
        }
      }    // for (inst_id ∈ inst_disp_remainder)
      inst_disp_remainder = std::move(inst_disp_remainder_copy);
    }  // while (!inst_disp_remainder.empty())

    if (selected_states.size() > 128) {
      LOG(WARNING) << "The number of selected states is greater than 128, "
                      "hence is not valid";
    } else {
      break;
    }
  }
  LOG(INFO) << "k=" << k;
  return disp_map_to_ret;
}


/*
double GetSyntheticWorkloadFlopCtFromState(const SearchTask& task,
                                           const State& state) {
  te::Schedule synthetic_sch;
  Array<te::Tensor> synthetic_tensors;
  std::tie(synthetic_sch, synthetic_tensors) =
      task->compute_dag.GenerateSyntheticWorkloadAndApplySteps(
        state, task->hardware_params);
  Array<te::Operation> synthetic_sch_ops;
  for (const te::Stage& stage : synthetic_sch->stages) {
    synthetic_sch_ops.push_back(stage->op);
  }
  return FlopEstimator().EstimateFlop(synthetic_sch_ops);
}
 */

std::pair<double, double>
GetCherryPickedWklInstFlopCtFromState(const SearchTask& task,
                                      const State& state) {
  // te::Schedule synthetic_sch;
  // Array<te::Tensor> synthetic_tensors;
  Array<IntImm> cherry_picked_wkl_inst =
      task->compute_dag.CherryPickWorkloadInstance(state, task);
  
  if (enable_verbose_logging) {
    LOG(INFO) << "Cherry picked workload inst="
              << ArrayToString(cherry_picked_wkl_inst);
  }

  // std::tie(synthetic_sch, synthetic_tensors) =
  //     task->compute_dag.InstantiateAndApplySteps(state,
  //                                                task->shape_vars.value(),
  //                                                cherry_picked_wkl_inst);
  // Array<te::Operation> synthetic_sch_ops;
  // for (const te::Stage& stage : synthetic_sch->stages) {
  //   synthetic_sch_ops.push_back(stage->op);
  // }
  double inst_flop =
      EstimateFlopForInst(task->compute_dag, task->shape_vars.value(),
                          cherry_picked_wkl_inst);

  const float base_score = 1.;
  float occupancy_penalty, padding_penalty, adapted_score;

  AdaptStateToWorkload(task, state, cherry_picked_wkl_inst, base_score,
                       &occupancy_penalty, &padding_penalty, &adapted_score);
  return std::make_pair(inst_flop, adapted_score);
}


double EstimateFlopForInst(const ComputeDAG& compute_dag,
                           // const Array<Step>& transform_steps,
                           const Array<String>& shape_vars,
                           const Array<IntImm>& shape_values) {
  CHECK(shape_vars.size() == shape_values.size());
  Map<String, IntImm> shape_var_value_map;
  for (size_t i = 0; i < shape_vars.size(); ++i) {
    shape_var_value_map.Set(shape_vars[i], shape_values[i]);
  }
  DynamicAxisReplacer replacer(
      [&shape_var_value_map](const DynamicAxisNode* op) -> PrimExpr {
        auto shape_var_value_map_iter =
            shape_var_value_map.find(op->name_hint);
        if (shape_var_value_map_iter != shape_var_value_map.end()) {
          return (*shape_var_value_map_iter).second;
        }
        LOG(FATAL) << "Dynamic Axis Node " << GetRef<DynamicAxis>(op)
                   << " has not been found in "
                   << MapToString(shape_var_value_map);
        return GetRef<DynamicAxis>(op);
      }
      );
  te::Schedule sch;
  Array<te::Tensor> tensors;
  std::tie(sch, tensors) = compute_dag.ApplySteps({});
  Array<te::Operation> sch_ops;
  for (const te::Stage& stage : sch->stages) {
    sch_ops.push_back(stage->op);
  }
  return FlopEstimator(replacer).EstimateFlop(sch_ops);
}


}  // namespace auto_scheduler
}  // namespace tvm
