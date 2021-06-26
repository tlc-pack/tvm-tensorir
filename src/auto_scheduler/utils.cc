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


std::unordered_map<size_t, size_t>
TopKDispatcher::dispatch(const std::vector<float>& scores,
                         const size_t num_states) {
  const size_t num_instances = scores.size() / num_states;
  float max_acc_score;
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

  do {
    max_acc_score = 1e-10;
    ++k;  // increment the number of candidates selected

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
            std::pop_heap(topK_candidates.begin(), topK_candidates.end(),
                          scoreboard_item_gt_cmp);
            topK_candidates.pop_back();
            topK_candidates.push_back({state_id, score});
            std::push_heap(topK_candidates.begin(), topK_candidates.end(),
                           scoreboard_item_gt_cmp);
          }
        }  // if (topK_candidates.size() < k)
      }    // for (state_id ∈ range(num_states))
    }      // for (inst_id ∈ range(num_instances))
    // Now that all the instances have their most-preferred states ready, we now
    // choose the minimum set that could cover all the candidates.
    
    // make a copy of the dispatch status
    std::unordered_set<size_t> inst_disp_remainder(inst_disp_init_remainder);

    while (!inst_disp_remainder.empty()) {
      // count the number of votes per state
      std::unordered_map<size_t, size_t> votes;

      for (const size_t inst_id : inst_disp_remainder) {
        for (const ScoreboardItem& cand : inst_topK_candidates[inst_id]) {
          // auto votes_it = votes.find
        }
      }
    }
  } while (max_acc_score > 1e-10);
  LOG(INFO) << "k=" << k;
}

}  // namespace auto_scheduler
}  // namespace tvm
