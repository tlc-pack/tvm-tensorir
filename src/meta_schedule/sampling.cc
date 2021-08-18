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

#include "./sampling.h"

#include "../tir/schedule/analysis.h"
#include "../tir/schedule/utils.h"
#include "./sampler.h"
#include "./utils.h"

namespace tvm {
namespace meta_schedule {

std::vector<int64_t> SamplePerfectTile(tir::ScheduleState self, int64_t* random_state,
                                       const tir::StmtSRef& loop_sref, int n,
                                       int max_innermost_factor,
                                       Optional<Array<Integer>>* decision) {
  const auto* loop = TVM_SREF_TO_FOR(loop, loop_sref);
  int64_t extent = GetLoopIntExtent(loop);
  std::vector<int64_t> result;
  if (extent == -1) {
    // Case 1. Handle loops with non-constant length
    result = std::vector<int64_t>(n, 1);
    result[0] = -1;
  } else if (decision->defined()) {
    // Case 2. Use previous decision
    result = AsVector<Integer, int64_t>(decision->value());
    int n = result.size();
    ICHECK_GE(n, 2);
    int64_t len = extent;
    for (int i = n - 1; i > 0; --i) {
      int64_t& l = result[i];
      // A previous decision could become invalid because of the change of outer tiles
      // To handle this case properly, we check if the tiling strategy is still perfect.
      // If not, we use a trivial default solution (1, 1, ..., 1, L) for rest of the tiles
      if (len % l != 0) {
        l = len;
      }
      len /= l;
    }
    result[0] = len;
  } else {
    // Case 3. Use fresh new sampling result
    std::vector<int> sampled =
        Sampler(random_state).SamplePerfectTile(n, extent, max_innermost_factor);
    result = std::vector<int64_t>(sampled.begin(), sampled.end());
    ICHECK_LE(sampled.back(), max_innermost_factor);
  }
  *decision = AsArray<int64_t, Integer>(result);
  return result;
}

int64_t SampleCategorical(tir::ScheduleState self, int64_t* random_state,
                          const Array<Integer>& candidates, const Array<FloatImm>& probs,
                          Optional<Integer>* decision) {
  int i = -1;
  int n = candidates.size();
  if (decision->defined()) {
    const auto* int_imm = decision->as<IntImmNode>();
    i = int_imm->value;
    CHECK(0 <= i && i < n) << "ValueError: Wrong decision value, where n = " << n
                           << ", but decision is: " << i;
  } else {
    i = Sampler(random_state).MakeMultinomial(AsVector<FloatImm, double>(probs))();
    ICHECK(0 <= i && i < n);
  }
  *decision = Integer(i);
  return candidates[i];
}

tir::StmtSRef SampleComputeLocation(tir::ScheduleState self, int64_t* random_state,
                                    const tir::StmtSRef& block_sref, Optional<Integer>* decision) {
  // Find all possible compute-at locations
  Array<tir::StmtSRef> loop_srefs = tir::CollectComputeLocation(self, block_sref);
  int n = loop_srefs.size();
  // Extract non-unit loops
  std::vector<int> choices;
  choices.reserve(n);
  for (int i = 0; i < n; ++i) {
    int64_t extent = tir::GetLoopIntExtent(loop_srefs[i]);
    if (extent != -1 && extent != 1) {
      choices.push_back(i);
    }
  }
  // The decision made, by default it is -1
  int i = -1;
  if (decision->defined()) {
    // Handle existing decision
    const auto* int_imm = decision->as<IntImmNode>();
    int64_t decided = int_imm->value;
    if (decided == -2 || decided == -1) {
      i = decided;
    } else {
      for (int choice : choices) {
        if (choice <= decided) {
          i = choice;
        } else {
          break;
        }
      }
    }
  } else {
    // Sample possible combinations
    i = Sampler(random_state).SampleInt(-2, choices.size());
    if (i >= 0) {
      i = choices[i];
    }
  }
  *decision = Integer(i);
  if (i == -2) {
    return tir::StmtSRef::InlineMark();
  }
  if (i == -1) {
    return tir::StmtSRef::RootMark();
  }
  return loop_srefs[i];
}

}  // namespace meta_schedule
}  // namespace tvm
