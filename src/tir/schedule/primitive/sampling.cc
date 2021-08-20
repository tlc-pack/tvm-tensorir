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
#include <tvm/target/target.h>
#include <tvm/tir/schedule/schedule.h>

#include <random>

#include "../primitive.h"
#include "../utils.h"

namespace tvm {
namespace tir {

std::function<int()> MakeMultinomial(support::LinearCongruentialEngine::TRandState* rand_state,
                                     const std::vector<double>& weights) {
  support::LinearCongruentialEngine rand_(rand_state);
  std::vector<double> sums;
  sums.reserve(weights.size());
  double sum = 0.0;
  for (double w : weights) {
    sums.push_back(sum += w);
  }
  std::uniform_real_distribution<double> dist(0.0, sum);
  auto sampler = [rand_state, dist = std::move(dist), sums = std::move(sums)]() mutable -> int {
    support::LinearCongruentialEngine rand_(rand_state);
    double p = dist(rand_);
    int idx = std::lower_bound(sums.begin(), sums.end(), p) - sums.begin();
    int n = sums.size();
    CHECK_LE(0, idx);
    CHECK_LE(idx, n);
    return (idx == n) ? (n - 1) : idx;
  };
  return sampler;
}

int64_t SampleCategorical(tir::ScheduleState self,
                          support::LinearCongruentialEngine::TRandState* rand_state,
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
    i = MakeMultinomial(rand_state, AsVector<FloatImm, double>(probs))();
    ICHECK(0 <= i && i < n);
  }
  *decision = Integer(i);
  return candidates[i];
}

struct SampleCategoricalTraits : public UnpackedInstTraits<SampleCategoricalTraits> {
  static constexpr const char* kName = "SampleCategorical";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 0;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 1;

  static ExprRV UnpackedApplyToSchedule(Schedule sch,               //
                                        Array<Integer> candidates,  //
                                        Array<FloatImm> probs,      //
                                        Optional<Integer> decision) {
    return sch->SampleCategorical(candidates, probs, decision);
  }

  static String UnpackedAsPython(Array<String> outputs,      //
                                 Array<Integer> candidates,  //
                                 Array<FloatImm> probs,      //
                                 Optional<Integer> decision) {
    PythonAPICall py("sample_categorical");
    py.Input("candidates", candidates);
    py.Input("probs", probs);
    py.Decision(decision);
    py.SingleOutput(outputs);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(SampleCategoricalTraits);

}  // namespace tir
}  // namespace tvm
