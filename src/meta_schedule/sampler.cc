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
#include "./sampler.h"  // NOLINT(build/include)

#include <random>

namespace tvm {
namespace meta_schedule {

int Sampler::SampleInt(int min_inclusive, int max_exclusive) {
  std::uniform_int_distribution<> dist(min_inclusive, max_exclusive - 1);
  return dist(rand);
}

std::vector<int> Sampler::SampleInts(int n, int min_inclusive, int max_exclusive) {
  std::uniform_int_distribution<> dist(min_inclusive, max_exclusive - 1);
  std::vector<int> result;
  result.reserve(n);
  for (int i = 0; i < n; ++i) {
    result.push_back(dist(rand));
  }
  return result;
}

std::vector<double> Sampler::SampleUniform(int n, double min, double max) {
  std::uniform_real_distribution<double> dist(min, max);
  std::vector<double> result;
  result.reserve(n);
  for (int i = 0; i < n; ++i) {
    result.push_back(dist(rand));
  }
  return result;
}

bool Sampler::SampleBernoulli(double p) {
  std::bernoulli_distribution dist(p);
  return dist(rand);
}

std::function<int()> Sampler::MakeMultinomial(const std::vector<double>& weights) {
  std::vector<double> sums;
  sums.reserve(weights.size());
  double sum = 0.0;
  for (double w : weights) {
    sums.push_back(sum += w);
  }
  std::uniform_real_distribution<double> dist(0.0, sum);
  auto sampler = [this, dist = std::move(dist), sums = std::move(sums)]() mutable -> int {
    double p = dist(rand);
    return std::lower_bound(sums.begin(), sums.end(), p) - sums.begin();
  };
  return sampler;
}

std::vector<int> Sampler::SampleTileFactor(int n, int extent, const std::vector<int>& candidates) {
  constexpr int kMaxTrials = 100;
  std::uniform_int_distribution<> dist(0, static_cast<int>(candidates.size()) - 1);
  std::vector<int> sample(n, -1);
  for (int trial = 0; trial < kMaxTrials; ++trial) {
    int64_t product = 1;
    for (int i = 1; i < n; ++i) {
      int value = candidates[dist(rand)];
      product *= value;
      if (product > extent) {
        break;
      }
      sample[i] = value;
    }
    if (product <= extent) {
      sample[0] = (extent + product - 1) / product;
      return sample;
    }
  }
  sample[0] = extent;
  for (int i = 1; i < n; ++i) {
    sample[i] = 1;
  }
  return sample;
}

}  // namespace meta_schedule
}  // namespace tvm
