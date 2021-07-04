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

struct PrimeTable {
  /*! \brief The table contains prime numbers in [2, kMaxPrime) */
  static constexpr const int kMaxPrime = 65536;
  /*! \brief The exact number of prime numbers in the table */
  static constexpr const int kNumPrimes = 6542;
  /*!
   * \brief For each number in [2, kMaxPrime), the index of its min factor.
   * For example, if min_factor_idx[x] = i, then the min factor of x is primes[i].
   */
  int min_factor_idx[kMaxPrime];
  /*! \brief The prime numbers in [2, kMaxPrime) */
  std::vector<int> primes;
  /*!
   * \brief The power of each prime number.
   * pow_table[i, j] stores the result of pow(prime[i], j + 1)
   */
  std::vector<std::vector<int>> pow_tab;

  /*! \brief Get a global instance of the prime table */
  static const PrimeTable* Global() {
    static const PrimeTable table;
    return &table;
  }

  /*! \brief Constructor, pre-computes all info in the prime table */
  PrimeTable() {
    constexpr const int64_t int_max = std::numeric_limits<int>::max();
    // Euler's sieve: prime number in linear time
    for (int i = 0; i < kMaxPrime; ++i) {
      min_factor_idx[i] = -1;
    }
    primes.reserve(kNumPrimes);
    for (int x = 2; x < kMaxPrime; ++x) {
      if (min_factor_idx[x] == -1) {
        min_factor_idx[x] = primes.size();
        primes.push_back(x);
      }
      for (size_t i = 0; i < primes.size(); ++i) {
        int factor = primes[i];
        int y = x * factor;
        if (y >= kMaxPrime) {
          break;
        }
        min_factor_idx[y] = i;
        if (x % factor == 0) {
          break;
        }
      }
    }
    CHECK_EQ(static_cast<int>(primes.size()), kNumPrimes);
    // Calculate the power table for each prime number
    pow_tab.reserve(primes.size());
    for (int prime : primes) {
      std::vector<int> tab;
      tab.reserve(32);
      for (int64_t pow = prime; pow <= int_max; pow *= prime) {
        tab.push_back(pow);
      }
      tab.shrink_to_fit();
      pow_tab.emplace_back(std::move(tab));
    }
  }
  /*!
   * \brief Factorize a number n, and return in a cryptic format
   * \param n The number to be factorized
   * \return A list of integer pairs [(i_1, j_1), (i_2, j_2), ..., (i_l, j_l)]
   * For each pair (i, j), we define
   *    (a, b) = (j, 1)             if i == -1 (in this case j must be a prime number)
   *             (primes[i], j)     if i != -1
   * Then the factorization is
   *    n = (a_1 ^ b_1) * (a_2 ^ b_2) ... (a_l ^ b_l)
   */
  std::vector<std::pair<int, int>> Factorize(int n) const {
    std::vector<std::pair<int, int>> result;
    result.reserve(16);
    int i = 0, n_primes = primes.size();
    // Phase 1: n >= kMaxPrime
    for (int j; n >= kMaxPrime && i < n_primes && primes[i] * primes[i] <= n; ++i) {
      for (j = 0; n % primes[i] == 0; n /= primes[i], ++j) {
      }
      if (j != 0) {
        result.emplace_back(i, j);
      }
    }
    // if i >= n_primes or primes[i] > sqrt(n), then n must be a prime number
    if (n >= kMaxPrime) {
      result.emplace_back(-1, n);
      return result;
    }
    // Phase 2: n < kMaxPrime
    for (int j; n > 1;) {
      int i = min_factor_idx[n];
      for (j = 0; n % primes[i] == 0; n /= primes[i], ++j) {
      }
      result.emplace_back(i, j);
    }
    return result;
  }
};

int Sampler::ForkSeed() {
  uint32_t a = this->rand();
  uint32_t b = this->rand();
  uint32_t c = this->rand();
  uint32_t d = this->rand();
  return (a ^ b) * (c ^ d) % 1145141;
}

void Sampler::Seed(int seed) { this->rand.seed(seed); }

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
    int idx = std::lower_bound(sums.begin(), sums.end(), p) - sums.begin();
    int n = sums.size();
    CHECK_LE(0, idx);
    CHECK_LE(idx, n);
    return (idx == n) ? (n - 1) : idx;
  };
  return sampler;
}

std::vector<int> Sampler::SampleWithoutReplacement(int n, int k) {
  if (k == 1) {
    return {SampleInt(0, n)};
  }
  if (k == 2) {
    int result0 = SampleInt(0, n);
    int result1 = SampleInt(0, n - 1);
    if (result1 >= result0) {
      result1 += 1;
    }
    return {result0, result1};
  }
  std::vector<int> order(n);
  for (int i = 0; i < n; ++i) {
    order[i] = i;
  }
  for (int i = 0; i < k; ++i) {
    int j = SampleInt(i, n);
    if (i != j) {
      std::swap(order[i], order[j]);
    }
  }
  return {order.begin(), order.begin() + k};
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

std::vector<int> Sampler::SamplePerfectTile(int n_splits, int extent) {
  CHECK_GE(extent, 1) << "ValueError: Cannot tile a loop with 0 or negative extent";
  CHECK_GE(n_splits, 1) << "ValueError: Cannot tile a loop to 0 or negative splits";
  // Handle special case that we can potentially accelerate
  if (n_splits == 1) {
    return {extent};
  }
  if (extent == 1) {
    return std::vector<int>(n_splits, 1);
  }
  // Enumerate each pair (i, j), we define
  //    (a, p) = (j, 1)             if i == -1 (in this case j must be a prime number)
  //             (primes[i], j)     if i != -1
  // Then the factorization is
  //    extent = (a_1 ^ p_1) * (a_2 ^ p_2) ... (a_l ^ p_l)
  const PrimeTable* prime_tab = PrimeTable::Global();
  std::vector<std::pair<int, int>> factorized = prime_tab->Factorize(extent);
  if (n_splits == 2) {
    // n_splits = 2, this can be taken special care of,
    // because general reservoir sampling can be avoided to accelerate the sampling
    int result0 = 1;
    int result1 = 1;
    for (const std::pair<int, int>& ij : factorized) {
      // Case 1: (a, p) = (j, 1), where j is a prime number
      if (ij.first == -1) {
        (SampleInt(0, 2) ? result1 : result0) *= ij.second;
        continue;
      }
      // Case 2: (a = primes[i], p = 1)
      int p = ij.second;
      const int* pow = prime_tab->pow_tab[ij.first].data() - 1;
      int x1 = SampleInt(0, p + 1);
      int x2 = p - x1;
      if (x1 != 0) {
        result0 *= pow[x1];
      }
      if (x2 != 0) {
        result1 *= pow[x2];
      }
    }
    return {result0, result1};
  }
  // Data range:
  //    2 <= extent <= 2^31 - 1
  //    3 <= n_splits <= max tiling splits
  //    1 <= p <= 31
  std::vector<int> result(n_splits, 1);
  for (const std::pair<int, int>& ij : factorized) {
    // Handle special cases to accelerate sampling
    // Case 1: (a, p) = (j, 1), where j is a prime number
    if (ij.first == -1) {
      result[SampleInt(0, n_splits)] *= ij.second;
      continue;
    }
    // Case 2: (a = primes[i], p = 1)
    int p = ij.second;
    if (p == 1) {
      result[SampleInt(0, n_splits)] *= prime_tab->primes[ij.first];
      continue;
    }
    // The general case. We have to sample uniformly from the solution of:
    //    x_1 + x_2 + ... + x_{n_splits} = p
    // where x_i >= 0
    // Data range:
    //    2 <= p <= 31
    //    3 <= n_splits <= max tiling splits
    std::vector<int> sampled = SampleWithoutReplacement(p + n_splits - 1, n_splits - 1);
    std::sort(sampled.begin(), sampled.end());
    sampled.push_back(p + n_splits - 1);
    const int* pow = prime_tab->pow_tab[ij.first].data() - 1;
    for (int i = 0, last = -1; i < n_splits; ++i) {
      int x = sampled[i] - last - 1;
      last = sampled[i];
      if (x != 0) {
        result[i] *= pow[x];
      }
    }
  }
  return result;
}

std::vector<int> Sampler::SamplePerfectTile(int n_splits, int extent, int max_innermost_factor) {
  CHECK_GE(n_splits, 2) << "ValueError: Cannot tile a loop into " << n_splits << " splits";
  std::vector<int> innermost_candidates;
  innermost_candidates.reserve(max_innermost_factor);
  for (int i = 1; i <= max_innermost_factor; ++i) {
    if (extent % i == 0) {
      innermost_candidates.push_back(i);
    }
  }
  // N.B. Theoretically sampling evenly breaks the uniform sampling of the global sampling space.
  // We should do multiple factorization to weight the choices. However, it would lead to slower
  // sampling speed. On the other hand, considering potential tricks we might do on the innermost
  // loop, in which sampling uniformly does not help, let's leave it as it is for now, and maybe add
  // more heuristics in the future
  int innermost = innermost_candidates[SampleInt(0, innermost_candidates.size())];
  std::vector<int> result = SamplePerfectTile(n_splits - 1, extent / innermost);
  result.push_back(innermost);
  return result;
}

}  // namespace meta_schedule
}  // namespace tvm
