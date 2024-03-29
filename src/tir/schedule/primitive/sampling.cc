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

#include "../utils.h"

namespace tvm {
namespace tir {

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
    ICHECK_EQ(static_cast<int>(primes.size()), int(kNumPrimes));
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

TRandState ForkSeed(TRandState* rand_state) {
  // In order for reproducibility, we computer the new seed using RNG's random state and a
  // different set of parameters. Note that both 32767 and 1999999973 are prime numbers.
  TRandState ret = (RandEngine(rand_state)() * 32767) % 1999999973;
  return ret;
}

int SampleInt(TRandState* rand_state, int min_inclusive, int max_exclusive) {
  RandEngine rand_(rand_state);

  if (min_inclusive + 1 == max_exclusive) {
    return min_inclusive;
  }
  std::uniform_int_distribution<> dist(min_inclusive, max_exclusive - 1);
  return dist(rand_);
}

std::vector<int> SampleInts(TRandState* rand_state, int n, int min_inclusive, int max_exclusive) {
  RandEngine rand_(rand_state);
  std::uniform_int_distribution<> dist(min_inclusive, max_exclusive - 1);
  std::vector<int> result;
  result.reserve(n);
  for (int i = 0; i < n; ++i) {
    result.push_back(dist(rand_));
  }
  return result;
}

template <typename RandomAccessIterator>
void SampleShuffle(TRandState* rand_state, RandomAccessIterator begin_it,
                   RandomAccessIterator end_it) {
  RandEngine rand_(rand_state);
  std::shuffle(begin_it, end_it, rand_);
}

std::vector<double> SampleUniform(TRandState* rand_state, int n, double min, double max) {
  RandEngine rand_(rand_state);
  std::uniform_real_distribution<double> dist(min, max);
  std::vector<double> result;
  result.reserve(n);
  for (int i = 0; i < n; ++i) {
    result.push_back(dist(rand_));
  }
  return result;
}

bool SampleBernoulli(TRandState* rand_state, double p) {
  RandEngine rand_(rand_state);
  std::bernoulli_distribution dist(p);
  return dist(rand_);
}

std::function<int()> MakeMultinomial(TRandState* rand_state, const std::vector<double>& weights) {
  RandEngine rand_(rand_state);
  std::vector<double> sums;
  sums.reserve(weights.size());
  double sum = 0.0;
  for (double w : weights) {
    sums.push_back(sum += w);
  }
  std::uniform_real_distribution<double> dist(0.0, sum);
  auto sampler = [rand_state, dist = std::move(dist), sums = std::move(sums)]() mutable -> int {
    RandEngine rand_(rand_state);
    double p = dist(rand_);
    int idx = std::lower_bound(sums.begin(), sums.end(), p) - sums.begin();
    int n = sums.size();
    CHECK_LE(0, idx);
    CHECK_LE(idx, n);
    return (idx == n) ? (n - 1) : idx;
  };
  return sampler;
}

std::vector<int> SampleWithoutReplacement(TRandState* rand_state, int n, int k) {
  if (k == 1) {
    return {SampleInt(rand_state, 0, n)};
  }
  if (k == 2) {
    int result0 = SampleInt(rand_state, 0, n);
    int result1 = SampleInt(rand_state, 0, n - 1);
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
    int j = SampleInt(rand_state, i, n);
    if (i != j) {
      std::swap(order[i], order[j]);
    }
  }
  return {order.begin(), order.begin() + k};
}

std::vector<int> SampleTileFactor(TRandState* rand_state, int n, int extent,
                                  const std::vector<int>& candidates) {
  RandEngine rand_(rand_state);
  constexpr int kMaxTrials = 100;
  std::uniform_int_distribution<> dist(0, static_cast<int>(candidates.size()) - 1);
  std::vector<int> sample(n, -1);
  for (int trial = 0; trial < kMaxTrials; ++trial) {
    int64_t product = 1;
    for (int i = 1; i < n; ++i) {
      int value = candidates[dist(rand_)];
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

std::vector<int> SamplePerfectTile(TRandState* rand_state, int n_splits, int extent) {
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
        (SampleInt(rand_state, 0, 2) ? result1 : result0) *= ij.second;
        continue;
      }
      // Case 2: (a = primes[i], p = 1)
      int p = ij.second;
      const int* pow = prime_tab->pow_tab[ij.first].data() - 1;
      int x1 = SampleInt(rand_state, 0, p + 1);
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
      result[SampleInt(rand_state, 0, n_splits)] *= ij.second;
      continue;
    }
    // Case 2: (a = primes[i], p = 1)
    int p = ij.second;
    if (p == 1) {
      result[SampleInt(rand_state, 0, n_splits)] *= prime_tab->primes[ij.first];
      continue;
    }
    // The general case. We have to sample uniformly from the solution of:
    //    x_1 + x_2 + ... + x_{n_splits} = p
    // where x_i >= 0
    // Data range:
    //    2 <= p <= 31
    //    3 <= n_splits <= max tiling splits
    std::vector<int> sampled = SampleWithoutReplacement(rand_state, p + n_splits - 1, n_splits - 1);
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

std::vector<int> SamplePerfectTile(TRandState* rand_state, int n_splits, int extent,
                                   int max_innermost_factor) {
  if (max_innermost_factor == -1) {
    return SamplePerfectTile(rand_state, n_splits, extent);
  }
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
  int innermost = innermost_candidates[SampleInt(rand_state, 0, innermost_candidates.size())];
  std::vector<int> result = SamplePerfectTile(rand_state, n_splits - 1, extent / innermost);
  result.push_back(innermost);
  return result;
}

static inline int ExtractInt(const Target& target, const char* name) {
  if (Optional<Integer> v = target->GetAttr<Integer>(name)) {
    return v.value()->value;
  }
  LOG(FATAL) << "AttributedError: \"" << name << "\" is not defined in the target";
  throw;
}

static inline bool IsCudaTarget(const Target& target) {
  if (Optional<String> v = target->GetAttr<String>("kind")) {
    return v.value() == "cuda";
  }
  return false;
}

std::vector<std::vector<int>> SampleShapeGenericTiles(TRandState* rand_state,
                                                      const std::vector<int>& n_splits,
                                                      const std::vector<int>& max_extents,
                                                      const Target& target,
                                                      int max_innermost_factor) {
  std::vector<std::vector<int>> ret_split_factors;

  if (IsCudaTarget(target)) {
    // The following factorization scheme is built under the assumption that: (1) The target is
    // CUDA, and (2) The tiling structure is SSSRRSRS.

    // extract all the hardware parameters
    const struct HardwareConstraints {
      int max_shared_memory_per_block;
      int max_local_memory_per_block;
      int max_threads_per_block;
      int max_innermost_factor;
      int max_vthread;
    } constraints = {ExtractInt(target, "shared_memory_per_block"),
                     ExtractInt(target, "registers_per_block"),
                     ExtractInt(target, "max_threads_per_block"), max_innermost_factor, 8};

    for (const int n_split : n_splits) {
      ret_split_factors.push_back(std::vector<int>(n_split, 1));
    }

    // sample the number of threads per block
    const int warp_size = ExtractInt(target, "warp_size");
    int num_threads_per_block =
        SampleInt(rand_state, 1, constraints.max_threads_per_block / warp_size) * warp_size;
    // find all the possible factors of the number of threads per block
    int num_spatial_axes = 0;
    size_t last_spatial_iter_id = -1;
    for (size_t iter_id = 0; iter_id < n_splits.size(); ++iter_id) {
      CHECK(n_splits[iter_id] == 4 || n_splits[iter_id] == 2)
          << "The tiling structure is not SSSRRSRS";
      if (n_splits[iter_id] == 4) {
        ++num_spatial_axes;
        last_spatial_iter_id = iter_id;
      }
    }

    bool all_below_max_extents;
    std::vector<int> num_threads_factor_scheme;
    do {
      all_below_max_extents = true;

      num_threads_factor_scheme =
          SamplePerfectTile(rand_state, num_spatial_axes, num_threads_per_block);
      for (size_t iter_id = 0, spatial_iter_id = 0; iter_id < n_splits.size(); ++iter_id) {
        if (n_splits[iter_id] == 4) {
          if (num_threads_factor_scheme[spatial_iter_id] > max_extents[iter_id]) {
            all_below_max_extents = false;
          }
          ++spatial_iter_id;
        }
      }  // for (iter_id ∈ [0, split_steps_info.size()))
    } while (!all_below_max_extents);

    // do the looping again and assign the factors
    for (size_t iter_id = 0, spatial_iter_id = 0; iter_id < n_splits.size(); ++iter_id) {
      if (n_splits[iter_id] == 4) {
        ret_split_factors[iter_id][1] = num_threads_factor_scheme[spatial_iter_id];
        ++spatial_iter_id;
      }
    }

    // factor[0] (vthread)
    int reg_usage = num_threads_per_block, shmem_usage = 0;

    auto sample_factors = [&](std::function<bool(const size_t)> continue_predicate,
                              std::function<int(const size_t)> max_extent,
                              std::function<int&(const size_t)> factor_to_assign) {
      std::vector<int> iter_max_extents;
      std::vector<int> factors_to_assign;
      for (size_t iter_id = 0; iter_id < n_splits.size(); ++iter_id) {
        if (continue_predicate(iter_id)) {
          continue;
        }
        size_t iter_max_extent = max_extent(iter_id), factor_to_assign;

        std::uniform_int_distribution<> dist(1, iter_max_extent);
        factor_to_assign = SampleInt(rand_state, 1, iter_max_extent);

        if (n_splits[iter_id] == 4) {
          reg_usage *= factor_to_assign;
        } else {
          shmem_usage *= factor_to_assign;
        }
        iter_max_extents.push_back(iter_max_extent);
        factors_to_assign.push_back(factor_to_assign);
      }
      // shuffle the factors
      std::vector<int> factors_to_assign_bak = factors_to_assign;
      SampleShuffle(rand_state, factors_to_assign.begin(), factors_to_assign.end());
      // make sure that the shuffle is valid
      bool valid_shuffle = true;
      std::vector<int>::iterator iter_max_extents_it = iter_max_extents.begin(),
                                 factors_to_assign_it = factors_to_assign.begin();

      for (size_t iter_id = 0; iter_id < n_splits.size(); ++iter_id) {
        if (continue_predicate(iter_id)) {
          continue;
        }
        int iter_max_extent = *iter_max_extents_it;
        if (*factors_to_assign_it > iter_max_extent) {
          valid_shuffle = false;
        }
        ++iter_max_extents_it;
        ++factors_to_assign_it;
      }
      if (!valid_shuffle) {
        factors_to_assign = std::move(factors_to_assign_bak);
      }
      // do the actual assignment
      factors_to_assign_it = factors_to_assign.begin();
      for (size_t iter_id = 0; iter_id < n_splits.size(); ++iter_id) {
        if (continue_predicate(iter_id)) {
          continue;
        }
        factor_to_assign(iter_id) = *factors_to_assign_it;
        ++factors_to_assign_it;
      }
    };

    sample_factors(
        [&](const size_t iter_id) -> bool {
          return (n_splits[iter_id] != 4) || (iter_id != last_spatial_iter_id);
        },
        [&](const size_t iter_id) -> int {
          size_t max_vthread_extent = std::min(
              constraints.max_vthread, max_extents[iter_id] / ret_split_factors[iter_id][1]);
          max_vthread_extent =
              std::min(constraints.max_vthread, constraints.max_local_memory_per_block / reg_usage);
          return max_vthread_extent;
        },
        [&](const size_t iter_id) -> int& { return ret_split_factors[iter_id][0]; });

    // factor[3] (innermost)
    sample_factors(
        [&](const size_t iter_id) -> bool {
          return (n_splits[iter_id] != 4) || (iter_id == last_spatial_iter_id);
        },
        [&](const size_t iter_id) -> int {
          int max_innermost_extent =
              std::min(max_innermost_factor, max_extents[iter_id] / ret_split_factors[iter_id][0] /
                                                 ret_split_factors[iter_id][1]);
          max_innermost_extent =
              std::min(max_innermost_extent, constraints.max_local_memory_per_block / reg_usage);
          return max_innermost_extent;
        },
        [&](const size_t iter_id) -> int& { return ret_split_factors[iter_id][3]; });
    // factor[2]
    sample_factors([&](const size_t iter_id) -> bool { return (n_splits[iter_id] != 4); },
                   [&](const size_t iter_id) -> size_t {
                     size_t max_2nd_innermost_extent =
                         std::min(max_extents[iter_id] / ret_split_factors[iter_id][0] /
                                      ret_split_factors[iter_id][1] / ret_split_factors[iter_id][3],
                                  constraints.max_local_memory_per_block / reg_usage);
                     return max_2nd_innermost_extent;
                   },
                   [&](const size_t iter_id) -> int& { return ret_split_factors[iter_id][2]; });

    for (size_t iter_id = 0; iter_id < n_splits.size(); ++iter_id) {
      if (n_splits[iter_id] == 4) {
        shmem_usage += ret_split_factors[iter_id][0] * ret_split_factors[iter_id][1] *
                       ret_split_factors[iter_id][2] * ret_split_factors[iter_id][3];
      }
    }
    if (shmem_usage > static_cast<int>(constraints.max_shared_memory_per_block / sizeof(float))) {
      LOG(FATAL) << "shmem_usage goes out-of-range";
    }
    // repeat similar procedure for reduction axes
    // rfactor[1] (innermost)
    sample_factors(
        [&](const size_t iter_id) -> bool { return (n_splits[iter_id] != 2); },
        [&](const size_t iter_id) -> int {
          int max_innermost_extent = std::min(max_innermost_factor, max_extents[iter_id]);
          max_innermost_extent = std::min(max_innermost_extent,
                                          static_cast<int>(constraints.max_shared_memory_per_block /
                                                           sizeof(float) / shmem_usage));
          return max_innermost_extent;
        },
        [&](const size_t iter_id) -> int& { return ret_split_factors[iter_id][1]; });
    // rfactor[0]
    sample_factors([&](const size_t iter_id) -> bool { return (n_splits[iter_id] != 2); },
                   [&](const size_t iter_id) -> size_t {
                     size_t max_2nd_innermost_extent =
                         std::min(max_extents[iter_id] / ret_split_factors[iter_id][1],
                                  static_cast<int>(constraints.max_shared_memory_per_block /
                                                   sizeof(float) / shmem_usage));
                     return max_2nd_innermost_extent;
                   },
                   [&](const size_t iter_id) -> int& { return ret_split_factors[iter_id][0]; });
  }  // if (IsCudaTarget(target))
  return ret_split_factors;
}

std::vector<int64_t> SamplePerfectTile(tir::ScheduleState self, TRandState* rand_state,
                                       const tir::StmtSRef& loop_sref, int n,
                                       int max_innermost_factor,
                                       Optional<Array<Integer>>* decision) {
  const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
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
    std::vector<int> sampled = SamplePerfectTile(rand_state, n, extent, max_innermost_factor);
    result = std::vector<int64_t>(sampled.begin(), sampled.end());
    ICHECK_LE(sampled.back(), max_innermost_factor);
  }
  *decision = AsArray<int64_t, Integer>(result);
  return result;
}

std::vector<std::vector<int64_t>> SampleShapeGenericTiles(tir::ScheduleState self,
                                                          TRandState* rand_state,
                                                          const Array<tir::StmtSRef>& loop_srefs,
                                                          const Array<IntImm>& ns,
                                                          const Target& target,
                                                          int max_innermost_factor,
                                                          Optional<Array<Array<Integer>>>* decision) {
  std::vector<int> extents;
  for (const StmtSRef& loop_sref : loop_srefs) {
    const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
    extents.push_back(GetLoopIntExtent(loop));
  }
  std::vector<int> n_splits;
  for (const IntImm& n : ns) {
    n_splits.push_back(n->value);
  }
  std::vector<std::vector<int>> sampled_tiles =
      SampleShapeGenericTiles(rand_state, n_splits, extents, target, max_innermost_factor);
  std::vector<std::vector<int64_t>> result;
  *decision = Array<Array<Integer>>();
  for (const std::vector<int>& sampled : sampled_tiles) {
    result.emplace_back(sampled.begin(), sampled.end());
    decision->value().push_back(AsArray<int64_t, Integer>(result.back()));
  }
  return result;
}

int64_t SampleCategorical(tir::ScheduleState self, TRandState* rand_state,
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

tir::StmtSRef SampleComputeLocation(tir::ScheduleState self, TRandState* rand_state,
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
    i = SampleInt(rand_state, -2, choices.size());
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

struct SamplePerfectTileTraits : public UnpackedInstTraits<SamplePerfectTileTraits> {
  static constexpr const char* kName = "SamplePerfectTile";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 1;

  static Array<ExprRV> UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv, Integer n,
                                               Integer max_innermost_factor,
                                               Optional<Array<Integer>> decision) {
    return sch->SamplePerfectTile(loop_rv, n->value, max_innermost_factor->value, decision);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv, Integer n,
                                 Integer max_innermost_factor, Optional<Array<Integer>> decision) {
    PythonAPICall py("sample_perfect_tile");
    py.Input("loop", loop_rv);
    py.Input("n", n->value);
    py.Input("max_innermost_factor", max_innermost_factor->value);
    py.Decision(decision);
    py.OutputList(outputs);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

struct SampleShapeGenericTilesTraits : public UnpackedInstTraits<SampleShapeGenericTilesTraits> {
  static constexpr const char* kName = "SampleShapeGenericTiles";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 3;
  static constexpr size_t kNumDecisions = 1;

  static Array<Array<ExprRV>> UnpackedApplyToSchedule(Schedule sch, Array<LoopRV> loop_rvs,
                                                      Array<IntImm> ns, Target target,
                                                      Integer max_innermost_factor,
                                                      Optional<Array<Array<Integer>>> decision) {
    return sch->SampleShapeGenericTiles(loop_rvs, ns, target, max_innermost_factor->value,
                                        decision);
  }

  static String UnpackedAsPython(Array<String> outputs, Array<String> loop_rvs, Array<Integer> ns,
                                 Target target, Integer max_innermost_factor,
                                 Optional<Array<Array<Integer>>> decision) {
    PythonAPICall py("sample_shape_generic_tiles");
    py.Input("loops", loop_rvs);
    py.Input("ns", ns);
    py.Input("max_innermost_factor", max_innermost_factor->value);
    py.Input("target", target->str());
    py.Decision(decision);
    py.OutputList(outputs);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

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

struct SampleComputeLocationTraits : public UnpackedInstTraits<SampleComputeLocationTraits> {
  static constexpr const char* kName = "SampleComputeLocation";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 1;

  static LoopRV UnpackedApplyToSchedule(Schedule sch,      //
                                        BlockRV block_rv,  //
                                        Optional<Integer> decision) {
    return sch->SampleComputeLocation(block_rv, decision);
  }

  static String UnpackedAsPython(Array<String> outputs,  //
                                 String block_rv,        //
                                 Optional<Integer> decision) {
    PythonAPICall py("sample_compute_location");
    py.Input("block", block_rv);
    py.Decision(decision);
    py.SingleOutput(outputs);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(SamplePerfectTileTraits);
TVM_REGISTER_INST_KIND_TRAITS(SampleCategoricalTraits);
TVM_REGISTER_INST_KIND_TRAITS(SampleComputeLocationTraits);

}  // namespace tir
}  // namespace tvm
