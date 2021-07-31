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
#ifndef TVM_TIR_SCHEDULE_SAMPLER_H_
#define TVM_TIR_SCHEDULE_SAMPLER_H_

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "../../support/rng.h"
namespace tvm {

class Target;

namespace tir {

/*!
 * \brief Sampler based on random number generator for sampling in meta schedule.
 * \note Typical usage is like Sampler(&random_state).SamplingFunc(...).
 */
class Sampler {
 public:
  /*! Random state type for random number generator. */
  using TRandomState = support::RandomNumberGenerator::result_type;
  /*!
   *  \brief Return a random state value that can be used as seed for new samplers.
   *  \return The random state value to be used as seed for new samplers.
   */
  TRandomState ForkSeed();
  /*!
   * \brief Re-seed the random number generator
   * \param seed The random state value given used to re-seed the RNG.
   */
  void Seed(TRandomState seed);
  /*!
   * \brief Sample an integer in [min_inclusive, max_exclusive)
   * \param min_inclusive The left boundary, inclusive
   * \param max_exclusive The right boundary, exclusive
   * \return The integer sampled
   */
  int SampleInt(int min_inclusive, int max_exclusive);
  /*!
   * \brief Sample n integers in [min_inclusive, max_exclusive)
   * \param min_inclusive The left boundary, inclusive
   * \param max_exclusive The right boundary, exclusive
   * \return The list of integers sampled
   */
  std::vector<int> SampleInts(int n, int min_inclusive, int max_exclusive);
  /*!
   * \brief Random shuffle from the begin iterator to the end.
   * \param begin_it The begin iterator
   * \param end_it The end iterator
   */
  template <typename RandomAccessIterator>
  void Shuffle(RandomAccessIterator begin_it, RandomAccessIterator end_it);
  /*!
   * \brief Sample n tiling factors of the specific extent
   * \param n The number of parts the loop is split
   * \param extent Length of the loop
   * \param candidates The possible tiling factors
   * \return A list of length n, the tiling factors sampled
   */
  std::vector<int> SampleTileFactor(int n, int extent, const std::vector<int>& candidates);
  /*!
   * \brief Sample perfect tiling factor of the specific extent
   * \param n_splits The number of parts the loop is split
   * \param extent Length of the loop
   * \return A list of length n_splits, the tiling factors sampled, the product of which strictly
   * equals to extent
   */
  std::vector<int> SamplePerfectTile(int n_splits, int extent);
  /*!
   * \brief Sample perfect tiling factor of the specific extent
   * \param n_splits The number of parts the loop is split
   * \param extent Length of the loop
   * \param max_innermost_factor A small number indicating the max length of the innermost loop
   * \return A list of length n_splits, the tiling factors sampled, the product of which strictly
   * equals to extent
   */
  std::vector<int> SamplePerfectTile(int n_splits, int extent, int max_innermost_factor);
  /*!
   * \brief Sample shape-generic tiling factors that are determined by the hardware constraints.
   * \param n_splits The number of parts the loops are split
   * \param max_extents Maximum length of the loops
   * \param is_spatial Whether each loop is a spatial axis or not
   * \param target Hardware target
   * \param max_innermost_factor A small number indicating the max length of the innermost loop
   * \return A list of list of length n_splits, the tiling factors sampled, all satisfying the
   * maximum extents and the hardware constraints
   */
  std::vector<std::vector<int>> SampleShapeGenericTiles(const std::vector<int>& n_splits,
                                                        const std::vector<int>& max_extents,
                                                        const Target& target,
                                                        int max_innermost_factor);
  /*!
   * \brief Sample n floats uniformly in [min, max)
   * \param min The left boundary
   * \param max The right boundary
   * \return The list of floats sampled
   */
  std::vector<double> SampleUniform(int n, double min, double max);
  /*!
   * \brief Sample from a Bernoulli distribution
   * \param p Parameter in the Bernoulli distribution
   * \return return true with probability p, and false with probability (1 - p)
   */
  bool SampleBernoulli(double p);
  /*!
   * \brief Create a multinomial sampler based on the specific weights
   * \param weights The weights, event probabilities
   * \return The multinomial sampler
   */
  std::function<int()> MakeMultinomial(const std::vector<double>& weights);
  /*!
   * \brief Classic sampling without replacement
   * \param n The population size
   * \param k The number of samples to be drawn from the population
   * \return A list of indices, samples drawn, unsorted and index starting from 0
   */
  std::vector<int> SampleWithoutReplacement(int n, int k);
  /*! \brief The default constructor function for Sampler */
  Sampler() = default;
  /*!
   * \brief Constructor. Construct a sampler with a given random state pointer for its RNG.
   * \param random_state The given pointer to random state used to construct the RNG.
   * \note The random state is neither initialized not modified by this constructor.
   */
  explicit Sampler(TRandomState* random_state) : rand_(random_state) {}

 private:
  /*! \brief The random number generator for sampling. */
  support::RandomNumberGenerator rand_;
};

template <typename RandomAccessIterator>
void Sampler::Shuffle(RandomAccessIterator begin_it, RandomAccessIterator end_it) {
  std::shuffle(begin_it, end_it, rand_);
}

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_SAMPLER_H_
