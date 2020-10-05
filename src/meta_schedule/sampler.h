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
#ifndef SRC_META_SCHEDULE_SAMPLER_H_
#define SRC_META_SCHEDULE_SAMPLER_H_

#include <tvm/node/node.h>

#include <memory>
#include <random>
#include <vector>

namespace tvm {
namespace meta_schedule {

namespace sampler {

/*! \brief A phantom type indicating the sampler uses device random as its seed */
struct DeviceRandType {};

}  // namespace sampler

/*! \brief A singleton indicating the sampler uses device random as its seed */
constexpr sampler::DeviceRandType DeviceRand{};

/*! \brief Random number sampler used for sampling in meta schedule */
class Sampler {
 public:
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
   * \brief Sample an array with replacement
   * \tparam T The type of elements
   * \param array The array to be indexed
   * \param num_samples Number of samples we want to draw
   * \return The result of sampling
   */
  template <class T>
  runtime::Array<T> SampleWithReplacement(const runtime::Array<T>& array, int num_samples) {
    std::vector<int> indices = SampleInts(num_samples, 0, array.size());
    runtime::Array<T> result;
    result.reserve(num_samples);
    for (int index : indices) {
      result.push_back(array[index]);
    }
    return result;
  }
  /*!
   * \brief Classic sampling without replacement
   * \param n The population size
   * \param k The number of samples to be drawn from the population
   * \return A list of indices, samples drawn, unsorted and index starting from 0
   */
  std::vector<int> SampleWithoutReplacement(int n, int k);
  /*! \brief Default constructor */
  Sampler() : Sampler(DeviceRand) {}
  /*!
   * \brief Constructor. Construct a sampler seeded with std::random_device
   */
  explicit Sampler(sampler::DeviceRandType) : Sampler(std::random_device /**/ {}()) {}
  /*!
   * \brief Constructor. Construct a sampler seeded with the specific integer
   * \param seed The random seed
   */
  explicit Sampler(int seed) : rand(seed) {}
  /*!
   * \brief Return a thread-local instance of the sampler
   * \return The thread-local sampler
   */
  static Sampler* ThreadLocal();

 private:
  /*! \brief The random number generator */
  std::minstd_rand rand;
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SAMPLER_H_
