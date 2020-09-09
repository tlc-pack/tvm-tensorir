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
   * \brief Sample n tiling factors of the given extent
   * \param n The number of parts the loop is split
   * \param extent Length of the loop
   * \param candidates The possible tiling factors
   * \return A list of length n, the tiling factors sampled
   */
  std::vector<int> SampleTileFactor(int n, int extent, const std::vector<int>& candidates);
  /*! \brief Default constructor */
  Sampler() : Sampler(DeviceRand) {}
  /*!
   * \brief Constructor. Construct a sampler seeded with std::random_device
   */
  explicit Sampler(sampler::DeviceRandType) : Sampler(std::random_device /**/ {}()) {}
  /*!
   * \brief Constructor. Construct a sampler seeded with the given integer
   * \param seed The random seed
   */
  explicit Sampler(int seed) : rand(seed) {}

 private:
  /*! \brief The random number generator */
  std::minstd_rand rand;
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SAMPLER_H_
