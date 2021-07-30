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
 * \file rng.h
 * \brief Random number generator, for Sampler and Sampling
 *  functions.
 */

#ifndef TVM_SUPPORT_RNG_H_
#define TVM_SUPPORT_RNG_H_

#include <tvm/runtime/logging.h>

#include <cstdint>  // for int64_t

namespace tvm {
namespace support {

/*!
 * \brief The random number generator is implemented as a linear congruential engine.
 */
class RandomNumberGenerator {
 public:
  /*! \brief The result type is defined as int64_t here for sampler usage. */
  using result_type = int64_t;

  /*! \brief The multiplier */
  static constexpr result_type multiplier = 48271;

  /*! \brief The increment */
  static constexpr result_type increment = 0;

  /*! \brief The modulus */
  static constexpr result_type modulus = 2147483647;

  /*! \brief Construct a null random number generator. */
  RandomNumberGenerator() { rand_state_ptr = nullptr; }

  /*!
   * \brief Construct a random number generator with a random state pointer.
   * \param random_state The random state pointer given in result_type*.
   * \note The random state is not initialized here. You may need to call seed function.
   */
  explicit RandomNumberGenerator(result_type* random_state) { rand_state_ptr = random_state; }

  /*!
   * \brief Change the start random state of RNG with the seed of a new random state value.
   * \param random_state The random state given in result_type.
   * \note The seed is used to initialize the random number generator and the random state would be
   * changed to next random state by calling the next_state() function.
   */
  void seed(result_type state = 1) {
    state %= modulus;                   // Make sure the seed is within the range of the modulus.
    if (state < 0) state += modulus;    // The congruential engine is always non-negative.
    ICHECK(rand_state_ptr != nullptr);  // Make sure the pointer is not null.
    *rand_state_ptr = state;            // Change pointed random state to given random state value.
    next_state();
  };

  /*! \brief The minimum possible value of random state here. */
  result_type min() { return 0; }

  /*! \brief The maximum possible value of random state here. */
  result_type max() { return modulus - 1; }

  /*!
   * \brief Fetch the current random state.
   * \return The current random state value in the type of result_type.
   */
  result_type random_state() { return *rand_state_ptr; }

  /*!
   * \brief Operator to fetch the current random state.
   * \return The current random state value in the type of result_type.
   */
  result_type operator()() { return next_state(); }

  /*!
   * \brief Move the random state to the next and return the new random state. According to
   *  definition of linear congruential engine, the new random state value is computed as
   *  new_random_state = (current_random_state * multiplier + increment) % modulus.
   * \return The next current random state value in the type of result_type.
   */
  result_type next_state() {
    (*rand_state_ptr) = ((*rand_state_ptr) * multiplier + increment) % modulus;
    return *rand_state_ptr;
  }

 private:
  result_type* rand_state_ptr;
};

}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_RNG_H_
