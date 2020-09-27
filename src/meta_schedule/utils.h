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
#ifndef SRC_META_SCHEDULE_UTILS_H_
#define SRC_META_SCHEDULE_UTILS_H_

#include <tvm/tir/expr.h>

#include <vector>

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Compute mean of a FloatImm array.
 * Taken from Ansor
 * \param float_array The array of floating point numbers to be averaged
 * \return The mean of the given array
 */
inline double FloatArrayMean(const Array<PrimExpr>& float_array) {
  double sum = 0;
  if (float_array.empty()) {
    return 0.0;
  }
  for (const auto& x : float_array) {
    const auto* float_imm = x.as<tir::FloatImmNode>();
    CHECK(float_imm != nullptr);
    sum += float_imm->value;
  }
  return sum / float_array.size();
}

/*!
 * \brief An empty output stream
 * Taken from Ansor
 */
class NullStream : public std::ostream {
 public:
  NullStream() : std::ostream(nullptr) {}
  NullStream(const NullStream&) : std::ostream(nullptr) {}
  static NullStream& Global();
};

template <class T>
NullStream& operator<<(NullStream& os, const T& value) {
  return os;
}

/*!
 * \brief Get std cout with verbose control
 * Taken from Ansor
 */
inline std::ostream& StdCout(int verbose, int setting = 1) {
  return verbose >= setting ? std::cout : NullStream::Global();
}

/*!
 * \brief Find all positions that the specific char occurs in the string
 * \param str The string to be examined
 * \param c The specific char
 * \return A list of integers indicating the occurrence position
 */
inline std::vector<int> FindCharPos(const String& str, char c) {
  std::vector<int> result;
  const char* data = str.data();
  int n = str.length();
  for (int i = 0; i < n; ++i) {
    if (data[i] == c) {
      result.push_back(i);
    }
  }
  return result;
}

template <class T>
std::vector<T> ConcatArray(const std::vector<std::vector<T> >& source) {
  std::vector<T> result;
  for (const std::vector<T>& item : source) {
    result.insert(result.end(), item.begin(), item.end());
  }
  return result;
}

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_UTILS_H_
