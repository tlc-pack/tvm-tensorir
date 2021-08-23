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

#include <tvm/support/nd_int_set.h>

namespace tvm {
namespace support {

using namespace arith;

IntSet IntSetFromMinExtent(const PrimExpr& min, const PrimExpr& extent) {
   return arith::IntSet::FromRange(Range::FromMinExtent(min, extent));
}

NDIntSet NDIntSetFromRegion(const tir::Region& region) {
  NDIntSet result;
  result.reserve(region.size());
  for (const Range& range : region) {
    result.push_back(IntSet::FromRange(range));
  }
  return result;
}

NDIntSet NDIntSetFromShape(const Array<PrimExpr>& shape) {
  PrimExpr zero = Integer(0);
  NDIntSet result;
  result.reserve(shape.size());
  for (const PrimExpr& extent : shape) {
    result.push_back(IntSetFromMinExtent(zero, extent));
  }
  return result;
}

NDIntSet NDIntSetFromPoint(const Array<PrimExpr>& indices) {
  NDIntSet result;
  result.reserve(indices.size());
  for (const PrimExpr& index : indices) {
    result.push_back(IntSet::SinglePoint(index));
  }
  return result;
}

void NDIntSetUnionWith(NDIntSet* lhs, const NDIntSet& rhs) {
  ICHECK_EQ(lhs->size(), rhs.size());
  int ndim = rhs.size();
  for (int i = 0; i < ndim; ++i) {
    IntSet& int_set = lhs->at(i);
    int_set = Union({int_set, rhs.at(i)});
  }
}

NDIntSet NDIntSetEmpty(int ndim) {
  return std::vector<IntSet>(ndim, IntSet::Nothing());
}

NDIntSet EvalNDIntSet(const NDIntSet& nd_int_set,
                      const std::unordered_map<const VarNode*, IntSet>& dom_map) {
  NDIntSet ret;
  ret.reserve(nd_int_set.size());
  for (const IntSet& s : nd_int_set) {
    ret.push_back(EvalSet(s, dom_map));
  }
  return ret;
}

}  // namespace support
}  // namespace tvm
