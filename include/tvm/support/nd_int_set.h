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
 * \file nd_int_set.h
 * \brief N-dimensional Integer set
 */
#ifndef TVM_SUPPORT_ND_INT_SET_H_
#define TVM_SUPPORT_ND_INT_SET_H_

#include <tvm/ir/expr.h>
#include <tvm/arith/int_set.h>

namespace tvm {
namespace support {

using NDIntSet = std::vector<tvm::arith::IntSet>;

/*!
 * \brief Construct an integer set representing a range.
 * \param min The minimum value of the range.
 * \param max The extent of the extent.
 * \return constructed set.
 */
arith::IntSet IntSetFromMinExtent(const PrimExpr& min, const PrimExpr& extent);

/*!
 * \brief Construct an N-dimensional integer set representing a region.
 * \param region The region.
 * \return constructed set.
 */
NDIntSet NDIntSetFromRegion(const tir::Region& region);

/*!
 * \brief Construct an N-dimensional integer set representing a shape.
 * \param shape The shape which is an array of the length of each dimension.
 * \return constructed set.
 */
NDIntSet NDIntSetFromShape(const Array<PrimExpr>& shape);

/*!
 * \brief Construct an N-dimensional integer set representing a point.
 * \param indices The N-dimensional indices representing the point.
 * \return constructed set.
 */
NDIntSet NDIntSetFromPoint(const Array<PrimExpr>& indices);

/*!
 * \brief Create a union set of two sets, possibly relaxed. The RHS set will be combined into the
 *        LHS set.
 * \param lhs The first N-dimensional integer set
 * \param rhs The second N-dimensional integer set
 */
void NDIntSetUnionWith(NDIntSet* lhs, const NDIntSet& rhs);

/*!
 * \brief Create an empty N-dimensional integer set.
 * \param ndim The number of dimensions.
 * \return constructed set.
 */
NDIntSet NDIntSetEmpty(int ndim);

/*!
 * \brief The N-dimensional version of EvalSet.
 * \param nd_int_set The N-dimensional integer set to be evaluated.
 * \param dom_map The domain of each variable.
 * \return An N-dimensional integer set that can cover all the possible values of the N-dimensional
 *         integer set.
 * \sa EvalSet
 */
NDIntSet EvalNDIntSet(const NDIntSet& nd_int_set,
                      const std::unordered_map<const tir::VarNode*, arith::IntSet>& dom_map);

}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_ND_INT_SET_H_
