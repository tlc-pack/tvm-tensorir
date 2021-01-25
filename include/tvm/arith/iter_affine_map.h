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
 * \file tvm/arith/iter_affine_map.h
 * \brief Iterator quasi-affine mapping patterns.
 *
 *  This file defines a collection of mapping patterns
 *  maps a collection of independent iterators to another
 *  collection of independent iterators.
 *
 *  There are two main kinds of mapping patterns:
 *
 *  - Fuse: fuse a collection of iterators into a single one
 *
 *    domain(x0) = [0, 4), domain(x1) = [0, 3), domain(x2) = [0, 2)
 *    fuse(x0, x1, x2): y = x2 * 12 + x1 * 4 + x0
 *    domain(y) = [0, 24)
 *
 *  - Split: split an iterator into multiple ones
 *
 *    domain(x) = [0, 24)
 *    split(x, 3, 12): [y0, y1, y2] = [x % 3, (x % 12) / 3, x / 12]
 *    domain(y0) = [0, 3), domain(y1) = [0, 4), domain(y2) = [0, 2)
 *
 *  We use the name "(quasi)affine" to be consistent with
 *  the terminology used in the polyhedral compilation.
 *  Notably, fuse is an affine transformation,
 *  while split corresponds to additional floordiv/mod operations
 *  that can appear in quasi-affine transformations.
 */
#ifndef TVM_ARITH_ITER_AFFINE_MAP_H_
#define TVM_ARITH_ITER_AFFINE_MAP_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ir/expr.h>
#include <tvm/tir/var.h>

namespace tvm {
namespace arith {

/*!
 * \brief Base class of all iter map expressions.
 *
 *  An IterMapExpr is a special expression to store
 *  the result of IterMapDetection.
 *  It should not appear in a legal TIR PrimFunc.
 */
class IterMapExprNode : public PrimExprNode {
 public:
  // overrides
  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "arith.IterMapExpr";
  static constexpr const uint32_t _type_child_slots = 3;
  TVM_DECLARE_BASE_OBJECT_INFO(IterMapExprNode, PrimExprNode);
};

/*!
 * \brief Managed reference to IterMapExprNode.
 * \sa IterMapExprNode
 */
class IterMapExpr : public PrimExpr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(IterMapExpr, PrimExpr, IterMapExprNode);
};

/*!
 * \brief Mark the source as an iterator in [0, extent).
 *
 *  IterMark is used to mark source expression as a valid
 *  iterator to make future analysis easy.
 */
class IterMarkNode : public Object {
 public:
  /*!
   * \brief The source expression, can either be
   *  a IterSumExpr or a Var.
   */
  PrimExpr source;
  /*!
   * \brief The extent of the iteration.
   */
  PrimExpr extent;

  // overrides
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("source", &source);
    v->Visit("extent", &extent);
  }

  bool SEqualReduce(const IterMarkNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(source, other->source) && equal(extent, other->extent);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(source);
    hash_reduce(extent);
  }

  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const char* _type_key = "arith.IterMark";
  TVM_DECLARE_FINAL_OBJECT_INFO(IterMarkNode, Object);
};

/*!
 * \brief Managed reference to IterMarkExprNode.
 * \sa IterMarkExprNode
 */
class IterMark : public ObjectRef {
 public:
  /*!
   * \brief constructor.
   * \param source The source expression.
   * \param extent The extent of the iterator.
   */
  TVM_DLL IterMark(PrimExpr source, PrimExpr extent);

  TVM_DEFINE_OBJECT_REF_METHODS(IterMark, ObjectRef, IterMarkNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IterMarkNode);
};

/*!
 * \brief Split of an iterator.
 *
 *  result = floormod(floordiv(source, lower_factor), extent) * scale
 */
class IterSplitExprNode : public IterMapExprNode {
 public:
  /*! \brief The source marked iterator. */
  IterMark source;
  /*! \brief The lower factor to split the source. */
  PrimExpr lower_factor;
  /*! \brief The extent of the split. */
  PrimExpr extent;
  /*! \brief Additional scale. */
  PrimExpr scale;

  // overrides
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("source", &source);
    v->Visit("lower_factor", &lower_factor);
    v->Visit("extent", &extent);
    v->Visit("scale", &scale);
  }

  bool SEqualReduce(const IterSplitExprNode* other, SEqualReducer equal) const {
    return equal(source, other->source) && equal(lower_factor, other->lower_factor) &&
           equal(extent, other->extent) && equal(scale, other->scale);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(source);
    hash_reduce(lower_factor);
    hash_reduce(extent);
    hash_reduce(scale);
  }

  static constexpr const char* _type_key = "arith.IterSplitExpr";
  TVM_DECLARE_FINAL_OBJECT_INFO(IterSplitExprNode, IterMapExprNode);
};

/*!
 * \brief Managed reference to IterSplitExprNode.
 * \sa IterSplitExprNode
 */
class IterSplitExpr : public IterMapExpr {
 public:
  /*!
   * \brief constructor from just source.
   * \param source The source expression.
   */
  TVM_DLL explicit IterSplitExpr(IterMark source);
  /*!
   * \brief constructor from just source.
   * \param source The source expression.
   */
  TVM_DLL explicit IterSplitExpr(IterMark source, PrimExpr scale);
  /*!
   * \brief constructor
   * \param source The source expression.
   * \param lower_factor The lower factor to split the source.
   * \param extent The extent of the split.
   * \param scale The additional scaling factor.
   */
  TVM_DLL explicit IterSplitExpr(IterMark source, PrimExpr lower_factor, PrimExpr extent,
                                 PrimExpr scale);

  TVM_DEFINE_OBJECT_REF_METHODS(IterSplitExpr, IterMapExpr, IterSplitExprNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IterSplitExprNode);
};

/*!
 * \brief Fuse multiple iterators by summing them with scaling.
 *
 *  result = sum(args) + base
 */
class IterSumExprNode : public IterMapExprNode {
 public:
  /*! \brief The args to the sum. */
  Array<IterSplitExpr> args;
  /*! \brief The base offset. */
  PrimExpr base;

  // overrides
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("args", &args);
    v->Visit("base", &base);
  }

  bool SEqualReduce(const IterSumExprNode* other, SEqualReducer equal) const {
    return equal(args, other->args) && equal(base, other->base);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(args);
    hash_reduce(base);
  }

  static constexpr const char* _type_key = "arith.IterSumExpr";
  TVM_DECLARE_FINAL_OBJECT_INFO(IterSumExprNode, IterMapExprNode);
};

/*!
 * \brief Managed reference to IterSumExprNode.
 * \sa IterSumExprNode
 */
class IterSumExpr : public IterMapExpr {
 public:
  /*!
   * \brief constructor.
   * \param args The args to the sum.
   * \param base The base offset.
   */
  TVM_DLL IterSumExpr(Array<IterSplitExpr> args, PrimExpr base);

  TVM_DEFINE_OBJECT_REF_METHODS(IterSumExpr, IterMapExpr, IterSumExprNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IterSumExprNode);
};

/*!
 * \brief Detect if indices can be written as
 *
 *  [y_0 + c_0, y_1 + c_1, ..., y_n + c_n]
 *
 *  Here y = some-quasi-affine-iter-map(input_iters)
 *  and c are symbolic constants.
 *
 *  We also requires that y_i and y_j to be independent for i != j.
 *
 *  For returned value rv, the following is always true:
 *  - rv[i]->args.size() <=1: only one iterator per element.
 *
 * \param indices The indices to detect pattern for.
 * \param input_iters Map from variable to iterator's range.
 * \param predicate The predicate input_iters should follow
 * \param analyzer Analyzer used to get context information.
 *
 * \return The detected pattern if a match exists,
 *         otherwise return an empty array.
 */
Array<IterSumExpr> DetectIterMap(const Array<PrimExpr>& indices, const Map<Var, Range>& input_iters,
                                 const PrimExpr& predicate, arith::Analyzer* analyzer);

/*!
 * \brief Use IterVarMap detector to rewrite and simplify the indices
 *
 * \param indices The indices to detect pattern for.
 * \param input_iters Map from variable to iterator's range
 * \param predicate The predicate input_iters should follow
 *
 * \return The indices after rewrite
 */
Array<PrimExpr> IterMapRewriteSimplify(const Array<PrimExpr>& indices,
                                       const Map<Var, Range>& input_iters,
                                       const PrimExpr& predicate);

Optional<IterSumExpr> DetectIter(const PrimExpr& index, const Map<Var, Range>& input_iters,
                                 arith::Analyzer* analyzer);

/*!
 * \brief Given an IterVar map, transform it to normal PrimExpr
 */
class IterVarMapConverter {
 public:
  explicit IterVarMapConverter(Analyzer* analyzer) : analyzer_(analyzer) {}

  PrimExpr Convert(const IterMapExpr& expr);
  PrimExpr ConvertIterSumExpr(const IterSumExpr& expr);
  PrimExpr ConvertIterSplitExpr(const IterSplitExpr& expr);

 private:
  Analyzer* analyzer_;
};

class DivisionForm;

/*!
 * \brief Denotes outer*inner_extent + inner, where outer and inner are IterVarMaps
 */
class DivisionFormNode : public Object {
 public:
  /*!
   * \brief IterMapExpr of outer iters
   */
  IterMapExpr outer;
  /*!
   * \brief IterMapExpr of inner iters
   */
  IterMapExpr inner;
  /*!
   * \brief extent of outer
   */
  PrimExpr outer_extent;
  /*!
   * \brief extent of inner
   */
  PrimExpr inner_extent;

  // overrides
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("outer", &outer);
    v->Visit("inner", &inner);
    v->Visit("outer_extent", &outer_extent);
    v->Visit("inner_extent", &inner_extent);
  }

  bool IsOuter() const;
  bool IsInner() const;
  bool OuterIsSplit() const;
  bool InnerIsSplit() const;

  static IterSplitExpr GetAsSplit(const IterMapExpr& expr, const PrimExpr& extent);
  IterSplitExpr GetOuterAsSplit() const;
  IterSplitExpr GetInnerAsSplit() const;
  static DivisionForm MakeInner(const IterMapExpr& iter, const PrimExpr& extent);
  static DivisionForm MakeOuter(const IterMapExpr& iter, const PrimExpr& extent);

  bool SEqualReduce(const DivisionFormNode* other, SEqualReducer equal) const {
    return equal(outer, other->outer) && equal(outer_extent, other->outer_extent) &&
           equal(inner, other->inner) && equal(inner_extent, other->inner_extent);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(outer);
    hash_reduce(outer_extent);
    hash_reduce(inner);
    hash_reduce(inner_extent);
  }

  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const char* _type_key = "arith.DivisionForm";
  TVM_DECLARE_FINAL_OBJECT_INFO(DivisionFormNode, Object);
};

/*!
 * \brief Managed reference to DivisionFormNode
 * \sa DivisionFormNode
 */
class DivisionForm : public ObjectRef {
 public:
  TVM_DLL DivisionForm(IterMapExpr outer, PrimExpr outer_extent, IterMapExpr inner,
                       PrimExpr inner_extent);

  TVM_DEFINE_OBJECT_REF_METHODS(DivisionForm, ObjectRef, DivisionFormNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DivisionFormNode);
};

/*!
 * \brief Detect if indices can be written as
 *
 * [a_0*e_0 + b_0 + c_0, a_1*e_1 + b_1, ..., a_n*e_n + b_n]
 *
 * where a = some-quasi-affine-iter-map(iter_range_map \set_minus sub_iters)
 *       b = some-quasi-affine-iter-map(sub_iters)
 *       c is constant symbols
 *       e is the extent of b
 *
 * \param indices The indices to detect pattern for.
 * \param iter_range_map Map from variable to iterator's range.
 * \param sub_iters Iterators of subspace
 * \param predicate The predicate for input_inters
 * \param analyzer Analyzer used to get context information.
 *
 * \return The detected a and b if a match exists,
 *         otherwise return an empty array.
 */
Array<DivisionForm> SubspaceDivision(const Array<PrimExpr>& indices,
                                     const Map<Var, Range>& iter_range_map,
                                     const Array<Var>& sub_iters, const PrimExpr& predicate,
                                     arith::Analyzer* analyzer);

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_ITER_AFFINE_MAP_H_
