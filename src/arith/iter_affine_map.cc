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
 * \file src/arith/iter_affine_map.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op.h>

#include "../support/utils.h"
#include "const_fold.h"
#include "pattern_match.h"

namespace tvm {
namespace arith {

using namespace tir;

IterMark::IterMark(PrimExpr source, PrimExpr extent) {
  auto n = make_object<IterMarkNode>();
  n->source = std::move(source);
  n->extent = std::move(extent);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("arith.IterMark").set_body_typed([](PrimExpr source, PrimExpr extent) {
  return IterMark(source, extent);
});

TVM_REGISTER_NODE_TYPE(IterMarkNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IterMarkNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IterMarkNode*>(node.get());
      p->stream << "IterMark(" << op->source << ", extent=" << op->extent << ")";
    });

IterSplitExpr::IterSplitExpr(IterMark source) {
  auto n = make_object<IterSplitExprNode>();
  auto one = make_const(source->source->dtype, 1);
  n->dtype = source->source->dtype;
  n->source = std::move(source);
  n->extent = n->source->extent;
  n->lower_factor = one;
  n->scale = one;
  data_ = std::move(n);
}

IterSplitExpr::IterSplitExpr(IterMark source, PrimExpr scale) {
  auto n = make_object<IterSplitExprNode>();
  auto one = make_const(source->source->dtype, 1);
  n->dtype = source->source->dtype;
  n->source = std::move(source);
  n->extent = n->source->extent;
  n->lower_factor = one;
  n->scale = std::move(scale);
  data_ = std::move(n);
}

IterSplitExpr::IterSplitExpr(IterMark source, PrimExpr lower_factor, PrimExpr extent,
                             PrimExpr scale) {
  auto n = make_object<IterSplitExprNode>();
  n->dtype = source->source->dtype;
  n->source = std::move(source);
  n->lower_factor = std::move(lower_factor);
  n->extent = std::move(extent);
  n->scale = std::move(scale);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("arith.IterSplitExpr")
    .set_body_typed([](IterMark source, PrimExpr lower_factor, PrimExpr extent, PrimExpr scale) {
      return IterSplitExpr(source, lower_factor, extent, scale);
    });

TVM_REGISTER_NODE_TYPE(IterSplitExprNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IterSplitExprNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IterSplitExprNode*>(node.get());
      p->stream << "IterSplit(" << op->source << ", lower_factor=" << op->lower_factor
                << ", extent=" << op->extent << ", scale=" << op->scale << ")";
    });

IterSumExpr::IterSumExpr(Array<IterSplitExpr> args, PrimExpr base) {
  auto n = make_object<IterSumExprNode>();
  n->dtype = base->dtype;
  n->args = std::move(args);
  n->base = std::move(base);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("arith.IterSumExpr")
    .set_body_typed([](Array<IterSplitExpr> args, PrimExpr base) {
      return IterSumExpr(args, base);
    });

TVM_REGISTER_NODE_TYPE(IterSumExprNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IterSumExprNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IterSumExprNode*>(node.get());
      p->stream << "IterSum(" << op->args << ", " << op->base << ")";
    });

/*!
 * \brief Given an IterVar map, transform it to normal PrimExpr
 */
class IterVarMapConverter {
 public:
  explicit IterVarMapConverter(Analyzer* analyzer) : analyzer_(analyzer) {}

  PrimExpr Convert(const IterMapExpr& expr) {
    if (const auto* op = expr.as<IterSplitExprNode>()) {
      return ConvertIterSplitExpr(GetRef<IterSplitExpr>(op));
    } else if (const auto* op = expr.as<IterSumExprNode>()) {
      return ConvertIterSumExpr(GetRef<IterSumExpr>(op));
    } else {
      LOG(FATAL);
      return 0;
    }
  }

  PrimExpr ConvertIterSumExpr(const IterSumExpr& expr) {
    PrimExpr res = 0;
    for (const auto& arg : expr->args) res += ConvertIterSplitExpr(arg);
    res += expr->base;
    return res;
  }

  PrimExpr ConvertIterSplitExpr(const IterSplitExpr& expr) {
    PrimExpr source;
    if (const auto* op = expr->source->source.as<VarNode>()) {
      source = GetRef<Var>(op);
    } else if (const auto& op = expr->source->source.as<IterSumExprNode>()) {
      source = ConvertIterSumExpr(GetRef<IterSumExpr>(op));
    }
    if (analyzer_->CanProve(expr->extent - expr->source->extent == 0) &&
        is_one(expr->lower_factor)) {
      return source * expr->scale;
    } else if (analyzer_->CanProve(div(expr->source->extent, expr->lower_factor) - expr->extent ==
                                   0)) {
      return floordiv(source, expr->lower_factor) * expr->scale;
    } else {
      return floormod(floordiv(source, expr->lower_factor), expr->extent) * expr->scale;
    }
  }

 private:
  Analyzer* analyzer_;
};

/*!
 * \brief Count the size of the PrimExpr
 */
class PrimExprSizeCounter : public ExprVisitor {
 public:
  explicit PrimExprSizeCounter() = default;

  void VisitExpr(const PrimExpr& expr) override {
    counter_++;
    ExprVisitor::VisitExpr(expr);
  }

  size_t Count(const PrimExpr& expr) {
    counter_ = 0;
    VisitExpr(expr);
    return counter_;
  }

 private:
  size_t counter_{0};
};

/*!
 * \brief Collector that collects
 *  the outgoing split reference of each IterMark.
 *
 *  These out-going splits can then be used to
 *  check if the iterators are independent.
 */
class IterMarkSplitCollector {
 public:
  // mark all IterMarks that are visited.
  std::unordered_set<IterMark, ObjectPtrHash, ObjectPtrEqual> visited_;
  // each iter mark to its outgoing splits that are referenced.
  std::unordered_map<IterMark, std::vector<IterSplitExpr>, ObjectPtrHash, ObjectPtrEqual>
      mark2splits_;
  /*!
   * \brief Collect all mark2splits recursively from indices.
   * \param indices The iterator of interest.
   */
  void Collect(const Array<IterSumExpr>& indices) {
    for (IterSumExpr sum_expr : indices) {
      for (IterSplitExpr split : sum_expr->args) {
        this->CollectInternal(split->source);

        mark2splits_[split->source].push_back(split);
      }
    }
  }

  void CollectInternal(const IterMark& mark) {
    if (visited_.count(mark)) return;
    visited_.insert(mark);
    if (auto* op = mark->source.as<IterSumExprNode>()) {
      for (IterSplitExpr split : op->args) {
        this->CollectInternal(split->source);

        mark2splits_[split->source].push_back(split);
      }
    }
  }
};

// Rewriter to rewrite PrimExpr to IterMapExpr
// when possible
class IterMapRewriter : public ExprMutator {
 public:
  using Parent = ExprMutator;

  explicit IterMapRewriter(Analyzer* analyzer, const Map<Var, Range>& input_iters)
      : analyzer_(analyzer) {
    for (auto kv : input_iters) {
      const auto& vrng = kv.second;
      if (is_one(vrng->extent)) {
        var_map_[kv.first] = IterSumExpr({}, vrng->min);
      } else {
        if (is_zero(vrng->min)) {
          IterMark mark(kv.first, vrng->extent);
          var_map_[kv.first] = IterSplitExpr(mark);
          input_marks_.push_back(mark);
        } else {
          IterMark mark(kv.first - vrng->min, vrng->extent);
          auto sum_expr = ToIterSumExpr(IterSplitExpr(mark));
          sum_expr.CopyOnWrite()->base = vrng->min;
          var_map_[kv.first] = sum_expr;
          input_marks_.push_back(mark);
        }
      }
    }
  }

  size_t unresolved_count() const { return unresolved_count_; }

  IterSumExpr Rewrite(const PrimExpr& expr, const Optional<PrimExpr>& extent = NullOpt) {
    return NormalizeToIterWithOffset(ToIterSumExpr(DirectMutate(expr)), extent);
  }

  bool CheckBijective(const Array<IterSumExpr>& indices) {
    // This function checks two conditions:
    // - C0: Each iter mark should be fully covered by non-overlapping splits.
    // - C1: All of the input iterators are used.
    //
    // Example: given x in [0, 8) y in [0, 6)
    // - indices = [x, x+1, y] won't pass because x and x+1 contribute
    //   two splits that overlaps with each other.
    // - indices = [x / 4, x % 4, y] will pass because x / 4 and x % 4
    //   contribute two non-overlapping splits that covers x.
    // - indices = [x / 4, x % 4] won't pass because y is not used.
    //
    IterMarkSplitCollector collector;
    // We can check that for each iter mark:
    // All the splits that refers to the iter_mark covers its extent.
    // The splits do not overlap with each other.

    collector.Collect(indices);
    for (const IterMark& mark : collector.visited_) {
      if (TryNormalizeSplits(mark, collector.mark2splits_[mark]).empty()) return false;
    }

    // all input marks must be visited
    for (const auto& mark : input_marks_) {
      if (collector.visited_.count(mark) == 0) return false;
    }
    return true;
  }

  // override the original mutate function.
  PrimExpr VisitExpr(const PrimExpr& input_expr) final {
    auto expr = ExprMutator::VisitExpr(input_expr);
    if (expr->IsInstance<IterMapExprNode>()) {
      ++unresolved_count_;
    }
    return expr;
  }

  // Normal mutation without normalization.
  PrimExpr DirectMutate(const PrimExpr& expr) { return ExprMutator::VisitExpr(expr); }

  PrimExpr VisitExpr_(const VarNode* op) final;
  PrimExpr VisitExpr_(const AddNode* op) final;
  PrimExpr VisitExpr_(const SubNode* op) final;
  PrimExpr VisitExpr_(const MulNode* op) final;
  PrimExpr VisitExpr_(const FloorDivNode* op) final;
  PrimExpr VisitExpr_(const FloorModNode* op) final;

  std::vector<IterSumExpr> pred_sum_exprs;

 private:
  // temp hash for de-duplication purposes.
  struct IterSumHash {
    size_t operator()(const IterSumExpr& value) const {
      // for now only hash on source index.
      size_t hash = value->args.size();
      for (const auto& arg : value->args) {
        hash = support::HashCombine(hash, std::hash<const Object*>()(arg->source.get()));
      }
      return hash;
    }
  };

  static bool IterSplitEqual(const IterSplitExpr& lhs, const IterSplitExpr& rhs,
                             bool check_scale = true) {
    tir::ExprDeepEqual equal;
    if (!lhs->source.same_as(rhs->source)) return false;
    if (!equal(lhs->lower_factor, rhs->lower_factor)) return false;
    if (check_scale && !equal(lhs->scale, rhs->scale)) return false;
    if (!equal(lhs->extent, rhs->extent)) return false;
    return true;
  };

  struct IterSumEqual {
    bool operator()(const IterSumExpr& lhs, const IterSumExpr& rhs) const {
      tir::ExprDeepEqual equal;
      if (lhs->args.size() != rhs->args.size()) return false;
      if (!equal(lhs->base, rhs->base)) return false;
      for (size_t i = 0; i < lhs->args.size(); ++i) {
        if (!IterSplitEqual(lhs->args[i], rhs->args[i])) return false;
      }
      return true;
    }
  };

  // Internal analyzer
  Analyzer* analyzer_;
  // Counter to keep track of unresolved cases.
  int unresolved_count_{0};
  // The var map
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> var_map_;
  // input iter marks
  std::vector<IterMark> input_marks_;
  // The canonical map for sum
  std::unordered_map<IterSumExpr, IterSplitExpr, IterSumHash, IterSumEqual> sum_fuse_map_;

  /*!
   * \brief Verify that splits fully covers mark in a non-overlapping fashion.
   *        If verification passes, return splits from outermost to inner most order.
   *        If not, return an empty array
   * \param mark The iterator of interest.
   * \param splits The splits to be verified.
   * \return The normalized splits.
   */
  Array<IterSplitExpr> TryNormalizeSplits(const IterMark& mark,
                                          const std::vector<IterSplitExpr>& splits) {
    std::vector<bool> used(splits.size(), false);
    std::vector<IterSplitExpr> iters;
    PrimExpr expected_lower_factor = make_const(mark->source->dtype, 1);

    for (size_t i = 0; i < splits.size(); ++i) {
      size_t j = 0;
      for (; j < splits.size(); ++j) {
        if (used[j]) continue;
        if (!used[j] && CanProveEqual(splits[j]->lower_factor, expected_lower_factor)) break;
      }
      if (j == splits.size()) {
        return Array<IterSplitExpr>();
      }
      used[j] = true;
      iters.push_back(splits[j]);
      expected_lower_factor *= splits[j]->extent;
    }
    if (!CanProveEqual(expected_lower_factor, mark->extent)) return Array<IterSplitExpr>();
    return Array<IterSplitExpr>(iters.rbegin(), iters.rend());
  }

  /*!
   * \brief Normalize expr to an iterator + offset.
   * \param expr The input expression.
   * \param extent Extent from predicate
   * \return The Normalized expression.
   */
  IterSumExpr NormalizeToIterWithOffset(IterSumExpr expr,
                                        const Optional<PrimExpr>& extent = NullOpt) {
    if (extent) {
      if (is_zero(expr->base)) {
        auto opt = TryFuseIters(expr, extent);
        if (opt && is_one(opt.value()->scale)) return expr;
      }
      ++unresolved_count_;
      return expr;
    } else {
      if (expr->args.size() <= 1) return expr;
      PrimExpr base = expr->base;
      expr.CopyOnWrite()->base = make_zero(expr->dtype);
      auto opt = TryFuseIters(expr);
      expr.CopyOnWrite()->base = base;
      if (opt) {
        expr.CopyOnWrite()->args = Array<IterSplitExpr>({opt.value()});
        return expr;
      } else {
        ++unresolved_count_;
        return expr;
      }
    }
  }

  bool CanProveEqual(const PrimExpr& lhs, const PrimExpr& rhs) {
    const auto* clhs = lhs.as<IntImmNode>();
    const auto* crhs = rhs.as<IntImmNode>();
    if (clhs && crhs) return clhs->value == crhs->value;
    return analyzer_->CanProve(lhs - rhs == 0);
  }

  bool CanProveLess(const PrimExpr& lhs, const PrimExpr& rhs) {
    const auto* clhs = lhs.as<IntImmNode>();
    const auto* crhs = rhs.as<IntImmNode>();
    if (clhs && crhs) return clhs->value < crhs->value;
    return analyzer_->CanProve(lhs - rhs < 0);
  }

  /*!
   * \brief Create a IterSumExpr from expr.
   * \param expr The input expr.
   * \return The transformed IterSumExpr.
   */
  static IterSumExpr ToIterSumExpr(const PrimExpr& expr) {
    if (const auto* op = expr.as<IterSumExprNode>()) {
      return GetRef<IterSumExpr>(op);
    } else if (const auto* op = expr.as<IterSplitExprNode>()) {
      return IterSumExpr({GetRef<IterSplitExpr>(op)}, make_zero(expr->dtype));
    } else {
      CHECK(!expr->IsInstance<IterMapExprNode>());
      return IterSumExpr({}, expr);
    }
  }

  // Try to normalize IterSum into a fused IterMark
  // IterSum = x1*c1 + x2*c2 + ... + xn*cn
  //         = (x1*s1 + x2*s2 + ... + xn)*cn
  //         = y*cn (IterMark y => x1*s1 + x2*s2 + ... + xn)
  //         = [IterSplit(IterMark(y), scale=cn)]
  // return a corresponding IterSplitExpr if needed.
  Optional<IterSplitExpr> TryFuseIters(IterSumExpr expr,
                                       const Optional<PrimExpr>& extent = NullOpt) {
    if (!is_zero(expr->base)) return NullOpt;
    if (expr->args.size() == 1) return expr->args[0];
    // select the iterators in order
    std::vector<bool> visited(expr->args.size(), false);
    std::vector<IterSplitExpr> iters, sub_iters;
    // canonicalize the expression
    // step0. check if find the base scale first
    IntImm base_scale(nullptr);
    size_t base_index = 0;
    for (size_t i = 0; i < expr->args.size(); ++i) {
      if (const auto* op = expr->args[i]->scale.as<IntImmNode>()) {
        if (!base_scale.defined() || op->value < base_scale->value) {
          base_scale = GetRef<IntImm>(op);
          base_index = i;
        }
      }
    }
    if (!base_scale.defined()) return NullOpt;
    // check if it can be remapped into a fused pattern.
    PrimExpr expected_scale = base_scale;
    for (size_t i = 0; i < expr->args.size();) {
      size_t j = i == 0 ? base_index : 0;
      for (; j < expr->args.size(); ++j) {
        if (!visited[j] && CanProveEqual(expr->args[j]->scale, expected_scale)) break;
      }
      if (j == expr->args.size()) return NullOpt;
      // look for the longest predicate started from expr->args[j]
      IterSumExpr sub_expr(nullptr);
      for (const auto& it : pred_sum_exprs) {
        if (IterSplitEqual(expr->args[j], it->args.back(), false)) {
          // find a predicate started from expr->args[j]
          if (!sub_expr.defined() || sub_expr->args.size() < it->args.size()) {
            sub_expr = it;
          }
        } else if (extent) {
          // The IterSum we are trying to fuse is also a predicate,
          // then we need to make sure the predicate doesn't intersect with this expr (independent)
          for (const auto& sub_arg : it->args) {
            if (IterSplitEqual(expr->args[j], sub_arg)) return NullOpt;
          }
        }
      }
      if (sub_expr.defined()) {
        // sub_expr found
        // mark the iterators in the sub_expr as visited
        for (auto it = sub_expr->args.rbegin(); it != sub_expr->args.rend(); ++it) {
          size_t j = 0;
          for (; j < expr->args.size(); ++j) {
            if (!visited[j] && IterSplitEqual(expr->args[j], *it, false)) {
              if (CanProveEqual((*it)->scale * expected_scale, expr->args[j]->scale)) break;
            }
          }
          if (j == expr->args.size()) return NullOpt;
          visited[j] = true;
          iters.push_back(expr->args[j]);
        }
        IterSplitExpr sub_iter = sum_fuse_map_[sub_expr];
        sub_iter.CopyOnWrite()->scale = expected_scale;
        expected_scale *= sum_fuse_map_[sub_expr]->source->extent;
        i += sub_expr->args.size();
        sub_iters.push_back(sub_iter);
      } else {
        // sub_expr not found, skip this iterator
        visited[j] = true;
        iters.push_back(expr->args[j]);
        sub_iters.push_back(expr->args[j]);
        expected_scale *= expr->args[j]->extent;
        ++i;
      }
    }
    // update the iterator to use the canonicalized form
    IterSumExpr full_expr = expr, canonical_expr = expr;
    canonical_expr.CopyOnWrite()->args = Array<IterSplitExpr>(iters.rbegin(), iters.rend());
    full_expr.CopyOnWrite()->args = Array<IterSplitExpr>(sub_iters.rbegin(), sub_iters.rend());
    auto it = sum_fuse_map_.find(canonical_expr);
    if (it != sum_fuse_map_.end()) {
      if (extent) {
        // another predicate on this iter
        IterMark updated_mark = it->second->source;
        updated_mark.CopyOnWrite()->extent = min(updated_mark->extent, extent.value());
        IterSplitExpr split = IterSplitExpr(updated_mark, base_scale);
        sum_fuse_map_[canonical_expr] = split;
        return split;
      } else {
        return it->second;
      }
    }
    expected_scale = div(expected_scale, base_scale);
    auto mark = IterMark(full_expr, extent ? min(extent.value(), expected_scale) : expected_scale);
    IterSplitExpr split(mark, base_scale);
    sum_fuse_map_[canonical_expr] = split;
    pred_sum_exprs.push_back(canonical_expr);
    return split;
  }

  bool CanProveDivisible(const PrimExpr& lhs, const PrimExpr& rhs) {
    const auto* clhs = lhs.as<IntImmNode>();
    const auto* crhs = rhs.as<IntImmNode>();
    if (clhs && crhs) return clhs->value % crhs->value == 0;
    return analyzer_->CanProve(floormod(lhs, rhs) == 0);
  }

  PrimExpr SplitFloorDivConst(IterSplitExpr lhs, PrimExpr rhs);
  PrimExpr SplitFloorModConst(IterSplitExpr lhs, PrimExpr rhs);

  static void AddToLhs(IterSumExprNode* lhs, IterSplitExpr rhs, int sign) {
    tir::ExprDeepEqual equal;
    for (size_t i = 0; i < lhs->args.size(); ++i) {
      IterSplitExpr lvalue = lhs->args[i];
      if (lvalue->source.same_as(rhs->source) && equal(lvalue->lower_factor, rhs->lower_factor) &&
          equal(lvalue->extent, rhs->extent)) {
        if (sign > 0) {
          rhs.CopyOnWrite()->scale = lvalue->scale + rhs->scale;
        } else {
          rhs.CopyOnWrite()->scale = lvalue->scale - rhs->scale;
        }
        lhs->args.Set(i, rhs);
        return;
      }
    }
    if (sign > 0) {
      lhs->args.push_back(rhs);
    } else {
      rhs.CopyOnWrite()->scale = make_zero(rhs->scale.dtype()) - rhs->scale;
      lhs->args.push_back(rhs);
    }
  }

  static void AddToLhs(IterSumExprNode* lhs, const IterSumExpr& rhs, int sign) {
    for (size_t i = 0; i < rhs->args.size(); ++i) {
      AddToLhs(lhs, rhs->args[i], sign);
    }
    if (sign > 0) {
      lhs->base += rhs->base;
    } else {
      lhs->base -= rhs->base;
    }
  }

  static void MulToLhs(IterSumExprNode* lhs, const PrimExpr& rhs) {
    for (size_t i = 0; i < lhs->args.size(); ++i) {
      IterSplitExpr lvalue = lhs->args[i];
      lvalue.CopyOnWrite()->scale *= rhs;
      lhs->args.Set(i, lvalue);
    }
    lhs->base *= rhs;
  }
};

/*!
 * \brief Split the predicate into `(a < b) && (c < d) && ...`
 * \param pred The predicate to be split
 * \return A list of pairs, each element of which are lhs and rhs of the '<' sign
 */
std::vector<std::tuple<size_t, PrimExpr, PrimExpr>> SplitPredicate(PrimExpr pred) {
  std::vector<std::tuple<size_t, PrimExpr, PrimExpr>> result;
  arith::PVar<PrimExpr> lhs, rhs, rest;
  for (;;) {
    if ((rest && (lhs < rhs)).Match(pred)) {
      result.emplace_back(0, lhs.Eval(), rhs.Eval());
      pred = rest.Eval();
    } else if ((lhs < rhs).Match(pred)) {
      result.emplace_back(0, lhs.Eval(), rhs.Eval());
      break;
    } else {
      return std::vector<std::tuple<size_t, PrimExpr, PrimExpr>>({});
    }
  }
  return result;
}

Array<IterSumExpr> DetectIterMap(const Array<PrimExpr>& indices, const Map<Var, Range>& input_iters,
                                 const PrimExpr& input_pred, arith::Analyzer* analyzer) {
  // Overall detection algorithm is divided into two steps:
  // - Step0: IterMapRewriter rewrites the expression to use IterMapExpr patterns.
  // - Step1: IterIndependenceChecker checks if the iterator are independent.

  using Predicate = std::tuple<size_t, PrimExpr, PrimExpr>;
  std::vector<Predicate> predicates = SplitPredicate(input_pred);
  if (!is_one(input_pred) && predicates.empty()) return Array<IterSumExpr>();

  PrimExprSizeCounter prim_expr_size_counter;
  for (auto& predicate : predicates) {
    std::get<0>(predicate) = prim_expr_size_counter.Count(std::get<1>(predicate));
  }

  std::sort(predicates.begin(), predicates.end(),
            [](const Predicate& a, const Predicate& b) { return std::get<0>(a) < std::get<0>(b); });

  IterMapRewriter rewriter(analyzer, input_iters);

  // Step0.0: rewrite predicates in the order from size-small ones to size-big ones
  for (Predicate predicate : predicates) {
    PrimExpr res = rewriter.Rewrite(std::get<1>(predicate), std::get<2>(predicate));
    if (rewriter.unresolved_count() != 0) return Array<IterSumExpr>();
  }
  // Step0.1: rewrite indices
  Array<IterSumExpr> results;
  for (PrimExpr value : indices) {
    results.push_back(rewriter.Rewrite(value));
    if (rewriter.unresolved_count() != 0) return Array<IterSumExpr>();
  }
  // Step1: IterIndependenceChecker checks if the iterator are independent.
  if (!rewriter.CheckBijective(results)) return Array<IterSumExpr>();

  return results;
}

TVM_REGISTER_GLOBAL("arith.DetectIterMap")
    .set_body_typed([](const Array<PrimExpr>& indices, const Map<Var, Range>& input_iters,
                       const PrimExpr& predicate) {
      arith::Analyzer ana;
      return DetectIterMap(indices, input_iters, predicate, &ana);
    });

Array<PrimExpr> IterMapRewriteSimplify(const Array<PrimExpr>& indices,
                                       const Map<Var, Range>& input_iters,
                                       const PrimExpr& predicate) {
  Analyzer analyzer;
  auto rewrite = DetectIterMap(indices, input_iters, predicate, &analyzer);
  if (rewrite.empty())
    return indices;
  else {
    std::vector<PrimExpr> res;
    IterVarMapConverter converter(&analyzer);
    for (const auto& expr : rewrite) res.push_back(converter.Convert(expr));
    return res;
  }
}

PrimExpr IterMapRewriter::VisitExpr_(const VarNode* op) {
  auto var = GetRef<Var>(op);
  auto it = var_map_.find(var);
  if (it != var_map_.end()) return it->second;
  return std::move(var);
}

PrimExpr IterMapRewriter::VisitExpr_(const AddNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Parent::VisitExpr_(op);
  }

  PrimExpr a = this->DirectMutate(op->a);
  PrimExpr b = this->DirectMutate(op->b);

  // const folding
  PrimExpr const_res = TryConstFold<Add>(a, b);
  if (const_res.defined()) return const_res;
  // does not contain iter map.
  if (!a->IsInstance<IterMapExprNode>() && !b->IsInstance<IterMapExprNode>()) {
    if (op->a.same_as(a) && op->b.same_as(b)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Add(a, b);
    }
  }

  // canonical form simplification.
  IterSumExpr ret = ToIterSumExpr(a);

  if (!b->IsInstance<IterMapExprNode>()) {
    ret.CopyOnWrite()->base += b;
  } else if (const auto* op = b.as<IterSumExprNode>()) {
    AddToLhs(ret.CopyOnWrite(), GetRef<IterSumExpr>(op), 1);
  } else if (const auto* op = b.as<IterSplitExprNode>()) {
    AddToLhs(ret.CopyOnWrite(), GetRef<IterSplitExpr>(op), 1);
  } else {
    AddToLhs(ret.CopyOnWrite(), ToIterSumExpr(b), 1);
  }
  return std::move(ret);
}

PrimExpr IterMapRewriter::VisitExpr_(const SubNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Parent::VisitExpr_(op);
  }

  PrimExpr a = this->DirectMutate(op->a);
  PrimExpr b = this->DirectMutate(op->b);

  // const folding
  PrimExpr const_res = TryConstFold<Sub>(a, b);
  if (const_res.defined()) return const_res;

  // does not contain iter map.
  if (!a->IsInstance<IterMapExprNode>() && !b->IsInstance<IterMapExprNode>()) {
    if (op->a.same_as(a) && op->b.same_as(b)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Sub(a, b);
    }
  }

  // canonical form simplification.
  IterSumExpr ret = ToIterSumExpr(a);

  if (!b->IsInstance<IterMapExprNode>()) {
    ret.CopyOnWrite()->base -= b;
  } else if (const auto* op = b.as<IterSumExprNode>()) {
    AddToLhs(ret.CopyOnWrite(), GetRef<IterSumExpr>(op), -1);
  } else if (const auto* op = b.as<IterSplitExprNode>()) {
    AddToLhs(ret.CopyOnWrite(), GetRef<IterSplitExpr>(op), -1);
  } else {
    AddToLhs(ret.CopyOnWrite(), ToIterSumExpr(b), -1);
  }
  return std::move(ret);
}

PrimExpr IterMapRewriter::VisitExpr_(const MulNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Parent::VisitExpr_(op);
  }
  // normalize
  PrimExpr a = this->DirectMutate(op->a);
  PrimExpr b = this->DirectMutate(op->b);

  // const folding
  PrimExpr const_res = TryConstFold<Mul>(a, b);
  if (const_res.defined()) return const_res;

  // does not contain iter map.
  if (!a->IsInstance<IterMapExprNode>() && !b->IsInstance<IterMapExprNode>()) {
    if (op->a.same_as(a) && op->b.same_as(b)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Mul(a, b);
    }
  }

  if (a->IsInstance<IterMapExprNode>() && b->IsInstance<IterMapExprNode>()) {
    // cannot multiply two iterators, mark as unresolved.
    ++unresolved_count_;
    return Mul(a, b);
  }

  if (!a->IsInstance<IterMapExprNode>()) {
    std::swap(a, b);
  }

  if (a->IsInstance<IterSumExprNode>()) {
    IterSumExpr ret = Downcast<IterSumExpr>(std::move(a));
    MulToLhs(ret.CopyOnWrite(), b);
    return std::move(ret);
  } else {
    CHECK(a->IsInstance<IterSplitExprNode>());
    IterSplitExpr ret = Downcast<IterSplitExpr>(std::move(a));
    ret.CopyOnWrite()->scale *= b;
    return std::move(ret);
  }
}

PrimExpr IterMapRewriter::SplitFloorDivConst(IterSplitExpr lhs, PrimExpr rhs) {
  // floordiv(x*scale, rhs)
  if (is_one(rhs)) return std::move(lhs);
  if (!is_one(lhs->scale)) {
    if (CanProveDivisible(lhs->scale, rhs)) {
      // floordiv(x*c1*c2, c2) = x*c1, c1=scale/rhs
      lhs.CopyOnWrite()->scale = floordiv(lhs->scale, rhs);
      return std::move(lhs);
    } else {
      if (CanProveDivisible(rhs, lhs->scale)) {
        // floordiv(x*c1, c1*c2) = floordiv(x, c2), c2=rhs/scale
        rhs = floordiv(rhs, lhs->scale);
        lhs.CopyOnWrite()->scale = make_const(rhs->dtype, 1);
      } else {
        // mark as unresolved.
        ++unresolved_count_;
        return floordiv(lhs, rhs);
      }
    }
  }

  // We handle scale!=1 in above code, hence we only consider floordiv(x, rhs) below
  // where x=floormod(floordiv(iter, lower_factor), extent)
  if (CanProveDivisible(lhs->extent, rhs)) {
    // floordiv(floormod(floordiv(iter, lower_factor), c1c2), c1)
    // = floordiv(floormod(y, c1c2), c1), where y=floordiv(iter, lower_factor)
    // = floordiv(floormod(sc1c2+tc1+u, c1c2), c1), where y=sc1c2+tc1+u, t<c2, u<c1
    // = t
    // = floormod(sc2+t, c2)
    // = floormod(floordiv(y, c1), c2)
    // = floormod(floordiv(iter, lower_factor*c1), c2), where c1=rhs, c2=extent/rhs
    auto* ptr_lhs = lhs.CopyOnWrite();
    ptr_lhs->lower_factor *= rhs;
    ptr_lhs->extent = analyzer_->Simplify(floordiv(ptr_lhs->extent, rhs));
    return std::move(lhs);
  } else {
    // mark as unresolved.
    ++unresolved_count_;
    return floordiv(lhs, rhs);
  }
}

PrimExpr IterMapRewriter::VisitExpr_(const FloorDivNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Parent::VisitExpr_(op);
  }

  PrimExpr a = this->DirectMutate(op->a);
  PrimExpr b = this->DirectMutate(op->b);

  // const folding
  PrimExpr const_res = TryConstFold<FloorDiv>(a, b);
  if (const_res.defined()) return const_res;

  // does not contain iter map.
  if (!a->IsInstance<IterMapExprNode>() && !b->IsInstance<IterMapExprNode>()) {
    if (op->a.same_as(a) && op->b.same_as(b)) {
      return GetRef<PrimExpr>(op);
    } else {
      return FloorDiv(a, b);
    }
  }

  if (b->IsInstance<IterMapExprNode>()) {
    // cannot divide an iterator, mark as unresolved.
    ++unresolved_count_;
    return FloorDiv(a, b);
  }

  if (a->IsInstance<IterSumExprNode>()) {
    IterSumExpr ret = Downcast<IterSumExpr>(a);
    if (auto opt = TryFuseIters(ret)) {
      return SplitFloorDivConst(opt.value(), b);
    } else {
      ++unresolved_count_;
      return FloorDiv(a, b);
    }
  } else {
    CHECK(a->IsInstance<IterSplitExprNode>());
    IterSplitExpr ret = Downcast<IterSplitExpr>(std::move(a));
    return SplitFloorDivConst(ret, b);
  }
}

PrimExpr IterMapRewriter::SplitFloorModConst(IterSplitExpr lhs, PrimExpr rhs) {
  // floormod(x*scale, rhs)
  if (is_one(rhs)) return make_zero(lhs->dtype);
  if (!is_one(lhs->scale)) {
    // floormod(x*c1*c2, c1) = 0
    if (CanProveDivisible(lhs->scale, rhs)) {
      return make_zero(lhs->dtype);
    } else {
      if (CanProveDivisible(rhs, lhs->scale)) {
        // floormod(x*c1, c1*c2) = (floormod(x, c2)) * c1, where c2 = rhs/scale
        rhs = floordiv(rhs, lhs->scale);
      } else {
        // mark as unresolved.
        ++unresolved_count_;
        return floormod(lhs, rhs);
      }
    }
  }

  // floormod(x, rhs) where x=floormod(floordiv(iter, lower_factor), extent)
  if (CanProveDivisible(lhs->extent, rhs)) {
    // floormod(floormod(floordiv(iter, lower_factor), c1c2), c1)
    // = floormod(floordiv(iter, lower_factor), c1), where c1=rhs
    lhs.CopyOnWrite()->extent = rhs;
    return std::move(lhs);
  } else {
    // mark as unresolved.
    ++unresolved_count_;
    return floormod(lhs, rhs);
  }
}

PrimExpr IterMapRewriter::VisitExpr_(const FloorModNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Parent::VisitExpr_(op);
  }

  PrimExpr a = this->DirectMutate(op->a);
  PrimExpr b = this->DirectMutate(op->b);

  // const folding
  PrimExpr const_res = TryConstFold<FloorMod>(a, b);
  if (const_res.defined()) return const_res;

  // does not contain iter map.
  if (!a->IsInstance<IterMapExprNode>() && !b->IsInstance<IterMapExprNode>()) {
    if (op->a.same_as(a) && op->b.same_as(b)) {
      return GetRef<PrimExpr>(op);
    } else {
      return FloorMod(a, b);
    }
  }

  if (b->IsInstance<IterMapExprNode>()) {
    // cannot mod an iterator, mark as unresolved.
    ++unresolved_count_;
    return FloorMod(a, b);
  }

  if (a->IsInstance<IterSumExprNode>()) {
    IterSumExpr ret = Downcast<IterSumExpr>(a);
    if (auto opt = TryFuseIters(ret)) {
      return SplitFloorModConst(opt.value(), b);
    } else {
      ++unresolved_count_;
      return FloorMod(a, b);
    }
  } else {
    CHECK(a->IsInstance<IterSplitExprNode>());
    IterSplitExpr ret = Downcast<IterSplitExpr>(std::move(a));
    return SplitFloorModConst(ret, b);
  }
}

/*!
 * \brief Inner class to wrap the result of subspace division
 *        Denotes outer*inner_extent + inner, where outer and inner are IterVarMaps
 */
class DivisionForm {
 public:
  explicit DivisionForm(const IterMapExpr& outer, const PrimExpr& outer_extent,
                        const IterMapExpr& inner, const PrimExpr& inner_extent)
      : outer(outer), inner(inner), outer_extent(outer_extent), inner_extent(inner_extent) {}

  bool IsOuter() const { return is_one(inner_extent); }

  bool IsInner() const { return is_one(outer_extent); }

  bool OuterIsSplit() const { return outer->IsInstance<IterSplitExprNode>(); }

  bool InnerIsSplit() const { return inner->IsInstance<IterSplitExprNode>(); }

  static IterSplitExpr GetAsSplit(const IterMapExpr& expr, const PrimExpr& extent) {
    if (const auto* op = expr.as<IterSplitExprNode>()) {
      return GetRef<IterSplitExpr>(op);
    } else if (const auto* op = expr.as<IterSumExprNode>()) {
      return IterSplitExpr(IterMark(GetRef<IterSumExpr>(op), extent));
    } else {
      LOG(FATAL);
      return NullValue<IterSplitExpr>();
    }
  }

  IterSplitExpr GetOuterAsSplit() const { return GetAsSplit(outer, outer_extent); }

  IterSplitExpr GetInnerAsSplit() const { return GetAsSplit(inner, inner_extent); }

  static DivisionForm MakeInner(const IterMapExpr& iter, const PrimExpr extent) {
    return DivisionForm(IterSumExpr({}, 0), 1, iter, extent);
  }

  static DivisionForm MakeOuter(const IterMapExpr& iter, const PrimExpr extent) {
    return DivisionForm(iter, extent, IterSumExpr({}, 0), 1);
  }

  void AddBase(const PrimExpr& base) {
    if (const auto* op = inner.as<IterSplitExprNode>()) {
      inner = IterSumExpr({GetRef<IterSplitExpr>(op)}, base);
    } else if (const auto* op = inner.as<IterSumExprNode>()) {
      const auto& expr = GetRef<IterSumExpr>(op);
      inner = IterSumExpr(expr->args, expr->base + base);
    }
  }

  IterMapExpr outer, inner;
  PrimExpr outer_extent, inner_extent;
};

std::ostream& operator<<(std::ostream& os, const DivisionForm& n) {
  os << n.outer << " " << n.outer_extent << " " << n.inner << " " << n.inner_extent;
  return os;
}

class SubspaceDivider {
 public:
  explicit SubspaceDivider(Analyzer* analyzer, const IterMarkSplitCollector& collector,
                           const std::unordered_set<const VarNode*>& sub_iters)
      : analyzer_(analyzer), collector_(collector), sub_iters_(sub_iters) {}

  size_t unresolved_count() const { return unresolved_count_; }

  DivisionForm Fail() {
    unresolved_count_++;
    return DivisionForm(IterSumExpr({}, 0), 0, IterSumExpr({}, 0), 0);
  }

  DivisionForm DivideIterSumExpr(const IterSumExpr& expr, const PrimExpr& mark_extent) {
    if (expr->args.size() == 1) {
      // arg + base, if arg=Y*E(X)+X, then arg+base = Y*E(X)+(X+base)
      if (!is_one(expr->args[0]->scale)) return Fail();
      auto res = DivideIterSplitExpr(expr->args[0]);
      if (!is_zero(expr->base)) res.AddBase(expr->base);
      return res;
    }
    // arg1 + arg2 + ... + argn + base
    // then we can write it as Y*E(X)+X if it starts with Y*1+0, followed by 0*E(X)+X
    PrimExpr extent = 1;
    std::vector<IterSplitExpr> outer_args, inner_args;
    bool inner = true, scale_is_one = false;
    for (auto it = expr->args.rbegin(); it != expr->args.rend(); ++it) {
      const IterSplitExpr& arg = *it;
      if (is_one(arg->scale)) scale_is_one = true;
      DivisionForm arg_division = DivideIterSplitExpr(arg);
      IterSplitExpr new_arg;
      if (arg_division.IsInner()) {
        // 0*E(Xi)+Xi
        if (!inner) return Fail();
        inner_args.push_back(new_arg = arg_division.GetInnerAsSplit());
        inner = true;
        // Y(i)*1+0
      } else if (arg_division.IsOuter()) {
        outer_args.push_back(new_arg = arg_division.GetOuterAsSplit());
        inner = false;
      } else {
        return Fail();
      }
      extent *= new_arg->extent;
    }
    if (!scale_is_one) return Fail();
    bool need_predicate = !CanProveEqual(extent, mark_extent);
    const auto& outer_mark = MarkFromArgsAndBase(outer_args, 0);
    const auto& inner_mark = MarkFromArgsAndBase(inner_args, expr->base);
    if (need_predicate) {
      // if we have a predicate on this sum expr, then we cannot divide it into Y*E+X
      // it should either be Y*1+0 or 0*E(X)+X
      IterVarMapConverter converter(analyzer_);
      if (inner_args.empty()) {
        // Y*1+0
        outer_preds = outer_preds && (converter.Convert(outer_mark.first) < mark_extent);
        return DivisionForm::MakeOuter(outer_mark.first, mark_extent);
      } else if (outer_args.empty()) {
        // 0*E(X)+X
        inner_preds = inner_preds && (converter.Convert(inner_mark.first) < mark_extent);
        return DivisionForm::MakeInner(inner_mark.first, mark_extent);
      } else {
        return Fail();
      }
    }
    return DivisionForm(outer_mark.first, outer_mark.second, inner_mark.first, inner_mark.second);
  }

  PrimExpr outer_preds{Bool(true)}, inner_preds{Bool(true)};

 private:
  static std::pair<IterSumExpr, PrimExpr> MarkFromArgsAndBase(
      const std::vector<IterSplitExpr>& args, PrimExpr base) {
    std::vector<IterSplitExpr> res;
    PrimExpr extent = 1;
    for (const auto& it : args) {
      auto arg = it;
      arg.CopyOnWrite()->scale = extent;
      extent *= arg->extent;
      res.push_back(arg);
    }
    return std::make_pair(IterSumExpr(Array<IterSplitExpr>(res.rbegin(), res.rend()), base),
                          extent);
  }

  DivisionForm DivideIterSplitExpr(const IterSplitExpr& expr) {
    auto it = split_map_.find(expr);
    if (it != split_map_.end()) {
      // We will calculate all the splits of an IterMark's division form when we first
      // encounter one of them. If we encounter another later, we directly return the record.
      return it->second;
    } else {
      const Array<IterSplitExpr>& splits = collector_.mark2splits_.at(expr->source);
      if (const auto* iter_ptr = expr->source->source.as<VarNode>()) {
        // source is input_iter,
        bool inner = sub_iters_.count(iter_ptr);
        for (const auto& split : splits) {
          if (inner) {
            // 0*E(split)+split
            split_map_.emplace(split, DivisionForm::MakeInner(split, split->extent));
          } else {
            // split*1 + 0
            split_map_.emplace(split, DivisionForm::MakeOuter(split, split->extent));
          }
        }
      } else if (const auto* iter_ptr = expr->source->source.as<IterSumExprNode>()) {
        // source = Y*E+X
        // splits = [s1, s2, ..., sn]
        // we can divide if there exists i, such that extent(s1)extent(s2)...extent(si)=extent(Y)
        //                                            extent(si+1)...extent(sn)=extent(X)
        auto mark_division = DivideIterSumExpr(GetRef<IterSumExpr>(iter_ptr), expr->source->extent);
        if (splits.size() == 1) {
          return mark_division;
        }
        IterMark outer_mark(Downcast<IterSumExpr>(mark_division.outer), mark_division.outer_extent);
        IterMark inner_mark(Downcast<IterSumExpr>(mark_division.inner), mark_division.inner_extent);
        bool encountered = mark_division.IsOuter();
        std::vector<bool> used(splits.size(), false);
        std::vector<IterSplitExpr> inner_iters, outer_iters;
        PrimExpr expected_lower_factor = make_const(expr->source->source->dtype, 1);
        for (size_t i = 0; i < splits.size(); ++i) {
          size_t j = 0;
          for (; j < splits.size(); ++j) {
            if (used[j]) continue;
            if (!used[j] && CanProveEqual(splits[j]->lower_factor, expected_lower_factor)) break;
          }
          if (j == splits.size()) return Fail();
          used[j] = true;
          if (!encountered) {
            inner_iters.push_back(splits[j]);
          } else {
            outer_iters.push_back(splits[j]);
          }
          expected_lower_factor *= splits[j]->extent;
          if (CanProveEqual(expected_lower_factor, mark_division.inner_extent)) encountered = true;
        }
        if (!encountered) return Fail();
        for (const auto& inner_iter : inner_iters) {
          IterSplitExpr new_iter = inner_iter;
          new_iter.CopyOnWrite()->source = inner_mark;
          split_map_.emplace(inner_iter, DivisionForm::MakeInner(new_iter, inner_iter->extent));
        }
        for (const auto& outer_iter : outer_iters) {
          IterSplitExpr new_iter = outer_iter;
          new_iter.CopyOnWrite()->source = outer_mark;
          new_iter.CopyOnWrite()->lower_factor =
              div(outer_iter->lower_factor, outer_iters[0]->lower_factor);
          split_map_.emplace(outer_iter, DivisionForm::MakeOuter(new_iter, outer_iter->extent));
        }
      } else {
        return Fail();
      }
      return split_map_.at(expr);
    }
  }

  bool CanProveEqual(const PrimExpr& lhs, const PrimExpr& rhs) {
    const auto* clhs = lhs.as<IntImmNode>();
    const auto* crhs = rhs.as<IntImmNode>();
    if (clhs && crhs) return clhs->value == crhs->value;
    return analyzer_->CanProve(lhs - rhs == 0);
  }

  size_t unresolved_count_{0};
  Analyzer* analyzer_;
  const IterMarkSplitCollector collector_;
  const std::unordered_set<const VarNode*>& sub_iters_;
  // map from SplitExpr to its corresponding DivisionForm(Y*E(X)+X)
  std::unordered_map<IterSplitExpr, DivisionForm, ObjectPtrHash, ObjectPtrEqual> split_map_;
};

Array<Array<PrimExpr>> SubspaceDivision(const Array<PrimExpr>& indices,
                                        const Map<Var, Range>& input_iters,
                                        const Array<Var>& sub_iters, const PrimExpr& predicate,
                                        arith::Analyzer* analyzer) {
  const auto& maps = DetectIterMap(indices, input_iters, predicate, analyzer);
  if (indices.empty()) return {};

  std::unordered_set<const VarNode*> sub_iters_set;
  for (const auto& sub_iter : sub_iters) sub_iters_set.insert(sub_iter.get());

  IterMarkSplitCollector collector;
  collector.Collect(maps);
  SubspaceDivider subspace_divider(analyzer, collector, sub_iters_set);
  IterVarMapConverter converter(analyzer);

  std::vector<Array<PrimExpr>> results;
  for (const auto& expr : maps) {
    auto res = subspace_divider.DivideIterSumExpr(expr, 0);
    results.push_back({converter.Convert(res.outer), converter.Convert(res.inner)});
    if (subspace_divider.unresolved_count()) return {};
  }
  results.push_back({subspace_divider.outer_preds, subspace_divider.inner_preds});

  return results;
}

TVM_REGISTER_GLOBAL("arith.SubspaceDivision")
    .set_body_typed([](const Array<PrimExpr>& indices, const Map<Var, Range>& input_iters,
                       const Array<Var>& sub_iters, const PrimExpr& predicate) {
      arith::Analyzer ana;
      return SubspaceDivision(indices, input_iters, sub_iters, predicate, &ana);
    });

}  // namespace arith
}  // namespace tvm
