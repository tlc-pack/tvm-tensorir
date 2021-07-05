// <bojian/DietCode>
#pragma once

#include <tvm/tir/dynamic_axis.h>
#include <tvm/tir/expr_functor.h>


namespace tvm {
namespace tir {

/**
 * @brief Replace the dynamic axis node with certain value.
 */
class DynamicAxisReplacer : public ExprMutator {
 private:
  std::function<PrimExpr(const DynamicAxisNode*)> freplace_expr_;
 protected:
  PrimExpr VisitExpr_(const DynamicAxisNode* op) override {
    if (freplace_expr_ == nullptr) {
      return GetRef<DynamicAxis>(op);
    } else {
      return freplace_expr_(op);
    }
  }
 public:
  explicit DynamicAxisReplacer(
      std::function<PrimExpr(const DynamicAxisNode*)> freplace_expr)
      : freplace_expr_(freplace_expr) {}
};

/**
 * @brief Find all the dynamic axis nodes of an expression.
 */
class DynamicAxisFinder : public ExprVisitor {
 public:
  std::unordered_set<const DynamicAxisNode*> dyn_axes;
 protected:
  void VisitExpr_(const DynamicAxisNode* op) override {
    dyn_axes.insert(op);
  }
};


/**
 * @brief Verify that a predicate holds true for all its dynamic axis nodes.
 */
bool canProveForAllDynamicAxes(arith::Analyzer& analyzer, PrimExpr predicate);


}  // namespace tir
}  // namespace tvm
