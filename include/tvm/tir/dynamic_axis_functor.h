// <bojian/DietCode>
#pragma once

#include <tvm/tir/dynamic_axis.h>
#include <tvm/tir/expr_functor.h>


namespace tvm {
namespace tir {

/**
 * @brief Replace the dynamic axis node with its maximum value.
 */
class DynamicAxisMaxReplacer : public ExprMutator {
 protected:
  PrimExpr VisitExpr_(const DynamicAxisNode* op) override {
    return op->possible_values[op->possible_values.size() - 1];
  }
};

/**
 * @brief Replace the dynamic axis node with its minimum value.
 */
class DynamicAxisMinReplacer : public ExprMutator {
 protected:
  PrimExpr VisitExpr_(const DynamicAxisNode* op) override {
    return op->possible_values[0];
  }
};

/**
 * @brief Replace the dynamic axis node with certain value.
 */
class DynamicAxisSubstituter : public ExprMutator {
 private:
  const DynamicAxisNode* op;
  IntImm v;
 protected:
  PrimExpr VisitExpr_(const DynamicAxisNode* op) override {
    if (op == this->op) {
      return v;
    } else {
      return GetRef<DynamicAxis>(op);
    }
  }
 public:
  DynamicAxisSubstituter(const DynamicAxisNode* const op, IntImm v)
      : op(op), v(v) {}
};

/**
 * @brief Find all the dynamic axis nodes of an expression.
 */
class DynamicAxisFinder : public ExprVisitor {
 public:
  std::unordered_set<const DynamicAxisNode*> dy_axes;
 protected:
  void VisitExpr_(const DynamicAxisNode* op) override {
    dy_axes.insert(op);
  }
};


/**
 * @brief Verify that a predicate holds true for all its dynamic axis nodes.
 */
bool canProveForAllDynamicAxes(arith::Analyzer& analyzer, PrimExpr predicate);


}  // namespace tir
}  // namespace tvm
