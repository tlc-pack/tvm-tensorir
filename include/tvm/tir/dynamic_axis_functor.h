// <bojian/TVM-SymbolicTuning>
#pragma once

#include <tvm/tir/dynamic_axis.h>
#include <tvm/tir/expr_functor.h>


namespace tvm {
namespace tir {


class DyAxisMaxReplacer : public ExprMutator {
 protected:
  PrimExpr VisitExpr_(const DyAxisNode* op) override {
    return op->possible_values[op->possible_values.size() - 1];
  }
};


class DyAxisMinReplacer : public ExprMutator {
 protected:
  PrimExpr VisitExpr_(const DyAxisNode* op) override {
    return op->possible_values[0];
  }
};


class DyAxisSubstituter : public ExprMutator {
 public:
  const DyAxisNode* op;
  int v;
 protected:
  PrimExpr VisitExpr_(const DyAxisNode* op) override {
    if (op == this->op) {
      return v;
    } else {
      return GetRef<DyAxis>(op);
    }
  }
};


class DyAxisFinder : public ExprVisitor {
 public:
  std::unordered_set<const DyAxisNode*> dy_axes;
 protected:
  void VisitExpr_(const DyAxisNode* op) override {
    dy_axes.insert(op);
  }
};


}  // namespace tir
}  // namespace tvm
