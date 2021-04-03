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
 private:
  const DyAxisNode* op;
  IntImm v;
 protected:
  PrimExpr VisitExpr_(const DyAxisNode* op) override {
    if (op == this->op) {
      return v;
    } else {
      return GetRef<DyAxis>(op);
    }
  }
 public:
  DyAxisSubstituter(const DyAxisNode* const op, IntImm v) : op(op), v(v) {
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


bool canProveForAllDyAxes(arith::Analyzer& analyzer, PrimExpr predicate);


class BlockIdxDivFinder : public ExprVisitor {
public:
  const FloorDivNode *floor_div = nullptr;
protected:
  void VisitExpr_(const FloorDivNode* op) override {
    if (const VarNode *const var = op->a.as<VarNode>()) {
      if (var->name_hint == "blockIdx.x") {
        // LOG(INFO) << "blockIdx.x / spotted";
        this->floor_div = op;
      }
    }
  }
};


class BlockIdxModFinder : public ExprVisitor {
public:
  const FloorModNode *floor_mod = nullptr;
protected:
  void VisitExpr_(const FloorModNode* op) override {
    if (const VarNode *const var = op->a.as<VarNode>()) {
      if (var->name_hint == "blockIdx.x") {
        // LOG(INFO) << "blockIdx.x % spotted";
        this->floor_mod = op;
      }
    }
  }
};


}  // namespace tir
}  // namespace tvm
