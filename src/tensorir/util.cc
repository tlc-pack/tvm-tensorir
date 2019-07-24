/*!
 *  Copyright (c) 2019 by Contributors
 */

#include <tvm/ir_mutator.h>
#include <tvm/ir_functor_ext.h>
#include "util.h"

namespace tvm {
namespace tensorir {

using namespace tvm::ir;

Array<Stmt> ExpandBlockChain(Stmt stmt) {
  std::vector<Stmt> ret;

  std::function<void(Stmt, std::vector<Stmt>*)> expand_func;
  expand_func = [&expand_func](Stmt stmt, std::vector<Stmt>* ret) {
    if (const Block* node = stmt.as<Block>()) {
      expand_func(node->first, ret);
      expand_func(node->rest, ret);
    } else {
      ret->push_back(stmt);
    }
  };

  expand_func(stmt, &ret);
  return ret;
}

Set<Var> GatherVars(const NodeRef& expr_or_stmt) {
  Set<Var> ret;

  ir::PostOrderVisit(expr_or_stmt, [&ret](const NodeRef& node) {
    if (node->is_type<Variable>() && Downcast<Var>(node).type().code() != halideir_type_handle) {
      ret.insert(Downcast<Var>(node));
    }
  });

  return ret;
}

// Rewrite expression with both equation_simplify (expr to var)
// and direct substitute (var to expr)
class SubstituteAndEquationSimplifier : public IRMutator {
 public:
  SubstituteAndEquationSimplifier(Map<Var, Expr> var_map, arith::Analyzer* analyzer):
    var_map_(var_map), analyzer_(analyzer) {}

  Expr Mutate(Expr expr) final {
    return Substitute(analyzer_->equation_simplify(expr), var_map_);
  }

  Stmt Mutate(Stmt stmt) final {
    return IRMutator::Mutate(stmt);
  }

 private:
  Map<Var, Expr> var_map_;
  arith::Analyzer* analyzer_;
};

Expr SubstituteAndEquationSimplify(Expr expr, Map<Var, Expr> var_map, arith::Analyzer* analyzer) {
  return Substitute(analyzer->equation_simplify(expr), var_map);
}


Stmt SubstituteAndEquationSimplify(Stmt stmt, Map<Var, Expr> var_map, arith::Analyzer* analyzer) {
  return SubstituteAndEquationSimplifier(var_map, analyzer).Mutate(stmt);
}


} // namespace tvm
} // namespace tensorir