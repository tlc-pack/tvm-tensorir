/*!
 *  Copyright (c) 2019 by Contributors
 */

#ifndef TVM_TENSORIR_UTIL_H_
#define TVM_TENSORIR_UTIL_H_

#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <tvm/arithmetic.h>
#include <tvm/operation.h>
#include "node_util.h"

namespace tvm {
namespace tensorir {

// Flatten a chain of block statements to a list of statements.
// It will gather all statements with depth of 1 (Ignore ir::Block when computing depth).
Array<Stmt> ExpandBlockChain(Stmt stmt);

// Gather all accesses to all tensors or a specific tensor
class TensorAccessGather : public ir::IRVisitor {
 public:
  TensorAccessGather() {}
  TensorAccessGather(Tensor target_tensor) : target_tensor_(target_tensor) {}

  void Visit_(const ir::Call* op) {
    if (op->call_type == ir::Call::CallType::Halide) {
      if (target_tensor_.defined()) {
        if (target_tensor_ == (Downcast<Operation>(op->func)).output(op->value_index)) {
          std::vector<Expr> args;
          for(const auto& x : op->args) {
            args.push_back(x);
          }
          access_one.push_back(args);
        }
      } else {
        std::pair<Tensor, std::vector<Expr> > acc;
        acc.first = (Downcast<Operation>(op->func)).output(op->value_index);
        for (const auto& x : op->args) {
          acc.second.push_back(x);
        }
        access_all.push_back(acc);
        if (!access_grouped.count(acc.first)) {
          tensor_order.push_back(acc.first);
        }
        access_grouped[acc.first].push_back(acc.second);
      }
    }
  }

  StdNodeMap<Tensor, std::vector<std::vector<Expr> > > access_grouped; // grouped accesses by target tensor
  std::vector<std::pair<Tensor, std::vector<Expr> > > access_all; // all accesses
  std::vector<std::vector<Expr> > access_one;                     // accesses to the target buffer

  std::vector<Tensor> tensor_order;  // a list to keep the original order of tensors
 private:
  Tensor target_tensor_;
};

// Gather all accessed tensors
class TensorGather : public ir::IRVisitor {
 public:

  void Visit_(const ir::Call* op) {
    if (op->call_type == ir::Call::CallType::Halide) {
      tensors.insert((Downcast<Operation>(op->func)).output(op->value_index));
    }
  }

  std::unordered_set<Tensor> tensors;
};

// Convert an array of Halide Stmt to a Halide::IR::Block
inline Stmt ArrayToBlock(Array<Stmt> stmts) {
  if (stmts.size() == 0) {
    return Stmt(nullptr);
  }
  if (stmts.size() == 1) {
    return stmts[0];
  }
  int ct = static_cast<int>(stmts.size()) - 2;
  Stmt now = stmts[ct + 1];
  do {
    if (const ir::AttrStmt* op = stmts[ct].as<ir::AttrStmt>()) {
      now = ir::AttrStmt::make(op->node, op->attr_key, op->value, now);
    } else if (const ir::Realize* op = stmts[ct].as<ir::Realize>()) {
      now = ir::Realize::make(op->func, op->value_index, op->type, op->bounds, op->condition, now);
    } else {
      now = ir::Block::make(stmts[ct], now);
    }
    ct--;
  } while (ct >= 0);
  return now;
}

// Flatten a two-dimensional array
template <typename T>
inline Array<T> Flatten2DArray(Array<Array<T> > input) {
  Array<T> ret;
  for (size_t i = 0; i < input.size(); ++i) {
    for (size_t j = 0; j < input[i].size(); ++j) {
      ret.push_back(input[i][j]);
    }
  }
  return ret;
}

// Gather all vars in an expression or statement
Set<Var> GatherVars(const NodeRef& expr_or_stmt);

// Rewrite expression with both equation_simplify (expr to var)
// and direct substitute (var to expr)
Expr SubstituteAndEquationSimplify(Expr expr, Map<Var, Expr> var_map, arith::Analyzer* analyzer);
Stmt SubstituteAndEquationSimplify(Stmt stmt, Map<Var, Expr> var_map, arith::Analyzer* analyzer);

// Substitute expressions in range
inline Range SubstituteRange(Range range, StdNodeMap<Var, Expr> var_map) {
  return Range::make_by_min_extent(ir::Substitute(range->min, var_map),
                                   ir::Substitute(range->extent, var_map));
}

inline Array<Range> SubstituteRange(Array<Range> ranges, StdNodeMap<Var, Expr> var_map) {
  Array<Range> ret;
  for (size_t i = 0; i < ranges.size(); ++i) {
    ret.push_back(Range::make_by_min_extent(ir::Substitute(ranges[i]->min, var_map),
                                            ir::Substitute(ranges[i]->extent, var_map)));
  }
  return ret;
}

} // namespace tensorir
} // namespace tvm

#endif // TVM_TENSORIR_UTIL_H_
