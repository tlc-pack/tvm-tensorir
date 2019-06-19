/*!
 *  Copyright (c) 2019 by Contributors
 */

#ifndef TVM_TENSORIR_UTIL_H_
#define TVM_TENSORIR_UTIL_H_

#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include "node_util.h"

namespace tvm {
namespace tensorir {

// Flatten a chain of block statements to a list of statements.
// Gather all statements with depth of 1 when ignoring Block statement.
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
      }
    }
  }

  void GatherAndGroup(Expr expr) {
    Visit(expr);
    for (auto x : access_all) {
      access_grouped[x.first].push_back(x.second);
    }
  }


  StdNodeMap<Tensor, std::vector<std::vector<Expr> > > access_grouped; // grouped accesses by target tensor
  std::vector<std::pair<Tensor, std::vector<Expr> > > access_all; // all accesses
  std::vector<std::vector<Expr> > access_one;                     // accesses to the target buffer
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

// Return whether a range is a single point
inline bool is_single_point(Range range) {
  return is_zero(ir::Simplify(range->extent));
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

// Gather all vars in an expression
Set<Var> GatherVars(Expr expr);

} // namespace tensorir
} // namespace tvm

#endif // TVM_TENSORIR_UTIL_H_
