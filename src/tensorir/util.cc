/*!
 *  Copyright (c) 2019 by Contributors
 */

#include "util.h"

namespace tvm {
namespace tensorir {

using namespace tvm::ir;

void ExpandBlockChain(Stmt stmt, std::vector<Stmt>* ret) {
  if (const Block* node = stmt.as<Block>()) {
    ExpandBlockChain(node->first, ret);
    ExpandBlockChain(node->rest, ret);
  } else {
    ret->push_back(stmt);
  }
}

Array<Stmt> ExpandBlockChain(Stmt stmt) {
  std::vector<Stmt> ret;
  ExpandBlockChain(stmt, &ret);
  return ret;
}

Set<Var> GatherVars(const NodeRef& expr_or_stmt) {
  Set<Var> ret;

  ir::PostOrderVisit(expr_or_stmt, [&ret](const NodeRef& node) {
    if (node->is_type<Variable>()) {
      ret.insert(Downcast<Var>(node));
    }
  });

  return ret;
}

} // namespace tvm
} // namespace tensorir