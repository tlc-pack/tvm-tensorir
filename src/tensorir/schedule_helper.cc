/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Helper functions to write schedule
 */

#include <tvm/operation.h>
#include <tvm/ir_visitor.h>
#include "schedule_helper.h"

namespace tvm {
namespace tensorir {

bool IsDirectAccess(Call call) {
  return true;
}

bool IsInjectiveBlock(BlockTreeNode block) {
  if (block->children.size() != 0) {
    return false;
  }
  const Provide* p = block->stmt.as<Provide>();
  Tensor output = Downcast<Operation>(p->func).output(static_cast<size_t>(p->value_index));

  if (p == nullptr) {
    return false;
  }

  // condition
  // 1. All indices on LHS are single variables or constants
  // 2. All indices on RHS are single variables or constants
  // 3. LHS tensor does not appear on the RHS

  // check LHS
  for (size_t i = 0; i < p->args.size(); ++i) {
    if (!p->args[i]->is_type<Variable>() && !p->args[i]->is_type<IntImm>()) {
      return false;
    }
  }

  // check RHS
  bool fail = false;
  bool found_tensor_in_rhs = false;   // do not inline init of reduction
  ir::PostOrderVisit(block->stmt, [&fail, &output, &found_tensor_in_rhs](const NodeRef& node) {
    if (fail) { return; }

    if (const Call* call = node.as<Call>()) {
      if (call->call_type == Call::CallType::Halide) {
        found_tensor_in_rhs = true;
        if (Downcast<Operation>(call->func).output(call->value_index) == output) {
            fail = true;
            return;
        }
  
        for (size_t i = 0; i < call->args.size(); ++i) {
          if (!call->args[i]->is_type<Variable>() && !call->args[i]->is_type<IntImm>()) {
            fail = true;
            return;
          }
        }
      }
    }
  });

  return !fail && found_tensor_in_rhs;
}

void InlineAllInjective(Schedule sch) {
  Array<BlockTreeNode> outputs = sch.output_blocks();
  Set<BlockTreeNode> output_set(outputs.begin(), outputs.end());

  for (const auto& x : sch.blocks()) {
    if (!output_set.count(x) && IsInjectiveBlock(x)) {
      sch.compute_inline(x);
    }
  }
}

}  // namespace tensorir
}  // namespace tvm

