/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Tensor intrinsics for tensorize
 */

#include <tvm/packed_func_ext.h>
#include <string>
#include "tree_node.h"
#include "tree_builder.h"
#include "intrinsic.h"
#include "util.h"

namespace tvm {
namespace tensorir {

// maker
TensorIntrinsic TensorIntrinsicNode::make(
    Operation op,
    TypedPackedFunc<NodeRef(Array<TensorRegion>, Array<TensorRegion>)> intrin_func,
    std::string name) {
  NodePtr<TensorIntrinsicNode> node = make_node<TensorIntrinsicNode>();

  node->op = std::move(op);
  node->intrin_func = std::move(intrin_func);
  node->name = std::move(name);

  // todo (lmzheng): build BlockTreeNode `from` for checking and untensorize

  return TensorIntrinsic(node);
}

ScheduleTreeNode TensorIntrinsic::Instantiate(Array<TensorRegion> inputs,
                                              Array<TensorRegion> outputs) const {
  NodeRef ret = operator->()->intrin_func(inputs, outputs);

  if (ret->derived_from<ScheduleTreeNodeNode>()) {
    return Downcast<ScheduleTreeNode>(ret);
  } else if (ret->derived_from<StmtNode>()) {
    Stmt stmt = Downcast<Stmt>(ret);
    Array<Expr> args;
    Array<Var> vars;
    Set<Var> used_vars;
    Map<Var, Expr> var_map;
    arith::Analyzer analyzer;

    // gather vars
    for (const auto& x : inputs) {
      for (const auto& ran : x->ranges) {
        used_vars.insert(GatherVars(ran->min));
        used_vars.insert(GatherVars(ran->extent));
      }
    }
    for (const auto& x : outputs) {
      for (const auto& ran : x->ranges) {
        used_vars.insert(GatherVars(ran->min));
        used_vars.insert(GatherVars(ran->extent));
      }
    }

    // canonicalize outputs
    Array<TensorRegion> new_outputs;
    std::tie(args, vars, new_outputs, var_map) = CreateOutputRegions(
        outputs, used_vars, &analyzer);

    // replace inputs
    Array<TensorRegion> new_inputs;
    for (const auto& x : inputs) {
      Array<Range> ranges;
      for (const auto& ran : x->ranges) {
        ranges.push_back(Range::make_by_min_extent(
            SubstituteAndEquationSimplify(ran->min, var_map, &analyzer),
            SubstituteAndEquationSimplify(ran->extent, var_map, &analyzer)));
      }
      new_inputs.push_back(TensorRegionNode::make(x->data, ranges));
    }

    // replace stmt
    stmt = SubstituteAndEquationSimplify(stmt, var_map, &analyzer);

    return BlockTreeNodeNode::make(args, vars, new_inputs, new_outputs,
                                   stmt, Array<ScheduleTreeNode>{});
  } else {
    LOG(FATAL) << "The intrin func returns invalid value";
  }
  return ScheduleTreeNode(nullptr);
}

}  // namespace tensorir
}  // namespace tvm
