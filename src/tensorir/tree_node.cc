/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Nodes in schedule tree
 */

#include <tvm/operation.h>
#include <tvm/ir_pass.h>
#include <limits>
#include <vector>
#include <algorithm>
#include "tree_node.h"
#include "schedule.h"

namespace tvm {
namespace tensorir {

// TensorRegion
TensorRegion TensorRegionNode::make(Tensor data, Array<Range> ranges) {
  NodePtr<TensorRegionNode> node = make_node<TensorRegionNode>();

  node->data = std::move(data);
  Array<Range> simplified_ranges;
  for (const auto& x : ranges) {
    simplified_ranges.push_back(
      Range::make_by_min_extent(ir::Simplify(x->min), ir::Simplify(x->extent)));
  }
  node->ranges = std::move(simplified_ranges);

  return TensorRegion(node);
}

TensorRegion TensorRegion::MakeView(Array<Expr> mins, Array<Expr> extents) const {
  CHECK_EQ(mins.size(), operator->()->data.ndim());
  CHECK_EQ(mins.size(), extents.size());

  Array<Range> ranges;
  for (size_t i = 0; i < mins.size(); ++i) {
    Expr min = operator->()->ranges[i]->min + mins[i];
    ranges.push_back(Range::make_by_min_extent(min, extents[i]));
  }

  return TensorRegionNode::make(operator->()->data, ranges);
}

// Tree nodes
const ScheduleTreeNodeNode* ScheduleTreeNode::operator->() const {
  return static_cast<const ScheduleTreeNodeNode*>(node_.get());
}

ScheduleTreeNodeNode* ScheduleTreeNode::operator->() {
  return static_cast<ScheduleTreeNodeNode*>(node_.get());
}

ScheduleTreeNode ScheduleTreeNode::Copy() const {
  Array<ScheduleTreeNode> new_children;
  if (operator->()->children.defined()) {
    for (const auto& x : operator->()->children) {
      new_children.push_back(x.Copy());
    }
  }

  if (const AxisTreeNodeNode* n = this->as<AxisTreeNodeNode>()) {
    return AxisTreeNodeNode::make(n->loop_var, n->min, n->extent, n->axis_type, new_children);
  } else if (const BlockTreeNodeNode* n = this->as<BlockTreeNodeNode>()) {
    return BlockTreeNodeNode::make(n->args, n->vars, n->inputs, n->outputs, n->stmt, new_children);
  } else {
    LOG(FATAL) << "Internal error : unknown tree node type";
  }
  return ScheduleTreeNode(nullptr);
}

AxisTreeNode AxisTreeNodeNode::make(Var loop_var, Expr min, Expr extent,
                                    AxisType axis_type,
                                    Array<ScheduleTreeNode> children) {
  NodePtr<AxisTreeNodeNode> node = make_node<AxisTreeNodeNode>();

  node->loop_var = std::move(loop_var);
  node->min = std::move(min);
  node->extent = std::move(extent);
  node->axis_type = axis_type;
  node->children = std::move(children);
  return AxisTreeNode(node);
}

BlockTreeNode BlockTreeNodeNode::make(Array<Expr> args,
                                      Array<Var> vars,
                                      Array<TensorRegion> inputs,
                                      Array<TensorRegion> outputs,
                                      Stmt stmt,
                                      Array<ScheduleTreeNode> children) {
  NodePtr<BlockTreeNodeNode> node = make_node<BlockTreeNodeNode>();

  // sort block args according to its appearance order in outputs
  StdNodeMap<Var, int> weight;
  int ct = 0;

  for (const auto& t : outputs) {
    for (const auto& ran : t->ranges) {
      if (ran->min->is_type<Variable>()) {
        weight[Downcast<Var>(ran->min)] = std::numeric_limits<int>::min() + (ct++);
      }
    }
  }

  std::vector<size_t> indices;
  for (size_t i = 0; i < vars.size(); ++i) {
    indices.push_back(i);
  }
  std::sort(indices.begin(), indices.end(),
            [&](int a, int b) -> bool { return weight[vars[a]] < weight[vars[b]]; });

  Array<Expr> sorted_args;
  Array<Var> sorted_vars;
  for (size_t i = 0; i < vars.size(); ++i) {
    sorted_args.push_back(args[indices[i]]);
    sorted_vars.push_back(vars[indices[i]]);
  }

  // set members
  node->args = std::move(sorted_args);
  node->vars = std::move(sorted_vars);
  node->inputs = std::move(inputs);
  node->outputs = std::move(outputs);
  node->stmt = std::move(stmt);
  node->children = std::move(children);

  return BlockTreeNode(node);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<TensorRegionNode>([](const TensorRegionNode *t, IRPrinter *p) {
  p->stream << t->data->op->name << "[";

  for (size_t i = 0; i < t->data->shape.size(); ++i) {
    p->stream << t->ranges[i]->min << ":" << t->ranges[i]->min + t->ranges[i]->extent;
    if (i != t->data->shape.size() - 1) {
      p->stream << ", ";
    }
  }
  p->stream << "]";
});

// Debug tools
void PrintTreeNode(std::ostream &output, ScheduleTreeNode node, size_t indent) {
  for (size_t i = 0; i < indent; ++i) {
    output << " ";
  }

  if (const AxisTreeNodeNode* n = node.as<AxisTreeNodeNode>()) {
    output << "for " << n->loop_var << " = " << n->min << " to " << n->min + n->extent << std::endl;
    for (auto x : n->children) {
      PrintTreeNode(output, x, indent + 2);
    }
  } else if (const BlockTreeNodeNode* n = node.as<BlockTreeNodeNode>()) {
    output << "Block(";
    for (size_t i = 0; i < n->args.size(); ++i) {
      output << n->vars[i] << "=" << n->args[i];
      if (i != n->args.size() - 1) {
        output << ", ";
      }
    }
    output << ")";

    output << " W: " << n->outputs;
    output << " R: " << n->inputs;
    output << std::endl;
  } else {
    output << "Error";
//    LOG(FATAL);
  }
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<AxisTreeNodeNode>([](const AxisTreeNodeNode* op, IRPrinter* p) {
  PrintTreeNode(p->stream, GetRef<ScheduleTreeNode>(op), 0);
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<BlockTreeNodeNode>([](const BlockTreeNodeNode* op, IRPrinter* p) {
  PrintTreeNode(p->stream, GetRef<ScheduleTreeNode>(op), 0);
});


}  // namespace tensorir
}  // namespace tvm
