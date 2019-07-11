/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Nodes in schedule tree
 */

#include <tvm/operation.h>
#include "tree_node.h"
#include "schedule.h"

namespace tvm {
namespace tensorir {

// access methods
const ScheduleTreeNodeNode* ScheduleTreeNode::operator->() const {
  return static_cast<const ScheduleTreeNodeNode*>(node_.get());
}

ScheduleTreeNodeNode* ScheduleTreeNode::operator->() {
  return static_cast<ScheduleTreeNodeNode*>(node_.get());
}

//inline ScheduleTreeNodeNode* ScheduleTreeNode::CopyOnWrite() {
//  CHECK(node_ != nullptr);
//  if (!node_.unique()) {
//    if (node_->is_type<AxisTreeNodeNode>()) {
//      NodePtr<AxisTreeNodeNode> n =
//          make_node<AxisTreeNodeNode>(*static_cast<const AxisTreeNodeNode*>((operator->())));
//      NodePtr<Node>(std::move(n)).swap(node_);
//    } else if (node_->is_type<StmtTreeNodeNode>()) {
//      NodePtr<StmtTreeNodeNode> n =
//          make_node<StmtTreeNodeNode>(*static_cast<const StmtTreeNodeNode*>((operator->())));
//      NodePtr<Node>(std::move(n)).swap(node_);
//    }
//  }
//  return static_cast<ScheduleTreeNodeNode*>(node_.get());
//}

ScheduleTreeNode ScheduleTreeNode::Copy() const {
  CHECK(node_ != nullptr);
  if (node_->is_type<AxisTreeNodeNode>()) {
    NodePtr<AxisTreeNodeNode> n =
        make_node<AxisTreeNodeNode>(*static_cast<const AxisTreeNodeNode*>((operator->())));
    return ScheduleTreeNode(n);
  } else if (node_->is_type<BlockTreeNodeNode>()) {
    NodePtr<BlockTreeNodeNode> n =
        make_node<BlockTreeNodeNode>(*static_cast<const BlockTreeNodeNode*>((operator->())));
    return ScheduleTreeNode(n);
  } else {
    LOG(FATAL) << "Internal error : unknown tree node type";
  }
  return ScheduleTreeNode(nullptr);
}

TensorRegion TensorRegionNode::make(Tensor data, Array<Range> ranges) {
  NodePtr<TensorRegionNode> node = make_node<TensorRegionNode>();

  node->data = data;
  node->ranges = ranges;

  return TensorRegion(node);
}

// maker
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
  node->args = std::move(args);
  node->vars = std::move(vars);
  node->inputs = std::move(inputs);
  node->outputs = std::move(outputs);
  node->stmt = std::move(stmt);
  node->children = std::move(children);
  return BlockTreeNode(node);
}

//BlockTreeNode BlockTreeNode::CopyWithNewArgs(Array<Expr> new_args) {
//  NodePtr<BlockTreeNodeNode> node = make_node<BlockTreeNodeNode>();
//
//  node->args = node->args;
//  node->vars = node->vars;
//  node->inputs = node->inputs;
//  node->outputs = node->outputs;
//  node->stmt = node->stmt;
//
//  return BlockTreeNode(node);
//}

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
    LOG(FATAL);
  }
}

} // namespace tensorir
} // namespace tvm
