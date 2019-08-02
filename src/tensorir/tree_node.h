/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Nodes in schedule tree
 */

#ifndef TVM_TENSORIR_TREE_NODE_H_
#define TVM_TENSORIR_TREE_NODE_H_

#include <tvm/base.h>
#include <tvm/ir.h>
#include "node_util.h"

namespace tvm {
namespace tensorir {

using namespace ir;

// represents a sub-region of a tensor
class TensorRegion;
class TensorRegionNode : public Node {
 public:
  Tensor data;
  Array<Range> ranges;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("data", &data);
    v->Visit("ranges", &ranges);
  }

  TVM_DLL static TensorRegion make(Tensor data, Array<Range> ranges);

  static constexpr const char* _type_key = "tensorir.TensorRegion";
  TVM_DECLARE_NODE_TYPE_INFO(TensorRegionNode, Node);
};

class TensorRegion : public NodeRef {
 public:
  TensorRegion() {}
  explicit TensorRegion(NodePtr<Node> n): NodeRef(n) {}

  const TensorRegionNode* operator->() const;
  TensorRegion MakeView(Array<Expr> mins, Array<Expr> extents) const;

  using ContainerType = TensorRegionNode;
};

// base class of the nodes in the schedule tree
class ScheduleTreeNodeNode;

class ScheduleTreeNode : public NodeRef {
 public:
  ScheduleTreeNode() {}
  explicit ScheduleTreeNode(NodePtr<Node> n): NodeRef(n) {}

  const ScheduleTreeNodeNode* operator->() const;
  ScheduleTreeNodeNode* operator->();

  ScheduleTreeNode Copy() const;
  using ContainerType = ScheduleTreeNodeNode;
};

class ScheduleTreeNodeNode : public Node {
 public:
  Array<ScheduleTreeNode> children;

  ScheduleTreeNodeNode() {}
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("children", &children);
  }

  static constexpr const char* _type_key = "tensorir.ScheduleTreeNode";
  TVM_DECLARE_BASE_NODE_INFO(ScheduleTreeNodeNode, Node);
};

// Type of AxisTreeNode
enum AxisType: int {
  kSpace = 0,
  kReduce = 1,
  kMix = 2,
  kOpaque = 3,
  vectorized = 4,
  unrolled = 5,
};

// Two kinds of node : 1. AxisTreeNode
// It represents a for loop
class AxisTreeNode;
class AxisTreeNodeNode : public ScheduleTreeNodeNode {
 public:
  Var loop_var;
  Expr min, extent;
  AxisType axis_type;

  void VisitAttrs(AttrVisitor* v) final {
    ScheduleTreeNodeNode::VisitAttrs(v);
    v->Visit("loop_var", &loop_var);
    v->Visit("min", &min);
    v->Visit("extent", &extent);
  }

  TVM_DLL static AxisTreeNode make(Var loop_var, Expr min, Expr extent,
                                   AxisType axis_type,
                                   Array<ScheduleTreeNode> children);

  static constexpr const char* _type_key = "tensorir.AxisTreeNode";
  TVM_DECLARE_NODE_TYPE_INFO(AxisTreeNodeNode, ScheduleTreeNodeNode);
};

TVM_DEFINE_MUTABLE_NODE_REF(AxisTreeNode, ScheduleTreeNode, AxisTreeNodeNode);

// Two kinds of node : 2. BlockTreeNode
// It represents a computation block
class BlockTreeNode;
class BlockTreeNodeNode : public ScheduleTreeNodeNode {
 public:
  Array<Expr> args;
  Array<Var> vars;
  Array<TensorRegion> inputs;
  Array<TensorRegion> outputs;

  Stmt stmt;               // terminal block has a halide stmt
  NodeRef intrin;          // tensorized block has a intrin

  Array<Expr> predicates; // ??

  void VisitAttrs(AttrVisitor* v) final {
    ScheduleTreeNodeNode::VisitAttrs(v);
    v->Visit("stmt", &stmt);
  }

  TVM_DLL static BlockTreeNode make(Array<Expr> args,
                                    Array<Var> vars,
                                    Array<TensorRegion> inputs,
                                    Array<TensorRegion> outputs,
                                    Stmt stmt,
                                    Array<ScheduleTreeNode> children);

  static constexpr const char* _type_key = "tensorir.BlockTreeNode";
  TVM_DECLARE_NODE_TYPE_INFO(BlockTreeNodeNode, ScheduleTreeNodeNode);
};

TVM_DEFINE_MUTABLE_NODE_REF(BlockTreeNode, ScheduleTreeNode, BlockTreeNodeNode);

class Attr;

class AttrNode : public Node {
 public:
  NodeRef node;
  std::string attr_key;
  Expr value;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("node", &node);
    v->Visit("attr_key", &attr_key);
    v->Visit("value", &value);
  }

  TVM_DLL static Attr make(NodeRef node, std::string attr_key, Expr value);

  static constexpr const char* _type_key = "tensorir.AttrNode";
  TVM_DECLARE_NODE_TYPE_INFO(AttrNode, Node);
};

TVM_DEFINE_NODE_REF(Attr, AttrNode);

// debug tools
void PrintTreeNode(std::ostream &output, ScheduleTreeNode node, size_t indent=0);

} // namespace tensorir
} // namespace tvm

#endif // TVM_TENSORIR_TREE_NODE_H_
