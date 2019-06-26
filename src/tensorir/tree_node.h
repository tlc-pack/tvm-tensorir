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

  static constexpr const char* _type_key = "tensorir.TensorRegionNode";
  TVM_DECLARE_NODE_TYPE_INFO(TensorRegionNode, Node);
};

TVM_DEFINE_NODE_REF(TensorRegion, TensorRegionNode);

// base class of the nodes in schedule tree
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
};

// Two kinds of node : 1. AxisTreeNode
class AxisTreeNode;
class AxisTreeNodeNode : public ScheduleTreeNodeNode {
 public:
  Var loop_var;
  Expr min, extent;
  AxisType axis_type;

  void VisitAttrs(AttrVisitor* v) final {
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
class BlockTreeNode;
class BlockTreeNodeNode : public ScheduleTreeNodeNode {
 public:
  Array<Expr> args;
  Array<Var> vars;
  Array<TensorRegion> inputs;
  Array<TensorRegion> outputs;
  Stmt stmt;              // terminal block has a halide stmt

  Array<Expr> predicates; // ??

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("stmt", &stmt);
  }

  TVM_DLL static BlockTreeNode make(Array<Expr> args,
                                    Array<Var> vars,
                                    Array<TensorRegion> inputs,
                                    Array<TensorRegion> outputs,
                                    Stmt stmt);

  static constexpr const char* _type_key = "tensorir.BlockTreeNode";
  TVM_DECLARE_NODE_TYPE_INFO(BlockTreeNodeNode, ScheduleTreeNodeNode);
};

class BlockTreeNode : public ScheduleTreeNode {
 public:
  BlockTreeNode CopyWithNewArgs(Array<Expr> new_args);

  BlockTreeNodeNode* operator->(){
    return static_cast<BlockTreeNodeNode*>(node_.get());
  }
  TVM_DEFINE_NODE_REF_METHODS(BlockTreeNode, ScheduleTreeNode, BlockTreeNodeNode);
};

// debug tools
void PrintTreeNode(std::ostream &output, ScheduleTreeNode node, size_t indent=0);

} // namespace tensorir
} // namespace tvm

#endif // TVM_TENSORIR_TREE_NODE_H_
