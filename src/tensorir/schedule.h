/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Schedule operations
 */

#ifndef TVM_TENSORIR_SCHEDULE_H_
#define TVM_TENSORIR_SCHEDULE_H_

#include <tvm/base.h>
#include <tvm/ir.h>
#include <tvm/tensor.h>
#include <tvm/arithmetic.h>
#include "dependency_graph.h"
#include "tree_node.h"

namespace tvm {
namespace tensorir {

using namespace ir;

using FatherMap = Map<ScheduleTreeNode, ScheduleTreeNode>;

// The schedule interface
class Schedule;
class ScheduleNode : public Node {
 public:
  Array<Tensor> inputs;
  Array<Tensor> outputs;

  AxisTreeNode root;
  DependencyGraph dep_graph;
  Array<BlockTreeNode> block_list;

  FatherMap father_map;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("root", &root);
  }

  TVM_DLL static Schedule make(Stmt stmt);

  static constexpr const char* _type_key = "tensorir.Schedule";
  TVM_DECLARE_NODE_TYPE_INFO(ScheduleNode, Node);
};

class Schedule : public NodeRef {
 public:
  Schedule() {}
  explicit Schedule(NodePtr<Node> n): NodeRef(n) {}

  ScheduleNode* operator->() {
    return static_cast<ScheduleNode*>(node_.get());
  }
  const ScheduleNode* operator->() const {
    return static_cast<const ScheduleNode*>(node_.get());
  }

  // getter
  Array<BlockTreeNode> blocks() const;
  Array<AxisTreeNode> axis(ScheduleTreeNode node) const;

  // schedule primitives
  Array<AxisTreeNode> split(AxisTreeNode axis, Expr factor);
  AxisTreeNode fuse(AxisTreeNode outer, AxisTreeNode inner);
  Array<AxisTreeNode> reorder(AxisTreeNode outer, AxisTreeNode inner);
  Array<ScheduleTreeNode> unroll(AxisTreeNode axis);
  void compute_inline(BlockTreeNode block);
  BlockTreeNode compute_at(BlockTreeNode block, AxisTreeNode axis);
  BlockTreeNode compute_after(BlockTreeNode block, AxisTreeNode axis);
  BlockTreeNode compute_root(BlockTreeNode block);

  void bind(AxisTreeNode axis, std::string name);

  // dependency analysis
  Array<Array<arith::IntSet> > GatherRegion(Array<Tensor> tensors, AxisTreeNode axis) const;

  // output
  Stmt ToHalide(ScheduleTreeNode node) const;

  using ContainerType = ScheduleNode;


  // tree manipulation (Because we need to update the father_map, these functions are
  // set to be the member functions of Schedule. Considering moving them to another place later)
  void UpdateFather(ScheduleTreeNode father, bool recursive = false);
  ScheduleTreeNode LeastCommonAncestor(ScheduleTreeNode a, ScheduleTreeNode b) const;
  inline void ReplaceChild(ScheduleTreeNode old_child, ScheduleTreeNode new_child);
  void RemoveLeaf(ScheduleTreeNode node);

};

} // namespace tensorir
} // namespace tvm

#endif // TVM_TENSORIR_SCHEDULE_H_
