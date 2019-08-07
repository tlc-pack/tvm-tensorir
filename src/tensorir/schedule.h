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
#include <string>
#include "dependency_graph.h"
#include "tree_node.h"
#include "intrinsic.h"

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

  StdNodeMap<Tensor, Region> raw_realize_region;  // todo(lmzheng): try to use tvm::Map instead
  StdNodeMap<FunctionRef, std::string> raw_realize_scope;
  StdNodeMap<Var, Stmt> bind_var;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("root", &root);
  }

  TVM_DLL static Schedule make(Stmt stmt);

  static constexpr const char* _type_key = "tensorir.Schedule";
  TVM_DECLARE_NODE_TYPE_INFO(ScheduleNode, Node);
};


class Schedule : public NodeRef {
 public:
  // Getter
  Array<BlockTreeNode> blocks() const;
  Array<BlockTreeNode> output_blocks() const;
  Array<BlockTreeNode> reduction_blocks() const;
  Array<BlockTreeNode> predecessor(BlockTreeNode block) const;
  Array<BlockTreeNode> successor(BlockTreeNode block) const;
  Array<AxisTreeNode> axis(ScheduleTreeNode node) const;

  // Schedule primitives
  Array<AxisTreeNode> split(AxisTreeNode axis, Expr factor);
  Array<AxisTreeNode> split_nparts(AxisTreeNode axis, Expr nparts);
  AxisTreeNode fuse(AxisTreeNode outer, AxisTreeNode inner);
  Array<AxisTreeNode> reorder(Array<AxisTreeNode> axes);
  Array<AxisTreeNode> binary_reorder(AxisTreeNode outer, AxisTreeNode inner);
  Array<ScheduleTreeNode> unroll(AxisTreeNode axis);

  void compute_inline(BlockTreeNode block);
  BlockTreeNode compute_at(BlockTreeNode block, AxisTreeNode axis);
  BlockTreeNode compute_after(BlockTreeNode block, AxisTreeNode axis);
  BlockTreeNode compute_root(BlockTreeNode block);

  void bind(AxisTreeNode axis, IterVar thread_iter);

  BlockTreeNode blockize(AxisTreeNode axis);
  ScheduleTreeNode unblockize(BlockTreeNode block);
  BlockTreeNode merge(Array<ScheduleTreeNode> nodes);

  BlockTreeNode tensorize(BlockTreeNode block, TensorIntrinsic intrin);
  ScheduleTreeNode untensorize(BlockTreeNode block);

  BlockTreeNode cache_read();        // unimplemented
  BlockTreeNode cache_write();       // unimplemented
  BlockTreeNode double_buffer();     // unimplemented
  void annotate(AxisTreeNode axis, std::string type);

  // Dependency analysis
  Array<Array<arith::IntSet> > GatherRegion(Array<Tensor> tensors,
                                            Set<BlockTreeNode> block_filter,
                                            AxisTreeNode axis,
                                            int start_child_index,
                                            bool gather_inputs,
                                            bool gather_outputs,
                                            char aggregate_mode) const;

  // Output
  Stmt ToHalide() const;

  // Tree manipulation
  // Note: Because we need to update the father_map, these functions are set to be the member 
  // functions of Schedule. Considering moving them to another place later
  void UpdateFather(ScheduleTreeNode father, bool recursive = false);
  ScheduleTreeNode LowestCommonAncestor(Array<ScheduleTreeNode> nodes, bool inclusive) const;
  inline void ReplaceChild(ScheduleTreeNode old_child, ScheduleTreeNode new_child);
  inline void ReplaceChild(ScheduleTreeNode old_child, Array<ScheduleTreeNode> new_children);
  void RemoveLeaf(ScheduleTreeNode node);
  bool IsAncestor(ScheduleTreeNode outer, ScheduleTreeNode inner) const;

  // Debug tools
  void CheckFatherLink();

  TVM_DEFINE_MUTABLE_NODE_REF_METHODS(Schedule, NodeRef, ScheduleNode);

 private:
  // Generate loop axes according to the solved iteration domain
  // This is the common part of compute_at and compute_after
  BlockTreeNode RegenerateLoopAxis(BlockTreeNode block, AxisTreeNode axis, 
                                   Array<arith::IntSet> iter_domain, int insert_pos);

  size_t ct_{0};
};

}  // namespace tensorir
}  // namespace tvm

#endif  // TVM_TENSORIR_SCHEDULE_H_
