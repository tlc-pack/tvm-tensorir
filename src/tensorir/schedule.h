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
  StdNodeMap<FunctionRef, Stmt> attrs;

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

  BlockTreeNode cache(Tensor, std::string scope, std::string type);
  BlockTreeNode cache_read(Tensor tensor, std::string scope);
  BlockTreeNode cache_write(Tensor tensor, std::string scope);
  BlockTreeNode double_buffer();     // unimplemented
  void annotate(AxisTreeNode axis, std::string type);
  void double_buffer_scope(Tensor tensor);

  // Dependency analysis
  /*
   * \brief Gather required regions for a list of tensors
   * \param tensors The tensor of interest
   * \param axis The root axis for gathering
   * \param start_child_index The blocks in the children of root axis with index less
   *                          then this number will be ignored
   * \param block_filter If this is defined, the func only considers the blocks in this set.
   *                     If this is not defined, the function will consider all blocks.
   * \param gather_inputs Whether gather inputs of blocks
   * \param gather_outputs Whether gather outputs of blocks
   * \param aggregate_mode Aggregation mode. 'U' for union, 'I' for intersect.
   *
   */
  Array<Array<arith::IntSet> > GatherRegion(Array<Tensor> tensors,
                                            AxisTreeNode axis,
                                            int start_child_index,
                                            Set<BlockTreeNode> block_filter,
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
  size_t BlockNum(ScheduleTreeNode node) const;

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
