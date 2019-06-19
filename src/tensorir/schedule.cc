/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Schedule operations
 */

#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/arithmetic.h>
#include "../arithmetic/int_set_internal.h"
#include "schedule.h"
#include "tree_builder.h"
#include "util.h"

namespace tvm {

namespace arith {
// move this to arithmetic.h when this is stable
Array<IntSet> SolveCover(Array<Var> vars, Array<IntSet> produce, Array<IntSet> requirement);
}

namespace tensorir {

using arith::IntSet;

void GatherChildrenBlocks(ScheduleTreeNode node, StdNodeSet<BlockTreeNode>* ret);
void GatherVarDomain(ScheduleTreeNode node,
                     std::unordered_map<const Variable*, arith::IntSet>* dom_map);

// maker
Schedule ScheduleNode::make(Stmt stmt) {
  return TreeBuilder().Build(stmt);
}

// accessor
Array<BlockTreeNode> Schedule::blocks() const {
  return operator->()->block_list;
}

Array<AxisTreeNode> Schedule::axis(ScheduleTreeNode stmt) const {
  Array<AxisTreeNode> ret;
  const FatherMap& father_map = operator->()->father_map;

  const AxisTreeNodeNode* now = father_map[stmt].as<AxisTreeNodeNode>();
  const AxisTreeNodeNode* root = operator->()->root.as<AxisTreeNodeNode>();

  while (now != root) {
    ret.push_back(GetRef<AxisTreeNode>(now));
    now = father_map[GetRef<AxisTreeNode>(now)].as<AxisTreeNodeNode>();
  }

  return Array<AxisTreeNode>(ret.rbegin(), ret.rend()); // reverse
}

// schedule primitives
BlockTreeNode Schedule::compute_at(BlockTreeNode block, AxisTreeNode axis) {
  ScheduleTreeNode father = operator->()->father_map[block];

  // check dependency:
  // 1. all successors are at the subtree
  // 2. WAW cannot across reduce axis

  StdNodeSet<BlockTreeNode> child_blocks;
  GatherChildrenBlocks(axis, &child_blocks);

  for (auto x : operator->()->dep_graph.GetSuccessor(block)) {
    if (!child_blocks.count(x)) {
      LOG(FATAL) << "This block cannot compute at this point because some other "
          "blocks outside the scope of this point are also dependent on this block.";
    }
  }

  // todo(lmzheng): check compute_at across reduction axis for WAW

  // gather required bound for all outputs
  Array<Tensor> output_tensors;
  for (const auto x : block->outputs) {
    output_tensors.push_back(x->data);
  }
  Array<Array<IntSet> > ranges = GatherRegion(output_tensors, axis);

  // copy domain for reduction axis
  StdNodeMap<Var, Range> dom_map;
  ScheduleTreeNode lca = LeastCommonAncestor(block, axis);
  ScheduleTreeNode now = block;
  while (now != lca) {
    if (const AxisTreeNodeNode* n = now.as<AxisTreeNodeNode>()) {
      dom_map[n->loop_var] = Range::make_by_min_extent(n->min, n->extent);
    }
    now = operator->()->father_map[now];
  }

  // solve range for vars
  Array<IntSet> produces;
  for (const auto x : block->outputs) {
    for (const auto ran : x->ranges) {
      produces.push_back(IntSet::range(ran));
    }
  }

  Array<IntSet> iter_domain = arith::SolveCover(block->vars, produces, Flatten2DArray(ranges));

  // generate for AxisNodes
  std::vector<Expr> new_args(iter_domain.size());
  ScheduleTreeNode last = block;
  for (int i = static_cast<int>(iter_domain.size()) - 1; i >= 0; --i) {
    Var iter_var("ax" + std::to_string(i));
    AxisTreeNode node;

    if (!iter_domain[i].defined()) {  // for unsolvable reduction axes, copy it from old trees
      CHECK(block->args[i].get()->is_type<Variable>()) << "Unsolvable reduction iteration domain "
                                                          "can only contains single variable";
      // todo(lmzheng): check it is reduction axis
      node = AxisTreeNodeNode::make(iter_var,
                                    dom_map[iter_var]->min,
                                    dom_map[iter_var]->extent,
                                    kOpaque,
                                    Array<ScheduleTreeNode>{last});
      new_args[i] = block->args[i];
    }

    if (const arith::IntervalSet* set = iter_domain[i].as<arith::IntervalSet>()) {
      node = AxisTreeNodeNode::make(iter_var,
                                    set->i.min,
                                    set->i.max - set->i.min,
                                    kOpaque, // todo(lmzheng): fill correct type to replace kOpaque x 3
                                    Array<ScheduleTreeNode>{last});
      new_args[i] = iter_var;
    } else if (const arith::StrideSet* set = iter_domain[i].as<arith::StrideSet>()) {
      CHECK(set->extents.size() == 1);
      CHECK(set->base.is_single_point());
      if (!is_zero(set->extents[0])) {
        node = AxisTreeNodeNode::make(iter_var,
                                      0,
                                      set->extents[0],
                                      kOpaque,
                                      Array<ScheduleTreeNode>{last});
        new_args[i] = iter_var * set->strides[0] + set->base.min;
      } else {
        new_args[i] = set->base.min;
        node = AxisTreeNode(nullptr);
      }
    } else {
      LOG(FATAL) << "Cannot handle int set : " << iter_domain[i];
    }

    if (node.defined()) {
      UpdateFather(node);
      last = node;
    }
  }
  block->args = new_args;

  // relink
  father->children.Remove(block);
  if (father->children.size() == 0) {
    RemoveLeaf(father);
  }

  ArrayNode* new_children = axis->children.CopyOnWrite();
  new_children->data.insert(new_children->data.begin(), last.node_);
  UpdateFather(axis);

  return block;
}

ScheduleTreeNode Schedule::LeastCommonAncestor(ScheduleTreeNode a, ScheduleTreeNode b) const {
  StdNodeSet<ScheduleTreeNode> father_set;

  ScheduleTreeNode now = a;
  while (now != operator->()->root) {
    father_set.insert(a);
    now = operator->()->father_map[now];
  }

  now = b;
  while (now != operator->()->root) {
    if (father_set.count(now)) {
      return now;
    }
    now = operator->()->father_map[now];
  }

  return operator->()->root;
}

// dependency analysis
Array<Array<IntSet> > Schedule::GatherRegion(Array<Tensor> tensors, AxisTreeNode axis) const {
  std::unordered_map<const Variable*, IntSet> dom_map;

  // set father as a single point
  ScheduleTreeNode father = axis;
  while (father != operator->()->root) {
    if (const AxisTreeNodeNode* n = father.as<AxisTreeNodeNode>()) {
      dom_map[n->loop_var.get()] = IntSet::single_point(n->loop_var);
    }
    father = operator->()->father_map[father];
  }

  for (auto x : axis->children) {
    GatherVarDomain(x, &dom_map);
  }

  StdNodeSet<BlockTreeNode> blocks;
  GatherChildrenBlocks(axis, &blocks);

  Array<Array<IntSet> > ret;

  // for all tensors, compute required regions
  for (const auto& tensor : tensors) {
    std::vector<Array<IntSet> > isets_to_merge(tensor.ndim());

    for (const auto& block : blocks) {
      // replace formal parameters with actual parameters
      std::unordered_map<const Variable*, Expr> arg_map;
      for (size_t i = 0; i < block->vars.size(); ++i) {
        arg_map[block->vars[i].get()] = block->args[i];
      }

      for (const auto& read : block->inputs) {
        if (read->data == tensor) {
          for (size_t i = 0; i < tensor.ndim(); ++i) {
            Range real = Range::make_by_min_extent(Substitute(read->ranges[i]->min, arg_map),
                                                   Substitute(read->ranges[i]->extent, arg_map));
            IntSet o = arith::EvalSet(real, dom_map);
            isets_to_merge[i].push_back(o);
          }
        }
      }
    }

    Array<IntSet> required_region;
    for (size_t i = 0; i < tensor.ndim(); ++i) {
      IntSet tmp = arith::Union(isets_to_merge[i]);
      required_region.push_back(tmp);
    }
    ret.push_back(required_region);
  }

  return ret;
}

// tree manipulation (requires father map)
void Schedule::UpdateFather(ScheduleTreeNode father, bool recursive) {
  for (auto x : father->children) {
    operator->()->father_map.Set(x, father);

    if (recursive) {
      UpdateFather(x, recursive);
    }
  }
}

void Schedule::RemoveLeaf(ScheduleTreeNode node) {
  ScheduleTreeNode next;

  CHECK(node->children.size() == 0 && node != operator->()->root);

  // find
  next = operator->()->father_map[node];
  while (next != operator->()->root && next->children.size() == 1) {
    node = next;
    next = operator->()->father_map[node];
    operator->()->father_map.erase(node);
  }

  // destroy to remove circular reference
  next->children.Remove(node);
}

// tree utility that do not require father map
void GatherChildrenBlocks(ScheduleTreeNode node, StdNodeSet<BlockTreeNode> *ret) {
  if (const AxisTreeNodeNode* n = node.as<AxisTreeNodeNode>()) {
    for (auto x : n->children) {
      GatherChildrenBlocks(x, ret);
    }
  } else if (const BlockTreeNodeNode* n = node.as<BlockTreeNodeNode>()) {
    ret->insert(GetRef<BlockTreeNode>(n));
  } else {
    LOG(FATAL) << "Internal Error";
  }
}

void GatherVarDomain(ScheduleTreeNode node,
                     std::unordered_map<const Variable*, arith::IntSet>* dom_map) {
  if (const AxisTreeNodeNode* n = node.as<AxisTreeNodeNode>()) {
    (*dom_map)[n->loop_var.get()] = IntSet::range(Range::make_by_min_extent(n->min, n->extent));
    for (auto x : n->children) {
      GatherVarDomain(x, dom_map);
    }
  } else if (const BlockTreeNodeNode* n = node.as<BlockTreeNodeNode>()) {
    CHECK(n != nullptr); // useless check to eliminate warnings
  }
}

} // namespace tensorir
} // namespace tvm
