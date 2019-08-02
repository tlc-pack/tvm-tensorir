/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Schedule operations
 */

#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/arithmetic.h>
#include <tvm/build_module.h>
#include <tvm/packed_func_ext.h>
#include <tvm/operation.h>
#include "../arithmetic/int_set_internal.h"
#include "schedule.h"
#include "tree_builder.h"
#include "util.h"
#include "../arithmetic/int_set.h"

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

ScheduleTreeNode SubstituteArgOnly(ScheduleTreeNode node, const Map<Var, Expr>& vmap);
Array<ScheduleTreeNode> SubstituteArgOnly(Array<ScheduleTreeNode> nodes, const Map<Var, Expr>& vmap);
ScheduleTreeNode SubstituteAndEquationSimplify(ScheduleTreeNode node, const Map<Var, Expr>& vmap,
                                               arith::Analyzer* analyzer);

// maker
Schedule ScheduleNode::make(Stmt stmt) {
  return TreeBuilder().Build(stmt);
}

// accessor
Array<BlockTreeNode> Schedule::blocks() const {
  return operator->()->block_list;
}

Array<AxisTreeNode> Schedule::axis(ScheduleTreeNode block) const {
  Array<AxisTreeNode> ret;
  const FatherMap& father_map = operator->()->father_map;

  const AxisTreeNodeNode* now = father_map[block].as<AxisTreeNodeNode>();
  const AxisTreeNodeNode* root = operator->()->root.as<AxisTreeNodeNode>();

  while (now != root) {
    ret.push_back(GetRef<AxisTreeNode>(now));
    now = father_map[GetRef<AxisTreeNode>(now)].as<AxisTreeNodeNode>();
  }

  return Array<AxisTreeNode>(ret.rbegin(), ret.rend()); // reverse
}

// schedule primitives
Array<AxisTreeNode> Schedule::split(AxisTreeNode axis, Expr factor) {
  Var outer_var = axis->loop_var.copy_with_suffix(".outer");
  Var inner_var = axis->loop_var.copy_with_suffix(".inner");

  Expr outer_min = axis->min;
  Expr outer_extent = (axis->extent + factor - 1) / factor;

  Expr inner_min = 0;
  Expr inner_extent = factor;

  Map<Var, Expr> vmap;
  vmap.Set(axis->loop_var, outer_var * factor + inner_var);

  AxisTreeNode inner_node = AxisTreeNodeNode::make(inner_var, inner_min, inner_extent,
                                                   axis->axis_type,
                                                   SubstituteArgOnly(axis->children, vmap));
  UpdateFather(inner_node);

  AxisTreeNode outer_node = AxisTreeNodeNode::make(outer_var, outer_min, outer_extent,
                                                   axis->axis_type,
                                                   Array<ScheduleTreeNode>{inner_node});
  UpdateFather(outer_node);

  // relink
  ReplaceChild(axis, outer_node);

  return Array<AxisTreeNode>{outer_node, inner_node};
}

Array<AxisTreeNode> Schedule::split_nparts(AxisTreeNode axis, Expr nparts) {
  return split(axis, (axis->extent + nparts - 1) / nparts);
}

AxisType FuseAxisType(AxisType t1, AxisType t2) {
  switch (t1) {
    case kSpace:
      switch (t2) {
        case kSpace: return kSpace;
        case kReduce: return kMix;
        default : return kOpaque;
      };
    case kReduce:
      switch (t2) {
        case kSpace: return kMix;
        case kReduce: return kReduce;
        default: return kOpaque;
      };
    default: return kOpaque;
  }
}

AxisTreeNode Schedule::fuse(AxisTreeNode outer, AxisTreeNode inner) {
  // Can only fuse neighbor axes without any extra branches.
  // Future Enhancement: this condition can be eliminated by lifting all siblings of inner
  // as the children of the father of outer
  CHECK(operator->()->father_map[inner] == outer);
  CHECK(outer->children.size() == 1 && outer->children[0] == inner);

  Expr min = 0;
  Expr extent = outer->extent * inner->extent;

  Var fused_var = outer->loop_var.copy_with_suffix("." + inner->loop_var.get()->name_hint + ".fused");

  Map<Var, Expr> vmap;
  vmap.Set(outer->loop_var, fused_var / inner->extent + outer->min);
  vmap.Set(inner->loop_var, fused_var % inner->extent + inner->min);

  AxisTreeNode fused_node = AxisTreeNodeNode::make(fused_var, min, extent,
                                                   FuseAxisType(outer->axis_type, inner->axis_type),
                                                   SubstituteArgOnly(inner->children, vmap));
  UpdateFather(fused_node);

  // relink
  ReplaceChild(outer, fused_node);

  return fused_node;
}

Array<AxisTreeNode> Schedule::reorder(Array<AxisTreeNode> axises) {
  // Get the depth of each axises
  std::map<int, AxisTreeNode> depth;
  for (auto axis : axises) {
    int axis_depth = 0;
    ScheduleTreeNode node = axis;
    while (node != operator->()->root) {
      ++axis_depth;
      node = operator->()->father_map[node];
    }
    CHECK_EQ(depth.count(axis_depth), 0) << "try to reorder two axises with same depth";
    depth[axis_depth] = axis;
  }

  Array<AxisTreeNode> origin;
  for (auto x : depth) {
    origin.push_back(x.second);
  }

  // reorder axises
  for (int i = axises.size() - 1; i >= 0; --i) {
    auto axis = axises[i];
    auto origin_axis = origin[i];
    if (origin_axis != axis) {
      auto tmp = binary_reorder(axis, origin_axis);
      auto new_outer = tmp[0], new_inner = tmp[1];
      for (int j = origin.Index(axis); j < i - 1; ++j) {
        origin.Set(j, origin[j + 1]);
      }
      origin.Set(i - 1, new_outer);
      origin.Set(i, new_inner);

      axises.Set(i, new_inner);
      axises.Set(axises.Index(origin_axis), new_outer);
    }
  }
  return origin;
}

Array<AxisTreeNode> Schedule::binary_reorder(AxisTreeNode outer, AxisTreeNode inner) {
  // Just lower the outer axis under the inner axis.
  // If there are extra branches, build a new outer axis above the branches
  AxisTreeNode root = operator->()->root;
  CHECK(outer != root) << "Cannot reorder root axis";
  ScheduleTreeNode now = inner;
  const auto& father_map = operator->()->father_map;
  while (father_map[now] != outer && now != root) {
    now = father_map[now];
  }
  CHECK(now != root) << "outer node is not the ancestor of the inner node";
  AxisTreeNode new_outer;

  // Update nodes in the path
  ScheduleTreeNode last = inner;
  now = inner;
  while (now != father_map[outer]) {
    if (now == inner || now->children.size() > 1) {
      for (const auto& child : now->children) {
        if (child != last) {
          AxisTreeNode new_node = AxisTreeNodeNode::make(outer->loop_var, outer->min, outer->extent, outer->axis_type, Array<ScheduleTreeNode>{child});
          ReplaceChild(child, new_node);
          UpdateFather(new_node);
          if (now == inner) {
            new_outer = new_node;
          }
        }
      }
    }
    last = now;
    now = father_map[now];
  }
  CHECK(now == father_map[outer]);

  auto index = now->children.Index(outer);
  if (outer->children.size() == 1) {
    now->children.Set(index, outer->children[0]);
  }
  else {
    size_t offset = outer->children.size() - 1;
    for (size_t i = 0; i < offset; ++i) {
      now->children.push_back(outer->children[i]);
    }
    for (size_t i = now->children.size() - 1 - offset; i > index; --i) {
      now->children.Set(i + offset, now->children[i]);
    }
    for (size_t i = 0; i < outer->children.size(); ++i) {
      now->children.Set(i + index, outer->children[i]);
    }
  }
  operator->()->father_map.erase(outer);
  UpdateFather(now);
  return Array<AxisTreeNode>{inner, new_outer};
}

// Inline a single provide
class StatementInliner : public IRMutator {
 public:
  StatementInliner(const Provide* op) : op_(op) {

    Set<Var> old_space_vars;

    for (size_t i = 0; i < op->args.size(); ++i) {
      Var var = Var("i" + std::to_string(i));
      vars_.push_back(var);
      analyzer.Bind(var, op->args[i]);
      old_space_vars.insert(GatherVars(op->args[i]));
    }

    value_ = analyzer.equation_simplify(op->value);
    Set<Var> simplified_vars = GatherVars(value_);

    // check simplification result : all old space vars should be simplified out
    CHECK(simplified_vars.Intersection(old_space_vars).size() == 0)
      << "Equation simplification fails to simplify all index vars: " << value_;
  };

  Expr Mutate_(const Call* op, const Expr& e) {
    if (op->call_type == Call::CallType::Halide && op->func == op_->func
        && op->value_index == op_->value_index) {
      StdNodeMap<Var, Expr> vmap;
      for (size_t i = 0; i < op->args.size(); ++i) {
        vmap[vars_[i]] = op->args[i];
      }
      return Substitute(value_, vmap);
    }
    return IRMutator::Mutate_(op, e);
  }

 private:
  const Provide* op_;
  Array<Var> vars_;
  Expr value_;
  arith::Analyzer analyzer;
};

void Schedule::compute_inline(BlockTreeNode block) {
  // conditions:
  // 1. only write to one element
  // 2. is terminal block
  // -> The inner stmt is a Provide

  CHECK(block->stmt->is_type<Provide>())
    << "Can only inline single assignment statement";
  CHECK_EQ(block->outputs.size(), 1)
    << "Can only inline statement with one output";  // Note: this can be relaxed.

  RemoveLeaf(block);
  operator->()->block_list.Remove(block);

  // check validity of inline : don't exist a dst that has both WAW and RAW relation
  // (e.g. cannot inline init statement to reduction statement)
  StdNodeMap<BlockTreeNode, std::set<EdgeType> > dst_edges;
  for (const auto& x : operator->()->dep_graph->forward_edges[block]) {
    dst_edges[x->dst].insert(x->type);
  }
  for (auto x : dst_edges) {
    if (x.second.count(kRAW) && x.second.count(kWAW)) {
      LOG(FATAL) << "Cannot inline this statements due to dependency constraint with: " << x.first->stmt;
    }
  }

  for (auto& x : operator->()->dep_graph->forward_edges[block]) {
    BlockTreeNode dst = x->dst;
    dst->stmt = StatementInliner(block->stmt.as<Provide>()).Mutate(x->dst->stmt);

    Array<TensorRegion> new_inputs = CreateInputRegions(dst->stmt);

    // Update vars : remove useless vars
    Array<Var> new_vars;
    Array<Expr> new_args;
    Set<Var> all_vars = GatherVars(dst->stmt);
    for (size_t i = 0; i < dst->vars.size(); ++i) {
      if (all_vars.count(dst->vars[i])) {
        new_vars.push_back(dst->vars[i]);
        new_args.push_back(dst->args[i]);
      }
    }

    dst->inputs = new_inputs;
    dst->vars = new_vars;
    dst->args = new_args;
  }

  // update in dependency graph
  operator->()->dep_graph.InlineNode(block);
}

Array<ScheduleTreeNode> Schedule::unroll(AxisTreeNode axis) {
  CHECK(axis->min->is_type<IntImm>()) << "Cannot unroll non-const loop";
  CHECK(axis->extent->is_type<IntImm>()) << "Cannot unroll non-const loop";

  Array<ScheduleTreeNode> expanded_stmts;

  // expand statements, replace var
  int64_t min_v = axis->min.as<IntImm>()->value, extent_v = axis->extent.as<IntImm>()->value;
  StdNodeMap<Var, Expr> var_map;
  for (int64_t v = min_v; v < min_v + extent_v; v++) {
    var_map[axis->loop_var] = IntImm::make(axis->loop_var.type(), v);
    for (size_t i = 0; i < axis->children.size(); i++) {
      expanded_stmts.push_back(SubstituteArgOnly(axis->children[i].Copy(), var_map));
    }
  }

  // relink
  ScheduleTreeNode father = operator->()->father_map[axis];
  Array<ScheduleTreeNode> father_new_children;
  size_t index = father->children.Index(axis);
  for (size_t i = 0; i < father->children.size(); ++i) {
    if (i == index) {
      for (size_t j = 0; j < expanded_stmts.size(); ++j) {
        father_new_children.push_back(expanded_stmts[j]);
      }
    } else {
      father_new_children.push_back(father->children[i]);
    }
  }
  father->children = father_new_children;
  UpdateFather(father, false);

  return expanded_stmts;
}

BlockTreeNode Schedule::compute_after(BlockTreeNode block, AxisTreeNode axis) {
  // todo (syfeng):
  // 1. check dependency using operator->()->DependencyGraph
  //    a) the iteration domains of all parent axes match
  //    b) all regions required by read is already produced
  //    c) cannot move across WAW relation
  // 2. move tree nodes
  return BlockTreeNode(nullptr);
}

// Return whether the subtree rooted by node has any block in `set`
bool FindAny(ScheduleTreeNode node, Set<BlockTreeNode> set) {
  if (const AxisTreeNodeNode* n = node.as<AxisTreeNodeNode>()) {
    for (const auto& x : n->children) {
      if (FindAny(x, set)) {
        return true;
      }
    }
  } else if (const BlockTreeNodeNode* n = node.as<BlockTreeNodeNode>()) {
    return static_cast<bool>(set.count(GetRef<BlockTreeNode>(n)));
  }
  return false;
}

BlockTreeNode Schedule::compute_at(BlockTreeNode block, AxisTreeNode axis) {
  ScheduleTreeNode father = operator->()->father_map[block];

  // check dependency:
  // 1. all successors are at the subtree
  // 2. WAW cannot across reduce axis
  StdNodeSet<BlockTreeNode> child_blocks;
  GatherChildrenBlocks(axis, &child_blocks);

  const auto& predecessors = operator->()->dep_graph.GetPredecessor(block);
  const auto& successor = operator->()->dep_graph.GetSuccessor(block);

  for (auto x : successor) {
    if (!child_blocks.count(x)) {
      LOG(FATAL) << "This block cannot compute at this point because some other "
                    "blocks outside the scope of this point are also dependent on this block.";
    }
  }
  // todo(lmzheng): check compute_at across reduction axis for WAW

  // find insert position : after all predecessors in dependency graph and before all successors in dep graph.
  int after_pos, before_pos;
  for (after_pos = static_cast<int>(axis->children.size()) - 1; after_pos >= 0; after_pos--) {
    if (FindAny(axis->children[after_pos], predecessors)) {
      break;
    }
  }
  after_pos++;
  for (before_pos = 0; before_pos < static_cast<int>(axis->children.size()); before_pos++) {
    if (FindAny(axis->children[before_pos], successor)) {
      break;
    }
  }
  if (after_pos > before_pos) {
    LOG(FATAL) << "Cannot satisfy dependency";
  }

  // gather required bound for all outputs
  Array<Tensor> output_tensors;
  for (const auto x : block->outputs) {
    output_tensors.push_back(x->data);
  }
  Array<Array<IntSet> > ranges = GatherRegion(output_tensors, axis, after_pos);
  auto& realize_region = operator->()->raw_realize_region;

  Array<ScheduleTreeNode> dependent_blocks;
  for (const auto& block_tree_node : blocks()) {
    if (block == block_tree_node) continue;
    bool finished = false;
    for (const auto& tensor_region : block_tree_node->inputs) {
      if (finished) break;
      for (const auto& tensor : output_tensors) {
        if (tensor->op == tensor_region->data->op) {
          dependent_blocks.push_back(block_tree_node);
          finished = true;
          break;
        }
      }
    }
  }
  dependent_blocks.push_back(axis);
  ScheduleTreeNode _lca = LowestCommonAncestor(dependent_blocks, true);

  std::unordered_map<const Variable*, IntSet> reduction_dom_map;
  for (ScheduleTreeNode node = axis; node != _lca; node = operator->()->father_map[node]) {
    if (const AxisTreeNodeNode* n = node.as<AxisTreeNodeNode>()) {
      reduction_dom_map[n->loop_var.get()] = IntSet::range(Range::make_by_min_extent(n->min, n->extent));
    }
  }

  for (size_t i = 0; i < output_tensors.size(); ++i) {
    Region region;
    const auto& tensor = output_tensors[i];
    for (const auto& int_set : ranges[i]) {
      Range real = Range::make_by_min_extent(int_set.min(), int_set.max() - int_set.min() + 1);
      IntSet o = arith::EvalSet(real, reduction_dom_map);
      if (const arith::IntervalSetNode* set = o.as<arith::IntervalSetNode>()) {
        region.push_back(Range::make_by_min_extent(set->min_value, set->max_value - set->min_value + 1));
      }
      else {
        LOG(FATAL);
      }
    }
    realize_region[tensor] = region;
  }
  // copy domain for reduction axis
  StdNodeMap<Var, Range> dom_map;
  ScheduleTreeNode lca = LowestCommonAncestor(Array<ScheduleTreeNode>{block, axis}, false);
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
      const auto * variable = block->args[i].as<Variable>();
      Var var(variable->GetNodePtr());
      node = AxisTreeNodeNode::make(var,
                                    dom_map[var]->min,
                                    dom_map[var]->extent,
                                    kOpaque,
                                    Array<ScheduleTreeNode>{last});
      new_args[i] = block->args[i];

    } else if (const arith::IntervalSetNode* set = iter_domain[i].as<arith::IntervalSetNode>()) {
      node = AxisTreeNodeNode::make(iter_var,
                                    set->min_value,
                                    set->max_value - set->min_value + 1,
                                    kOpaque, // todo(lmzheng): fill correct type to replace kOpaque x 3
                                    Array<ScheduleTreeNode>{last});
      new_args[i] = iter_var;
    } else if (const arith::StrideSetNode* set = iter_domain[i].as<arith::StrideSetNode>()) {
      CHECK(set->extents.size() == 1);
      CHECK(is_one(set->base_extent));
      if (is_one(set->extents[0])) {
        node = AxisTreeNode(nullptr);
        new_args[i] = set->base_min;
      } else {
        node = AxisTreeNodeNode::make(iter_var,
                                      0,
                                      set->extents[0],
                                      kOpaque,
                                      Array<ScheduleTreeNode>{last});
        new_args[i] = iter_var * set->strides[0] + set->base_min;
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

  // insert to father's children list
  ArrayNode* new_children = axis->children.CopyOnWrite();
  new_children->data.insert(new_children->data.begin() + after_pos, last.node_);
  UpdateFather(axis);

  return block;
}

BlockTreeNode Schedule::compute_root(BlockTreeNode block) {
  return compute_at(block, operator->()->root);
}

// Build domain map context for an axis.
void BuildDomMapContext(const FatherMap& father_map,
                        AxisTreeNode root,
                        AxisTreeNode axis,
                        std::unordered_map<const Variable*, IntSet>* dom_map) {
  // upwards : set them as single points
  ScheduleTreeNode father = father_map[root];
  while (father != root) {
    if (const AxisTreeNodeNode* n = father.as<AxisTreeNodeNode>()) {
      (*dom_map)[n->loop_var.get()] = IntSet::single_point(n->loop_var);
    }
    father = father_map[father];
  }

  // self and downwards : set them as interval
  GatherVarDomain(axis, dom_map);
}

BlockTreeNode Schedule::blockize(AxisTreeNode axis) {
  // gather output regions
  Set<Var> used_vars, deleted_vars;
  StdNodeMap<Tensor, std::vector<std::vector<IntSet> > > gathered_output_regions;
  StdNodeMap<Tensor, std::vector<std::vector<IntSet> > > gathered_input_regions;
  std::vector<Tensor> raw_output_order; // To keep the original order of input/output arguments
  std::vector<Tensor> raw_input_order;  //

  std::unordered_map<const Variable*, IntSet> dom_map;
  BuildDomMapContext(operator->()->father_map, operator->()->root, axis, &dom_map);

  std::function<void(ScheduleTreeNode)> gather_vars_and_regions;
  gather_vars_and_regions = [&gather_vars_and_regions, &used_vars, &deleted_vars, &dom_map,
                             &gathered_input_regions, &raw_input_order,
                             &gathered_output_regions, &raw_output_order]
      (ScheduleTreeNode node) {
    if (const AxisTreeNodeNode* n = node.as<AxisTreeNodeNode>()) {
      for (auto x : n->children) {
        gather_vars_and_regions(x);
      }
      used_vars.insert(GatherVars(n->min));
      used_vars.insert(GatherVars(n->extent));
      deleted_vars.insert(n->loop_var);
    } else if (const BlockTreeNodeNode* n = node.as<BlockTreeNodeNode>()) {
      StdNodeMap<Var, Expr> var_map;

      for (size_t i = 0; i < n->vars.size(); ++i) {
        var_map[n->vars[i]] = n->args[i];
        used_vars.insert(GatherVars(n->args[i]));
      }

      for (const TensorRegion& tensor: n->inputs) {
        std::vector<IntSet> ranges;
        for (const Range& range: tensor->ranges) {
          ranges.push_back(arith::EvalSet(
              Range::make_by_min_extent(Substitute(range->min, var_map),
                                        Substitute(range->extent, var_map)), dom_map));
        }
        if (!gathered_input_regions.count(tensor->data)) {
          raw_input_order.push_back(tensor->data);
        }
        gathered_input_regions[tensor->data].push_back(ranges);
      }

      for (const TensorRegion& tensor: n->outputs) {
        std::vector<IntSet> ranges;
        for (const Range& range: tensor->ranges) {
          ranges.push_back(arith::EvalSet(
              Range::make_by_min_extent(Substitute(range->min, var_map),
                                        Substitute(range->extent, var_map)), dom_map));
        }
        if (!gathered_output_regions.count(tensor->data)) {
          raw_output_order.push_back(tensor->data);
        }
        gathered_output_regions[tensor->data].push_back(ranges);
      }
    } else {
      LOG(FATAL) << "Internal Error";
    }
  };
  gather_vars_and_regions(axis);

  Array<TensorRegion> raw_outputs;
  for (const auto& tensor : raw_output_order) {
    Array<Range> ranges;
    for (size_t i = 0; i < tensor.ndim(); ++i) {
      Array<IntSet> to_merge;
      for (const auto& y : gathered_output_regions[tensor]) {
        to_merge.push_back(y[i]);
      }
      IntSet merged = arith::Union(to_merge);
      ranges.push_back(Range::make_by_min_extent(merged.min(), merged.max() - merged.min() + 1));
    }
    raw_outputs.push_back(TensorRegionNode::make(tensor, ranges));
  }

  // canonicalize output regions
  arith::Analyzer analyzer;
  Array<Expr> args;
  Array<Var> vars;
  Array<TensorRegion> outputs;
  Map<Var, Expr> var_map;
  std::tie(args, vars, outputs, var_map) =
    CreateOutputRegions(raw_outputs, used_vars.Difference(deleted_vars), &analyzer);

  // create inputs regions
  Array<TensorRegion> inputs;
  for (const auto tensor : raw_input_order) {
    Array<Range> ranges;
    for (size_t i = 0; i < tensor.ndim(); ++i) {
      Array<IntSet> to_merge;
      for (const std::vector<IntSet>& y : gathered_input_regions[tensor]) {
        const arith::IntervalSetNode* set = y[i].as<arith::IntervalSetNode>();
        CHECK(set != nullptr);
        IntSet b = arith::IntervalSet(
            SubstituteAndEquationSimplify(set->min_value, var_map, &analyzer),
            SubstituteAndEquationSimplify(set->max_value, var_map, &analyzer));
        to_merge.push_back(b);
      }
      IntSet merged = arith::Union(to_merge);
      ranges.push_back(Range::make_by_min_extent(merged.min(), merged.max() - merged.min() + 1));
    }
    inputs.push_back(TensorRegionNode::make(tensor, ranges));
  }

  // make node
  ScheduleTreeNode child = SubstituteAndEquationSimplify(axis, var_map, &analyzer);
  BlockTreeNode ret = BlockTreeNodeNode::make(args, vars, inputs, outputs,
                                              Stmt(nullptr), Array<ScheduleTreeNode>{child});

  // relink
  ReplaceChild(axis, ret);
  UpdateFather(ret, false);

  // Todo(lmzheng) : Update dependency graph, block_list

  return ret;
}

ScheduleTreeNode Schedule::unblockize(BlockTreeNode block) {
  // check it is a block created by schedule primitive blockize
  CHECK(!block->stmt.defined()) << "Cannot unblockize a block that is not created by blockize" << std::endl;
  CHECK_EQ(block->children.size(), 1) << "Can only unblockize a block that is created by blockize" << std::endl;

  ScheduleTreeNode child = block->children[0];

  // expand the block
  StdNodeMap<Var, Expr> replace_map;
  for (size_t i = 0; i < block->args.size(); ++i) {
    replace_map[block->vars[i]] = block->args[i];
  }
  SubstituteArgOnly(child, replace_map);

  // relink
  ReplaceChild(block, child);

  // Todo(lmzheng) : Update dependency graph, block_list

  return child;
}

BlockTreeNode Schedule::tensorize(BlockTreeNode block, TensorIntrinsic intrin) {
  // todo (lmzheng): 1. check whether the shapes match   2. check the semantics

  // get new body
  NodeRef ret = intrin->intrin_func(block->inputs, block->outputs);
  BlockTreeNode new_block;
  if (ret->derived_from<ScheduleTreeNodeNode>()) {
    new_block = BlockTreeNodeNode::make(block->args, block->vars,
                                        block->inputs, block->outputs,
                                        Stmt(NodePtr<Node>(nullptr)),
                                        Array<ScheduleTreeNode>{Downcast<ScheduleTreeNode>(ret)});
  } else if (ret->derived_from<StmtNode>()) {
    new_block = BlockTreeNodeNode::make(block->args, block->vars,
                                        block->inputs, block->outputs,
                                        Downcast<Stmt>(ret), Array<ScheduleTreeNode>{});
  } else {
    LOG(FATAL) << "The intrin func returns invalid value";
  }

  // Relink
  ReplaceChild(block, new_block);

  // todo (lmzheng): update block_list and dependency graph

  return new_block;
}

ScheduleTreeNode Schedule::untensorize(BlockTreeNode block) {
  return ScheduleTreeNode(nullptr);
}

void Schedule::annotate(AxisTreeNode axis, std::string type) {
  if (type == "vectorize") {
    axis->axis_type = AxisType::vectorized;
  } else if (type == "unroll") {
    axis->axis_type = AxisType::unrolled;
  } else {
    LOG(FATAL) << "Unsupported type of " << type;
  }
}

void Schedule::bind(AxisTreeNode axis, IterVar thread_iter) {
  CHECK(thread_iter->iter_type == kThreadIndex)
    << "Cannot rebase by " << axis->loop_var
    << ", only thread axis is allowed so far";
  auto& bind_var = operator->()->bind_var;
  Var old_var = axis->loop_var;
  CHECK(bind_var.count(old_var) == 0) << "The axis has been bind to another axis";

  // Create new loop var
  Var new_var = thread_iter->var;
  std::string attr_type = thread_iter->thread_tag == "vthread" ? attr::virtual_thread : attr::thread_extent;
  bind_var[new_var] = AttrNode::make(thread_iter, attr_type, axis->extent);

  // Replace the axis loop var to the new one
  Map<Var, Expr> bind_map;
  bind_map.Set(old_var, new_var);
  SubstituteArgOnly(axis, bind_map);
  axis->loop_var = new_var;
}

// dependency analysis
Array<Array<IntSet> > Schedule::GatherRegion(Array<Tensor> tensors, AxisTreeNode axis, int start_child_index) const {
  std::unordered_map<const Variable*, IntSet> dom_map;
  std::unordered_map<const Variable*, IntSet> shared_dom_map;

  ScheduleTreeNode father = axis;

  while (father != operator->()->root) {
    if (const AxisTreeNodeNode* n = father.as<AxisTreeNodeNode>()) {
      dom_map[n->loop_var.get()] = IntSet::single_point(n->loop_var);
      const auto& bind_var = operator->()->bind_var;
      const auto& var = n->loop_var;
      if (bind_var.count(var) &&
          (bind_var.at(var)->attr_key == attr::virtual_thread ||
          (bind_var.at(var)->attr_key == attr::thread_extent &&
          (var->name_hint.find("threadIdx.") == 0)))) {
        const auto& attr = operator->()->bind_var.at(var);
        shared_dom_map[n->loop_var.get()] = IntSet::range(Range::make_by_min_extent(0, attr->value));
      }
      else {
        shared_dom_map[n->loop_var.get()] = IntSet::single_point(n->loop_var);
      }
    }
    father = operator->()->father_map[father];
  }

  for (auto x : axis->children) {
    GatherVarDomain(x, &dom_map);
    GatherVarDomain(x, &shared_dom_map);
  }

  StdNodeSet<BlockTreeNode> blocks;
  for (size_t i = start_child_index; i < axis->children.size(); ++i) {
    GatherChildrenBlocks(axis->children[i], &blocks);
  }

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
            CHECK(operator->()->raw_realize_scope.count(tensor->op));
            IntSet o;
            if (operator->()->raw_realize_scope.at(tensor->op) == "shared") {
              o = arith::EvalSet(real, shared_dom_map);
            }
            else {
              o = arith::EvalSet(real, dom_map);
            }
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

  // go upwards until find a father with more than two children
  next = operator->()->father_map[node];
  while (next != operator->()->root && next->children.size() == 1) {
    node = next;
    next = operator->()->father_map[node];
    operator->()->father_map.erase(node);
  }

  // destroy to remove circular reference
  next->children.Remove(node);
}

void Schedule::ReplaceChild(ScheduleTreeNode old_child, ScheduleTreeNode new_child) {
  ScheduleTreeNode father = operator->()->father_map[old_child];
  father->children.Set(father->children.Index(old_child), new_child);
  operator->()->father_map.Set(new_child, father);
}

ScheduleTreeNode Schedule::LowestCommonAncestor(Array<ScheduleTreeNode> nodes, bool inclusive) const {
  // Todo (lmzheng): better alg?
  std::vector<StdNodeSet<ScheduleTreeNode> > father_set;
  ScheduleTreeNode now;

  for (size_t i = 1; i < nodes.size(); ++i) {
    StdNodeSet<ScheduleTreeNode> set;
    now = inclusive ? nodes[i] : operator->()->father_map[nodes[i]];
    while (now != operator->()->root) {
      set.insert(now);
      now = operator->()->father_map[now];
    }
    father_set.push_back(set);
  }

  now = inclusive ? nodes[0] : operator->()->father_map[nodes[0]];
  while (now != operator->()->root) {
    bool all_found = true;
    for (const auto& set : father_set) {
      if (!set.count(now)) {
        all_found = false;
        break;
      }
    }

    if (all_found) {
      return now;
    }
    now = operator->()->father_map[now];
  }

  return now;
}

void Schedule::CheckFatherLink() {
  std::function<void(ScheduleTreeNode)> check_func;

  check_func = [this, &check_func](ScheduleTreeNode node) {
    if (const AxisTreeNodeNode* n = node.as<AxisTreeNodeNode>()) {
      for (auto x : n->children) {
        if (operator->()->father_map[x] != node) {
          std::cerr << "Father link error (f to c):  " << n->loop_var << " to " << x << std::endl;
        }
        check_func(x);
      }
    }
  };

  check_func(operator->()->root);
}

// tree utilities that do not require father map
void GatherChildrenBlocks(ScheduleTreeNode node, StdNodeSet<BlockTreeNode>* ret) {
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
  // todo(@siyuan)
  if (const AxisTreeNodeNode* n = node.as<AxisTreeNodeNode>()) {
    if ((*dom_map).count(n->loop_var.get()) == 0) {
      (*dom_map)[n->loop_var.get()] = IntSet::range(Range::make_by_min_extent(n->min, n->extent));
    }
    for (auto x : n->children) {
      GatherVarDomain(x, dom_map);
    }
  } else if (const BlockTreeNodeNode* n = node.as<BlockTreeNodeNode>()) {
    CHECK(n != nullptr); // useless check to eliminate warnings
  }
}

// Note : inplace substitute
Array<ScheduleTreeNode> SubstituteArgOnly(Array<ScheduleTreeNode> nodes, const Map<Var, Expr>& vmap) {
  for (auto x : nodes) {
    SubstituteArgOnly(x, vmap);
  }
  return nodes;
}

// Note : inplace substitute
ScheduleTreeNode SubstituteArgOnly(ScheduleTreeNode node, const Map<Var, Expr>& vmap) {
  if (const AxisTreeNodeNode* n = node.as<AxisTreeNodeNode>()) {
    AxisTreeNode axis = GetRef<AxisTreeNode>(n); // get ref to remove const qualifier
    axis->min = Substitute(axis->min, vmap);
    axis->extent = Substitute(axis->extent, vmap);
    for (auto x : axis->children) {
      SubstituteArgOnly(x, vmap);
    }
  } else if (const BlockTreeNodeNode* n = node.as<BlockTreeNodeNode>()) {
    BlockTreeNode block = GetRef<BlockTreeNode>(n); // get ref to remove const qualifier
    Array<Expr> args;
    for (const auto& x : block->args) {
      args.push_back(Substitute(x, vmap));
    }
    block->args = args;
  } else {
    LOG(FATAL) << "Invalid node in schedule tree";
  }
  return node;
}

// Note : inplace substitute
ScheduleTreeNode SubstituteAndEquationSimplify(ScheduleTreeNode node, const Map<Var, Expr>& vmap,
                                               arith::Analyzer* analyzer) {
  if (const AxisTreeNodeNode* n = node.as<AxisTreeNodeNode>()) {
    AxisTreeNode axis = GetRef<AxisTreeNode>(n);
    axis->min = SubstituteAndEquationSimplify(axis->min, vmap, analyzer);
    axis->extent = SubstituteAndEquationSimplify(axis->extent, vmap, analyzer);
    for (auto x : axis->children) {
      SubstituteAndEquationSimplify(x, vmap, analyzer);
    }
  } else if (const BlockTreeNodeNode* n = node.as<BlockTreeNodeNode>()) {
    BlockTreeNode block = GetRef<BlockTreeNode>(n);
    Array<Expr> args;
    for (const auto& x : block->args) {
      args.push_back(SubstituteAndEquationSimplify(x, vmap, analyzer));
    }
    block->args = args;
  } else {
    LOG(FATAL) << "Invalid node in schedule tree";
  }
  return node;
}

} // namespace tensorir
} // namespace tvm
