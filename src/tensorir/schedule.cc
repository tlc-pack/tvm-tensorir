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
#include <map>
#include <set>
#include <vector>
#include "schedule.h"
#include "tree_builder.h"
#include "util.h"
#include "../arithmetic/int_set.h"

namespace tvm {

namespace arith {
// TODO(lmzheng): move this section to arithmetic.h when this is stable

/*!
 * \brief Return int sets with minimal sizes to make `produce` cover `requirement`
 * \param vars The unknown vars to solve
 * \param produces The produced int sets
 * \param requirements The required int sets
 * \return sets The set of unknown vars
 */
Array<IntSet> SolveCover(Array<Var> vars, Array<IntSet> produces, Array<IntSet> requirements);

/*!
 * \brief Return int sets with maximum sizes to make `consume` covered by `requirement`
 * \param vars The unknown vars to solve
 * \param produces The produced int sets
 * \param requirements The required int sets
 * \return sets The set of unknown vars
 */
Array<IntSet> SolveCoverBy(Array<Var> vars, Array<IntSet> produces, Array<IntSet> requirements);
}

namespace tensorir {

using arith::IntSet;

// Return all first-level block nodes under the `node`
void GatherChildrenBlocks(ScheduleTreeNode node, StdNodeSet<BlockTreeNode>* ret);

// Gather domain information for all variables under the `node`
void GatherVarDomain(ScheduleTreeNode node,
                     std::unordered_map<const Variable*, arith::IntSet>* dom_map);

// Substitute variables with expressions.
// When meeting a block, this function only replace args and stops
// Note: This does inplace substitution
ScheduleTreeNode SubstituteArgOnly(ScheduleTreeNode node, const Map<Var, Expr>& vmap);
Array<ScheduleTreeNode> SubstituteArgOnly(Array<ScheduleTreeNode> nodes,
                                          const Map<Var, Expr>& vmap);
ScheduleTreeNode SubstituteTensorOnly(ScheduleTreeNode node,
                                      const StdNodeMap<Tensor, Tensor>& vmap,
                                      bool read, bool write);


// Substitute variables with expression and do equation simplification
// using the information from analyzer.
// When meeting a block, this function only replace args and stops
// Note: This does inplace substitution
ScheduleTreeNode SubstituteAndEquationSimplify(ScheduleTreeNode node,
                                               const Map<Var, Expr>& vmap,
                                               arith::Analyzer* analyzer);

// maker
Schedule ScheduleNode::make(Stmt stmt) {
  return TreeBuilder().Build(stmt);
}

// accessor
Array<BlockTreeNode> Schedule::blocks() const {
  return operator->()->block_list;
}

Array<BlockTreeNode> Schedule::reduction_blocks() const {
  Array<BlockTreeNode> ret;
  for (const auto& block : operator->()->block_list) {
    // TODO(lmzheng) : the condition can be stricter
    for (size_t i = 0; i < block->outputs.size(); ++i) {
      for (size_t j = 0; j < block->inputs.size(); ++j) {
        if (block->outputs[i]->data == block->inputs[j]->data) {
          bool match = true;
          for (size_t k = 0; k < block->outputs[i]->ranges.size(); ++k) {
            if (!is_zero(Simplify(block->outputs[i]->ranges[k]->min
                                  - block->outputs[j]->ranges[k]->min)) ||
                !is_zero(Simplify(block->outputs[i]->ranges[k]->extent
                                  - block->outputs[j]->ranges[k]->extent))) {
              match = false;
              break;
            }
          }

          if (match) {
            ret.push_back(block);
            continue;
          }
        }
      }
    }
  }
  return ret;
}

Array<BlockTreeNode> Schedule::output_blocks() const {
  // FIXME(lmzheng) : add concept of output tensor and fix this function
  return Array<BlockTreeNode>{operator->()->block_list.back()};
}

Array<BlockTreeNode> Schedule::successor(BlockTreeNode block) const {
  Set<BlockTreeNode> tmp = operator->()->dep_graph.GetSuccessor(block);
  return Array<BlockTreeNode>(tmp.begin(), tmp.end());
}

Array<BlockTreeNode> Schedule::predecessor(BlockTreeNode block) const {
  Set<BlockTreeNode> tmp = operator->()->dep_graph.GetPredecessor(block);
  return Array<BlockTreeNode>(tmp.begin(), tmp.end());
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

  return Array<AxisTreeNode>(ret.rbegin(), ret.rend());  // reverse
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
      }
    case kReduce:
      switch (t2) {
        case kSpace: return kMix;
        case kReduce: return kReduce;
        default: return kOpaque;
      }
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

  Var fused_var = outer->loop_var.copy_with_suffix(
      "." + inner->loop_var.get()->name_hint + ".fused");

  Map<Var, Expr> vmap;
  vmap.Set(outer->loop_var, fused_var / inner->extent + outer->min);
  vmap.Set(inner->loop_var, fused_var % inner->extent + inner->min);

  AxisTreeNode fused_node = AxisTreeNodeNode::make(
      fused_var, min, extent,
      FuseAxisType(outer->axis_type, inner->axis_type),
      SubstituteArgOnly(inner->children, vmap));
  UpdateFather(fused_node);

  // relink
  ReplaceChild(outer, fused_node);

  return fused_node;
}

Array<AxisTreeNode> Schedule::reorder(Array<AxisTreeNode> axes) {
  // Get the depth of each axes
  std::map<int, AxisTreeNode> depth;
  for (auto axis : axes) {
    int axis_depth = 0;
    ScheduleTreeNode node = axis;
    while (node != operator->()->root) {
      ++axis_depth;
      node = operator->()->father_map[node];
    }
    CHECK_EQ(depth.count(axis_depth), 0) << "try to reorder two axes with same depth";
    depth[axis_depth] = axis;
  }

  Array<AxisTreeNode> origin;
  for (auto x : depth) {
    origin.push_back(x.second);
  }

  // reorder axes
  for (int i = static_cast<int>(axes.size()) - 1; i >= 0; --i) {
    auto axis = axes[i];
    auto origin_axis = origin[i];
    if (origin_axis != axis) {
      auto tmp = binary_reorder(axis, origin_axis);
      auto new_outer = tmp[0], new_inner = tmp[1];
      for (int j = origin.Index(axis); j < i - 1; ++j) {
        origin.Set(j, origin[j + 1]);
      }
      origin.Set(i - 1, new_outer);
      origin.Set(i, new_inner);

      axes.Set(i, new_inner);
      axes.Set(axes.Index(origin_axis), new_outer);
    }
  }
  return origin;
}

Array<AxisTreeNode> Schedule::binary_reorder(AxisTreeNode outer, AxisTreeNode inner) {
  // Just lower the outer axis under the inner axis.
  // If there are extra branches, build a new outer axis above the branches
  AxisTreeNode root = operator->()->root;
  const auto &father_map = operator->()->father_map;
  CHECK(outer != root) << "Cannot reorder root axis";
  CHECK(IsAncestor(outer, inner)) << "outer node is not the ancestor of the inner node";
  AxisTreeNode new_outer;

  // Update nodes in the path
  ScheduleTreeNode last = inner, now = inner;
  while (now != father_map[outer]) {
    if (now == inner || now->children.size() > 1) {
      for (const auto& child : now->children) {
        if (child != last) {
          AxisTreeNode new_node = AxisTreeNodeNode::make
              (outer->loop_var, outer->min,
               outer->extent, outer->axis_type, Array<ScheduleTreeNode>{child});
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

  ReplaceChild(outer, outer->children);
  operator->()->father_map.erase(outer);
  return Array<AxisTreeNode>{inner, new_outer};
}

// Inline a single provide
class StatementInliner : public IRMutator {
 public:
  explicit StatementInliner(const Provide* op) : op_(op) {
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
  }

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
      LOG(FATAL) << "Cannot inline this statements due to dependency constraint with: "
                 << x.first->stmt;
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

BlockTreeNode Schedule::RegenerateLoopAxis(BlockTreeNode block, AxisTreeNode axis,
                                           Array<IntSet> iter_domain, int insert_pos) {
  ScheduleTreeNode father = operator->()->father_map[block];

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
      Var var = Downcast<Var>(block->args[i]);
      CHECK(dom_map.count(var));
      node = AxisTreeNodeNode::make(var,
                                    dom_map[var]->min,
                                    dom_map[var]->extent,
                                    kOpaque,
                                    Array<ScheduleTreeNode>{last});
      new_args[i] = block->args[i];

    } else if (const arith::IntervalSetNode* set = iter_domain[i].as<arith::IntervalSetNode>()) {
      // todo(lmzheng): fill correct type to replace kOpaque x 3
      node = AxisTreeNodeNode::make(iter_var,
                                    set->min_value,
                                    set->max_value - set->min_value + 1,
                                    kOpaque,
                                    Array<ScheduleTreeNode>{last});
      new_args[i] = iter_var;
    } else if (const arith::StrideSetNode* set = iter_domain[i].as<arith::StrideSetNode>()) {
      CHECK_EQ(set->extents.size(), 1);
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
  axis->children.insert(axis->children.begin() + insert_pos, last);
  UpdateFather(axis);

  return block;
}

BlockTreeNode Schedule::compute_at(BlockTreeNode block, AxisTreeNode axis) {
  // check dependency:
  // 1. all successors are in the subtree rooted by `axis`
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

  // find insert position : after all predecessors in dependency graph
  // and before all successors in dep graph.
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
  for (const auto& x : block->outputs) {
    output_tensors.push_back(x->data);
  }
  Array<Array<IntSet> > ranges = GatherRegion(output_tensors, axis, after_pos,
                                              Set<BlockTreeNode>(nullptr), true, false, 'U');

  // solve range for vars
  Array<IntSet> produces;
  for (const auto& x : block->outputs) {
    for (const auto& ran : x->ranges) {
      produces.push_back(IntSet::range(ran));
    }
  }

  Array<IntSet> iter_domain = arith::SolveCover(block->vars, produces, Flatten2DArray(ranges));

  return RegenerateLoopAxis(block, axis, iter_domain, after_pos);
}

BlockTreeNode Schedule::compute_after(BlockTreeNode block, AxisTreeNode axis) {
  // check dependency
  // 1. all predecessors are in the subtree rooted by `axis`
  StdNodeSet<BlockTreeNode> child_blocks;
  GatherChildrenBlocks(axis, &child_blocks);

  const auto& predecessor = operator->()->dep_graph.GetPredecessor(block);

  for (auto x : predecessor) {
    if (!child_blocks.count(x)) {
      LOG(FATAL) << "This block cannot compute after this point because some other "
                    "blocks outside the scope of this point are also dependent on this block.";
    }
  }

  // gather the generated input regions by predecessor blocks
  Array<Tensor> input_tensors;
  for (const auto& x : block->inputs) {
    input_tensors.push_back(x->data);
  }
  Array<Array<IntSet> > ranges = GatherRegion(input_tensors, axis, 0,
                                              predecessor, false, true, 'I');
  // solve range for vars
  Array<IntSet> consumes;
  Array<IntSet> flatten_ranges;

  for (size_t i = 0; i < block->inputs.size(); ++i) {
    if (ranges[i].size() > 0 && ranges[i][0].is_nothing()) {  // it is an input placeholder
      for (size_t j = 0; j < ranges[i].size(); ++j) {
        // TODO(lmzheng): replace this by strictly checking it is an input placeholder
        CHECK(ranges[i][j].is_nothing());
      }
      continue;
    }

    for (const auto& iset : ranges[i]) {
      flatten_ranges.push_back(iset);
    }
    for (const auto& ran : block->inputs[i]->ranges) {
      consumes.push_back(IntSet::range(ran));
    }
  }

  Array<IntSet> iter_domain = arith::SolveCoverBy(block->vars, consumes, flatten_ranges);

  // TOOD(lmzheng): check the output region of the moved block keeps the same
  return RegenerateLoopAxis(block, axis, iter_domain, axis->children.size());
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
  std::vector<Tensor> raw_output_order;  // To keep the original order of input/output arguments
  std::vector<Tensor> raw_input_order;

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

      for (const TensorRegion& tensor : n->inputs) {
        std::vector<IntSet> ranges;
        for (const Range& range : tensor->ranges) {
          ranges.push_back(arith::EvalSet(
              Range::make_by_min_extent(Substitute(range->min, var_map),
                                        Substitute(range->extent, var_map)), dom_map));
        }
        if (!gathered_input_regions.count(tensor->data)) {
          raw_input_order.push_back(tensor->data);
        }
        gathered_input_regions[tensor->data].push_back(ranges);
      }

      for (const TensorRegion& tensor : n->outputs) {
        std::vector<IntSet> ranges;
        for (const Range& range : tensor->ranges) {
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
  CHECK(!block->stmt.defined())
    << "Cannot unblockize a block that is not created by blockize";
  CHECK_EQ(block->children.size(), 1)
    << "Can only unblockize a block that is created by blockize";

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

void Schedule::double_buffer_scope(Tensor tensor) {
  operator->()->attrs[tensor->op] = AttrStmt::make(tensor->op, attr::double_buffer_scope, 1, Stmt());
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
  std::string attr_type =
      thread_iter->thread_tag == "vthread" ? attr::virtual_thread : attr::thread_extent;
  bind_var[new_var] = AttrStmt::make(thread_iter, attr_type, axis->extent, Stmt());

  // Replace the axis loop var to the new one
  Map<Var, Expr> bind_map;
  bind_map.Set(old_var, new_var);
  SubstituteArgOnly(axis, bind_map);
  axis->loop_var = new_var;

  // Handle with cooperative fetch
  // Down to leaf
  std::function<void(const ScheduleTreeNode& n)> delete_duplicate_axis;
  delete_duplicate_axis = [this, &thread_iter, &delete_duplicate_axis]
      (const ScheduleTreeNode& node) {
    if (const AxisTreeNodeNode* n = node.as<AxisTreeNodeNode>()) {
      if (operator->()->bind_var.count(n->loop_var) &&
          operator->()->bind_var[n->loop_var].as<AttrStmt>()->node == thread_iter) {
        AxisTreeNode axis = GetRef<AxisTreeNode>(n);
        ReplaceChild(axis, axis->children);
      } else {
        for (const auto& child : n->children) {
          delete_duplicate_axis(child);
        }
      }
    }
  };
  for (const auto& child : axis->children) {
    delete_duplicate_axis(child);
  }

  // Up to root
  ScheduleTreeNode now = operator->()->father_map[axis];
  while (now != operator->()->root) {
    if (const AxisTreeNodeNode* n = now.as<AxisTreeNodeNode>()) {
      if (operator->()->bind_var.count(n->loop_var) &&
          operator->()->bind_var[n->loop_var].as<AttrStmt>()->node == thread_iter) {
        ReplaceChild(axis, axis->children);
        // We guarantee there is only one bound axis to the root.
        break;
      }
      now = operator->()->father_map[now];
    }
  }
}

BlockTreeNode Schedule::cache(Tensor tensor, std::string scope, std::string type) {
  CHECK(type == "read" || type == "write") << "Not supported type: " + type;
  Set<BlockTreeNode> predecessors, successor;
  for (const auto& block : operator->()->block_list) {
    if (FindOutput(block, tensor)) {
      predecessors.insert(block);
    }
    else if (FindInput(block, tensor)) {
      successor.insert(block);
    }
  }
  auto root = operator->()->root;
  int after_pos, before_pos;
  for (after_pos = static_cast<int>(root->children.size()) - 1; after_pos >= 0; --after_pos) {
    if (FindAny(root->children[after_pos], predecessors)) {
      break;
    }
  }
  after_pos++;
  for (before_pos = 0; before_pos < static_cast<int>(root->children.size()); ++before_pos) {
    if (FindAny(root->children[before_pos], successor)) {
      break;
    }
  }
  if (after_pos > before_pos) {
    LOG(FATAL) << "Cannot satisfy dependency";
  }
  Array<Range> ranges;
  for (const auto& expr : tensor->shape) {
    ranges.push_back(Range::make_by_min_extent(0, expr));
  }

  Array<Var> iter_vars;
  Set<Var> used_vars;
  Array<Expr> args;
  arith::Analyzer analyzer;
  Array<Expr> halide_call_args;
  Array<IterVar> tmp_iter_vars;
  Map<Var, Expr> var_map;
  Array<Var> vars;

  for (size_t i = 0; i < tensor->shape.size(); ++i) {
    Var iter_var("ax" + std::to_string(i));
    tmp_iter_vars.push_back(IterVarNode::make(ranges[i], iter_var, IterVarType::kDataPar, iter_var->name_hint));
    iter_vars.push_back(iter_var);
    used_vars.insert(iter_var);
    args.push_back(iter_var);
  }

  Tensor new_tensor;
  Expr value;
  if (type == "read") {
    value = Call::make(tensor->dtype, tensor->op->name, args, Call::CallType::Halide, tensor->op);
    new_tensor = ComputeOpNode::make(
        tensor->op->name + "." + scope, tensor->op->tag, Map<std::string, NodeRef>(),
        tmp_iter_vars, Array<Expr>{value}).output(0);
  } else if (type == "write") {
    new_tensor = PlaceholderOpNode::make(tensor->op->name + "." + scope, tensor->shape, tensor->dtype).output(0);
    value = Call::make(new_tensor->dtype, new_tensor->op->name, args, Call::CallType::Halide, new_tensor->op);
  }

  Array<Range> output_range;
  for (size_t i = 0; i < iter_vars.size(); ++i) {
    output_range.push_back(Range::make_by_min_extent(iter_vars[i], 1));
  }

  Array<TensorRegion> raw_outputs;
  if (type == "read") {
    raw_outputs.push_back(TensorRegionNode::make(new_tensor, output_range));
  } else if (type == "write") {
    raw_outputs.push_back(TensorRegionNode::make(tensor, output_range));
  }
  Array<TensorRegion> outputs;

  std::tie(args, vars, outputs, var_map) = CreateOutputRegions(raw_outputs, used_vars, &analyzer);

  for (const auto& x : iter_vars) {
    halide_call_args.push_back(SubstituteAndEquationSimplify(x, var_map, &analyzer));
  }
  value = SubstituteAndEquationSimplify(value, var_map, &analyzer);

  Stmt stmt;
  if (type == "read") {
    stmt = Provide::make(new_tensor->op, 0, value, halide_call_args);
  } else if (type == "write"){
    stmt = Provide::make(tensor->op, 0, value, halide_call_args);
  }

  Array<TensorRegion> inputs = CreateInputRegions(value);
  BlockTreeNode new_block = BlockTreeNodeNode::make(args, vars, inputs, outputs, stmt, Array<ScheduleTreeNode>{});
  ScheduleTreeNode last = new_block;
  for (int i = static_cast<int>(ranges.size()) - 1; i >= 0; --i) {
    last = AxisTreeNodeNode::make(iter_vars[i], ranges[i]->min,
                                  ranges[i]->extent, AxisType::kOpaque,
                                  Array<ScheduleTreeNode>{last});
  }
  UpdateFather(last, true);
  // Update children of root
  auto& block_list = operator->()->block_list;
  root->children.insert(root->children.begin() + after_pos, last);
  UpdateFather(root);

  size_t block_pos = 0;
  for (int i = 0; i < after_pos; ++i) {
    block_pos += BlockNum(root->children[i]);
  }

  // Update block_list
  block_list.insert(block_list.begin() + block_pos, new_block);

  // Calculate the last block produce the tensor
  int last_block;
  for (last_block = static_cast<int>(block_list.size()) - 1; last_block >= 0; --last_block) {
    if (predecessors.count(block_list[last_block]) && FindOutput(block_list[last_block], tensor)) {
      break;
    }
  }

  StdNodeMap<Tensor, Tensor> vmap;
  vmap[tensor] = new_tensor;

  if (type == "read") {
    for (auto& block : successor) {
      SubstituteTensorOnly(block, vmap, true, false);
    }

    // Update dependency graph
    operator->()->dep_graph.CacheReadNode(last_block >= 0 ? block_list[last_block] : BlockTreeNode(),
                                          new_block, Array<BlockTreeNode>(successor.begin(), successor.end()));

  } else if (type == "write") {
    for (auto& block : predecessors) {
      SubstituteTensorOnly(block, vmap, block != new_block, true);
    }

    // Update dependency graph
    operator->()->dep_graph.CacheWriteNode(last_block >= 0 ? block_list[last_block] : BlockTreeNode(),
                                           new_block, Array<BlockTreeNode>(predecessors.begin(), predecessors.end()));
  }
  operator->()->raw_realize_region[new_tensor] = ranges;
  operator->()->raw_realize_scope[new_tensor->op] = scope;
  return new_block;
}

BlockTreeNode Schedule::cache_read(Tensor tensor, std::string scope) {
  return cache(tensor, scope, "read");
}

BlockTreeNode Schedule::cache_write(Tensor tensor, std::string scope) {
  return cache(tensor, scope, "write");
}

// dependency analysis
Array<Array<IntSet> > Schedule::GatherRegion(Array<Tensor> tensors,
                                             AxisTreeNode axis,
                                             int start_child_index,
                                             Set<BlockTreeNode> block_filter,
                                             bool gather_inputs,
                                             bool gather_outputs,
                                             char aggregate_mode) const {
  std::unordered_map<const Variable*, IntSet> dom_map;
  std::unordered_map<const Variable*, IntSet> shared_dom_map;

  ScheduleTreeNode father = axis;

  while (father != operator->()->root) {
    if (const AxisTreeNodeNode* n = father.as<AxisTreeNodeNode>()) {
      dom_map[n->loop_var.get()] = IntSet::single_point(n->loop_var);
      const auto& bind_var = operator->()->bind_var;
      const auto& var = n->loop_var;
      if (bind_var.count(var) &&
          (bind_var.at(var).as<AttrStmt>()->attr_key == attr::virtual_thread ||
          (bind_var.at(var).as<AttrStmt>()->attr_key == attr::thread_extent &&
          (var->name_hint.find("threadIdx.") == 0)))) {
        const auto& attr = operator->()->bind_var.at(var).as<AttrStmt>();
        shared_dom_map[n->loop_var.get()] = IntSet::range(
            Range::make_by_min_extent(0, attr->value));
      } else {
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

      if (block_filter.defined() && !block_filter.count(block)) {
        continue;
      }

      // replace formal parameters with actual parameters
      std::unordered_map<const Variable*, Expr> arg_map;
      for (size_t i = 0; i < block->vars.size(); ++i) {
        arg_map[block->vars[i].get()] = block->args[i];
      }

      auto gather_func = [&tensor, &arg_map, this, &isets_to_merge,
                          &dom_map, &shared_dom_map]
          (const Array<TensorRegion>& regions) {
        for (const auto& reg : regions) {
          if (reg->data == tensor) {
            for (size_t i = 0; i < tensor.ndim(); ++i) {
              Range real = Range::make_by_min_extent(Substitute(reg->ranges[i]->min, arg_map),
                                                     Substitute(reg->ranges[i]->extent, arg_map));
              CHECK(operator->()->raw_realize_scope.count(tensor->op));
              IntSet o;
              if (operator->()->raw_realize_scope.at(tensor->op) == "shared") {
                o = arith::EvalSet(real, shared_dom_map);
              } else {
                o = arith::EvalSet(real, dom_map);
              }
              isets_to_merge[i].push_back(o);
            }
          }
        }
      };

      if (gather_inputs) {
        gather_func(block->inputs);
      }
      if (gather_outputs) {
        gather_func(block->outputs);
      }
    }

    Array<IntSet> aggregated_region;
    for (size_t i = 0; i < tensor.ndim(); ++i) {
      if (aggregate_mode == 'U') {
        aggregated_region.push_back(arith::Union(isets_to_merge[i]));
      } else {
        aggregated_region.push_back(arith::Intersect(isets_to_merge[i]));
      }
    }
    ret.push_back(aggregated_region);
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

size_t Schedule::BlockNum(ScheduleTreeNode node) const {
  size_t ret = 0;
  if (node.as<BlockTreeNodeNode>()) {
    ret += 1;
  }
  for (const auto& x : node->children) {
    ret += BlockNum(x);
  }
  return ret;
}

void Schedule::ReplaceChild(ScheduleTreeNode old_child, ScheduleTreeNode new_child) {
  ScheduleTreeNode father = operator->()->father_map[old_child];
  father->children.Set(father->children.Index(old_child), new_child);
  operator->()->father_map.Set(new_child, father);
}

void Schedule::ReplaceChild(ScheduleTreeNode old_child, Array<ScheduleTreeNode> new_children) {
  if (new_children.size() == 1) {
    ReplaceChild(old_child, new_children[0]);
  } else {
    ScheduleTreeNode father = operator->()->father_map[old_child];
    auto index = father->children.Index(old_child);
    size_t offset = new_children.size() - 1;
    for (size_t i = 0; i < offset; ++i) {
      father->children.push_back(new_children[i]);
    }
    for (size_t i = father->children.size() - 1 - offset; i > index; --i) {
      father->children.Set(i + offset, father->children[i]);
    }
    for (size_t i = 0; i < new_children.size(); ++i) {
      father->children.Set(i + index, new_children[i]);
    }

    UpdateFather(father);
  }
}

ScheduleTreeNode Schedule::LowestCommonAncestor(Array<ScheduleTreeNode> nodes,
                                                bool inclusive) const {
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

bool Schedule::IsAncestor(ScheduleTreeNode outer, ScheduleTreeNode inner) const {
  ScheduleTreeNode now = inner;
  while (now != operator->()->root) {
    now = operator->()->father_map[now];
    if (now == outer) {
      return true;
    }
  }
  return false;
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
  if (const AxisTreeNodeNode* n = node.as<AxisTreeNodeNode>()) {
    if ((*dom_map).count(n->loop_var.get()) == 0) {
      (*dom_map)[n->loop_var.get()] = IntSet::range(Range::make_by_min_extent(n->min, n->extent));
    }
    for (auto x : n->children) {
      GatherVarDomain(x, dom_map);
    }
  } else if (const BlockTreeNodeNode* n = node.as<BlockTreeNodeNode>()) {
    CHECK(n != nullptr);  // useless check to eliminate warnings
  }
}

Array<ScheduleTreeNode> SubstituteArgOnly(Array<ScheduleTreeNode> nodes,
                                          const Map<Var, Expr>& vmap) {
  for (auto x : nodes) {
    SubstituteArgOnly(x, vmap);
  }
  return nodes;
}

ScheduleTreeNode SubstituteArgOnly(ScheduleTreeNode node, const Map<Var, Expr>& vmap) {
  if (const AxisTreeNodeNode* n = node.as<AxisTreeNodeNode>()) {
    AxisTreeNode axis = GetRef<AxisTreeNode>(n);  // get ref to remove const qualifier
    axis->min = Substitute(axis->min, vmap);
    axis->extent = Substitute(axis->extent, vmap);
    for (auto x : axis->children) {
      SubstituteArgOnly(x, vmap);
    }
  } else if (const BlockTreeNodeNode* n = node.as<BlockTreeNodeNode>()) {
    BlockTreeNode block = GetRef<BlockTreeNode>(n);  // get ref to remove const qualifier
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

ScheduleTreeNode SubstituteTensorOnly(ScheduleTreeNode node, const StdNodeMap<Tensor, Tensor>& vmap, bool read, bool write) {
  std::unordered_map<TensorKey, TensorKey> smap;
  for (const auto x : vmap) {
    TensorKey key = {x.first->op, x.first->value_index};
    TensorKey value = {x.second->op, x.second->value_index};
    smap[key] = value;
  }
  if (const BlockTreeNodeNode* n = node.as<BlockTreeNodeNode>()) {
    BlockTreeNode block = GetRef<BlockTreeNode>(n);  // get ref to remove const qualifier
    if (read) {
      Array<TensorRegion> inputs;
      for (const auto& x : block->inputs) {
        Tensor t = vmap.count(x->data) ? vmap.at(x->data) : x->data;
        inputs.push_back(TensorRegionNode::make(t, x->ranges));
      }
      block->inputs = inputs;
    }
    if (write) {
      Array<TensorRegion> outputs;
      for (const auto& x : block->outputs) {
        Tensor t = vmap.count(x->data) ? vmap.at(x->data) : x->data;
        outputs.push_back(TensorRegionNode::make(t, x->ranges));
      }
      block->outputs = outputs;
    }
    block->stmt = Substitute(block->stmt, smap, read, write);
  }
  return node;
}

}  // namespace tensorir
}  // namespace tvm
