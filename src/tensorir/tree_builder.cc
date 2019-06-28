/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Build Schedule Tree from Halide IR
 */

#include <tuple>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/api_registry.h>
#include <tvm/arithmetic.h>
#include <ir/Expr.h>
#include "tree_builder.h"
#include "dependency_graph.h"
#include "util.h"

namespace tvm {
namespace tensorir {

// remove realize, produce, and attr annotations to get a cleaner IR
class IRCleaner : public IRMutator {
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) {
    if (op->attr_key == attr::realize_scope) {
      const StringImm* str = op->value.as<StringImm>();
      CHECK(str != nullptr);
      raw_realize_scope[FunctionRef(op->node.node_)] = str->value;
    }

    return Mutate(op->body);
  }

  Stmt Mutate_(const Realize* op, const Stmt& s) {
    Tensor t = Downcast<Operation>(op->func).output(op->value_index);
    raw_realize_region[t] = op->bounds;
    return Mutate(op->body);
  }

  Stmt Mutate_(const ProducerConsumer* op, const Stmt& s) {
    return Mutate(op->body);
  }

 public:
  StdNodeMap<Tensor, Region> raw_realize_region;
  StdNodeMap<FunctionRef, std::string> raw_realize_scope;
};


Schedule TreeBuilder::Build(Stmt stmt) {
  // remove redundant ast nodes from Halide
  IRCleaner ir_cleaner;
  stmt = ir_cleaner.Mutate(stmt);

  // insert an auxiliary root node and start building schedule tree
  stmt = For::make(Var("root"), 0, 1, ForType::Parallel, DeviceAPI::None, stmt);
  AxisTreeNode root_node = Downcast<AxisTreeNode>(VisitStmt(stmt));
  DependencyGraph dep_graph = DependencyGraphNode::make(root_node);

  NodePtr<ScheduleNode> node = make_node<ScheduleNode>();

  node->root = std::move(root_node);
  node->dep_graph = std::move(dep_graph);
  node->block_list = std::move(block_list_);
  node->raw_realize_region = std::move(ir_cleaner.raw_realize_region);
  node->raw_realize_scope = std::move(ir_cleaner.raw_realize_scope);

  Schedule ret(node);
  ret.UpdateFather(ret->root, true);
  ret->father_map.Set(ret->root, ret->root);

  return ret;
}

ScheduleTreeNode TreeBuilder::VisitStmt_(const For* op) {
  Var var = Var(op->loop_var);
  dom_map_[var] = Range::make_by_min_extent(op->min, op->extent);
  var_order_[var] = var_ct_++;

  Array<Stmt> children_stmt = ExpandBlockChain(op->body);
  Array<ScheduleTreeNode> children;

  // TODO (lmzheng): Merge consecutive blocks
  for (auto x : children_stmt) {
    children.push_back(VisitStmt(x));
  }
  return AxisTreeNodeNode::make(var, op->min, op->extent, AxisType::kOpaque, children);
}

ScheduleTreeNode TreeBuilder::VisitStmt_(const Allocate* op) {
  return ScheduleTreeNode(NodePtr<Node>(nullptr));
}

ScheduleTreeNode TreeBuilder::VisitStmt_(const AssertStmt* op) {
  return ScheduleTreeNode(NodePtr<Node>(nullptr));
}

std::pair<Array<TensorRegion>, Map<Var, Expr> > CanonicalizeBlockRegion(
    Array<TensorRegion> outputs, std::shared_ptr<arith::Analyzer> analyzer) {
  static size_t var_ct = 0;
  Map<Var, Expr> replace_map;

  // simplify min and store them
  std::vector<Expr> mins;
  for (const auto& t : outputs) {
    for (const auto& ran : t->ranges) {
      Expr min = ran->min;
      if (!min->is_type<Variable>()) {
        min = analyzer->equation_simplify(min);
      }
      if (!min->is_type<Variable>()) {
        Var min_var("p" + std::to_string(var_ct++));
        replace_map.Set(min_var, min);
        analyzer->Bind(min_var, min);
        min = min_var;
      }
      mins.push_back(min);
    }
  }

  // simplify extent
  Array<TensorRegion> ret;
  size_t ct = 0;
  for (const auto& t : outputs) {
    Array<Range> ranges;
    for (const auto& ran : t->ranges) {
      Expr extent = analyzer->equation_simplify(ran->extent);
      ranges.push_back(Range::make_by_min_extent(mins[ct++], extent));
    }
    ret.push_back(TensorRegionNode::make(t->data, ranges));
  }

  return std::make_pair(ret, replace_map);
}

BlockTreeNode SortBlockArgs(BlockTreeNode node) {
  StdNodeMap<Var, int> weight;
  int ct = 0;

  for (const auto& t : node->inputs) {
    for (const auto& ran : t->ranges) {
      if (ran->min->is_type<Variable>()) {
        weight[Downcast<Var>(ran->min)] = std::numeric_limits<int>::min() + (ct++);
      }
    }
  }

  std::vector<int> indices;
  for (size_t i = 0; i < node->vars.size(); ++i) {
    indices.push_back(i);
  }
  std::sort(indices.begin(), indices.end(),
            [&](int a, int b) -> bool { return weight[node->vars[a]] < weight[node->vars[b]]; });

  Array<Var> vars;
  Array<Expr> args;
  for (size_t i = 0; i < node->vars.size(); ++i) {
    vars.push_back(node->vars[indices[i]]);
    args.push_back(node->args[indices[i]]);
  }
  return BlockTreeNodeNode::make(args, vars, node->inputs, node->outputs, node->stmt);
}

Array<TensorRegion> CreateInputRegions(const NodeRef& expr_or_stmt) {
  Array<TensorRegion> inputs;

  TensorAccessGather gather;
  gather.GatherAndGroup(expr_or_stmt);

  for (auto iter : gather.access_grouped) { // for all tensors
    Array<Range> ranges;

    for (size_t i = 0; i < iter.first->shape.size(); ++i) { // for all dimensions
      Array<arith::IntSet> sets;
      for (const auto &x : iter.second) {   // for multiple accesses
        sets.push_back(arith::IntSet::single_point(x[i]));
      }
      arith::IntSet unioned = arith::Union(sets);
      ranges.push_back(Range::make_by_min_extent(unioned.min(), simplify(unioned.max() - unioned.min()+1)));
    }

    inputs.push_back(TensorRegionNode::make(iter.first, ranges));
  }
  return inputs;
}

ScheduleTreeNode TreeBuilder::VisitStmt_(const Provide* op) {
  size_t ct = 0;

  // canonicalize output regions
  Tensor output = Downcast<Operation>(op->func).output(op->value_index);
  Array<Range> output_range;
  for (size_t i = 0; i < op->args.size(); ++i) {
    output_range.push_back(Range::make_by_min_extent(op->args[i], 1));
  }
  Array<TensorRegion> raw_outputs;
  raw_outputs.push_back(TensorRegionNode::make(output, output_range));

  auto analyzer = std::make_shared<arith::Analyzer>();
  Map<Var, Expr> simplify_map;
  std::tie(raw_outputs, simplify_map) = CanonicalizeBlockRegion(raw_outputs, analyzer);

  // create formal and actual arguments
  Array<Var> vars;
  Array<Expr> args;
  Set<Var> deleted_vars;
  for (const auto& x : simplify_map) {
    vars.push_back(x.first);
    args.push_back(x.second);
    deleted_vars = deleted_vars.Union(GatherVars(x.second));
  }

  Set<Var> used_vars = GatherVars(op->value);
  for (size_t i = 0; i < op->args.size(); ++i) {
    used_vars = used_vars.Union(GatherVars(op->args[i]));
  }
  used_vars = used_vars.Difference(deleted_vars);

  StdNodeMap<Var, Expr> var_map;
  for (const auto& x : used_vars) {
    Var var("v" + std::to_string(ct++));
    var_map[x] = var;
    vars.push_back(var);
    args.push_back(x);
  }

  // create output regions
  Array<TensorRegion> outputs;
  for (const auto& t : raw_outputs) {
    Array<Range> ranges;
    for (const auto& ran : t->ranges) {
      ranges.push_back(Range::make_by_min_extent(Substitute(ran->min, var_map),
                                                 Substitute(ran->extent, var_map)));
    }
    outputs.push_back(TensorRegionNode::make(t->data, ranges));
  }

  // create input regions
  Expr value = Substitute(analyzer->equation_simplify(op->value), var_map);
  Array<TensorRegion> inputs = CreateInputRegions(value);

  // make node
  Array<Expr> halide_call_args;
  for (const auto& x : op->args) {
    halide_call_args.push_back(Substitute(analyzer->equation_simplify(x), var_map));
  }
  Stmt stmt = Provide::make(op->func,
                            op->value_index,
                            value,
                            halide_call_args);

  BlockTreeNode ret = BlockTreeNodeNode::make(args, vars, inputs, outputs, stmt);
  ret = SortBlockArgs(ret);
  block_list_.push_back(ret);
  return ret;
}

} // namespace tensorir
} // namespace tvm
