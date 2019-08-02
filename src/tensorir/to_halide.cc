/*!
 *  Copyright (c) 2019 by Contributors
 *  \brief Lower TensorIR to HalideIR
 */

#include <tvm/ir_mutator.h>
#include <tvm/buffer.h>
#include "schedule.h"
#include "node_util.h"
#include "util.h"
#include "../arithmetic/int_set.h"

namespace tvm {
namespace tensorir {

// Substitute by making new copy
ScheduleTreeNode SubstituteCopy(ScheduleTreeNode node, const Map<Var, Expr>& vmap) {
  if (const AxisTreeNodeNode* n = node.as<AxisTreeNodeNode>()) {
    Array<ScheduleTreeNode> children;
    for (const auto& x : n->children) {
      children.push_back(SubstituteCopy(x, vmap));
    }
    return AxisTreeNodeNode::make(n->loop_var, Substitute(n->min, vmap), Substitute(n->extent, vmap),
                                  n->axis_type, children);
  } else if (const BlockTreeNodeNode* n = node.as<BlockTreeNodeNode>()) {
    Array<Expr> args;
    for (const auto& x : n->args) {
      args.push_back(Substitute(x, vmap));
    }
    return BlockTreeNodeNode::make(args, n->vars, n->inputs, n->outputs, n->stmt, n->children);
  } else {
    LOG(FATAL) << "Invalid node in schedule tree";
    return ScheduleTreeNode(nullptr);
  }
}

// return whether the subtree rooted by node accesses tensor t
bool FindAccess(ScheduleTreeNode node, Tensor t) {
  if (const AxisTreeNodeNode* n = node.as<AxisTreeNodeNode>()) {
    for (const auto& x : n->children) {
      if (FindAccess(x, t)) {
        return true;
      }
    }
  } else if (const BlockTreeNodeNode* n = node.as<BlockTreeNodeNode>()) {
    for (const auto& x : n->inputs) {
      if (x->data == t) {
        return true;
      }
    }
    for (const auto& x : n->outputs) {
      if (x->data == t) {
        return true;
      }
    }
  }
  return false;
}

// output
Stmt Schedule::ToHalide() const {
  // # 1. Set allocation position for all tensors
  StdNodeMap<Tensor, Array<ScheduleTreeNode> > related_nodes;

  // get root node for computation blocks
  std::function<void(const ScheduleTreeNode& n)> get_tensor_location;
  get_tensor_location = [this, &get_tensor_location, &related_nodes](const ScheduleTreeNode& node) {
    if (const AxisTreeNodeNode* n = node.as<AxisTreeNodeNode>()) {
      for (const auto& x : n->children) {
        get_tensor_location(x);
      }
    } else if (const BlockTreeNodeNode* n = node.as<BlockTreeNodeNode>()) {
      Set<Var> used_vars;
      Set<Var> seen;
      Set<Var> used_thread_vars;
      Set<Var> seen_thread_vars;
      for (size_t i = 0; i < n->args.size(); ++i) {
        const auto vars = GatherVars(n->args[i]);
        used_vars.insert(vars);
        for (const auto & var : vars) {
          if (var->name_hint.find("threadIdx.") == 0) {
            used_thread_vars.insert(var);
          }
        }
      }

      // Go upwards until all used vars are fetched.
      // Then we find the root node of this block
      ScheduleTreeNode now = node;
      ScheduleTreeNode shared_memory_node = node;
      while (operator->()->father_map[now] != operator->()->root
          && used_vars.size() > seen.size()) {
        now = operator->()->father_map[now];
        if (used_thread_vars.size() > seen_thread_vars.size()) {
          shared_memory_node = now;
        }
        if (const AxisTreeNodeNode* loop = now.as<AxisTreeNodeNode>()) {
          Var v = loop->loop_var;
          if (used_vars.count(v) && !seen.count(v)) {
            seen.insert(v);
            if (v->name_hint.find("threadIdx.") == 0) {
              seen_thread_vars.insert(v);
            }
          }
        }
      }

      for (const auto& x : n->outputs) {
        if (operator->()->raw_realize_scope.count(x->data->op)) {
          if (operator->()->raw_realize_scope.at(x->data->op) != "") {
            related_nodes[x->data].push_back(node);
          }
          else {
            related_nodes[x->data].push_back(now);
          }
        }
      }
      for (const auto& x : n->inputs) {
        if (operator->()->raw_realize_scope.count(x->data->op)) {
          if (operator->()->raw_realize_scope.at(x->data->op) != "") {
            related_nodes[x->data].push_back(node);
          }
          else {
            related_nodes[x->data].push_back(now);
          }
        }
      }
    }
  };
  get_tensor_location(operator->()->root);

  auto realize_region = operator->()->raw_realize_region;

  // For tensor T, let A be the lowest common ancestor of all nodes accessing it.
  // We place its allocation before the first children of A that accesses T.
  StdNodeMap<ScheduleTreeNode, std::vector<Tensor> > attached_allocation;
  for (const auto x : related_nodes) {
    ScheduleTreeNode lca = LowestCommonAncestor(x.second, false);
    for (const auto& child : lca->children) {
      if (FindAccess(child, x.first)) {
        attached_allocation[child].push_back(x.first);
        if (operator->()->raw_realize_scope.at(x.first->op) != "") {
          const auto *axis = lca.as<AxisTreeNodeNode>();
          auto regions = GatherRegion(Array<Tensor>{x.first}, GetRef<AxisTreeNode>(axis), 0);
          Region region;
          for (const auto& int_set : regions[0]) {
            arith::Analyzer analyzer;
            Range real = Range::make_by_min_extent(int_set.min(), analyzer.canonical_simplify(int_set.max() - int_set.min() + 1));
            region.push_back(real);
          }
          realize_region[x.first] = region;
        }
        break;
      }
    }
  }

  std::unordered_set<std::string> binded_threads;
  // # 2. Translate to HalideIR
  std::function<Array<Stmt> (const ScheduleTreeNode& n)> to_halide_stmt;
  to_halide_stmt = [this, &to_halide_stmt, &attached_allocation, &binded_threads, &realize_region](const ScheduleTreeNode& node) {
    std::vector<Stmt> ret;

    // attach realize scope
    for (const auto& tensor : attached_allocation[node]) {
      if (realize_region.count(tensor)) {
        CHECK_GE(operator->()->raw_realize_scope.count(tensor->op), 1);

        Region region = realize_region.at(tensor);
        ret.push_back(AttrStmt::make(tensor->op,
                                     attr::realize_scope,
                                     operator->()->raw_realize_scope.at(tensor->op),
                                     Evaluate::make(0)));

        ret.push_back(Realize::make(tensor->op,
                                    tensor->value_index,
                                    tensor->dtype,
                                    region,
                                    const_true(1),
                                    Evaluate::make(0)));
      }
    }

    // translate nodes
    if (const AxisTreeNodeNode* n = node.as<AxisTreeNodeNode>()) {
      bool new_thread_axis;
      if (operator->()->bind_var.count(n->loop_var)) {
        const auto& name = n->loop_var->name_hint;
        if (binded_threads.count(name)) {
          new_thread_axis = false;
        }
        else {
          binded_threads.insert(name);
          new_thread_axis = true;
        }
      }
      Array<Stmt> stmts;
      for (const auto& child : n->children) {
        for (const auto& stmt : to_halide_stmt(child)) {
          stmts.push_back(stmt);
        }
      }
      if (node == operator->()->root) {
        ret.push_back(ArrayToBlock(stmts));
      } else if (is_one(n->extent)) {
        Map<Var, Expr> vmap;
        vmap.Set(n->loop_var, 0);
        Array<Stmt> new_stmts;
        for (const auto &stmt : stmts) {
          arith::Analyzer analyzer;
          new_stmts.push_back(SubstituteAndEquationSimplify(stmt, vmap, &analyzer));
        }
        ret.push_back(ArrayToBlock(new_stmts));
      } else {
          auto var = n->loop_var;
          auto bind_var = operator->()->bind_var;
          auto it = bind_var.find(var);
          if (it != bind_var.end()) {
            Attr attr = it->second;
            if (new_thread_axis) {
              ret.push_back(AttrStmt::make(attr->node, attr->attr_key, attr->value, ArrayToBlock(stmts)));
              binded_threads.erase(var->name_hint);
            }
            else {
              ret.push_back(ArrayToBlock((stmts)));
            }
          } else {
            ForType type;
            if (n->axis_type == AxisType::vectorized) {
              type = ForType::Vectorized;
            } else if (n->axis_type == AxisType::unrolled) {
              type = ForType::Unrolled;
            } else {
              type = ForType::Serial;
            }
            ret.push_back(For::make(var, n->min, n->extent,
                                    type, DeviceAPI::None,
                                    ArrayToBlock(stmts)));
          }
        }
      }
    else if (const BlockTreeNodeNode* n = node.as<BlockTreeNodeNode>()) {
      StdNodeMap<Var, Expr> var_map;
      for (size_t i = 0; i < n->args.size(); ++i) {
        var_map[n->vars[i]] = n->args[i];
      }
      if (n->stmt.defined()) {
        ret.push_back(Substitute(n->stmt, var_map));
      } else {
        CHECK_EQ(n->children.size(), 1) << "Encounter invalid block" << std::endl;
        Array<Stmt> tmp = to_halide_stmt(SubstituteCopy(n->children[0], var_map));
        CHECK_EQ(tmp.size(), 1);
        ret.push_back(tmp[0]);
      }
    } else {
      LOG(FATAL) << "Internal error: unknown node type";
    }

    return ret;
  };

  return ArrayToBlock(to_halide_stmt(operator->()->root));
}

} // namespace tensorir
} // namespace tvm
