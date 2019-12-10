/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/te/schedule.h>
#include <tvm/api_registry.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/te/transform.h>
#include <tvm/ir_mutator.h>
#include "../util.h"

namespace tvm {
namespace te {

Array<Stmt> Schedule::GetChildren(Stmt stmt) {
  Stmt body;
  if (const auto* block = stmt.as<BlockNode>()) {
    body = block->body;
  } else if (const auto* loop = stmt.as<LoopNode>()) {
    body = loop->body;
  } else {
    return Array<Stmt>();
  }
  // Don't support ir::Block in schedule
  CHECK(!body.as<ir::Block>());
  if (const auto* seq = body.as<SeqStmtNode>()) {
    return seq->seq;
  } else {
    Array<Stmt> children;
    children.push_back(body);
    return children;
  }
}

Schedule::Schedule(Function func,
                   ScheduleTreeNodeRef root,
                   DependencyGraph dependency_graph,
                   Map<Buffer, Array<BlockTreeNodeRef>> write_map,
                   std::unordered_map<const StmtNode*, ScheduleTreeNodeRef> stmt_map) {
  NodePtr<ScheduleNode> node = make_node<ScheduleNode>();
  node->func = std::move(func);
  node->dependency_graph = std::move(dependency_graph);
  node->write_map = std::move(write_map);
  node->root = std::move(root);
  node->stmt_map = std::move(stmt_map);
  data_ = std::move(node);
}

void Schedule::UpdateChildren(Stmt stmt, ScheduleTreeNodeRef father) {
  const auto* stmt_ptr = stmt.as<StmtNode>();
  if (operator->()->stmt_map.count(stmt_ptr) == 0) {
    ScheduleTreeNodeRef ref;
    if (const auto* loop = stmt.as<LoopNode>()) {
      ref = AxisTreeNodeRef(loop, father);
    } else {
      const auto* block = stmt.as<BlockNode>();
      CHECK(block);
      ref = BlockTreeNodeRef(block, father);
    }
    operator->()->stmt_map[stmt_ptr] = ref;
    if (stmt.as<LoopNode>()) {
      for (auto child : GetChildren(stmt)) {
        UpdateChildren(child, ref);
      }
    }
  }
}

class SubReplacer : public IRMutator {
 public:
  SubReplacer(ScheduleNode* schedule, Stmt old_stmt, Stmt new_stmt)
      : schedule_(schedule), old_stmt_(old_stmt), new_stmt_(new_stmt) {}

  Stmt Mutate_(const SeqStmtNode* op, const Stmt& s) final {
    if (s.unique()) {
      SeqStmtNode* seq = const_cast<SeqStmtNode*>(op);
      for (size_t i = 0; i < seq->size(); ++i) {
        if (seq->operator[](i).same_as(old_stmt_)) {
          seq->operator[](i) = new_stmt_;
        }
      }
      need_copy = false;
      return s;
    } else {
      Array<Stmt> stmt;
      for (size_t i = 0; i < op->size(); ++i) {
        if (op->operator[](i).same_as(old_stmt_)) {
          stmt.push_back(new_stmt_);
        } else {
          stmt.push_back(op->operator[](i));
        }
      }
      return SeqStmt(stmt);
    }
  }
  Stmt Mutate_(const BlockNode* op, const Stmt& s) final {
    Stmt body;
    if (op->body.as<SeqStmtNode>()) {
      body = Mutate(op->body);
    } else {
      CHECK(op->body.same_as(old_stmt_));
      body = new_stmt_;
    }
    LOG(INFO) << op->predicate;
    if (need_copy && s.unique()) {
      auto* seq = const_cast<BlockNode*>(op);
      seq->body = body;
      need_copy = false;
      return s;
    } else if (need_copy && !s.unique()) {
      Block new_block(op->iter_vars, op->values,
                      op->reads, op->writes,
                      body, op->predicate,
                      op->annotations, op->tag);
      const auto* stmt_node = new_block.as<StmtNode>();
      const auto* old_stmt = static_cast<const StmtNode*>(op);
      schedule_->stmt_map[stmt_node] = schedule_->stmt_map[old_stmt];
      schedule_->stmt_map.erase(old_stmt);
      return std::move(new_block);
    } else {
      return s;
    }

  }
  Stmt Mutate_(const LoopNode* op, const Stmt& s) final {
    Stmt body;
    if (op->body.as<SeqStmtNode>()) {
      body = Mutate(op->body);
    } else {
      CHECK(op->body.same_as(old_stmt_));
      body = new_stmt_;
    }
    if (need_copy && s.unique()) {
      auto* seq = const_cast<LoopNode*>(op);
      seq->body = body;
      need_copy = false;
      return s;
    } else if (need_copy && !s.unique()) {
      Loop new_loop(op->loop_var, op->min, op->extent,
                    op->annotations, body);
      const auto* stmt_node = new_loop.as<StmtNode>();
      const auto* old_stmt = static_cast<const StmtNode*>(op);
      schedule_->stmt_map[stmt_node] = schedule_->stmt_map[old_stmt];
      schedule_->stmt_map.erase(old_stmt);
      return std::move(new_loop);
    } else {
      return s;
    }
  }

  bool need_copy{true};
 private:
  ScheduleNode* schedule_;
  Stmt old_stmt_;
  Stmt new_stmt_;
};

void Schedule::Replace(ScheduleTreeNodeRef old_node, Stmt new_stmt) {
  CHECK(new_stmt.as<LoopNode>() || new_stmt.as<BlockNode>());
  UpdateChildren(new_stmt, old_node->father);
  bool need_copy = true;
  ScheduleTreeNodeRef node = old_node;
  Stmt old_stmt = GetRef<Stmt>(node->stmt());
  while (node->father != operator->()->root &&
      node != operator->()->root) {
    node = node->father;
    old_stmt = GetRef<Stmt>(node->stmt());
    SubReplacer sub_replacer(operator->(), old_stmt, new_stmt);
    new_stmt = sub_replacer.Mutate(GetRef<Stmt>(node->stmt()));
    if (!sub_replacer.need_copy) {
      need_copy = false;
      break;
    }
  }
  if (need_copy) {
    const auto& func = operator->()->func;
    if (func->body != old_stmt) {
      Array<Stmt> stmts;
      const auto* seq = func->body.as<SeqStmtNode>();
      CHECK(seq);
      bool found = false;
      for (size_t i = 0; i < seq->size(); ++i) {
        if (seq->operator[](i) == old_stmt) {
          stmts.push_back(new_stmt);
          found = true;
        } else {
          stmts.push_back(seq->operator[](i));
        }
      }
      CHECK(found) << "Can not find stmt to be replace";
      new_stmt = SeqStmt(stmts);
    }
    if (func.unique()) {
      operator->()->func.Mutable()->body = new_stmt;
    } else {
      operator->()->func = Function(func->params, func->buffer_map, func->name, new_stmt);
    }
  }
}

Array<BlockTreeNodeRef> Schedule::GetBlock(std::string tag) const {
  Array<BlockTreeNodeRef> ret;
  for (const auto& block : Blocks()) {
    if (block->block->tag == tag) {
      ret.push_back(block);
    }
  }
  return ret;
}

Array<BlockTreeNodeRef> Schedule::GetBlock(Buffer buffer) const {
  if (operator->()->write_map.count(buffer)) {
    return operator->()->write_map.at(buffer);
  } else {
    return Array<BlockTreeNodeRef>();
  }
}

Array<BlockTreeNodeRef> Schedule::Blocks() const {
  Array<BlockTreeNodeRef> ret;
  for (const auto& x : operator->()->write_map) {
    for (const auto& block : x.second) {
      ret.push_back(block);
    }
  }
  return ret;
}

Array<AxisTreeNodeRef> Schedule::GetAxes(BlockTreeNodeRef block) const {
  Array<AxisTreeNodeRef> ret;
  ScheduleTreeNodeRef node = block->father;
  while (!node.same_as(operator->()->root)) {
    if (node.as<AxisTreeNode>()) {
      ret.push_back(Downcast<AxisTreeNodeRef>(node));
    }
    node = node->father;
  }
  return Array<AxisTreeNodeRef>(ret.rbegin(), ret.rend());
}

AxisTreeNodeRef Schedule::fuse(AxisTreeNodeRef outer, AxisTreeNodeRef inner) {
  // Can only fuse neighbor axes without any extra branches.
  // Future Enhancement: this condition can be eliminated by lifting all siblings of inner
  // as the children of the father of outer
  Loop outer_loop = GetRef<Loop>(outer->loop);
  Loop inner_loop = GetRef<Loop>(inner->loop);

  CHECK(inner->father == outer);
  auto outer_children = GetChildren(outer_loop);
  CHECK(outer_children.size() == 1 && outer_children[0] == inner_loop);

  // Currently, can not fuse Loops with annotations
  if (!outer->loop->annotations.empty() || !inner->loop->annotations.empty()) {
    // TODO(tvm-team): Add ReportError
    LOG(FATAL) << "InvalidScheduleError: " << "Cannot fuse loops that already has annotations";
  }

  Expr min = 0;
  Expr extent = outer_loop->extent * inner_loop->extent;

  Var fused_var = outer_loop->loop_var.copy_with_suffix(
      "." + inner_loop->loop_var.get()->name_hint + ".fused");

  auto vmap = [&](const Variable* v) -> Expr {
    if (GetRef<Var>(v).same_as(outer_loop->loop_var)) {
      return truncdiv(fused_var, inner_loop->extent) + outer_loop->min;
    } else if (GetRef<Var>(v).same_as(inner_loop->loop_var)) {
      return truncmod(fused_var, inner_loop->extent) + inner_loop->min;
    } else {
      return Expr(NodePtr<Node>(nullptr));
    }
  };

  Loop fused_node = Loop(
      fused_var, min, extent, outer_loop->annotations,
      Substitute(inner_loop->body, vmap));

  // relink
  Replace(outer, fused_node);

  return Downcast<AxisTreeNodeRef>(
      operator->()->stmt_map[fused_node.as<StmtNode>()]);
}

class PredicateAdder : public IRMutator {
 public:
  PredicateAdder(Expr predicate) : predicate_(predicate) {}

  Stmt Mutate_(const BlockNode* op, const Stmt& s) final {
    return Block(op->iter_vars, op->values,
                 op->reads, op->writes,
                 op->body, op->predicate && predicate_,
                 op->annotations, op->tag);
  }
 private:
  Expr predicate_;
};

Array<AxisTreeNodeRef> Schedule::split(AxisTreeNodeRef loop, Expr factor) {
  Var outer_var = loop->loop->loop_var.copy_with_suffix(".outer");
  Var inner_var = loop->loop->loop_var.copy_with_suffix(".inner");

  Expr outer_min = loop->loop->min;
  Expr outer_extent = (loop->loop->extent + factor - 1) / factor;

  Expr inner_min = 0;
  Expr inner_extent = factor;

  auto vmap = [&](const Variable* v) -> Expr {
    if (GetRef<Var>(v).same_as(loop->loop->loop_var)) {
      return outer_var * factor + inner_var;
    } else {
      return Expr(NodePtr<Node>(nullptr));
    }
  };

  Map<Var, Range> vrange;
  vrange.Set(outer_var, Range::make_by_min_extent(outer_min, outer_extent));
  vrange.Set(inner_var, Range::make_by_min_extent(inner_min, inner_extent));
  Expr predicate = Simplify(outer_var * factor + inner_var < loop->loop->extent, vrange);
  Stmt new_stmt = PredicateAdder(predicate).Mutate(Substitute(loop->loop->body, vmap));

  Loop inner_loop(inner_var, inner_min, inner_extent, loop->loop->annotations, new_stmt);
  Loop outer_loop(outer_var, outer_min, outer_extent, loop->loop->annotations, inner_loop);

  // relink
  Replace(loop, outer_loop);

  AxisTreeNodeRef inner_axis = Downcast<AxisTreeNodeRef>(
      operator->()->stmt_map[inner_loop.as<StmtNode>()]);
  AxisTreeNodeRef outer_axis = Downcast<AxisTreeNodeRef>(
      operator->()->stmt_map[outer_loop.as<StmtNode>()]);

  return Array<AxisTreeNodeRef>{outer_axis, inner_axis};
}

bool Schedule::IsCompleteBlock(BlockTreeNodeRef block) {
  // Check the block is the only producer for every output tensors
  for (const auto& write : block->block->writes) {
    Buffer buffer = write->buffer;
    if (operator->()->write_map[buffer].size() != 1) {
      CHECK(operator->()->write_map[buffer][0].same_as(block));
      return false;
    }
  }

  // Check all the block vars are at data_par IterType
  for (const auto& iter_var : block->block->iter_vars) {
    if (iter_var->iter_type != kDataPar) {
      return false;
    }
  }
  return true;
}

TVM_REGISTER_NODE_TYPE(ScheduleNode);

}  // namespace te
}  // namespace tvm
