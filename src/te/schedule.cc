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

namespace tvm {
namespace te {

/*!
 * \brief A help function to get children of Loop and Block
 * \param stmt the Loop or the Block
 * \return the child nodes
 */


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

void Schedule::SetChild(Stmt father, Stmt child, size_t index) {
  if (const auto* s = father.as<BlockNode>()) {
    Block block = GetRef<Block>(s);
    if (const auto* seq = block->body.as<SeqStmtNode>()) {
      Array<Stmt> seq_stmt = seq->seq;
      seq_stmt.Set(index, child);
      block.Mutable()->body = SeqStmt(seq_stmt);
    } else {
      CHECK_EQ(index, 0);
      block.Mutable()->body = child;
    }
  } else if (const auto* s = father.as<LoopNode>()) {
    Loop loop = GetRef<Loop>(s);
    if (const auto* seq = loop->body.as<SeqStmtNode>()) {
      Array<Stmt> seq_stmt = seq->seq;
      seq_stmt.Set(index, child);
      loop.Mutable()->body = SeqStmt(seq_stmt);
    } else {
      CHECK_EQ(index, 0);
      loop.Mutable()->body = child;
    }
  } else {
    LOG(FATAL) << "Only support set child to Block or Loop";
  }
}

void Schedule::ReplaceStmt(Stmt old_stmt, Stmt new_stmt) {
  Stmt father = operator->()->father_map_[old_stmt];
  if (father.same_as(old_stmt)) {
    Stmt stmt = operator->()->func->body;
    if (const auto* seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> seq_stmt = seq->seq;
      size_t index = std::find(seq->seq.begin(), seq->seq.end(), old_stmt) - seq->seq.begin();
      seq_stmt.Set(index, new_stmt);
      Mutable()->father_map_.Set(new_stmt, new_stmt);
      Mutable()->func.Mutable()->body = SeqStmt(seq_stmt);
    } else {
      Mutable()->func.Mutable()->body = new_stmt;
    }
  } else {
    const auto& children = GetChildren(father);
    size_t index = std::find(children.begin(), children.end(), old_stmt) - children.begin();
    SetChild(father, new_stmt, index);
    UpdateFather(father);
  }
}

Schedule::Schedule(Function func,
                   DependencyGraph dependency_graph,
                   Map<Buffer, Array<Block>> write_map) {
  NodePtr<ScheduleNode> node = make_node<ScheduleNode>();
  node->func = std::move(func);
  node->dependency_graph_ = std::move(dependency_graph);
  node->write_map_ = std::move(write_map);
  node->father_map_ = std::move(Map<Stmt, Stmt>());
  data_ = std::move(node);
  Stmt stmt = operator->()->func->body;
  if (const auto* seq = stmt.as<SeqStmtNode>()) {
    for (const auto s : seq->seq) {
      if (s.as<LoopNode>() || s.as<BlockNode>()) {
        UpdateFather(s, true);
        Mutable()->father_map_.Set(s, s);
      }
    }
  } else {
    if (stmt.as<LoopNode>() || stmt.as<BlockNode>()) {
      UpdateFather(stmt, true);
      Mutable()->father_map_.Set(stmt, stmt);
    }
  }
}

void Schedule::AddPredicate(Stmt stmt, Expr predicate) {
  if (const auto* n = stmt.as<LoopNode>()) {
    auto loop = GetRef<Loop>(n);
    for (auto child : GetChildren(loop)) {
      AddPredicate(child, predicate);
    }
  } else if (const auto* n = stmt.as<BlockNode>()) {
    Block block = GetRef<Block>(n);
    block.Mutable()->predicate = block->predicate && predicate;
  }
}

void Schedule::UpdateFather(Stmt stmt, bool recursive) {
  for (auto x : GetChildren(stmt)) {
    Mutable()->father_map_.Set(x, stmt);
    if (recursive) {
      UpdateFather(x, recursive);
    }
  }
}

Array<Block> Schedule::GetBlock(std::string tag) const {
  Array<Block> ret;
  for (const auto& block : Blocks()) {
    if (block->tag == tag) {
      ret.push_back(block);
    }
  }
  return ret;
}

Array<Block> Schedule::GetBlock(Buffer buffer) const {
  if (operator->()->write_map_.count(buffer)) {
    return operator->()->write_map_.at(buffer);
  } else {
    return Array<Block>();
  }
}

Array<Block> Schedule::Blocks() const {
  Array<Block> ret;
  for (const auto& x : operator->()->write_map_) {
    for (const auto& block : x.second) {
      ret.push_back(block);
    }
  }
  return ret;
}

Array<Loop> Schedule::GetAxes(Block block) const {
  Array<Loop> ret;
  Stmt stmt = operator->()->father_map_[block];
  while (stmt) {
    if (stmt.as<LoopNode>()) {
      ret.push_back(Downcast<Loop>(stmt));
    }
    Stmt father = operator->()->father_map_[stmt];
    if (stmt != father) {
      stmt = father;
    } else {
      break;
    }
  }
  return Array<Loop>(ret.rbegin(), ret.rend());
}

Loop Schedule::fuse(Loop outer, Loop inner) {
  // Can only fuse neighbor axes without any extra branches.
  // Future Enhancement: this condition can be eliminated by lifting all siblings of inner
  // as the children of the father of outer
  CHECK(operator->()->father_map_[inner] == outer);
  auto outer_children = GetChildren(outer);
  CHECK(outer_children.size() == 1 && outer_children[0] == inner);

  // Currently, can not fuse Loops with annotations
  if (outer->annotations.size() != 0 || inner->annotations.size() != 0) {
    // TODO(tvm-team): Add ReportError
    LOG(FATAL) << "InvalidScheduleError: " << "Cannot fuse loops that already has annotations";
  }

  Expr min = 0;
  Expr extent = outer->extent * inner->extent;

  Var fused_var = outer->loop_var.copy_with_suffix(
      "." + inner->loop_var.get()->name_hint + ".fused");

  auto vmap = [&](const Variable* v) -> Expr {
    if (GetRef<Var>(v).same_as(outer->loop_var)) {
      return truncdiv(fused_var, inner->extent) + outer->min;
    } else if (GetRef<Var>(v).same_as(inner->loop_var)) {
      return truncmod(fused_var, inner->extent) + inner->min;
    } else {
      return Expr(NodePtr<Node>(nullptr));
    }
  };

  Loop fused_node = Loop(
      fused_var, min, extent, outer->annotations,
      Substitute(inner->body, vmap));

  UpdateFather(fused_node);

  // relink
  ReplaceStmt(outer, fused_node);

  return fused_node;
}

Array<Loop> Schedule::split(Loop loop, Expr factor) {
  Var outer_var = loop->loop_var.copy_with_suffix(".outer");
  Var inner_var = loop->loop_var.copy_with_suffix(".inner");

  Expr outer_min = loop->min;
  Expr outer_extent = (loop->extent + factor - 1) / factor;

  Expr inner_min = 0;
  Expr inner_extent = factor;

  auto vmap = [&](const Variable* v) -> Expr {
    if (GetRef<Var>(v).same_as(loop->loop_var)) {
      return outer_var * factor + inner_var;
    } else {
      return Expr(NodePtr<Node>(nullptr));
    }
  };

  Map<Var, Range> vrange;
  vrange.Set(outer_var, Range::make_by_min_extent(outer_min, outer_extent));
  vrange.Set(inner_var, Range::make_by_min_extent(inner_min, inner_extent));
  Expr predicate = Simplify(outer_var * factor + inner_var < loop->extent, vrange);

  Loop inner_loop(inner_var, inner_min, inner_extent, loop->annotations,
                  Substitute(loop->body, vmap));
  UpdateFather(inner_loop);

  Loop outer_loop(outer_var, outer_min, outer_extent, loop->annotations, inner_loop);
  UpdateFather(outer_loop);

  AddPredicate(outer_loop, predicate);

  // relink
  ReplaceStmt(loop, outer_loop);

  return Array<Loop>{outer_loop, inner_loop};
}

TVM_REGISTER_NODE_TYPE(ScheduleNode);

}  // namespace te
}  // namespace tvm
