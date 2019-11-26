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

namespace tvm {
namespace te {

/*!
 * \brief A help function to get children of Loop and Block
 * \param stmt the Loop or the Block
 * \return the child nodes
 */

bool is_block(Stmt stmt) {
  return stmt.as<BlockNode>() != nullptr;
}

bool is_loop(Stmt stmt) {
  return stmt.as<LoopNode>() != nullptr;
}

Array<Stmt> GetChildren(Stmt stmt) {
  Stmt body;
  Array<Stmt> children;
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
    for (size_t i = 0; i < seq->size(); ++i) {
      children.push_back(seq->operator[](i));
    }
  } else {
    children.push_back(body);
  }
  return children;
}

Schedule::Schedule(Function func,
                   DependencyGraph dependency_graph,
                   Map<Buffer, Array<Block>> write_map) {
  NodePtr<ScheduleNode> node = make_node<ScheduleNode>();
  node->func_ = std::move(func);
  node->dependency_graph_ = std::move(dependency_graph);
  node->write_map_ = std::move(write_map);
  node->father_map_ = std::move(Map<Stmt, Stmt>());
  data_ = std::move(node);
  Stmt stmt = operator->()->func_->body;
  if (const auto* seq = stmt.as<SeqStmtNode>()) {
    for (const auto s : seq->seq) {
      if (is_loop(s) || is_block(s)) {
        UpdateFather(s, true);
        operator->()->father_map_.Set(s, s);
      }
    }
  } else {
    if (is_loop(stmt) || is_block(stmt)) {
      UpdateFather(stmt, true);
      operator->()->father_map_.Set(stmt, stmt);
    }
  }
}

void Schedule::UpdateFather(Stmt stmt, bool recursive) {
  for (auto x : GetChildren(stmt)) {
    operator->()->father_map_.Set(x, stmt);
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
    if (is_loop(stmt)) {
      ret.push_back(Downcast<Loop>(stmt));
    }
    Stmt father = operator->()->father_map_[stmt];
    if (stmt != father) {
      stmt = father;
    } else {
      break;
    }
  }
  return ret;
}

TVM_REGISTER_NODE_TYPE(ScheduleNode);

}  // namespace te
}  // namespace tvm
