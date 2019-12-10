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
 * "AS IS" BASIS, WITHOUT WAStmtStmtANTIES OStmt CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) by Contributors
 * \file schedule_creator.cc
 */

#include <vector>
#include "schedule_creator.h"

namespace tvm {
namespace te {

ScheduleCreator::ScheduleCreator(Function func)
    : func_(func),
      dependency_graph_(make_node<DependencyGraphNode>()),
      father_node_(make_node<ScheduleTreeNode>()) {}

Schedule ScheduleCreator::Create() {
  NodePtr<ScheduleNode> node = make_node<ScheduleNode>();
  Stmt new_stmt = Mutate(func_->body);
  Function new_func = Function(func_->params, func_->buffer_map, func_->name, new_stmt);
  return Schedule(new_func, father_node_, dependency_graph_, write_map_, stmt_map);
}

Stmt ScheduleCreator::Mutate_(const BlockNode* op, const Stmt& s) {
  ScheduleTreeNodeRef father = father_node_;
  BlockTreeNodeRef block_node(nullptr, father_node_);
  Stmt body = this->Mutate(op->body);
  Block new_block = te::Block(op->iter_vars,
                              op->values,
                              op->reads,
                              op->writes,
                              body,
                              op->predicate,
                              op->annotations,
                              op->tag);
  block_node->block = new_block.as<BlockNode>();
  stmt_map[new_block.as<StmtNode>()] = block_node;

  for (const auto& write : op->writes) {
    Array<BlockTreeNodeRef> array;
    if (write_map_.count(write->buffer)) {
      array = write_map_.at(write->buffer);
    }
    array.push_back(block_node);
    write_map_.Set(write->buffer, array);
  }
  for (const auto& read : op->reads) {
    const auto& read_buffer = read->buffer;
    if (write_map_.count(read_buffer)) {
      for (const auto& write_block : write_map_[read_buffer]) {
        dependency_graph_.AddEdge(block_node, write_block, kWAR);
      }
    }
  }
  return new_block;
}

Stmt ScheduleCreator::Mutate_(const LoopNode* op, const Stmt& s) {
  ScheduleTreeNodeRef father = father_node_;
  AxisTreeNodeRef axis_node(nullptr, father_node_);
  father_node_ = axis_node;
  Stmt body = this->Mutate(op->body);
  Loop loop(op->loop_var, op->min, op->extent, op->annotations, body);
  axis_node->loop = loop.as<LoopNode>();
  stmt_map[loop.as<StmtNode>()] = axis_node;
  father_node_ = father;
  return loop;
}

Stmt ScheduleCreator::Mutate_(const SeqStmtNode* op, const Stmt& s) {
  std::vector<Stmt> new_stmt(op->size());
  for (size_t i = 0; i < op->size(); ++i) {
    Stmt old_elem = (*op)[i];
    Stmt new_elem = Mutate(old_elem);
    new_stmt[i] = new_elem;
  }
  return te::SeqStmt(new_stmt);
}

Stmt ScheduleCreator::Mutate_(const ir::Block* op, const Stmt& s) {
  std::vector<Stmt> new_stmt;
  do {
    new_stmt.push_back(Mutate(op->first));
    if (const ir::Block* t = op->rest.as<ir::Block>()) {
      op = t;
    } else {
      new_stmt.push_back(Mutate(op->rest));
      break;
    }
  } while (op);
  return te::SeqStmt(new_stmt);
}

}  // namespace te
}  // namespace tvm
