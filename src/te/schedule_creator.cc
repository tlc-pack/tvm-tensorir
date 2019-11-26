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
    : func_(func) {}

Schedule ScheduleCreator::Create() {
  NodePtr<ScheduleNode> node = make_node<ScheduleNode>();
  Stmt new_stmt = Mutate(func_->body);
  Function new_func = Function(func_->params, func_->buffer_map, func_->name, new_stmt);
  DependencyGraph dependency_graph(func_);
  return Schedule(new_func, DependencyGraph(func_), write_map_);
}

Stmt ScheduleCreator::Mutate_(const BlockNode* op, const tvm::Stmt& s) {
  Stmt body = this->Mutate(op->body);
  Block new_block = te::Block(op->iter_vars,
                              op->values,
                              op->reads,
                              op->writes,
                              body,
                              op->predicate,
                              op->annotations,
                              op->tag);

  for (const auto& write : op->writes) {
    Array<Block> array;
    if (write_map_.count(write->buffer)) {
      array = write_map_.at(write->buffer);
    }
    array.push_back(new_block);
    write_map_.Set(write->buffer, array);
  }
  return new_block;
}

Stmt ScheduleCreator::Mutate_(const te::LoopNode* op, const Stmt& s) {
  Stmt body = this->Mutate(op->body);
  return te::Loop(op->loop_var, op->min, op->extent, op->annotations, body);
}

Stmt ScheduleCreator::Mutate_(const te::SeqStmtNode* op, const Stmt& s) {
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
  new_stmt.push_back(Mutate(op->first));
  op = op->rest.as<ir::Block>();
  while (op) {
    new_stmt.push_back(Mutate(op->first));
    if (const ir::Block* t = op->rest.as<ir::Block>()) {
      op = t;
    } else {
      new_stmt.push_back(Mutate(op->rest));
      break;
    }
  }
  return te::SeqStmt(new_stmt);
}

}  // namespace te
}  // namespace tvm
