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

#include "sub_replacer.h"

namespace tvm {
namespace te {

Stmt SubReplacer::Mutate_(const LoopNode* op) {
  Stmt body;
  if (const auto* seq = op->body.as<SeqStmtNode>()) {
    body = Mutate_(seq);
  } else {
    CHECK(op->body.same_as(old_stmt_));
    body = new_stmt_;
  }
  if (need_copy && op->unique()) {
    auto* seq = const_cast<LoopNode*>(op);
    seq->body = body;
    need_copy = false;
    return GetRef<Loop>(op);
  } else if (need_copy && !op->unique()) {
    Loop new_loop(op->loop_var, op->min, op->extent,
                  op->annotations, body);
    const auto* stmt_node = new_loop.as<StmtNode>();
    const auto* old_stmt = static_cast<const StmtNode*>(op);
    schedule_->stmt_map[stmt_node] = schedule_->stmt_map[old_stmt];
    schedule_->stmt_map.erase(old_stmt);
    return std::move(new_loop);
  } else {
    return GetRef<Loop>(op);
  }
}

Stmt SubReplacer::Mutate_(const BlockNode* op) {
  Stmt body;
  if (const auto* seq = op->body.as<SeqStmtNode>()) {
    body = Mutate_(seq);
  } else {
    CHECK(op->body.same_as(old_stmt_));
    body = new_stmt_;
  }
  if (need_copy && op->unique()) {
    auto* seq = const_cast<BlockNode*>(op);
    seq->body = body;
    need_copy = false;
    return GetRef<Block>(op);
  } else if (need_copy && !op->unique()) {
    Block new_block(op->iter_vars, op->values,
                    op->reads, op->writes,
                    body, op->predicate,
                    op->allocations,
                    op->annotations, op->tag);
    const auto* stmt_node = new_block.as<StmtNode>();
    const auto* old_stmt = static_cast<const StmtNode*>(op);
    schedule_->stmt_map[stmt_node] = schedule_->stmt_map[old_stmt];
    schedule_->stmt_map.erase(old_stmt);
    return std::move(new_block);
  } else {
    return GetRef<Block>(op);
  }
}

Stmt SubReplacer::Mutate_(const SeqStmtNode* op) {
  if (op->unique()) {
    SeqStmtNode* seq = const_cast<SeqStmtNode*>(op);
    for (size_t i = 0; i < seq->size(); ++i) {
      if (seq->operator[](i).same_as(old_stmt_)) {
        seq->operator[](i) = new_stmt_;
      }
    }
    need_copy = false;
    return GetRef<SeqStmt>(op);
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

Stmt SubReplacer::Mutate(const StmtNode* op) {
  if (op->IsInstance<BlockNode>()) {
    return Mutate_(static_cast<const BlockNode*>(op));
  } else if (op->IsInstance<LoopNode>()) {
    return Mutate_(static_cast<const LoopNode*>(op));
  } else {
    LOG(FATAL) << "Unsupported AST Node";
    return Stmt();
  }
}

}  // namespace te
}  // namespace tvm
