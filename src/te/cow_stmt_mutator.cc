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

/*!
 * \brief Stmt mutator that implements COW semantics
 * \file cow_stmt_mutator.cc
 */
#include "cow_stmt_mutator.h"
#include <tvm/te/ir.h>
#include <utility>
#include "../pass/ir_util.h"

namespace tvm {
namespace te {

Stmt COWStmtMutator::VisitStmt_(const IfThenElse* op) {
  Expr condition = this->Mutate(op->condition);
  Stmt then_case = this->Mutate(op->then_case);
  Stmt else_case;
  if (op->else_case.defined()) {
    else_case = this->Mutate(op->else_case);
  }
  if (condition.same_as(op->condition) &&
      then_case.same_as(op->then_case) &&
      else_case.same_as(op->else_case)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = this->CopyOnWrite(op);
    n->condition = std::move(condition);
    n->then_case = std::move(then_case);
    n->else_case = std::move(else_case);
    return Stmt(n);
  }
}

Stmt COWStmtMutator::VisitStmt_(const Evaluate* op) {
  Expr value = this->Mutate(op->value);
  if (value.same_as(op->value)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->value = std::move(value);
    return Stmt(n);
  }
}

Stmt COWStmtMutator::VisitStmt_(const LoopNode* op) {
  Expr min = this->Mutate(op->min);
  Expr extent = this->Mutate(op->extent);
  Stmt body = this->Mutate(op->body);
  if (min.same_as(op->min) && extent.same_as(op->extent) &&
      body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->min = std::move(min);
    n->extent = std::move(extent);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt COWStmtMutator::VisitStmt_(const BlockNode* op) {
  auto v = UpdateArray(op->values, [this](const Expr& e) { return Mutate(e); });
  Expr pred = this->Mutate(op->predicate);
  Stmt body = this->Mutate(op->body);
  if (v.same_as(op->values) && pred.same_as(op->predicate)
      && body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->values = std::move(v);
    n->predicate = std::move(pred);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt COWStmtMutator::VisitStmt_(const SeqStmtNode* op) {
  NodePtr<SeqStmtNode> new_seq;
  for (size_t i = 0; i < op->size(); ++i) {
    Stmt old_elem = (*op)[i];
    Stmt new_elem = Mutate(old_elem);
    if (!new_elem.same_as(old_elem)) {
      if (new_seq == nullptr) new_seq = CopyOnWrite(op);
      new_seq->seq.Set(i, new_elem);
    }
  }
  if (new_seq == nullptr) {
    return GetRef<Stmt>(op);
  } else {
    return Stmt(new_seq);
  }
}

}  // namespace te
}  // namespace tvm
