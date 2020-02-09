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

#include <tvm/tir/ir.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/schedule.h>
#include <utility>
#include "schedule_common.h"


namespace tvm {
namespace tir {

/*! \note Nested SeqStmt is not allowed in schedule. */
Array<Stmt> GetChildren(const Stmt& stmt, bool keep_realize) {
  Stmt body;
  if (const auto* block = stmt.as<BlockNode>()) {
    body = block->body;
  } else if (const auto* loop = stmt.as<LoopNode>()) {
    body = loop->body;
  } else {
    return Array<Stmt>();
  }
  if (const auto* seq = body.as<SeqStmtNode>()) {
    Array<Stmt> ret;
    for (const Stmt& child : seq->seq)
      if (child->IsInstance<BlockRealizeNode>() && !keep_realize) {
        ret.push_back(child.as<BlockRealizeNode>()->block);
      } else {
        ret.push_back(child);
      }
    return ret;
  } else {
    return Array<Stmt>{body};
  }
}

class IRSubstitueInScope : public StmtExprMutator {
 public:
  explicit IRSubstitueInScope(
      std::function<PrimExpr(const VarNode*)> fmap)
      : fmap_(std::move(fmap)) {}

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = fmap_(op);
    if (it.defined()) {
      return it;
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    auto fmutate = [this](const PrimExpr& e) { return this->VisitExpr(e); };
    Array<PrimExpr> v = op->binding_values;
    v.MutateByApply(fmutate);
    PrimExpr pred = this->VisitExpr(op->predicate);
    if (v.same_as(op->binding_values) && pred.same_as(op->predicate)) {
      return GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->binding_values = std::move(v);
      n->predicate = std::move(pred);
      return Stmt(n);
    }
  }

 private:
  const std::function<PrimExpr(const VarNode*)> fmap_;
};

Stmt SubstituteInScope(const Stmt& stmt,
                       const std::function<PrimExpr(const VarNode*)>& value_func) {
  return IRSubstitueInScope(value_func)(stmt);
}

// Only Block and Loop are allowed here.
template <typename T>
Stmt GetStmtFromSeq(const T* op,
                    const Stmt& target,
                    const std::function<bool(const Stmt&, const Stmt&)>& f_equal,
                    int64_t seq_index) {
  if (const auto* seq = op->body.template as<SeqStmtNode>()) {
    if (seq_index >= 0) {
      // fast path
      CHECK(f_equal((*seq)[seq_index], target));
      return (*seq)[seq_index];
    } else {
      // apply slow path when seq_index == -1
      for (const auto& s : seq->seq) {
        if (f_equal(s, target)) return (*seq)[seq_index];
      }
      LOG(FATAL) << "Can not find target stmt";
    }
  } else {
    CHECK(f_equal(op->body, target));
    return op->body;
  }
  return NullValue<Stmt>();
}

template <typename T>
Stmt GetStmtFromSeq(const T* op, const Stmt& target, int64_t seq_index) {
  auto f_equal = [](const Stmt& s, const Stmt& target) {
    return s.same_as(target);
  };
  return GetStmtFromSeq(op, target, f_equal, seq_index);
}

BlockRealize GetBlockRealize(const StmtSRef& block_sref) {
  Stmt s = GetRef<Stmt>(block_sref->node);
  CHECK(GetRef<Stmt>(block_sref->node).as<BlockNode>());
  const auto* parent = block_sref->parent;
  Stmt parent_stmt = GetRef<Stmt>(parent->node);

  auto f_equal = [](const Stmt& s, const Stmt& target) {
    CHECK(target.as<BlockNode>());
    const auto* block_realize = s.as<BlockRealizeNode>();
    if (block_realize != nullptr) {
      return block_realize->block.same_as(target);
    } else {
      return false;
    }
  };

  if (const auto* block = parent_stmt.as<BlockNode>()) {
    return Downcast<BlockRealize>(GetStmtFromSeq(block, s, f_equal, block_sref->seq_index));
  } else if (const auto* loop = parent_stmt.as<LoopNode>()) {
    return Downcast<BlockRealize>(GetStmtFromSeq(loop, s, f_equal, block_sref->seq_index));
  } else {
    LOG(FATAL) << "Unknown SRef Type";
  }
  return NullValue<BlockRealize>();
}

}  // namespace tir
}  // namespace tvm
