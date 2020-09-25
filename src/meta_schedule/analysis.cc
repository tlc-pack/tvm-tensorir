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
#include "./analysis.h"  // NOLINT(build/include)

#include <tvm/arith/analyzer.h>
#include <tvm/tir/stmt_functor.h>

#include "../tir/schedule/schedule_common.h"  // TODO(@junrushao1994): replace it

namespace tvm {
namespace meta_schedule {

bool IsTrivialBinding(Schedule sch, BlockRV block_rv) {
  tir::StmtSRef block_sref = sch->Eval(block_rv);
  const auto* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block) << "TypeError: Expects Block, but gets: " << block_sref->stmt->GetTypeKey();
  tir::BlockRealize realize = tir::GetBlockRealize(block_sref);
  Array<tir::StmtSRef> loops = sch->sch->GetLoopsInScope(block_sref);
  const Array<PrimExpr>& bindings = realize->binding_values;
  if (loops.size() != bindings.size()) {
    return false;
  }
  int n = loops.size();
  arith::Analyzer analyzer;
  for (int i = 0; i < n; ++i) {
    const PrimExpr& bind = bindings[i];
    const auto* loop = loops[i]->GetStmt<tir::LoopNode>();
    CHECK(loop) << "TypeError: Expects Loop, but gets: " << loops[i]->stmt->GetTypeKey();
    if (!analyzer.CanProve(bind == loop->loop_var)) {
      return false;
    }
  }
  return true;
}

Array<Integer> GetIterType(Schedule sch, BlockRV block_rv) {
  tir::StmtSRef block_sref = sch->Eval(block_rv);
  const auto* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block) << "TypeError: Expects Block, but gets: " << block_sref->stmt->GetTypeKey();
  Array<Integer> result;
  for (const tir::IterVar& iter_var : block->iter_vars) {
    int iter_type = iter_var->iter_type;
    result.push_back(iter_type);
  }
  return result;
}

bool IsLeaf(Schedule sch, BlockRV block_rv) {
  tir::StmtSRef block_sref = sch->Eval(block_rv);
  const auto* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block) << "TypeError: Expects Block, but gets: " << block_sref->stmt->GetTypeKey();
  bool is_leaf = true;
  tir::PreOrderVisit(block->body, [&is_leaf](const ObjectRef& obj) -> bool {
    if (is_leaf == false) {
      return false;
    }
    if (obj->IsInstance<tir::BlockNode>()) {
      is_leaf = false;
      return false;
    }
    return true;
  });
  return is_leaf;
}

bool IsBodySingleStmt(Schedule sch, BlockRV block_rv) {
  tir::StmtSRef block_sref = sch->Eval(block_rv);
  const auto* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block) << "TypeError: Expects Block, but gets: " << block_sref->stmt->GetTypeKey();
  const tir::Stmt& body = block->body;
  if (body->IsInstance<tir::BufferStoreNode>()) {
    return true;
  }
  if (body->IsInstance<tir::ReduceStepNode>()) {
    return true;
  }
  return false;
}

tir::BufferLoad GetBufferStore(Schedule sch, BlockRV block_rv) {
  tir::StmtSRef block_sref = sch->Eval(block_rv);
  const auto* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block) << "TypeError: Expects Block, but gets: " << block_sref->stmt->GetTypeKey();
  if (const auto* body = block->body.as<tir::BufferStoreNode>()) {
    return tir::BufferLoad(body->buffer, body->indices);
  }
  if (const auto* body = block->body.as<tir::ReduceStepNode>()) {
    const auto* buffer_update = body->lhs.as<tir::BufferLoadNode>();
    CHECK(buffer_update) << "TypeError: LHS of ReduceStep is expected to be BufferLoad, but gets: "
                         << body->lhs->GetTypeKey();
    return GetRef<tir::BufferLoad>(buffer_update);
  }
  LOG(FATAL) << "ValueError: `GetBufferStore` only applies to a leaf block whose body is single "
                "statement, but get: "
             << GetRef<tir::Block>(block);
  throw;
}

Array<tir::BufferLoad> GetBufferLoad(Schedule sch, BlockRV block_rv) {
  tir::StmtSRef block_sref = sch->Eval(block_rv);
  const auto* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block) << "TypeError: Expects Block, but gets: " << block_sref->stmt->GetTypeKey();
  Array<tir::BufferLoad> reads;
  auto f_visit = [&reads](const ObjectRef& obj) {
    if (const auto* load = obj.as<tir::BufferLoadNode>()) {
      reads.push_back(GetRef<tir::BufferLoad>(load));
    }
  };
  if (const auto* body = block->body.as<tir::BufferStoreNode>()) {
    tir::PostOrderVisit(body->value, f_visit);
    return reads;
  }
  if (const auto* body = block->body.as<tir::ReduceStepNode>()) {
    tir::PostOrderVisit(body->rhs, f_visit);
    return reads;
  }
  LOG(FATAL) << "ValueError: `GetBufferLoad` only applies to a leaf block whose body is single "
                "statement, but get: "
             << GetRef<tir::Block>(block);
  throw;
}

int CountOp(Schedule sch, BlockRV block_rv, Op op) {
  tir::StmtSRef block_sref = sch->Eval(block_rv);
  const auto* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block) << "TypeError: Expects Block, but gets: " << block_sref->stmt->GetTypeKey();
  int count = 0;
  tir::PostOrderVisit(block->body, [&count, &op](const ObjectRef& obj) {
    if (const auto* call = obj.as<tir::CallNode>()) {
      if (call->op.same_as(op)) {
        ++count;
      }
    }
  });
  return count;
}

int HasBranch(Schedule sch, BlockRV block_rv) {
  tir::StmtSRef block_sref = sch->Eval(block_rv);
  const auto* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block) << "TypeError: Expects Block, but gets: " << block_sref->stmt->GetTypeKey();
  bool has_branch = false;
  arith::Analyzer analyzer;
  const Op& op_if_then_else = Op::Get("tir.if_then_else");
  auto f_visit = [&has_branch, &analyzer, &op_if_then_else](const ObjectRef& obj) -> bool {
    if (has_branch) {
      // stop visiting
      return false;
    }
    if (const auto* realize = obj.as<tir::BlockRealizeNode>()) {
      // Case 1: BlockRealize
      if (!analyzer.CanProve(realize->predicate == 1)) {
        has_branch = true;
        return false;
      }
    } else if (obj->IsInstance<tir::IfThenElseNode>() || obj->IsInstance<tir::SelectNode>()) {
      // Case 2: IfThenElse / Select
      has_branch = true;
      return false;
    } else if (const auto* call = obj.as<tir::CallNode>()) {
      // Case 3: Call
      if (call->op.same_as(op_if_then_else)) {
        has_branch = true;
        return false;
      }
    }
    // continue visiting
    return true;
  };
  tir::PreOrderVisit(tir::GetBlockRealize(block_sref), f_visit);
  return has_branch;
}

TVM_REGISTER_GLOBAL("meta_schedule.analysis.IsTrivialBinding").set_body_typed(IsTrivialBinding);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.GetIterType").set_body_typed(GetIterType);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.IsLeaf").set_body_typed(IsLeaf);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.IsBodySingleStmt").set_body_typed(IsBodySingleStmt);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.GetBufferStore").set_body_typed(GetBufferStore);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.GetBufferLoad").set_body_typed(GetBufferLoad);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.CountOp").set_body_typed(CountOp);
TVM_REGISTER_GLOBAL("meta_schedule.analysis.HasBranch").set_body_typed(HasBranch);

}  // namespace meta_schedule
}  // namespace tvm
