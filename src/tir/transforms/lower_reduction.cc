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
 * Lower block init stmt into branch stmt
 * \file lower_reduction.cc
 */
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

class ReductionTransformer : public StmtMutator {
 public:
  ReductionTransformer() = default;
 private:
  Stmt VisitStmt_(const BlockNode* block) final {
    if (!block->init.defined()) {
      return StmtMutator::VisitStmt_(block);
    }
    Stmt init = RealizeInitBlock(block->init.value(), block->iter_vars);
    Stmt body = VisitStmt(block->body);
    ObjectPtr<BlockNode> new_block = make_object<BlockNode>(*block);
    new_block->init = NullOpt;
    new_block->body = SeqStmt::Flatten(init, body);
    return Stmt(std::move(new_block));
  }

  static Stmt RealizeInitBlock(const Stmt& init, const Array<IterVar>& iter_vars) {
    std::vector<PrimExpr> conditions;
    for (const IterVar& var : iter_vars) {
      if (var->iter_type == IterVarType::kCommReduce) {
        conditions.push_back(equal(var->var, var->dom->min));
      }
    }
    int n = conditions.size();
    // Handle the case where there is no condition
    if (n == 0) {
      return init;
    }
    // Concate the conditions with logical and (&&)
    PrimExpr cond = conditions[0];
    for (int i = 1; i < n; ++i) {
      cond = logical_and(cond, conditions[i]);
    }
    return IfThenElse(cond, init);
  }
};

PrimFunc LowerReduction(PrimFunc func) {
  auto fptr = func.CopyOnWrite();
  fptr->body = ReductionTransformer()(std::move(fptr->body));
  return func;
}

namespace transform {

Pass LowerReduction() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerReduction(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerReduction", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerReduction").set_body_typed(LowerReduction);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
