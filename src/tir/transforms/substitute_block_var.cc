/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file substitute_block_var.cc
 * \brief Compact the buffer size into its exact need.
 */

#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

/*! \brief Helper class to mutate the buffer access. */
class BlockVarSubstituter : public StmtExprMutator {
 public:
  static Stmt Substitute(const PrimFunc& f) {
    BlockVarSubstituter substituter;
    return substituter.VisitStmt(f->body);
  }

 private:
  explicit BlockVarSubstituter() = default;

  PrimExpr VisitExpr_(const VarNode* var) final {
    auto it = var_substitutes_.find(var);
    if (it != var_substitutes_.end()) {
      return it->second;
    }
    return GetRef<Var>(var);
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    ICHECK(!block->init.defined())
        << "Block Init part is not allowed in pass substituter_block_var";
    Stmt s = StmtExprMutator::VisitStmt_(block);
    block = s.as<BlockNode>();
    ICHECK(block != nullptr);
    if (block->iter_vars.empty()) {
      return GetRef<Stmt>(block);
    } else {
      auto n = CopyOnWrite(block);
      n->iter_vars = {};
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    const auto* block_op = realize->block.get();
    ICHECK(!block_op->init.defined());
    // Step 1. Update "block vars => loop vars" for substitution, add reduction loop vars
    ICHECK_EQ(block_op->iter_vars.size(), realize->iter_values.size());
    for (int i = 0, n = block_op->iter_vars.size(); i < n; ++i) {
      IterVar block_var = block_op->iter_vars[i];
      PrimExpr v = this->VisitExpr(realize->iter_values[i]);
      var_substitutes_.emplace(block_var->var.get(), v);
    }
    // Step 2. Visit recursively
    Stmt s = StmtExprMutator::VisitStmt_(realize);
    realize = s.as<BlockRealizeNode>();
    ICHECK(realize != nullptr);
    if (realize->iter_values.empty()) {
      return GetRef<Stmt>(realize);
    } else {
      auto n = CopyOnWrite(realize);
      n->iter_values = {};
      return Stmt(n);
    }
  }
  std::unordered_map<const VarNode*, PrimExpr> var_substitutes_;
};

PrimFunc SubstituteBlockVar(PrimFunc f) {
  PrimFuncNode* fptr = f.CopyOnWrite();
  fptr->body = BlockVarSubstituter::Substitute(f);
  return f;
}

namespace transform {

Pass SubstituteBlockVar() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return SubstituteBlockVar(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.SubstituteBlockVar", {});
}

TVM_REGISTER_GLOBAL("tir.transform.SubstituteBlockVar").set_body_typed(SubstituteBlockVar);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
