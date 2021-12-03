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
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../support/utils.h"
namespace tvm {
namespace tir {
class Remover : public StmtExprMutator{
 private:
  Stmt VisitStmt_(const ForNode* op) final{
    PrimExpr min = this->VisitExpr(op->min);
    PrimExpr extent = this->VisitExpr(op->extent);
    if(is_one(op->extent)){
      // handling unit loop
      unit_loop_vars_[op->loop_var] = min;
    }
    // Step 2. Visit recursively
    Stmt body = this->VisitStmt(op->body);
    if (op->annotations.count(tir::attr::pipeline_scope)) {
      pipeline_scope=Downcast<PrimExpr>(op->annotations.Get(tir::attr::pipeline_scope).value());
    }
    if (pipeline_scope.defined()) {
      if (!is_one(extent)) {
        Map<String, ObjectRef> annotations = op->annotations;
        if (op->kind ==ForKind::kSerial) {
          annotations.Set(tvm::tir::attr::pipeline_scope, pipeline_scope);
        }
        body = For(op->loop_var, std::move(min), std::move(extent), op->kind, std::move(body), op
                                                                                                   ->thread_binding,
                   annotations);
        pipeline_scope=NullOpt;
      }
    } else if (!is_one(extent) || !op->annotations.empty()) {
      body = For(op->loop_var, std::move(min), std::move(extent), op->kind, std::move(body), op
                                                                                                 ->thread_binding, op->annotations);
    }
    return body;
  }
  
  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    auto it = unit_loop_vars_.find(var);
    if (it == unit_loop_vars_.end()) {
      return std::move(var);
    } else {
      PrimExpr expr = it->second;
      if (expr.dtype() != var.dtype()) {
        expr = Cast(var.dtype(), std::move(expr));
      }
      return expr;
    }
  }
  
  /*! \brief Record the loop_var and loop start value of unit loops, whose extent is one. */
  Optional<PrimExpr> pipeline_scope = NullOpt;
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> unit_loop_vars_;
};


namespace transform {
Pass RemoveUnitLoop() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body=Remover()(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.RemoveUnitLoop", {});
}

TVM_REGISTER_GLOBAL("tir.transform.RemoveUnitLOop").set_body_typed(RemoveUnitLoop);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
