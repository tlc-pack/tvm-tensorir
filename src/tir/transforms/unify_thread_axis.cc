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
 * \file unify_thread_axis.cc
 * \brief The pass to unify thread axis into the same IterVar.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "../../runtime/thread_storage_scope.h"
#include "ir_utils.h"
#include "storage_access.h"

namespace tvm {
namespace tir {

class ThreadAxisUnifier : private StmtExprMutator {
 public:

  Stmt Unify(const Stmt& stmt) {
    Stmt new_stmt = VisitStmt(stmt);
    for (const auto& kv : thread_extents_) {
      const auto& attr_stmt = kv.second;
      new_stmt = AttrStmt(attr_stmt->node, attr_stmt->attr_key, attr_stmt->value, new_stmt);
    }
    return new_stmt;
  }

 private:
  Stmt VisitStmt_(const AttrStmtNode* op) {
    if (op->attr_key == attr::thread_extent) {
      IterVar iter_var = Downcast<IterVar>(op->node);
      auto it = thread_extents_.find(iter_var->thread_tag);

      if (it != thread_extents_.end()) {
        ICHECK(analyzer_.CanProveEqual((*it).second->value, op->value)) << "Conflicting extends for thread axis " << iter_var->thread_tag;
        var_map_.Set(iter_var->var, Downcast<IterVar>((*it).second->node)->var);
        // thread_multiple_defined.insert(iter_var->thread_tag);
      }
      else {
        // first definition of the thread axis
        thread_extents_.Set(iter_var->thread_tag, GetRef<AttrStmt>(op));
        // auto body = VisitStmt(op->body);
        // if (thread_multiple_defined.count(iter_var->thread_tag)) {
        //   return body;
        // } else {
        //   if (body.same_as(op->body)) {
        //     return GetRef<Stmt>(op);
        //   } else {
        //     return AttrStmt()
        //   }
        // }
      }
     return VisitStmt(op->body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const VarNode* op) {
    auto it = var_map_.find(GetRef<Var>(op));
    if (it != var_map_.end()) {
      return (*it).second;
    }
    return GetRef<Var>(op);
  }

  Map<String, AttrStmt> thread_extents_;
  // std::unordered_set<String, ObjectPtrHash, ObjectPtrEqual> thread_multiple_defined;
  Map<Var, Var> var_map_;
  arith::Analyzer analyzer_;
};

namespace transform {

Pass UnifyThreadAxis() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = ThreadAxisUnifier().Unify(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.UnifyThreadAxis", {});
}

TVM_REGISTER_GLOBAL("tir.transform.UnifyThreadAxis").set_body_typed(UnifyThreadAxis);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
