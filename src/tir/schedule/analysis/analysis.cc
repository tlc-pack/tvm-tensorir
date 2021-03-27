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
#include "../utils.h"

namespace tvm {
namespace tir {

/******** ContainsVar ********/

bool ContainsVar(const ObjectRef& stmt_or_expr, const Array<Var>& vars) {
  std::unordered_set<const VarNode*> vars_set;
  vars_set.reserve(vars.size());
  for (const Var& var : vars) {
    vars_set.insert(var.get());
  }
  return ContainsVar(stmt_or_expr, vars_set);
}

bool ContainsVar(const ObjectRef& stmt_or_expr, const Var& var) {
  return ContainsVar(stmt_or_expr, {var.get()});
}

bool ContainsVar(const ObjectRef& stmt_or_expr, const std::unordered_set<const VarNode*>& vars) {
  bool found = false;
  auto f_find = [&found, &vars](const ObjectRef& obj) -> bool {
    if (found) {
      return false;
    }
    if (const VarNode* var = obj.as<VarNode>()) {
      if (vars.count(var)) {
        found = true;
        return false;
      }
    }
    return true;
  };
  PreOrderVisit(stmt_or_expr, f_find);
  return found;
}

/******** Block-loop relation ********/

Array<StmtSRef> GetChildBlocks(const ScheduleState& self, const StmtSRef& parent_sref) {
  struct Collector : public StmtVisitor {
   public:
    static Array<StmtSRef> Collect(const ScheduleState& self, const Stmt& stmt) {
      Collector collector(self);
      collector(stmt);
      return std::move(collector.result_);
    }

   private:
    explicit Collector(const ScheduleState& self) : self_(self) {}

    void VisitStmt_(const BlockNode* block) final {
      auto it = self_->stmt2ref.find(block);
      ICHECK(it != self_->stmt2ref.end());
      result_.push_back(it->second);
    }

    const ScheduleState& self_;
    Array<StmtSRef> result_;
  };

  if (parent_sref->stmt->IsInstance<ForNode>()) {
    const auto* loop = static_cast<const ForNode*>(parent_sref->stmt);
    return Collector::Collect(self, loop->body);
  } else if (parent_sref->stmt->IsInstance<BlockNode>()) {
    const auto* block = static_cast<const BlockNode*>(parent_sref->stmt);
    return Collector::Collect(self, block->body);
  }
  ICHECK(false) << "Unreachable";
  throw;
}

}  // namespace tir
}  // namespace tvm
