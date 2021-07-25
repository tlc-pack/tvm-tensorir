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
#include <tvm/arith/analyzer.h>

#include "../utils.h"

namespace tvm {
namespace tir {

class LoopNormalizer : public StmtExprMutator {
 public:
  explicit LoopNormalizer(std::unordered_set<const VarNode*>* vars,
                          std::unordered_map<Var, PrimExpr, ObjectPtrHash,
                                             ObjectPtrEqual>* loop_map)
      : loop_map_(loop_map), vars_(vars){};

  Stmt VisitStmt_(const ForNode* op) final {
    PrimExpr min = this->VisitExpr(op->min);
    PrimExpr extent = this->VisitExpr(op->extent);
    bool normalize = vars_->find(op->loop_var.get()) != vars_->end();
    PrimExpr new_min = normalize ? Integer(0) : std::move(min);
    if (normalize) {
      (*loop_map_)[op->loop_var] = min;
    }
    Stmt body = this->VisitStmt(op->body);

    if (!normalize && new_min.same_as(op->min) && extent.same_as(op->extent) &&
        body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->min = std::move(new_min);
      n->extent = std::move(extent);
      n->body = std::move(body);
      return Stmt(n);
    }
  }

  PrimExpr VisitExpr_(const VarNode* v) final {
    Var v_ref = GetRef<Var>(v);
    auto it = loop_map_->find(v_ref);
    if (it != loop_map_->end()) {
      return v_ref + it->second;
    } else {
      return GetRef<PrimExpr>(v);
    }
  }

 private:
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual>* loop_map_;
  std::unordered_set<const VarNode*>* vars_;
};

void Normalize(ScheduleState self, const Array<StmtSRef>& loop_srefs) {
  CHECK(!loop_srefs.empty()) << "ValueError: 'normalize' expects 'loop_srefs' "
                                "to be an non-empty list.";
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> loop_map;
  std::unordered_set<const VarNode*> vars;
  for (const StmtSRef& loop_sref : loop_srefs) {
    const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
    vars.insert(loop->loop_var.get());
    CHECK(GetScopeRoot(loop_sref).get() == GetScopeRoot(loop_srefs[0]).get())
        << "Normalize expects input loops to be in the same scope.";
  }

  LoopNormalizer normalizer(&vars, &loop_map);
  const BlockNode* root =
      TVM_SREF_TO_BLOCK(root, GetScopeRoot(loop_srefs[0]).value());
  auto new_block = normalizer(GetRef<Block>(root));
  self->Replace(GetScopeRoot(loop_srefs[0]).value(), new_block, {});
}

struct NormalizeTraits : public UnpackedInstTraits<NormalizeTraits> {
  static constexpr const char* kName = "Normalize";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  template <size_t delta>
  static TVM_ALWAYS_INLINE void _SetInputs(const runtime::TVMArgsSetter& setter,
                                           const Array<ObjectRef>& inputs) {
    setter(delta, inputs);
  }

  static void UnpackedApplyToSchedule(Schedule sch, Array<LoopRV> loop_rvs) {
    return sch->Normalize(loop_rvs);
  }

  static String UnpackedAsPython(Array<String> outputs,
                                 Array<String> loop_rvs) {
    PythonAPICall py("normalize");
    for (const String& loop_rv : loop_rvs) {
      py.Input("", loop_rv);
    }
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND(NormalizeTraits);

}  // namespace tir
}  // namespace tvm