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

#include <tvm/tir/op.h>

#include "../utils.h"

namespace tvm {
namespace tir {

/*! \brief The auxiliary info used for normalization. */
struct NormalizerInfo {
  /*! \brief the map from Variable to loop min */
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> var_map;
  /*! \brief The map used for ScheduleStateNode::Replace */
  Map<Block, Block> block_map;
};

class LoopNormalizer : public StmtExprMutator {
 public:
  explicit LoopNormalizer(NormalizerInfo* info) : info_(info){};

  Stmt VisitStmt_(const ForNode* op) final {
    PrimExpr min = this->VisitExpr(op->min);
    PrimExpr extent = this->VisitExpr(op->extent);
    bool normalize = info_->var_map.find(op->loop_var) != info_->var_map.end();
    PrimExpr new_min = normalize ? Integer(0) : std::move(min);
    if (normalize) {
      info_->var_map[op->loop_var] = min;
    }
    Stmt body = this->VisitStmt(op->body);

    auto n = CopyOnWrite(op);
    n->min = std::move(new_min);
    n->extent = std::move(extent);
    n->body = std::move(body);
    return Stmt(n);
  }

  PrimExpr VisitExpr_(const VarNode* v) final {
    Var old_v = GetRef<Var>(v);
    auto it = info_->var_map.find(old_v);
    if (it != info_->var_map.end()) {
      return old_v + it->second;
    } else {
      return std::move(old_v);
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    Block old_stmt = GetRef<Block>(op);
    Block new_stmt = Downcast<Block>(StmtMutator::VisitStmt_(op));
    info_->block_map.Set(old_stmt, new_stmt);
    return std::move(new_stmt);
  }

 private:
  NormalizerInfo* info_;
};

void Normalize(ScheduleState self, const Array<StmtSRef>& loop_srefs) {
  if (loop_srefs.empty()) {  // input length is zero.
    return;
  }
  NormalizerInfo info;
  StmtSRef root = GetScopeRoot(self, loop_srefs[0], /*require_stage_pipeline=*/false);
  for (const StmtSRef& loop_sref : loop_srefs) {
    const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
    if (!is_zero(loop->min)) {
      info.var_map[loop->loop_var] = Integer(0);  // placeholder
    }
    CHECK(GetScopeRoot(self, loop_sref, /*require_stage_pipeline=*/false).get() == root.get())
        << "Normalize expects input loops to be in the same scope.";
  }

  LoopNormalizer normalizer(&info);
  const BlockNode* root_blk = TVM_SREF_TO_BLOCK(root_blk, root);
  Block old_blk = GetRef<Block>(root_blk);
  auto new_blk = normalizer(old_blk);
  self->Replace(root, new_blk, info.block_map);
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

TVM_REGISTER_INST_KIND_TRAITS(NormalizeTraits);

}  // namespace tir
}  // namespace tvm
