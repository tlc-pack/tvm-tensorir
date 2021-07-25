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

void Normalize(ScheduleState self, const Array<StmtSRef>& loop_srefs) {
  CHECK(!loop_srefs.empty()) << "ValueError: 'normalize' expects 'loop_srefs' "
                                "to be an non-empty list.";
  // Collect unique loops
  std::unordered_set<const StmtSRefNode*> loops;
  for (const StmtSRef& loop_sref : loop_srefs) {
    const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
    loops.insert(loop_sref.operator->());
  }
  // Shift the range of all loops.
  for (const auto& loop_sref : loops) {
    const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
    Stmt new_loop_body =
        SubstituteInScope(loop->body, [&](const VarNode* v) -> PrimExpr {
          if (v == loop->loop_var.get()) {
            return loop->loop_var + loop->min;
          } else {
            return PrimExpr{nullptr};
          }
        });
    For new_loop(loop->loop_var, Integer(0), loop->extent, loop->kind,
                 new_loop_body);
    self->Replace(self->stmt2ref.at(loop), new_loop, {});
  }
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