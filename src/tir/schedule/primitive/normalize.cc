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
  // Type check and collect unique loops
  std::unordered_set<const StmtSRefNode*> loops;
  for (const StmtSRef& loop_sref : loop_srefs) {
    // type check
    const auto* loop = loop_sref->StmtAs<ForNode>();
    CHECK(loop) << "TypeError: 'normalize' expects an array of loops.";
    // collect unique loops
    loops.insert(loop_sref.operator->());
  }
  // Shift the range of all loops.
  // TODO(zihao): the implementation is slow, will improve later.
  for (const auto& loop_sref : loops) {
    const ForNode* loop = loop_sref->StmtAs<ForNode>();
    Stmt new_loop_body =
        SubstituteInScope(loop->body, [&](const VarNode* v) -> PrimExpr {
          if (GetRef<Var>(v).same_as(loop->loop_var)) {
            return loop->loop_var + loop->min;
          } else {
            return PrimExpr{nullptr};
          }
        });
    For new_loop(loop->loop_var, IntImm(loop->loop_var->dtype, 0, loop->span),
                 loop->extent, loop->kind, new_loop_body);
    self->Replace(self->stmt2ref.at(loop), new_loop, {});
  }
}

}  // namespace tir
}  // namespace tvm