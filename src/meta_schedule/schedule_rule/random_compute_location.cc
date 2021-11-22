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
namespace meta_schedule {

class RandomComputeLocationNode : public ScheduleRuleNode {
 public:
  bool IsFreeBlock(const tir::Schedule sch, const tir::StmtSRef& block_sref) const {
    if (block_sref->parent == nullptr) {
      return false;
    }
    if (!tir::IsSubrootBlock(sch->state(), block_sref)) {
      return false;
    }
    tir::ScheduleState state = sch->state();
    if (!tir::IsCompleteBlock(state, block_sref,
                              tir::GetScopeRoot(state, block_sref, false, false))) {
      return false;
    }
    Array<tir::StmtSRef> loop_srefs = tir::GetLoops(block_sref);
    for (const tir::StmtSRef& loop_sref : loop_srefs) {
      if (!tir::HasSingleChild(loop_sref)) {
        return false;
      }
    }
    Array<PrimExpr> binds = tir::GetBlockRealize(state, block_sref)->iter_values;
    for (const PrimExpr& bind : binds) {
      if (!bind->IsInstance<IntImmNode>() && !bind->IsInstance<tir::VarNode>()) {
        return false;
      }
    }
    return true;
  }

  // Inherited from ScheduleRuleNode
  void InitializeWithTuneContext(const TuneContext& context) final {}

  // Inherited from ScheduleRuleNode
  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) final {
    tir::StmtSRef block_sref = sch->GetSRef(block_rv);
    if (!IsFreeBlock(sch, block_sref)) {
      return {sch};
    }
    Array<tir::BlockRV> consumers = sch->GetConsumers(block_rv);
    if (consumers.size() != 1) {
      return {sch};
    }
    tir::BlockRV consumer = consumers[0];
    // Try to compute `block_rv` at `consumer`
    for (;;) {
      tir::LoopRV compute_at_loc = sch->SampleComputeLocation(consumer);
      try {
        sch->ComputeAt(block_rv, compute_at_loc, true);
      } catch (const dmlc::Error& e) {
        // ComputeAt fails, cleanup the following before re-try:
        // 1) trace: instruction & decisions
        // 2) sym_tab
        sch->trace().value()->Pop();
        sch->RemoveRV(compute_at_loc);
        continue;
      }
      break;
    }
    return {sch};
  }

 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "meta_schedule.RandomComputeLocation";
  TVM_DECLARE_FINAL_OBJECT_INFO(RandomComputeLocationNode, ScheduleRuleNode);
};

ScheduleRule ScheduleRule::RandomComputeLocation() {
  ObjectPtr<RandomComputeLocationNode> n = make_object<RandomComputeLocationNode>();
  return ScheduleRule(n);
}

TVM_REGISTER_NODE_TYPE(RandomComputeLocationNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleRandomComputeLocation")
    .set_body_typed(ScheduleRule::RandomComputeLocation);

}  // namespace meta_schedule
}  // namespace tvm
