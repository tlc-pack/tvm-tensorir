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
  bool CheckConditions(const tir::Schedule sch, const tir::BlockRV& block_rv) const {
    const tir::StmtSRef& block_sref = sch->GetSRef(block_rv);
    const tir::BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);

    // Cond 1. The block is not the root block.
    if (block_sref->parent == nullptr) {
      return false;
    }
    // Cond 2. The block should be the direct child block of the root block.
    if (GetScopeRoot(sch->state(), block_sref,          //
                     /*require_stage_pipeline=*/false,  //
                     /*require_subtree_compact_dataflow=*/false)
            ->parent != nullptr) {
      return false;
    }
    // Cond 3 & 4. The block has at least one outer loop, and the outermost loop has only one child
    // block.
    Array<tir::StmtSRef> loop_srefs = tir::GetLoops(block_sref);
    if (loop_srefs.empty()) {
      return false;
    }
    if (tir::GetChildBlockSRefOnSRefTree(sch->state(), loop_srefs[0]).size() > 1) {
      return false;
    }
    // Cond 5. The block has at lease one consumer.
    if (tir::GetConsumers(sch->state(), sch->GetSRef(block_rv)).empty()) {
      return false;
    }

    return true;
  }

  // Inherited from ScheduleRuleNode
  void InitializeWithTuneContext(const TuneContext& context) final {}

  // Inherited from ScheduleRuleNode
  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) final {
    if (!CheckConditions(sch, block_rv)) {
      return {sch};
    }

    for (;;) {
      tir::LoopRV compute_at_loc = sch->SampleComputeLocation(block_rv);
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
