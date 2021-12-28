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

class AddDataCopyConstraintsMemHammerNode : public ScheduleRuleNode{
 public:
  
  // Inherited from ScheduleRuleNode
  void InitializeWithTuneContext(const TuneContext& context) final {}
  
  // Inherited from ScheduleRuleNode
  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) final {
    tir::Block block = sch->Get(block_rv);
    if (block->annotations.count("auto_copy") && tir::is_one
        (Downcast<PrimExpr>(block->annotations["auto_copy"]))) {
      ICHECK_EQ(block->reads.size(),1);
      ICHECK_EQ(block->writes.size(),1);
      tir::Buffer read_buffer = block->reads[0]->buffer;
      tir::Buffer write_buffer = block->writes[0]->buffer;
      runtime::StorageScope read_scope = runtime::StorageScope::Create(read_buffer.scope());
      runtime::StorageScope write_scope = runtime::StorageScope::Create(write_buffer.scope());
      Array<FloatImm> probs(3, FloatImm(DataType::Float(64), 1.0/3));
      if ((read_scope.rank == runtime::StorageRank::kGlobal &&
           write_scope.rank == runtime::StorageRank::kShared) ||
          (read_scope.rank == runtime::StorageRank::kShared &&
           write_scope.rank == runtime::StorageRank::kGlobal) ||
          (read_scope.rank == runtime::StorageRank::kWMMAAccumulator &&
           write_scope.rank == runtime::StorageRank::kGlobal)) {
        PrimExpr ann_val = sch->SampleCategorical({4, 8, 16}, probs);
        sch->Annotate(block_rv, tir::attr::vector_bytes, ann_val);
      }
    }
    return {sch};
  }
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}
  
  static constexpr const char* _type_key = "meta_schedule.AddDataCopyConstraintsMemHammer";
  TVM_DECLARE_FINAL_OBJECT_INFO(AddDataCopyConstraintsMemHammerNode, ScheduleRuleNode);
};

ScheduleRule ScheduleRule::AddDataCopyConstraintsMemHammer() {
  ObjectPtr<AddDataCopyConstraintsMemHammerNode> n = make_object<AddDataCopyConstraintsMemHammerNode>();
  return ScheduleRule(n);
}

TVM_REGISTER_NODE_TYPE(AddDataCopyConstraintsMemHammerNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleAddDataCopyConstraintsMemHammer")
    .set_body_typed(ScheduleRule::AddDataCopyConstraintsMemHammer);

}  // namespace meta_schedule
}  // namespace tvm
