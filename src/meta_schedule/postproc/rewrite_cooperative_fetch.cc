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

/*!
 * \brief Parse instruction: sch.bind(..., "threadIdx.x")
 * \param sch The schedule
 * \param inst The instruction to be parsed
 * \return NullOpt if parsing fails; Otherwise, the extent of thread axis
 */
Optional<Integer> ParseThreadBinding(const Schedule& sch, const Instruction& inst) {
  static InstructionKind inst_kind_bind = InstructionKind::Get("Bind");
  if (!inst->kind.same_as(inst_kind_bind)) {
    return NullOpt;
  }
  ICHECK_EQ(inst->inputs.size(), 1);
  ICHECK_EQ(inst->attrs.size(), 1);
  String thread_axis = Downcast<String>(inst->attrs[0]);
  if (thread_axis != "threadIdx.x") {
    return NullOpt;
  }
  return Downcast<Integer>(sch->Get(Downcast<LoopRV>(inst->inputs[0]))->extent);
}

/*!
 * \brief Parse instruction: sch.annotate(..., attr::meta_schedule_cooperative_fetch)
 * \param sch The schedule
 * \param inst The instruction to be parsed
 * \param vector_lane The length of vector lane in vectorized cooperative fetching
 * \return NullOpt if parsing fails; Otherwise, the annotated block
 */
Optional<BlockRV> ParseAnnotate(const Schedule& sch, const Instruction& inst, int* vector_lane) {
  static InstructionKind inst_kind_annotate = InstructionKind::Get("Annotate");
  if (!inst->kind.same_as(inst_kind_annotate)) {
    return NullOpt;
  }
  ICHECK_EQ(inst->inputs.size(), 2);
  ICHECK_EQ(inst->attrs.size(), 1);
  String ann_key = Downcast<String>(inst->attrs[0]);
  if (ann_key != attr::meta_schedule_cooperative_fetch) {
    return NullOpt;
  }
  *vector_lane = Downcast<Integer>(sch->Get(Downcast<ExprRV>(inst->inputs[1])))->value;
  return Downcast<BlockRV>(inst->inputs[0]);
}

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Rewrite the cooperative fetch annotation to actual vectorized cooperative fetching
 * in loop bindings.
 */
class RewriteCooperativeFetchNode : public PostprocNode {
 public:
  // Inherited from PostprocNode
  void InitializeWithTuneContext(const TuneContext& context) final {}
  // Inherited from PostprocNode
  bool Apply(const tir::Schedule& sch) final;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "meta_schedule.RewriteCooperativeFetch";
  TVM_DECLARE_FINAL_OBJECT_INFO(RewriteCooperativeFetchNode, PostprocNode);
};

bool RewriteCooperativeFetchNode::Apply(const tir::Schedule& sch) {
  using tir::BlockRV;
  using tir::Instruction;
  using tir::LoopRV;
  using tir::Schedule;
  using tir::Trace;
  Trace trace = sch->trace().value();
  int thread_extent = -1;
  int vector_lane = -1;
  std::vector<std::function<void()>> tasks;
  for (const Instruction& inst : trace->insts) {
    if (Optional<Integer> new_thread_extent = tir::ParseThreadBinding(sch, inst)) {
      thread_extent = new_thread_extent.value()->value;
    }
    if (Optional<BlockRV> block_rv = tir::ParseAnnotate(sch, inst, &vector_lane)) {
      ICHECK_NE(thread_extent, -1);
      if (vector_lane > 1) {
        tasks.push_back([thread_extent, vector_lane, sch, block = block_rv.value()]() -> void {
          LoopRV fused = sch->GetLoops(block).back();
          Array<LoopRV> split = sch->Split(fused, {NullOpt,                 //
                                                   Integer(thread_extent),  //
                                                   Integer(vector_lane)});
          sch->Vectorize(split[2]);
          sch->Bind(split[1], "threadIdx.x");
        });
      } else {
        tasks.push_back([thread_extent, sch, block = block_rv.value()]() -> void {
          LoopRV fused = sch->GetLoops(block).back();
          Array<LoopRV> split = sch->Split(fused, {NullOpt, Integer(thread_extent)});
          sch->Bind(split[1], "threadIdx.x");
        });
      }
    }
  }
  for (auto&& task : tasks) {
    task();
  }
  return true;
}

Postproc Postproc::RewriteCooperativeFetch() {
  ObjectPtr<RewriteCooperativeFetchNode> n = make_object<RewriteCooperativeFetchNode>();
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(RewriteCooperativeFetchNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocRewriteCooperativeFetch")
    .set_body_typed(Postproc::RewriteCooperativeFetch);

}  // namespace meta_schedule
}  // namespace tvm
