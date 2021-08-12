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
#include <tvm/target/target.h>

#include "./traced_schedule.h"

namespace tvm {
namespace tir {

Schedule Schedule::Traced(IRModule mod, int64_t seed, int debug_mode,
                          ScheduleErrorRenderLevel error_render_level) {
  ObjectPtr<TracedScheduleNode> n = make_object<TracedScheduleNode>();
  n->state_ = ScheduleState(mod, debug_mode);
  n->error_render_level_ = error_render_level;
  n->sampler_.Seed(seed);
  n->symbol_table_ = {};
  n->analyzer_ = std::make_unique<arith::Analyzer>();
  n->trace_ = Trace();
  return Schedule(std::move(n));
}

Schedule TracedScheduleNode::Copy(int64_t new_seed) const {
  ObjectPtr<TracedScheduleNode> n = make_object<TracedScheduleNode>();
  ConcreteScheduleNode::Copy(&n->state_, &n->symbol_table_);
  n->error_render_level_ = this->error_render_level_;
  n->analyzer_ = std::make_unique<arith::Analyzer>();
  n->sampler_.Seed(new_seed);
  n->trace_ = Trace(this->trace_->insts, this->trace_->decisions);
  return Schedule(std::move(n));
}

/******** Schedule: Sampling ********/

Array<ExprRV> TracedScheduleNode::SamplePerfectTile(const LoopRV& loop_rv, int n,
                                                    int max_innermost_factor,
                                                    Optional<Array<Integer>> decision) {
  Array<ExprRV> results = CreateRV(tir::SamplePerfectTile(
      this->state_, &this->sampler_, this->GetSRef(loop_rv), n, max_innermost_factor, &decision));

  static const InstructionKind& kind = InstructionKind::Get("SamplePerfectTile");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{Integer(n), Integer(max_innermost_factor)},
                                      /*outputs=*/{results.begin(), results.end()}),
                 /*decision=*/decision);
  return results;
}

Array<Array<ExprRV>> TracedScheduleNode::SampleShapeGenericTiles(
    const Array<LoopRV>& loop_rvs, const std::vector<int>& ns, const Target& target,
    int max_innermost_factor, Optional<Array<Array<Integer>>> decision) {
  Array<StmtSRef> stmt_srefs;
  for (const LoopRV& loop_rv : loop_rvs) {
    stmt_srefs.push_back(GetSRef(loop_rv));
  }
  std::vector<std::vector<int64_t>> result =
      tir::SampleShapeGenericTiles(state_, &sampler_, stmt_srefs, ns, target, max_innermost_factor,
                                   &decision);
  Array<Array<ExprRV>> result_rvs;
  for (const std::vector<int64_t>& sampled : result) {
    result_rvs.push_back(CreateRV(sampled));
  }
  static const InstructionKind& kind = InstructionKind::Get("SampleShapeGenericTiles");
  trace_->Append(Instruction(kind,
                             {loop_rvs},
                             {AsArray<int, Integer>(ns), target, Integer(max_innermost_factor)},
                             {result_rvs.begin(), result_rvs.end()}),
                 decision);
  return result_rvs;
}


ExprRV TracedScheduleNode::SampleCategorical(const Array<Integer>& candidates,
                                             const Array<FloatImm>& probs,
                                             Optional<Integer> decision) {
  ExprRV result =
      CreateRV(tir::SampleCategorical(this->state_, &this->sampler_, candidates, probs, &decision));

  static const InstructionKind& kind = InstructionKind::Get("SampleCategorical");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{},
                                      /*attrs=*/{candidates, probs},
                                      /*outputs=*/{result}),
                 /*decision=*/decision);
  return result;
}

LoopRV TracedScheduleNode::SampleComputeLocation(const BlockRV& block_rv,
                                                 Optional<Integer> decision) {
  LoopRV result = CreateRV<LoopRV>(tir::SampleComputeLocation(this->state_, &this->sampler_,
                                                              this->GetSRef(block_rv), &decision));

  static const InstructionKind& kind = InstructionKind::Get("SampleComputeLocation");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{result}),
                 /*decision=*/decision);
  return result;
}

/******** Schedule: Get blocks & loops ********/

BlockRV TracedScheduleNode::GetBlock(const String& name, const String& func_name) {
  BlockRV result = ConcreteScheduleNode::GetBlock(name, func_name);

  static const InstructionKind& kind = InstructionKind::Get("GetBlock");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{},
                                      /*attrs=*/{name, func_name},
                                      /*outputs=*/{result}));
  return result;
}

Array<LoopRV> TracedScheduleNode::GetLoops(const BlockRV& block_rv) {
  Array<LoopRV> results = ConcreteScheduleNode::GetLoops(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("GetLoops");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{results.begin(), results.end()}));
  return results;
}

Array<BlockRV> TracedScheduleNode::GetChildBlocks(const BlockRV& block_rv) {
  Array<BlockRV> results = ConcreteScheduleNode::GetChildBlocks(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("GetChildBlocks");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{results.begin(), results.end()}));
  return results;
}

Array<BlockRV> TracedScheduleNode::GetChildBlocks(const LoopRV& loop_rv) {
  Array<BlockRV> results = ConcreteScheduleNode::GetChildBlocks(loop_rv);

  static const InstructionKind& kind = InstructionKind::Get("GetChildBlocks");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{results.begin(), results.end()}));
  return results;
}

Array<BlockRV> TracedScheduleNode::GetProducers(const BlockRV& block_rv) {
  Array<BlockRV> results = ConcreteScheduleNode::GetProducers(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("GetProducers");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{results.begin(), results.end()}));
  return results;
}

Array<BlockRV> TracedScheduleNode::GetConsumers(const BlockRV& block_rv) {
  Array<BlockRV> results = ConcreteScheduleNode::GetConsumers(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("GetConsumers");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{results.begin(), results.end()}));
  return results;
}

/******** Schedule: Transform loops ********/

LoopRV TracedScheduleNode::Fuse(const Array<LoopRV>& loop_rvs) {
  LoopRV result = ConcreteScheduleNode::Fuse(loop_rvs);

  static const InstructionKind& kind = InstructionKind::Get("Fuse");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rvs.begin(), loop_rvs.end()},
                                      /*attrs=*/{},
                                      /*outputs=*/{result}));
  return result;
}

Array<LoopRV> TracedScheduleNode::Split(const LoopRV& loop_rv,
                                        const Array<Optional<ExprRV>>& factor_rvs) {
  Array<LoopRV> results = ConcreteScheduleNode::Split(loop_rv, factor_rvs);

  std::vector<ObjectRef> inputs;
  inputs.reserve(1 + factor_rvs.size());
  inputs.push_back(loop_rv);
  for (const ObjectRef& obj : factor_rvs) {
    inputs.push_back(obj);
  }

  static const InstructionKind& kind = InstructionKind::Get("Split");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/inputs,
                                      /*attrs=*/{},
                                      /*outputs=*/{results.begin(), results.end()}));
  return results;
}

void TracedScheduleNode::Reorder(const Array<LoopRV>& order) {
  ConcreteScheduleNode::Reorder(order);

  static const InstructionKind& kind = InstructionKind::Get("Reorder");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{order.begin(), order.end()},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

/******** Schedule: Manipulate ForKind ********/

void TracedScheduleNode::Parallel(const LoopRV& loop_rv) {
  ConcreteScheduleNode::Parallel(loop_rv);

  static const InstructionKind& kind = InstructionKind::Get("Parallel");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::Vectorize(const LoopRV& loop_rv) {
  ConcreteScheduleNode::Vectorize(loop_rv);

  static const InstructionKind& kind = InstructionKind::Get("Vectorize");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::Unroll(const LoopRV& loop_rv) {
  ConcreteScheduleNode::Unroll(loop_rv);

  static const InstructionKind& kind = InstructionKind::Get("Unroll");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::Bind(const LoopRV& loop_rv, const String& thread) {
  ConcreteScheduleNode::Bind(loop_rv, thread);

  static const InstructionKind& kind = InstructionKind::Get("Bind");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{thread},
                                      /*outputs=*/{}));
}

/******** Schedule: Insert cache stages ********/

BlockRV TracedScheduleNode::CacheRead(const BlockRV& block_rv, int i, const String& storage_scope) {
  BlockRV result = ConcreteScheduleNode::CacheRead(block_rv, i, storage_scope);

  static const InstructionKind& kind = InstructionKind::Get("CacheRead");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{Integer(i), storage_scope},
                                      /*outputs=*/{result}));
  return result;
}

BlockRV TracedScheduleNode::CacheWrite(const BlockRV& block_rv, int i,
                                       const String& storage_scope) {
  BlockRV result = ConcreteScheduleNode::CacheWrite(block_rv, i, storage_scope);

  static const InstructionKind& kind = InstructionKind::Get("CacheWrite");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{Integer(i), storage_scope},
                                      /*outputs=*/{result}));
  return result;
}

/******** Schedule: Compute location ********/

void TracedScheduleNode::ComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                                   bool preserve_unit_loop) {
  ConcreteScheduleNode::ComputeAt(block_rv, loop_rv, preserve_unit_loop);

  static const InstructionKind& kind = InstructionKind::Get("ComputeAt");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv, loop_rv},
                                      /*attrs=*/{Integer(preserve_unit_loop)},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::ReverseComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                                          bool preserve_unit_loop) {
  ConcreteScheduleNode::ReverseComputeAt(block_rv, loop_rv, preserve_unit_loop);

  static const InstructionKind& kind = InstructionKind::Get("ReverseComputeAt");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv, loop_rv},
                                      /*attrs=*/{Integer(preserve_unit_loop)},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::ComputeInline(const BlockRV& block_rv) {
  ConcreteScheduleNode::ComputeInline(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("ComputeInline");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::ReverseComputeInline(const BlockRV& block_rv) {
  ConcreteScheduleNode::ReverseComputeInline(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("ReverseComputeInline");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

/******** Schedule: Reduction ********/

BlockRV TracedScheduleNode::RFactor(const LoopRV& loop_rv, int factor_axis) {
  BlockRV result = ConcreteScheduleNode::RFactor(loop_rv, factor_axis);
  static const InstructionKind& kind = InstructionKind::Get("RFactor");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{Integer(factor_axis)},
                                      /*outputs=*/{result}));
  return result;
}

BlockRV TracedScheduleNode::DecomposeReduction(const BlockRV& block_rv,
                                               const Optional<LoopRV>& loop_rv) {
  BlockRV result = ConcreteScheduleNode::DecomposeReduction(block_rv, loop_rv);
  static const InstructionKind& kind = InstructionKind::Get("DecomposeReduction");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv, loop_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{result}));
  return result;
}

void TracedScheduleNode::MergeReduction(const BlockRV& init_block_rv,
                                        const BlockRV& update_block_rv) {
  ConcreteScheduleNode::MergeReduction(init_block_rv, update_block_rv);
  static const InstructionKind& kind = InstructionKind::Get("MergeReduction");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{init_block_rv, update_block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

/******** Schedule: Blockize & Tensorize ********/

BlockRV TracedScheduleNode::Blockize(const LoopRV& loop_rv) {
  BlockRV result = ConcreteScheduleNode::Blockize(loop_rv);
  static const InstructionKind& kind = InstructionKind::Get("Blockize");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{result}));
  return result;
}

void TracedScheduleNode::Tensorize(const LoopRV& loop_rv, const String& intrin_name) {
  ConcreteScheduleNode::Tensorize(loop_rv, intrin_name);
  static const InstructionKind& kind = InstructionKind::Get("Tensorize");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{intrin_name},
                                      /*outputs=*/{}));
}

/******** Schedule: Annotation ********/

void TracedScheduleNode::MarkLoop(const LoopRV& loop_rv, const String& ann_key,
                                  const ObjectRef& ann_val) {
  ConcreteScheduleNode::MarkLoop(loop_rv, ann_key, ann_val);
  static const InstructionKind& kind = InstructionKind::Get("MarkLoop");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv, ann_val},
                                      /*attrs=*/{ann_key},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::MarkBlock(const BlockRV& block_rv, const String& ann_key,
                                   const ObjectRef& ann_val) {
  ConcreteScheduleNode::MarkBlock(block_rv, ann_key, ann_val);
  static const InstructionKind& kind = InstructionKind::Get("MarkBlock");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv, ann_val},
                                      /*attrs=*/{ann_key},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::Pragma(const LoopRV& loop_rv, const String& pragma_type,
                                const ExprRV& pragma_value) {
  ConcreteScheduleNode::Pragma(loop_rv, pragma_type, pragma_value);
  static const InstructionKind& kind = InstructionKind::Get("Pragma");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv, pragma_value},
                                      /*attrs=*/{pragma_type},
                                      /*outputs=*/{}));
}

/******** Schedule: Misc ********/

void TracedScheduleNode::EnterPostproc() {
  ConcreteScheduleNode::EnterPostproc();
  static const InstructionKind& kind = InstructionKind::Get("EnterPostproc");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::DoubleBuffer(const BlockRV& block_rv) {
  ConcreteScheduleNode::DoubleBuffer(block_rv);
  static const InstructionKind& kind = InstructionKind::Get("DoubleBuffer");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::SetScope(const BlockRV& block_rv, int i, const String& storage_scope) {
  ConcreteScheduleNode::SetScope(block_rv, i, storage_scope);
  static const InstructionKind& kind = InstructionKind::Get("SetScope");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{Integer(i), storage_scope},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::StorageAlign(const BlockRV& block_rv, int buffer_index, int axis,
                                      int factor, int offset) {
  ConcreteScheduleNode::StorageAlign(block_rv, buffer_index, axis, factor, offset);
  static const InstructionKind& kind = InstructionKind::Get("StorageAlign");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{block_rv},
      /*attrs=*/{Integer(buffer_index), Integer(axis), Integer(factor), Integer(offset)},
      /*outputs=*/{}));
}

void TracedScheduleNode::InlineArgument(int i, const String& func_name) {
  ConcreteScheduleNode::InlineArgument(i, func_name);
  static const InstructionKind& kind = InstructionKind::Get("InlineArgument");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{},
      /*attrs=*/{Integer(i), func_name},
      /*outputs=*/{}));
}

/******** FFI ********/

TVM_REGISTER_NODE_TYPE(TracedScheduleNode);
TVM_REGISTER_GLOBAL("tir.schedule.TracedSchedule")
    .set_body_typed([](IRModule mod, int64_t seed, int debug_mode,
                       int error_render_level) -> Schedule {
      return Schedule::Traced(mod, seed, debug_mode,
                              static_cast<ScheduleErrorRenderLevel>(error_render_level));
    });

}  // namespace tir
}  // namespace tvm
