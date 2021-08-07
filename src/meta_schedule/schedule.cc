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
#include "./schedule.h"  // NOLINT(build/include)

#include <tvm/arith/analyzer.h>
#include <tvm/tir/stmt_functor.h>

#include "../tir/schedule/analysis.h"
#include "../tir/schedule/primitives/primitives.h"
#include "./sampling.h"
#include "./utils.h"

namespace tvm {

namespace tir {
tir::Schedule tir::Schedule::Meta(tir::PrimFunc func, int64_t seed, int debug_mode) {
  return meta_schedule::Schedule(func, seed, debug_mode);
}
tir::Schedule tir::Schedule::Meta(IRModule mod, int64_t seed, int debug_mode) {
  return meta_schedule::Schedule(mod, seed, debug_mode);
}
}  // namespace tir

namespace meta_schedule {

Schedule::Schedule(tir::PrimFunc func, int64_t seed, int debug_mode)
    : Schedule(IRModule({{GlobalVar("main"), func}}), seed, debug_mode) {}

Schedule::Schedule(IRModule mod, int64_t seed, int debug_mode) {
  ObjectPtr<ScheduleNode> n = make_object<ScheduleNode>();
  n->state_ = tir::ScheduleState(mod, debug_mode);
  n->symbol_table_ = {};
  n->analyzer_ = std::make_unique<arith::Analyzer>();
  if (seed != -1) {
    n->random_state = seed;
  }
  n->trace = Trace();
  this->data_ = std::move(n);
}

/**************** Utility ****************/

Schedule ScheduleNode::Copy(int64_t new_seed) const {
  ObjectPtr<ScheduleNode> n = make_object<ScheduleNode>();
  tir::ConcreteScheduleNode::MakeCopy(&n->state_, &n->symbol_table_);
  n->analyzer_ = std::make_unique<arith::Analyzer>();
  n->trace = Trace(this->trace->insts, this->trace->decisions);
  n->random_state = new_seed;
  Sampler(&n->random_state);
  return Schedule(std::move(n));
}

void ScheduleNode::Seed(int64_t seed) {
  this->random_state = seed;
  Sampler(&this->random_state);
}

/**************** Sampling ****************/

Array<tir::Var> ScheduleNode::SamplePerfectTile(const LoopRV& loop_rv, int n,
                                                int max_innermost_factor,
                                                Optional<Array<Integer>> decision) {
  std::vector<int64_t> result = meta_schedule::SamplePerfectTile(
      state_, &(this->random_state), this->GetSRef(loop_rv), n, max_innermost_factor, &decision);
  Array<tir::Var> result_rvs = SetRV(AsArray<int64_t, Integer>(result));
  // Record the instruction
  this->trace->Append(SamplePerfectTileAttrs::Make(loop_rv, n, max_innermost_factor, result_rvs),
                      decision);
  return result_rvs;
}

tir::Var ScheduleNode::SampleCategorical(const Array<Integer>& candidates,  //
                                         const Array<FloatImm>& probs,      //
                                         Optional<Integer> decision) {
  int64_t result =
      meta_schedule::SampleCategorical(state_, &(this->random_state), candidates, probs, &decision);
  tir::Var result_rv = SetRV(result);
  this->trace->Append(SampleCategoricalAttrs::Make(candidates, probs, result_rv), decision);
  return result_rv;
}

LoopRV ScheduleNode::SampleComputeLocation(const BlockRV& block_rv, Optional<Integer> decision) {
  tir::StmtSRef result = meta_schedule::SampleComputeLocation(state_, &(this->random_state),
                                                              this->GetSRef(block_rv), &decision);
  LoopRV result_rv = SetRV<LoopRV>(result);
  this->trace->Append(SampleComputeLocationAttrs::Make(block_rv, result_rv), decision);
  return result_rv;
}

/**************** Block/Loop Relationship ****************/

BlockRV ScheduleNode::GetBlock(const String& name) {
  BlockRV result = tir::ConcreteScheduleNode::GetBlock(name);
  this->trace->Append(GetBlockAttrs::Make(name, result));
  return result;
}

Array<LoopRV> ScheduleNode::GetAxes(const BlockRV& block_rv) {
  Array<LoopRV> results = tir::ConcreteScheduleNode::GetAxes(block_rv);
  this->trace->Append(GetAxesAttrs::Make(block_rv, results));
  return results;
}

Array<BlockRV> ScheduleNode::GetChildBlocks(const BlockRV& block_rv) {
  Array<BlockRV> results = tir::ConcreteScheduleNode::GetChildBlocks(block_rv);
  // TODO
  return results;
}

Array<BlockRV> ScheduleNode::GetChildBlocks(const LoopRV& loop_rv) {
  Array<BlockRV> results = tir::ConcreteScheduleNode::GetChildBlocks(loop_rv);
  // TODO
  return results;
}

Array<BlockRV> ScheduleNode::GetProducers(const BlockRV& block_rv) {
  Array<BlockRV> results = tir::ConcreteScheduleNode::GetProducers(block_rv);
  this->trace->Append(GetProducersAttrs::Make(block_rv, results));
  return results;
}

Array<BlockRV> ScheduleNode::GetConsumers(const BlockRV& block_rv) {
  Array<BlockRV> results = tir::ConcreteScheduleNode::GetConsumers(block_rv);
  this->trace->Append(GetConsumersAttrs::Make(block_rv, results));
  return results;
}

/******** Schedule: loops ********/

LoopRV ScheduleNode::Fuse(const Array<LoopRV>& loop_rvs) {
  LoopRV result = tir::ConcreteScheduleNode::Fuse(loop_rvs);
  this->trace->Append(FuseAttrs::Make(loop_rvs, result));
  return result;
}

Array<LoopRV> ScheduleNode::Split(const LoopRV& loop_rv, const Array<Optional<ExprRV>>& factors) {
  Array<LoopRV> results = tir::ConcreteScheduleNode::Split(loop_rv, factors);
  this->trace->Append(SplitAttrs::Make(loop_rv, factors, results));
  return results;
}

void ScheduleNode::Reorder(const Array<LoopRV>& order) {
  tir::ConcreteScheduleNode::Reorder(order);
  this->trace->Append(ReorderAttrs::Make(order));
}

/**************** Schedule Primitives ****************/

void ScheduleNode::ComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                             bool preserve_unit_loop) {
  tir::ConcreteScheduleNode::ComputeAt(block_rv, loop_rv, preserve_unit_loop);
  this->trace->Append(ComputeAtAttrs::Make(block_rv, loop_rv, preserve_unit_loop));
}

void ScheduleNode::ReverseComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                                    bool preserve_unit_loop) {
  tir::ConcreteScheduleNode::ReverseComputeAt(block_rv, loop_rv, preserve_unit_loop);
  this->trace->Append(ReverseComputeAtAttrs::Make(block_rv, loop_rv, preserve_unit_loop));
}

void ScheduleNode::ComputeInline(const BlockRV& block_rv) {
  tir::ConcreteScheduleNode::ComputeInline(block_rv);
  this->trace->Append(ComputeInlineAttrs::Make(block_rv));
}

void ScheduleNode::ReverseComputeInline(const BlockRV& block_rv) {
  tir::ConcreteScheduleNode::ReverseComputeInline(block_rv);
  this->trace->Append(ReverseComputeInlineAttrs::Make(block_rv));
}

/******** Schedule: parallelize / annotate ********/

void ScheduleNode::Vectorize(const LoopRV& loop_rv) {
  tir::ConcreteScheduleNode::Vectorize(loop_rv);
  this->trace->Append(VectorizeAttrs::Make(loop_rv));
}

void ScheduleNode::Parallel(const LoopRV& loop_rv) {
  tir::ConcreteScheduleNode::Parallel(loop_rv);
  this->trace->Append(ParallelAttrs::Make(loop_rv));
}

void ScheduleNode::Unroll(const LoopRV& loop_rv) {
  tir::ConcreteScheduleNode::Unroll(loop_rv);
  this->trace->Append(UnrollAttrs::Make(loop_rv));
}

void ScheduleNode::Bind(const LoopRV& loop_rv, const tir::IterVar& thread) {
  LOG(FATAL) << "NotImplemented";
}

void ScheduleNode::Bind(const LoopRV& loop_rv, const String& thread) {
  tir::ConcreteScheduleNode::Bind(loop_rv, thread);
  this->trace->Append(BindAttrs::Make(loop_rv, thread));
}

void ScheduleNode::DoubleBuffer(const BlockRV& block_rv) {
  tir::ConcreteScheduleNode::DoubleBuffer(block_rv);
  // TODO
}

void ScheduleNode::SetScope(const BlockRV& block_rv, int i, const String& storage_scope) {
  tir::ConcreteScheduleNode::SetScope(block_rv, i, storage_scope);
  this->trace->Append(SetScopeAttrs::Make(block_rv, i, storage_scope));
}

void ScheduleNode::Pragma(const LoopRV& loop_rv, const String& pragma_type,
                          const ExprRV& pragma_value) {
  tir::ConcreteScheduleNode::Pragma(loop_rv, pragma_type, pragma_value);
  // TODO
}

void ScheduleNode::StorageAlign(const BlockRV& block_rv, int buffer_index, int axis, int factor,
                                int offset) {
  tir::ConcreteScheduleNode::StorageAlign(block_rv, buffer_index, axis, factor, offset);
  this->trace->Append(StorageAlignAttrs::Make(block_rv, buffer_index, axis, factor, offset));
}

/******** Schedule: cache read/write ********/

BlockRV ScheduleNode::CacheRead(const BlockRV& block_rv, int i, const String& storage_scope) {
  BlockRV result = tir::ConcreteScheduleNode::CacheRead(block_rv, i, storage_scope);
  this->trace->Append(CacheReadAttrs::Make(block_rv, i, storage_scope, result));
  return result;
}

BlockRV ScheduleNode::CacheWrite(const BlockRV& block_rv, int i, const String& storage_scope) {
  BlockRV result = tir::ConcreteScheduleNode::CacheWrite(block_rv, i, storage_scope);
  this->trace->Append(CacheWriteAttrs::Make(block_rv, i, storage_scope, result));
  return result;
}

/******** Schedule: reduction ********/

BlockRV ScheduleNode::RFactor(const LoopRV& loop_rv, int factor_axis) {
  BlockRV result = tir::ConcreteScheduleNode::RFactor(loop_rv, factor_axis);
  this->trace->Append(RFactorAttrs::Make(loop_rv, factor_axis, result));
  return result;
}

BlockRV ScheduleNode::DecomposeReduction(const BlockRV& block_rv, const Optional<LoopRV>& loop_rv) {
  BlockRV result = tir::ConcreteScheduleNode::DecomposeReduction(block_rv, loop_rv);
  this->trace->Append(DecomposeReductionAttrs::Make(block_rv, loop_rv, result));
  return result;
}

void ScheduleNode::MergeReduction(const BlockRV& init_block_rv, const BlockRV& update_block_rv) {
  tir::ConcreteScheduleNode::MergeReduction(init_block_rv, update_block_rv);
  // TODO
}

/******** Schedule: blockize / tensorize ********/

BlockRV ScheduleNode::Blockize(const LoopRV& loop_rv) {
  BlockRV result = tir::ConcreteScheduleNode::Blockize(loop_rv);
  this->trace->Append(BlockizeAttrs::Make(loop_rv, result));
  return result;
}

void ScheduleNode::Tensorize(const LoopRV& loop_rv, const tir::TensorIntrin& intrin) {
  LOG(FATAL) << "NotImplemented";
}

void ScheduleNode::Tensorize(const LoopRV& loop_rv, const String& intrin_name) {
  tir::ConcreteScheduleNode::Tensorize(loop_rv, intrin_name);
  this->trace->Append(TensorizeAttrs::Make(loop_rv, intrin_name));
}

/**************** Marks and NO-OPs ****************/

void ScheduleNode::MarkLoop(const LoopRV& loop_rv, const String& ann_key, const PrimExpr& ann_val) {
  ICHECK(ann_val->IsInstance<tir::StringImmNode>() || ann_val->IsInstance<IntImmNode>())
      << "TypeError: Only StringImm and IntImm are supported for now, but gets: "
      << ann_val->GetTypeKey();
  AddAnn(state_, this->GetSRef(loop_rv), ann_key, ann_val);
  this->trace->Append(MarkLoopAttrs::Make(loop_rv, ann_key, ann_val));
}

void ScheduleNode::MarkBlock(const BlockRV& block_rv, const String& ann_key,
                             const PrimExpr& ann_val) {
  PrimExpr value = this->Get(ann_val);
  const auto* int_imm = TVM_TYPE_AS(int_imm, value, IntImmNode);
  AddAnn(state_, this->GetSRef(block_rv), ann_key, tir::StringImm(std::to_string(int_imm->value)));
  this->trace->Append(MarkBlockAttrs::Make(block_rv, ann_key, ann_val));
}

void ScheduleNode::EnterPostProc() { this->trace->Append(EnterPostProcAttrs::Make()); }

/******** Misc ********/

void ScheduleNode::InlineArgument(int i, const String& func_name) {
  tir::ConcreteScheduleNode::InlineArgument(i, func_name);
  // TODO: add trace
}

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(ScheduleNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleMarkLoop")
    .set_body_method<Schedule>(&ScheduleNode::MarkLoop);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleMarkBlock")
    .set_body_method<Schedule>(&ScheduleNode::MarkBlock);
TVM_REGISTER_GLOBAL("meta_schedule.Schedule")
    .set_body_typed([](ObjectRef obj, int64_t seed, int debug_mode) -> Schedule {
      if (const auto* func = obj.as<tir::PrimFuncNode>()) {
        return Schedule(GetRef<tir::PrimFunc>(func), seed, debug_mode);
      }
      if (const auto* mod = obj.as<IRModuleNode>()) {
        return Schedule(GetRef<IRModule>(mod), seed, debug_mode);
      }
      LOG(FATAL) << "TypeError: Expects `IRModule` or `PrimFunc`, but gets: " << obj->GetTypeKey();
      throw;
    });
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleCopy").set_body_typed([](Schedule self, int new_seed) {
  return self->Copy(new_seed);
});

}  // namespace meta_schedule
}  // namespace tvm
