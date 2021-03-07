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
#include <tvm/tir/schedule/schedule.h>

namespace tvm {
namespace tir {

/**************** Constructor ****************/

BlockRV::BlockRV() { this->data_ = make_object<BlockRVNode>(); }

LoopRV::LoopRV() { this->data_ = make_object<LoopRVNode>(); }

/**************** GetSRef ****************/

StmtSRef ScheduleNode::GetSRef(const StmtNode* stmt) const {
  ScheduleState state = this->state();
  auto it = state->stmt2ref.find(stmt);
  if (it == state->stmt2ref.end()) {
    LOG(FATAL) << "IndexError: The stmt doesn't exist in the IR";
  }
  return it->second;
}

StmtSRef ScheduleNode::GetSRef(const Stmt& stmt) const { return this->GetSRef(stmt.get()); }

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(BlockRVNode);
TVM_REGISTER_NODE_TYPE(LoopRVNode);
TVM_REGISTER_OBJECT_TYPE(ScheduleNode);

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleModule")  //
    .set_body_method<Schedule>(&ScheduleNode::mod);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetState")  //
    .set_body_method<Schedule>(&ScheduleNode::state);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleSeed")  //
    .set_body_method<Schedule>(&ScheduleNode::Seed);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleCopy")  //
    .set_body_method<Schedule>(&ScheduleNode::Copy);

/******** (FFI) Lookup random variables ********/

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGet")
    .set_body_typed([](Schedule self, ObjectRef obj) -> ObjectRef {
      if (const auto* loop_rv = obj.as<LoopRVNode>()) {
        return self->Get(GetRef<LoopRV>(loop_rv));
      }
      if (const auto* block_rv = obj.as<BlockRVNode>()) {
        return self->Get(GetRef<BlockRV>(block_rv));
      }
      if (const auto* expr_rv = obj.as<PrimExprNode>()) {
        return self->Get(GetRef<PrimExpr>(expr_rv));
      }
      LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: " << obj->GetTypeKey()
                 << ". Its value is: " << obj;
      throw;
    });
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetSRef")
    .set_body_typed([](Schedule self, ObjectRef obj) -> Optional<ObjectRef> {
      if (const auto* loop_rv = obj.as<LoopRVNode>()) {
        return self->GetSRef(GetRef<LoopRV>(loop_rv));
      }
      if (const auto* block_rv = obj.as<BlockRVNode>()) {
        return self->GetSRef(GetRef<BlockRV>(block_rv));
      }
      if (const auto* stmt = obj.as<StmtNode>()) {
        return self->GetSRef(GetRef<Stmt>(stmt));
      }
      LOG(FATAL) << "TypeError: Invalid type: " << obj->GetTypeKey();
      throw;
    });

/***** (FFI) Sampling *****/

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleSamplePerfectTile")
    .set_body_method<Schedule>(&ScheduleNode::SamplePerfectTile);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleSampleCategorical")
    .set_body_method<Schedule>(&ScheduleNode::SampleCategorical);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleSampleComputeLocation")
    .set_body_method<Schedule>(&ScheduleNode::SampleComputeLocation);

/***** (FFI) Block/Loop relation *****/

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetBlock")
    .set_body_method<Schedule>(&ScheduleNode::GetBlock);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetAxes")
    .set_body_method<Schedule>(&ScheduleNode::GetAxes);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetChildBlocks")
    .set_body_typed([](Schedule self, ObjectRef rv) {
      if (const auto* block_rv = rv.as<BlockRVNode>()) {
        return self->GetChildBlocks(GetRef<BlockRV>(block_rv));
      }
      if (const auto* loop_rv = rv.as<LoopRVNode>()) {
        return self->GetChildBlocks(GetRef<LoopRV>(loop_rv));
      }
      LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: " << rv->GetTypeKey()
                 << ". Its value is: " << rv;
      throw;
    });
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetProducers")
    .set_body_method<Schedule>(&ScheduleNode::GetProducers);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetConsumers")
    .set_body_method<Schedule>(&ScheduleNode::GetConsumers);

/***** (FFI) Schedule: loops *****/

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleFuse")  //
    .set_body_method<Schedule>(&ScheduleNode::Fuse);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleSplit")  //
    .set_body_method<Schedule>(&ScheduleNode::Split);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleReorder")
    .set_body_method<Schedule>(&ScheduleNode::Reorder);

/***** (FFI) Schedule: compute location *****/

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleComputeAt")
    .set_body_method<Schedule>(&ScheduleNode::ComputeAt);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleReverseComputeAt")
    .set_body_method<Schedule>(&ScheduleNode::ReverseComputeAt);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleComputeInline")
    .set_body_method<Schedule>(&ScheduleNode::ComputeInline);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleReverseComputeInline")
    .set_body_method<Schedule>(&ScheduleNode::ReverseComputeInline);

/***** (FFI) Schedule: parallelize / annotate *****/

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleVectorize")
    .set_body_method<Schedule>(&ScheduleNode::Vectorize);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleParallel")
    .set_body_method<Schedule>(&ScheduleNode::Parallel);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleUnroll")  //
    .set_body_method<Schedule>(&ScheduleNode::Unroll);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleBind")
    .set_body_typed([](Schedule self, LoopRV loop_rv, ObjectRef thread) {
      if (const auto* iter_var = thread.as<IterVarNode>()) {
        return self->Bind(loop_rv, GetRef<IterVar>(iter_var));
      }
      if (const auto* str = thread.as<StringObj>()) {
        return self->Bind(loop_rv, GetRef<String>(str));
      }
      LOG(FATAL) << "TypeError: Schedule.Bind doesn't support type: " << thread->GetTypeKey()
                 << ", and the value is: " << thread;
      throw;
    });
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleDoubleBuffer")
    .set_body_method<Schedule>(&ScheduleNode::DoubleBuffer);
TVM_REGISTER_GLOBAL("tir.schedule.SchedulePragma")  //
    .set_body_method<Schedule>(&ScheduleNode::Pragma);

/***** (FFI) Schedule: cache read/write *****/

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleCacheRead")
    .set_body_method<Schedule>(&ScheduleNode::CacheRead);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleCacheWrite")
    .set_body_method<Schedule>(&ScheduleNode::CacheWrite);

/***** (FFI) Schedule: reduction *****/

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleRFactor")
    .set_body_method<Schedule>(&ScheduleNode::RFactor);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleDecomposeReduction")
    .set_body_method<Schedule>(&ScheduleNode::DecomposeReduction);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleMergeReduction")
    .set_body_method<Schedule>(&ScheduleNode::MergeReduction);

/***** (FFI) Schedule: blockize / tensorize *****/

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleBlockize")
    .set_body_method<Schedule>(&ScheduleNode::Blockize);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleTensorize")
    .set_body_typed([](Schedule self, LoopRV loop_rv, ObjectRef intrin) {
      if (const auto* str = intrin.as<runtime::StringObj>()) {
        return self->Tensorize(loop_rv, GetRef<String>(str));
      }
      if (const auto* p_intrin = intrin.as<TensorIntrinNode>()) {
        return self->Tensorize(loop_rv, GetRef<TensorIntrin>(p_intrin));
      }
      LOG(FATAL) << "TypeError: Cannot handle type: " << intrin->GetTypeKey();
      throw;
    });

}  // namespace tir
}  // namespace tvm
