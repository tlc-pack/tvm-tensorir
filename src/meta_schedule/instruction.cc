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
#include "./instruction.h"  // NOLINT(build/include)

namespace tvm {
namespace meta_schedule {

SampleTileFactorInst::SampleTileFactorInst(LoopRV loop, Array<Integer> where,
                                           Array<tir::Var> outputs) {
  ObjectPtr<SampleTileFactorInstNode> n = make_object<SampleTileFactorInstNode>();
  n->loop = std::move(loop);
  n->where = std::move(where);
  n->outputs = std::move(outputs);
  data_ = std::move(n);
}

GetBlockInst::GetBlockInst(String name, BlockRV output) {
  ObjectPtr<GetBlockInstNode> n = make_object<GetBlockInstNode>();
  n->name = std::move(name);
  n->output = std::move(output);
  data_ = std::move(n);
}

GetAxesInst::GetAxesInst(BlockRV block, Array<LoopRV> outputs) {
  ObjectPtr<GetAxesInstNode> n = make_object<GetAxesInstNode>();
  n->block = std::move(block);
  n->outputs = std::move(outputs);
  data_ = std::move(n);
}

SplitInst::SplitInst(LoopRV loop, Array<PrimExpr> factors, Array<LoopRV> outputs) {
  ObjectPtr<SplitInstNode> n = make_object<SplitInstNode>();
  n->loop = std::move(loop);
  n->factors = std::move(factors);
  n->outputs = std::move(outputs);
  data_ = std::move(n);
}

ReorderInst::ReorderInst(Array<LoopRV> after_axes) {
  ObjectPtr<ReorderInstNode> n = make_object<ReorderInstNode>();
  n->after_axes = std::move(after_axes);
  data_ = std::move(n);
}

ComputeInlineInst::ComputeInlineInst(BlockRV block) {
  ObjectPtr<ComputeInlineInstNode> n = make_object<ComputeInlineInstNode>();
  n->block = std::move(block);
  data_ = std::move(n);
}

DecomposeReductionInst::DecomposeReductionInst(BlockRV block, LoopRV loop, BlockRV output) {
  ObjectPtr<DecomposeReductionInstNode> n = make_object<DecomposeReductionInstNode>();
  n->block = std::move(block);
  n->loop = std::move(loop);
  n->output = std::move(output);
  data_ = std::move(n);
}

GetOnlyConsumerInst::GetOnlyConsumerInst(BlockRV block, BlockRV output) {
  ObjectPtr<GetOnlyConsumerInstNode> n = make_object<GetOnlyConsumerInstNode>();
  n->block = std::move(block);
  n->output = std::move(output);
  data_ = std::move(n);
}

CacheWriteInst::CacheWriteInst(BlockRV block, String storage_scope) {
  ObjectPtr<CacheWriteInstNode> n = make_object<CacheWriteInstNode>();
  n->block = std::move(block);
  n->storage_scope = std::move(storage_scope);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(InstructionNode);
TVM_REGISTER_NODE_TYPE(SampleTileFactorInstNode);
TVM_REGISTER_NODE_TYPE(GetBlockInstNode);
TVM_REGISTER_NODE_TYPE(GetAxesInstNode);
TVM_REGISTER_NODE_TYPE(SplitInstNode);
TVM_REGISTER_NODE_TYPE(ReorderInstNode);
TVM_REGISTER_NODE_TYPE(ComputeInlineInstNode);
TVM_REGISTER_NODE_TYPE(CacheWriteInstNode);
TVM_REGISTER_NODE_TYPE(DecomposeReductionInstNode);
TVM_REGISTER_NODE_TYPE(GetOnlyConsumerInstNode);

}  // namespace meta_schedule
}  // namespace tvm
