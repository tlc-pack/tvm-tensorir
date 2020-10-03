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

/**************** Constructors ****************/

Instruction::Instruction(Array<ObjectRef> inputs, Array<ObjectRef> outputs, Attrs attrs) {
  ObjectPtr<InstructionNode> n = make_object<InstructionNode>();
  n->inputs = std::move(inputs);
  n->outputs = std::move(outputs);
  n->attrs = std::move(attrs);
  data_ = std::move(n);
}

/**************** MakeInst: Sampling  ****************/

Instruction SamplePerfectTileAttrs::MakeInst(int n_splits, const LoopRV& loop,
                                             int max_innermost_factor,
                                             const Array<tir::Var>& outputs) {
  ObjectPtr<SamplePerfectTileAttrs> n = make_object<SamplePerfectTileAttrs>();
  n->n_splits = n_splits;
  n->max_innermost_factor = max_innermost_factor;
  return Instruction(/*inputs=*/{loop},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/Attrs(std::move(n)));
}

Instruction SampleTileFactorAttrs::MakeInst(int n_splits, const LoopRV& loop,
                                            const Array<Integer>& where,
                                            const Array<tir::Var>& outputs) {
  ObjectPtr<SampleTileFactorAttrs> n = make_object<SampleTileFactorAttrs>();
  n->n_splits = n_splits;
  n->where = where;
  return Instruction(/*inputs=*/{loop},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/Attrs(std::move(n)));
}

/**************** MakeInst: Block/Loop Relationship  ****************/

Instruction GetOnlyConsumerAttrs::MakeInst(const BlockRV& block, const BlockRV& output) {
  ObjectPtr<GetOnlyConsumerAttrs> n = make_object<GetOnlyConsumerAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{output},
                     /*attrs=*/Attrs(std::move(n)));
}

Instruction GetBlockAttrs::MakeInst(const String& name, const BlockRV& output) {
  ObjectPtr<GetBlockAttrs> n = make_object<GetBlockAttrs>();
  n->name = name;
  return Instruction(/*inputs=*/{},
                     /*outputs=*/{output},
                     /*attrs=*/Attrs(std::move(n)));
}

Instruction GetAxesAttrs::MakeInst(const BlockRV& block, const Array<LoopRV>& outputs) {
  ObjectPtr<GetAxesAttrs> n = make_object<GetAxesAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/Attrs(std::move(n)));
}

/**************** MakeInst: Scheduling Primitives  ****************/

Instruction SplitAttrs::MakeInst(const LoopRV& loop, const Array<PrimExpr>& factors,
                                 const Array<LoopRV>& outputs) {
  ObjectPtr<SplitAttrs> n = make_object<SplitAttrs>();
  n->factors = factors;
  return Instruction(/*inputs=*/{loop},
                     /*outputs=*/{outputs.begin(), outputs.end()},
                     /*attrs=*/Attrs(std::move(n)));
}

Instruction ReorderAttrs::MakeInst(const Array<LoopRV>& after_axes) {
  ObjectPtr<ReorderAttrs> n = make_object<ReorderAttrs>();
  return Instruction(/*inputs=*/{after_axes.begin(), after_axes.end()},
                     /*outputs=*/{},
                     /*attrs=*/Attrs(std::move(n)));
}

Instruction ComputeInlineAttrs::MakeInst(const BlockRV& block) {
  ObjectPtr<ComputeInlineAttrs> n = make_object<ComputeInlineAttrs>();
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{},
                     /*attrs=*/Attrs(std::move(n)));
}

Instruction CacheWriteAttrs::MakeInst(const BlockRV& block, const String& storage_scope,
                                      const BlockRV& output) {
  ObjectPtr<CacheWriteAttrs> n = make_object<CacheWriteAttrs>();
  n->storage_scope = storage_scope;
  return Instruction(/*inputs=*/{block},
                     /*outputs=*/{output},
                     /*attrs=*/Attrs(std::move(n)));
}

Instruction DecomposeReductionAttrs::MakeInst(const BlockRV& block, const LoopRV& loop,
                                              const BlockRV& output) {
  ObjectPtr<DecomposeReductionAttrs> n = make_object<DecomposeReductionAttrs>();
  return Instruction(/*inputs=*/{block, loop},
                     /*outputs=*/{output},
                     /*attrs=*/Attrs(std::move(n)));
}

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(InstructionNode);
TVM_REGISTER_NODE_TYPE(SamplePerfectTileAttrs);
TVM_REGISTER_NODE_TYPE(SampleTileFactorAttrs);
TVM_REGISTER_NODE_TYPE(GetBlockAttrs);
TVM_REGISTER_NODE_TYPE(GetAxesAttrs);
TVM_REGISTER_NODE_TYPE(SplitAttrs);
TVM_REGISTER_NODE_TYPE(ReorderAttrs);
TVM_REGISTER_NODE_TYPE(ComputeInlineAttrs);
TVM_REGISTER_NODE_TYPE(CacheWriteAttrs);
TVM_REGISTER_NODE_TYPE(DecomposeReductionAttrs);
TVM_REGISTER_NODE_TYPE(GetOnlyConsumerAttrs);

}  // namespace meta_schedule
}  // namespace tvm
