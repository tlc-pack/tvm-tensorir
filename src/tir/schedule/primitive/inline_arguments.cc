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

void InlineArgument(ScheduleState self, int i, const String& func_name) {
  GlobalVar g_var = self->mod->GetGlobalVar(func_name);
  PrimFunc func = Downcast<PrimFunc>(self->mod->Lookup(g_var));
  Buffer buffer = func->buffer_map.at(func->params[i]);
  BlockRealize root_realize = Downcast<BlockRealize>(func->body);
  // Create the new block
  ObjectPtr<BlockNode> new_block = make_object<BlockNode>(*root_realize->block.get());
  new_block->alloc_buffers.push_back(buffer);
  // Find the corresponding sref
  StmtSRef src_sref = self->stmt2ref.at(root_realize->block.get());
  // Replace the func
  self->Replace(src_sref, Block(new_block), {});
  // Create the new block realize
  ObjectPtr<BlockRealizeNode> new_realize = make_object<BlockRealizeNode>(*root_realize.get());
  new_realize->block = Block(new_block);
  // Create the new PrimFunc
  ObjectPtr<PrimFuncNode> new_func = make_object<PrimFuncNode>(*func.get());
  new_func->buffer_map.erase(new_func->params[i]);
  new_func->body = BlockRealize(new_realize);
  new_func->params.erase(new_func->params.begin() + i);
  self->mod->Update(g_var, PrimFunc(new_func));
}

struct InlineArgumentTraits : public UnpackedInstTraits<InlineArgumentTraits> {
  static constexpr const char* kName = "InlineArgument";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 0;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, Integer i, String func_name) {
    return sch->InlineArgument(i, func_name);
  }

  static String UnpackedAsPython(Array<String> outputs, Integer i, String func_name) {
    PythonAPICall py("inline_argument");
    py.Input("i", i->value);
    py.Input("func_name", func_name);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(InlineArgumentTraits);

}  // namespace tir
}  // namespace tvm
