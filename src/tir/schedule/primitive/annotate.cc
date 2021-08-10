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

void Pragma(ScheduleState self, const StmtSRef& loop_sref, const String& pragma_type,
            const PrimExpr& pragma_value) {
  AddAnn(self, loop_sref, "pragma_" + pragma_type, pragma_value);
}

void MarkLoop(ScheduleState self, const StmtSRef& loop_sref, const String& ann_key,
              const PrimExpr& ann_val) {
  AddAnn(self, loop_sref, ann_key, ann_val);
}

void MarkBlock(ScheduleState self, const StmtSRef& block_sref, const String& ann_key,
               const PrimExpr& ann_val) {
  AddAnn(self, block_sref, ann_key, ann_val);
}

void SoftwarePipeline(ScheduleState self, const StmtSRef& loop_sref, int num_stages) {
  const auto* loop = loop_sref->StmtAs<ForNode>();
  CHECK(loop != nullptr) << "TypeError: 'software_pipeline' expects a loop as its first argument";
  CHECK(!loop->thread_binding.defined())
      << "ValueError: 'software_pipeline' does not work on a loop with thread bindings.";
  AddAnn(self, loop_sref, tir::attr::pipeline_scope, IntImm(DataType::Int(32), num_stages));
}

struct PragmaTraits : public UnpackedInstTraits<PragmaTraits> {
  static constexpr const char* kName = "Pragma";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv, ExprRV pragma_value,
                                      String pragma_type) {
    return sch->Pragma(loop_rv, pragma_type, pragma_value);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv, String pragma_value,
                                 String pragma_type) {
    PythonAPICall py("pragma");
    py.Input("loop", loop_rv);
    py.Input("pragma_type", pragma_type);
    py.Input("pragma_value", pragma_value);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

struct MarkLoopTraits : public UnpackedInstTraits<MarkLoopTraits> {
  static constexpr const char* kName = "MarkLoop";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv, ObjectRef ann_val,
                                      String ann_key) {
    return sch->MarkLoop(loop_rv, ann_key, ann_val);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv, ObjectRef ann_val,
                                 String ann_key) {
    PythonAPICall py("mark_loop");
    py.Input("loop", loop_rv);
    py.Input("ann_key", ann_key);
    if (const auto* int_imm = ann_val.as<IntImmNode>()) {
      py.Input("ann_val", std::to_string(int_imm->value));
    } else if (const auto* str_imm = ann_val.as<StringObj>()) {
      py.Input("ann_val", GetRef<String>(str_imm));
    } else if (const auto* expr = ann_val.as<PrimExprNode>()) {
      std::ostringstream os;
      os << GetRef<PrimExpr>(expr);
      py.Input("ann_val", os.str());
    } else {
      LOG(FATAL) << "TypeError: Cannot handle type: " << ann_val->GetTypeKey();
      throw;
    }
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

struct MarkBlockTraits : public UnpackedInstTraits<MarkBlockTraits> {
  static constexpr const char* kName = "MarkBlock";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, ObjectRef ann_val,
                                      String ann_key) {
    return sch->MarkBlock(block_rv, ann_key, ann_val);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, ObjectRef ann_val,
                                 String ann_key) {
    PythonAPICall py("mark_block");
    py.Input("block", block_rv);
    py.Input("ann_key", ann_key);
    if (const auto* int_imm = ann_val.as<IntImmNode>()) {
      py.Input("ann_val", std::to_string(int_imm->value));
    } else if (const auto* str_imm = ann_val.as<StringObj>()) {
      py.Input("ann_val", GetRef<String>(str_imm));
    } else if (const auto* expr = ann_val.as<PrimExprNode>()) {
      std::ostringstream os;
      os << GetRef<PrimExpr>(expr);
      py.Input("ann_val", os.str());
    } else {
      LOG(FATAL) << "TypeError: Cannot handle type: " << ann_val->GetTypeKey();
      throw;
    }
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

struct SoftwarePipelineTraits : public UnpackedInstTraits<SoftwarePipelineTraits> {
  static constexpr const char* kName = "SoftwarePipeline";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv, Integer num_stages) {
    return sch->SoftwarePipeline(loop_rv, num_stages->value);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv, Integer num_stages) {
    PythonAPICall py("software_pipeline");
    py.Input("loop", loop_rv);
    py.Input("num_stages", std::to_string(num_stages->value));
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(PragmaTraits);
TVM_REGISTER_INST_KIND_TRAITS(MarkLoopTraits);
TVM_REGISTER_INST_KIND_TRAITS(MarkBlockTraits);
TVM_REGISTER_INST_KIND_TRAITS(SoftwarePipelineTraits);

}  // namespace tir
}  // namespace tvm
