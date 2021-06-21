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
    py.Attr("pragma_type", pragma_type);
    py.Input("pragma_value", pragma_value);
    return py.Str();
  }

  template <typename>
  friend struct UnpackedInstTraits;
};

struct MarkLoopTraits : public UnpackedInstTraits<MarkLoopTraits> {
  static constexpr const char* kName = "MarkLoop";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv, ExprRV ann_val,
                                      String ann_key) {
    return sch->MarkLoop(loop_rv, ann_key, ann_val);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv, String ann_val,
                                 String ann_key) {
    PythonAPICall py("mark_loop");
    py.Input("loop", loop_rv);
    py.Attr("ann_key", ann_key);
    py.Input("ann_val", ann_val);
    return py.Str();
  }

  template <typename>
  friend struct UnpackedInstTraits;
};

struct MarkBlockTraits : public UnpackedInstTraits<MarkBlockTraits> {
  static constexpr const char* kName = "MarkBlock";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, ExprRV ann_val,
                                      String ann_key) {
    return sch->MarkBlock(block_rv, ann_key, ann_val);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, String ann_val,
                                 String ann_key) {
    PythonAPICall py("mark_block");
    py.Input("block", block_rv);
    py.Attr("ann_key", ann_key);
    py.Input("ann_val", ann_val);
    return py.Str();
  }

  template <typename>
  friend struct UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND(PragmaTraits);
TVM_REGISTER_INST_KIND(MarkLoopTraits);
TVM_REGISTER_INST_KIND(MarkBlockTraits);

}  // namespace tir
}  // namespace tvm
