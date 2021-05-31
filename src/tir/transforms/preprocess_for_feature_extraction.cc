/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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

/*!
 * \file preprocess_for_feature_extraction.cc
 */
#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>

#include "../schedule/utils.h"

namespace tvm {
namespace tir {

class SimpifyConstMatrix : public StmtExprMutator {
 public:
  static Stmt simplifyConstMatrix(const PrimFunc& func) {
    SimpifyConstMatrix simp;
    return simp.VisitStmt(func->body);
  }

 private:
  PrimExpr VisitExpr_(const SelectNode* node) { return make_const(node->dtype, 1.0f); }
};
PrimFunc PreprocessForFeatureExtraction(PrimFunc f) {
  
  auto pass_list = Array<tvm::transform::Pass>();
  pass_list.push_back(tir::transform::LowerInitBlock());
  pass_list.push_back(tir::transform::PlanAndUpdateBufferAllocationLocation());
  pass_list.push_back(tir::transform::ConvertBlocksToOpaque());
  pass_list.push_back(tir::transform::CompactBufferAllocation());
  Map<GlobalVar,BaseFunc> tmp_map;
  tmp_map.Set(GlobalVar("main"),f);
  IRModule mod(tmp_map);
  mod=tir::transform::Sequential(pass_list)(mod);
  f= Downcast<PrimFunc>(mod->Lookup("main"));
  PrimFuncNode* fptr = f.CopyOnWrite();
  fptr->body = SimpifyConstMatrix::simplifyConstMatrix(f);
  return f;
}

namespace transform {
Pass PreprocessForFeatureExtraction() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return PreprocessForFeatureExtraction(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.PreProcessForFeatureExtraction", {});
}

TVM_REGISTER_GLOBAL("tir.transform.PreProcessForFeatureExtraction").set_body_typed(BufferFlatten);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
