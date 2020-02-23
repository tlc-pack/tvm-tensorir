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

/*!
 *  Exposure of pass functions.
 * \file api_pass.cc
 */
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/module.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace tir {

TVM_REGISTER_GLOBAL("ir_pass.Simplify")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args[0].IsObjectRef<Stmt>()) {
      if (args.size() > 1) {
        *ret = Simplify(args[0].operator Stmt(), args[1]);
      } else {
        *ret = Simplify(args[0].operator Stmt());
      }
    } else {
      if (args.size() > 1) {
        *ret = Simplify(args[0].operator PrimExpr(), args[1]);
      } else {
        *ret = Simplify(args[0].operator PrimExpr());
      }
    }
  });

TVM_REGISTER_GLOBAL("ir_pass.CanonicalSimplify")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args[0].IsObjectRef<Stmt>()) {
      if (args.size() > 1) {
        *ret = CanonicalSimplify(args[0].operator Stmt(), args[1]);
      } else {
        *ret = CanonicalSimplify(args[0].operator Stmt());
      }
    } else {
      if (args.size() > 1) {
        *ret = CanonicalSimplify(args[0].operator PrimExpr(), args[1]);
      } else {
        *ret = CanonicalSimplify(args[0].operator PrimExpr());
      }
    }
  });

TVM_REGISTER_GLOBAL("ir_pass.Substitute")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args[0].IsObjectRef<Stmt>()) {
      *ret = Substitute(args[0].operator Stmt(), args[1].operator Map<Var, PrimExpr>());
    } else {
      *ret = Substitute(args[0].operator PrimExpr(), args[1].operator Map<Var, PrimExpr>());
    }
  });


bool Equal(const IRModule& lhs,
           const IRModule& rhs,
           bool remap_free_var,
           bool assert_mode) {
  std::unordered_set<std::string> lhs_func_set;
  std::unordered_set<std::string> rhs_func_set;
  for (auto it = lhs->functions.begin(); it != lhs->functions.end(); ++it) {
    const BaseFunc& lhsFunc = (*it).second;
    if (lhsFunc->IsInstance<tir::FunctionNode>()) {
      lhs_func_set.insert(Downcast<Function>((*it).second)->name);
    }
  }
  for (auto it = rhs->functions.begin(); it != rhs->functions.end(); ++it) {
    const BaseFunc& rhsFunc = (*it).second;
    if (rhsFunc->IsInstance<tir::FunctionNode>()) {
      rhs_func_set.insert(Downcast<Function>((*it).second)->name);
    }
  }
  for (const auto & name : lhs_func_set)
    if (rhs_func_set.find(name) == rhs_func_set.end()) {
      return false;
    } else {
      if (!Equal(Downcast<Function>(lhs->Lookup(name)),
                 Downcast<Function>(rhs->Lookup(name)),
                 remap_free_var, assert_mode))
        return false;
      rhs_func_set.erase(name);
    }
  return rhs_func_set.empty();
}

#define REGISTER_EQUAL_PASS(PassName, remap_free_var, assert_mode)                           \
  TVM_REGISTER_GLOBAL("ir_pass."#PassName)                                                   \
  .set_body([](TVMArgs args, TVMRetValue *ret) {                                             \
    if (args[0].IsObjectRef<Stmt>()) {                                                       \
      *ret = Equal(args[0].operator Stmt(), args[1].operator Stmt(),                         \
                   remap_free_var, assert_mode);                                             \
    } else if (args[0].IsObjectRef<PrimExpr>()) {                                            \
      *ret = Equal(args[0].operator PrimExpr(), args[1].operator PrimExpr(),                 \
                   remap_free_var, assert_mode);                                             \
    } else if (args[0].IsObjectRef<Function>()){                                             \
      *ret = Equal(args[0].operator Function(), args[1].operator Function(),                 \
                   remap_free_var, assert_mode);                                             \
    } else {                                                                                 \
      *ret = Equal(args[0].operator IRModule(), args[1].operator IRModule(),                 \
                   remap_free_var, assert_mode);                                             \
    }                                                                                        \
  });

// Basic equal pass
REGISTER_EQUAL_PASS(Equal, false, false);
// Basic equal pass with assert mode
REGISTER_EQUAL_PASS(AssertEqual, false, true);
// Struct equal pass, which can remap free vars
REGISTER_EQUAL_PASS(StructEqual, true, false);
// Struct equal pass with assert mode
REGISTER_EQUAL_PASS(AssertStructEqual, true, true);

TVM_REGISTER_GLOBAL("ir_pass.StorageFlatten")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args.size() <= 3) {
      *ret = StorageFlatten(args[0], args[1], args[2]);
    } else {
      *ret = StorageFlatten(args[0], args[1], args[2], args[3]);
    }
  });

TVM_REGISTER_GLOBAL("ir_pass.RewriteForTensorCore")
.set_body_typed
  ([](const Stmt& stmt,
      const te::Schedule& schedule,
      const Map<te::Tensor, Buffer>& extern_buffer) {
      return RewriteForTensorCore(stmt, schedule, extern_buffer);
  });

TVM_REGISTER_GLOBAL("ir_pass.AttrsEqual")
.set_body_typed(
  [](const ObjectRef& lhs, const ObjectRef& rhs) {
    return AttrsEqual()(lhs, rhs);
  });

TVM_REGISTER_GLOBAL("ir_pass.AttrsHash")
.set_body_typed([](const ObjectRef &node) -> int64_t {
    return AttrsHash()(node);
});


TVM_REGISTER_GLOBAL("ir_pass.ExprUseVar")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = ExprUseVar(args[0].operator PrimExpr(), args[1].operator Var());
  });

TVM_REGISTER_GLOBAL("ir_pass.PostOrderVisit")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    PackedFunc f = args[1];
    tir::PostOrderVisit(args[0], [f](const ObjectRef& n) {
        f(n);
      });
  });

TVM_REGISTER_GLOBAL("ir_pass.LowerStorageAccess")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  LoweredFunc f = args[0];
  auto n = make_object<LoweredFuncNode>(*f.operator->());
  n->body = LowerStorageAccessInfo(f->body);
  *ret = LoweredFunc(n);
});

// make from two arguments
#define REGISTER_PASS(PassName)                                   \
  TVM_REGISTER_GLOBAL("ir_pass."#PassName)                           \
  .set_body_typed(PassName);                                     \


REGISTER_PASS(ConvertSSA);
REGISTER_PASS(VerifySSA);
REGISTER_PASS(RewriteUnsafeSelect);
REGISTER_PASS(Inline);
REGISTER_PASS(IRTransform);
REGISTER_PASS(VectorizeLoop);
REGISTER_PASS(SkipVectorize);
REGISTER_PASS(UnrollLoop);
REGISTER_PASS(InjectCopyIntrin);
REGISTER_PASS(ThreadSync);
REGISTER_PASS(MakeAPI);
REGISTER_PASS(BindDeviceType);
REGISTER_PASS(SplitHostDevice);
REGISTER_PASS(StorageRewrite);
REGISTER_PASS(CoProcSync);
REGISTER_PASS(LowerStorageAccessInfo);
REGISTER_PASS(LowerDeviceStorageAccessInfo)
REGISTER_PASS(InjectVirtualThread);
REGISTER_PASS(InjectPrefetch);
REGISTER_PASS(InjectDoubleBuffer);
REGISTER_PASS(LoopPartition);
REGISTER_PASS(RemoveNoOp);
REGISTER_PASS(LiftAttrScope);
REGISTER_PASS(LowerThreadAllreduce);
REGISTER_PASS(LowerWarpMemory);
REGISTER_PASS(RemapThreadAxis);
REGISTER_PASS(LowerIntrin);
REGISTER_PASS(LowerCustomDatatypes);
REGISTER_PASS(LowerTVMBuiltin);
REGISTER_PASS(CombineContextCall);
REGISTER_PASS(VerifyMemory);
REGISTER_PASS(VerifyGPUCode);
REGISTER_PASS(DecorateDeviceScope);
REGISTER_PASS(InstrumentBoundCheckers);
REGISTER_PASS(VerifyCompactBuffer);
REGISTER_PASS(HoistIfThenElse);
REGISTER_PASS(InferFragment);
REGISTER_PASS(BufferFlatten);
}  // namespace tir
}  // namespace tvm
