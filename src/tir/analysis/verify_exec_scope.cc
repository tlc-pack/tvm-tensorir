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
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

//class GPUValidator : public StmtVisitor {
// public:
//  void VisitStmt_(const ForNode* loop) final {
//    std::string thread_tag = "";
//    if (loop->kind == ForKind::kThreadBinding && loop->thread_binding.defined())
//      thread_tag = loop->thread_binding.value()->thread_tag;
//
//    bool new_kernel = false;
//    if ((IsBlockIdx(thread_tag) || IsThreadIdx(thread_tag)) && thread_tag != "vthread") {
//      // Check thread binding extents are same in one single kernel
//
//      // If there is no binding, we can regard it as a new kernel
//      new_kernel = thread_extents_.empty();
//
//      auto it = thread_extents_.find(thread_tag);
//      if (it != thread_extents_.end()) {
//        CHECK(ExprDeepEqual()(loop->extent, it->second))
//            << "All loops with the same thread binding must have the same extent, but get "
//            << loop->extent << " vs " << it->second;
//      } else {
//        thread_extents_[thread_tag] = loop->extent;
//      }
//    }
//
//    // Check execution scope and
//    if ((current_scope_ == "gpu_thread") && (IsBlockIdx(thread_tag) || IsThreadIdx(thread_tag))) {
//      // If the current scope is gpu_thread, any inside threadIdx or blockIdx is illegal.
//      LOG(FATAL) << "threadIdx or blockIdx can not be binded under the exec_scope gpu_thread";
//    } else if (current_scope_ == "gpu_warp" &&
//               ((IsBlockIdx(thread_tag) || IsThreadIdx(thread_tag)) &&
//                (thread_tag != "threadIdx.x" || !ExprDeepEqual()(loop->extent, 32)))) {
//      LOG(FATAL) << "threadIdx or blockIdx can not be binded under the exec_scope "
//                    "gpu_thread except threadIdx.x with extents 32";
//    } else if (current_scope_ == "gpu_block" && IsBlockIdx(thread_tag)) {
//      // If the current scope is gpu_block, any inside blockIdx is illegal.
//      LOG(FATAL) << "blockIdx can not be binded under the exec_scope gpu_block";
//    }
//
//    bool contain_thread_x = contain_thread_x_ || thread_tag == "threadIdx.x";
//    std::swap(contain_thread_x, contain_thread_x_);
//    StmtVisitor::VisitStmt_(loop);
//    std::swap(contain_thread_x, contain_thread_x_);
//
//    if (new_kernel) {
//      if (check_thread_x_) {
//        auto it = thread_extents_.find("threadIdx.x");
//        CHECK(it != thread_extents_.end())
//            << "can not find threadIdx.x but find warp level execution scope";
//        CHECK(ExprDeepEqual()(it->second, 32))
//            << "threadIdx.x extent is expected to be 32 with warp level block but get "
//            << it->second;
//      }
//      check_thread_x_ = false;
//      thread_extents_.clear();
//    }
//  }
//
//  void VisitStmt_(const BlockNode* block) final {
//    std::string exec_scope = block->exec_scope;
//    std::string current_scope;
//    std::swap(current_scope, current_scope_);
//
//    if (!exec_scope.empty() && !current_scope.empty()) {
//      if (exec_scope == "gpu_block") {
//        CHECK(current_scope == "gpu_block" || current_scope == "gpu_global");
//      } else if (exec_scope == "gpu_warp") {
//        CHECK(current_scope == "gpu_warp" || current_scope == "gpu_block" ||
//              current_scope == "gpu_global");
//      } else if (exec_scope == "gpu_warp") {
//        CHECK(exec_scope == "gpu_thread" || current_scope == "gpu_warp" ||
//              current_scope == "gpu_block" || current_scope == "gpu_global");
//      }
//    }
//    if (exec_scope == "gpu_warp") {
//      check_thread_x_ = true;
//      ICHECK(!contain_thread_x_);
//    }
//    current_scope_ = exec_scope;
//    StmtVisitor::VisitStmt_(block);
//    std::swap(current_scope, current_scope_);
//  }
//
//  /*! \brief The final result */
//
// private:
//  static inline bool IsThreadIdx(const std::string& thread_tag) {
//    return thread_tag.substr(0, 9) == "threadIdx" || thread_tag.substr(0, 7) == "vthread";
//  }
//
//  static inline bool IsBlockIdx(const std::string& thread_tag) {
//    return thread_tag.substr(0, 9) == "BlockIdx";
//  }
//
//  /*! \brief The current execution scope (gpu_global, gpu_block, gpu_warp or gpu_thread) */
//  std::string current_scope_ = "gpu_global";
//  /*! \brief The extents of each threadIdx or blockIdx */
//  std::unordered_map<std::string, PrimExpr> thread_extents_;
//  /*! \brief Whether need to check threadIdx.x extents = 32 */
//  bool check_thread_x_ = false;
//  /*! \brief The loop stack from current node up to root contain thread_x */
//  bool contain_thread_x_ = false;
//};
//
//bool VerifyExecScope(const PrimFunc& func) {
//  GPUValidator gpu_validator;
//  gpu_validator(func->body);
//  return true;
//}
//
//TVM_REGISTER_GLOBAL("tir.analysis.VerifyExecScope").set_body_typed(VerifyExecScope);

}  // namespace tir
}  // namespace tvm
