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

#include <tvm/arith/iter_affine_map.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../runtime/thread_storage_scope.h"
#include "./ir_utils.h"
/*!
 * \brief Automatically generate thread binding for auto copy blocks
 * \file lower_auto_copy.cc
 */

namespace tvm {
namespace tir {



class AutoCopyMutator : public StmtExprMutator {
 private:
  Stmt FuseNestLoops(Stmt stmt) {
    if (!stmt->IsInstance<ForNode>()) {
      return stmt;
    }
    std::vector<const ForNode*> loops;
    Stmt body = stmt;
    while (const ForNode* loop = body.as<ForNode>()) {
      loops.push_back(loop);
      body = loop->body;
    }
    Var fused_var = loops[0]->loop_var.copy_with_suffix("_fused");
    Array<PrimExpr> substitute_value;
    substitute_value.resize(loops.size());
    PrimExpr tot = fused_var;
    for (int i = static_cast<int>(loops.size()) - 1; i >= 0; i--) {
      substitute_value.Set(i, floormod(tot, loops[i]->extent));
      tot = floordiv(tot, loops[i]->extent);
    }
    auto f_substitute = [&](const Var& v) -> Optional<PrimExpr> {
      for (int i = 0; i < static_cast<int>(loops.size()); i++) {
        if (v.same_as(loops[i]->loop_var)) {
          return substitute_value[i];
        }
      }
      return NullOpt;
    };
    PrimExpr fused_extent = 1;
    for (int i = 0; i < static_cast<int>(loops.size()); i++) {
      fused_extent *= loops[i]->extent;
    }
    Stmt new_stmt = Substitute(body, f_substitute);
    new_stmt = For(fused_var, 0, fused_extent, ForKind::kSerial, new_stmt);
    return new_stmt;
  }
  
  Stmt SplitBindVectorize(Stmt body, int vector_bytes) {
    const ForNode* loop = body.as<ForNode>();
    int tot_threads=threadIdx_x_*threadIdx_y_*threadIdx_z_;
    if (!loop || !is_zero(indexmod(loop->extent, (vector_bytes * tot_threads)))) {
      return body;
    }
    PrimExpr outer_loop_extent = indexdiv(loop->extent, tot_threads * vector_bytes);
    Array<PrimExpr> factors{outer_loop_extent};
    std::vector<std::string> thread_axis;
    int new_loop_num=2;
    if (threadIdx_z_ != 1) {
      factors.push_back(threadIdx_z_);
      thread_axis.push_back("threadIdx.z");
      new_loop_num++;
    }
    if (threadIdx_y_ != 1) {
      factors.push_back(threadIdx_y_);
      thread_axis.push_back("threadIdx.y");
      new_loop_num++;
    }
    if (threadIdx_x_ != 1) {
      factors.push_back(threadIdx_x_);
      thread_axis.push_back("threadIdx.x");
      new_loop_num++;
    }
    factors.push_back(vector_bytes);
    std::vector<Var> new_loop_vars;
    new_loop_vars.reserve(new_loop_num);
    for (int i = 0; i < new_loop_num; i++) {
      new_loop_vars.push_back(loop->loop_var.copy_with_suffix("_" + std::to_string(i)));
    }
    
    PrimExpr substitute_value =0;
    for (int i = 0;i<new_loop_num;i++) {
      substitute_value*=factors[i];
      substitute_value+=new_loop_vars[i];
    }
    body = Substitute(loop->body, [&](const Var& v) -> Optional<PrimExpr> {
      if (v.same_as(loop->loop_var)) {
        return substitute_value;
      } else {
        return NullOpt;
      }
    });

    For new_loop = For(new_loop_vars[new_loop_num-1], 0, vector_bytes, ForKind::kVectorized, body);
    
    for (int i = new_loop_num-2; i >= 1; i--) {
      new_loop = For(new_loop_vars[i],0, factors[i], ForKind::kThreadBinding, new_loop,
                     IterVar(Range(nullptr), Var(thread_axis[i-1]), kThreadIndex,
                             thread_axis[i-1]));
    }
    
    new_loop = For(new_loop_vars[0], 0, outer_loop_extent, ForKind::kSerial, new_loop);
    return std::move(new_loop);
  }


  
  Stmt UnbindThreads(Stmt stmt){
    class ThreadBindingVarFinder: public StmtExprVisitor{
     public:
      ThreadBindingVarFinder(const std::vector<const ForNode*>& thread_binding_loops):thread_binding_loops_(thread_binding_loops){}
      static std::unordered_set<const ForNode*> FindLoopsToUnbind(Stmt stmt, const
                                                                  std::vector<const ForNode*>&
                                                                      thread_binding_loops){
        ThreadBindingVarFinder finder(thread_binding_loops);
        finder(stmt);
        return finder.loops_to_unbind_;
      }
     private:
      void VisitExpr_(const VarNode* op) final{
        for (const ForNode* loop : thread_binding_loops_) {
          const String& thread_tag = loop->thread_binding.value()->thread_tag;
          runtime::ThreadScope thread_scope=runtime::ThreadScope::Create(thread_tag);
          if (op == loop->loop_var.get() && static_cast<int>(runtime::StorageRank::kShared) <=
                                                static_cast<int>(thread_scope.rank)) {
            loops_to_unbind_.insert(loop);
          }
        }
      }
    
      const std::vector<const ForNode*>& thread_binding_loops_;
      std::unordered_set<const ForNode*> loops_to_unbind_;
    };
    
    std::unordered_set<const ForNode*> loops_to_unbind=ThreadBindingVarFinder::FindLoopsToUnbind
        (stmt, thread_binding_loops_);
    std::unordered_map<const VarNode*, PrimExpr> substitute_map;
    Stmt new_stmt = stmt;
    for (const ForNode* loop:loops_to_unbind) {
      Var var = loop->loop_var.copy_with_suffix("_unbind");
      substitute_map[loop->loop_var.get()]=var;
      new_stmt=For(var, loop->min, loop->extent, ForKind::kSerial, std::move(new_stmt));
    }
    new_stmt= Substitute(std::move(new_stmt),substitute_map);
    return new_stmt;
  }


  Stmt CoalesceGlobalLoad(Stmt stmt, int vector_bytes, int warp_size) {
    stmt = UnbindThreads(std::move(stmt));
    stmt = FuseNestLoops(std::move(stmt));
    stmt = SplitBindVectorize(std::move(stmt), vector_bytes);
    return stmt;
  }
  
  Stmt VisitStmt_(const BlockNode* op) final {
    Block block;
    if (op->annotations.count("auto_copy") &&
        is_one(Downcast<PrimExpr>(op->annotations["auto_copy"]))) {
      in_auto_copy_ = true;
      block = runtime::Downcast<Block>(StmtMutator::VisitStmt_(op));
      BlockNode* n = block.CopyOnWrite();
      if ((src_scope_.rank == runtime::StorageRank::kGlobal &&
           tgt_scope_.rank == runtime::StorageRank::kShared) ||
          (src_scope_.rank == runtime::StorageRank::kShared &&
           tgt_scope_.rank == runtime::StorageRank::kGlobal)) {
        int vector_bytes;
        if (op->annotations.count("vector_bytes")) {
          IntImm vec_bytes = Downcast<IntImm>(op->annotations["vector_bytes"]);
          vector_bytes = vec_bytes->value;
        } else {
          vector_bytes = 1;
        }
        n->body = CoalesceGlobalLoad(op->body, vector_bytes, threadIdx_x_);
      }
    } else {
      block = runtime::Downcast<Block>(StmtMutator::VisitStmt_(op));
    }
    return std::move(block);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    if (in_auto_copy_) {
      if (const BufferLoadNode* buf_load = op->value.as<BufferLoadNode>()) {
        src_scope_ = runtime::StorageScope::Create(buf_load->buffer.scope());
      }
      tgt_scope_ = runtime::StorageScope::Create(op->buffer.scope());
    }
    return GetRef<BufferStore>(op);
  }
  
  Stmt VisitStmt_(const ForNode* op) final{
    if (op->thread_binding.defined()) {
      IterVar binding=op->thread_binding.value();
      if (binding->iter_type == kThreadIndex) {
        if(binding->thread_tag=="threadIdx.x") {
          threadIdx_x_ = Downcast<IntImm>(op->extent)->value;
        } else if(binding->thread_tag=="threadIdx.y"){
          threadIdx_y_=Downcast<IntImm>(op->extent)->value;
        } else if (binding->thread_tag == "threadIdx.z") {
          threadIdx_z_=Downcast<IntImm>(op->extent)->value;
        }
        thread_binding_loops_.push_back(op);
      }
    }
    return StmtMutator::VisitStmt_(op);
  }
  
  bool in_auto_copy_;
  runtime::StorageScope src_scope_;
  runtime::StorageScope tgt_scope_;
  int threadIdx_x_ = 1;
  int threadIdx_y_ = 1;
  int threadIdx_z_= 1;
  std::vector<const ForNode*> thread_binding_loops_;
};

namespace transform {
Pass LowerAutoCopy() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = AutoCopyMutator()(std::move(f->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerAutoCopy", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerAutoCopy").set_body_typed(LowerAutoCopy);
}  // namespace transform
}  // namespace tir
}  // namespace tvm