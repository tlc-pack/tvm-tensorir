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
 * \brief Lower logical intrinsics
 * \file lower_logical_intrin.cc
 */
#include <tvm/arith/iter_affine_map.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

struct LogicalIntrinRegistry {
  static Map<String, PrimFunc> registry;
};

class LogicalIntrinBufferReplacer : public StmtExprMutator {
 public:
  explicit LogicalIntrinBufferReplacer(Map<Var, Buffer> buffer_var_to_new_buffer)
      : buffer_var_to_new_buffer_(std::move(buffer_var_to_new_buffer)) {
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = buffer_var_to_new_buffer_.find(GetRef<Var>(op));
    if (it != buffer_var_to_new_buffer_.end()) {
      return (*it).second->data;
    }
    return GetRef<Var>(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = buffer_var_to_new_buffer_.find(load->buffer->data);
    if (it != buffer_var_to_new_buffer_.end()) {
      auto* n = load.CopyOnWrite();
      n->buffer = (*it).second;
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = buffer_var_to_new_buffer_.find(store->buffer->data);
    if (it != buffer_var_to_new_buffer_.end()) {
      auto* n = store.CopyOnWrite();
      n->buffer = (*it).second;
    }
    return store;
  }

 private:
  Map<Var, Buffer> buffer_var_to_new_buffer_;
};

class LogicalIntrinMutator : public StmtMutator {
 public:
  using FLowerLogicalIntrin = runtime::TypedPackedFunc<PrimFunc(PrimExpr)>;

  explicit LogicalIntrinMutator(const PrimFunc& func) {
    for (const auto& kv : func->buffer_map) {
      const Buffer& buffer = kv.second;
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
  }

  Stmt VisitStmt_(const BlockNode* op) {
    for (const auto& buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const EvaluateNode* op) {
    static const auto& f_lower_logical_intrin = Op::GetAttrMap<PrimFunc>("FLowerLogicalIntrin");
    if (const auto* call = op->value.as<CallNode>()) {
      if (const auto* call_op = call->op.as<OpNode>()) {
        PrimFunc intrin_impl = f_lower_logical_intrin.get(GetRef<Op>(call_op), NullValue<PrimFunc>());
        if (intrin_impl.defined()) {
          // Make inlined call to intrin_impl
          CHECK(intrin_impl->params.size() == call->args.size());
          Map<Var, PrimExpr> subst_map;
          for (size_t i = 0; i < call->args.size(); i++) {
            subst_map.Set(intrin_impl->params[i], call->args[i]);
          }
          Map<Var, Buffer> new_buffer_map;
          for (size_t i = 0; i < call->args.size(); i++) {
            const auto& param = intrin_impl->params[i];
            if (const auto* var = param.as<VarNode>()) {
              if (var->dtype.is_handle()) {
                Var buffer_var = Downcast<Var>(param);
                auto it = intrin_impl->buffer_map.find(buffer_var);
                CHECK(it != intrin_impl->buffer_map.end()) << buffer_var;
                if (it != intrin_impl->buffer_map.end()) {
                  new_buffer_map.Set((*it).second->data,
                                     buffer_data_to_buffer_.at(Downcast<Var>(call->args[i])));
                }
              }
            }
          }

          auto body = Substitute(intrin_impl->body, subst_map);
          return LogicalIntrinBufferReplacer(new_buffer_map)(body);
        }
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
};

namespace transform {

Pass LowerLogicalIntrin() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = LogicalIntrinMutator(f)(std::move(f->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerLogicalLayout", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerLogicalIntrin").set_body_typed(LowerLogicalIntrin);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
