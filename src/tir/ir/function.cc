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
 * \file src/tir/ir/function.cc
 * \brief The function data structure.
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "functor_common.h"

namespace tvm {
namespace tir {

// Get the function type of a PrimFunc
PrimFunc::PrimFunc(Array<tir::Var> params, Stmt body, Type ret_type,
                   Map<tir::Var, Buffer> buffer_map, DictAttrs attrs) {
  // Assume void-return type for now
  // TODO(tvm-team) consider type deduction from body.
  if (!ret_type.defined()) {
    ret_type = VoidType();
  }
  auto n = make_object<PrimFuncNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->buffer_map = std::move(buffer_map);
  n->attrs = std::move(attrs);
  n->checked_type_ = n->func_type_annotation();
  data_ = std::move(n);
}

FuncType PrimFuncNode::func_type_annotation() const {
  Array<Type> param_types;
  for (auto param : this->params) {
    param_types.push_back(GetType(param));
  }
  return FuncType(param_types, ret_type, {}, {});
}

TensorIntrin::TensorIntrin(PrimFunc desc_func, PrimFunc intrin_func) {
  // check the number of func var is equal
  CHECK_EQ(desc_func->params.size(), intrin_func->params.size());
  CHECK_EQ(desc_func->buffer_map.size(), intrin_func->buffer_map.size());

  // check both functions' bodies are directly block
  const auto* desc_realize = desc_func->body.as<BlockRealizeNode>();
  const auto* intrin_realize = intrin_func->body.as<BlockRealizeNode>();
  CHECK(desc_realize != nullptr);
  CHECK(intrin_realize != nullptr);
  CHECK_EQ(desc_realize->exec_scope, intrin_realize->exec_scope);

  const Block& desc_block = desc_realize->block;
  const Block& intrin_block = intrin_realize->block;

  // check block var number and iter type
  CHECK_EQ(desc_block->iter_vars.size(), intrin_block->iter_vars.size());
  for (size_t i = 0; i < desc_block->iter_vars.size(); i++) {
    const IterVar& desc_var = desc_block->iter_vars[i];
    const IterVar& intrin_var = intrin_block->iter_vars[i];
    CHECK(desc_var->iter_type == intrin_var->iter_type);
  }

  auto n = make_object<TensorIntrinNode>();
  n->description = std::move(desc_func);
  n->implementation = std::move(intrin_func);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(PrimFuncNode);
TVM_REGISTER_NODE_TYPE(TensorIntrinNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimFuncNode>([](const ObjectRef& ref, ReprPrinter* p) {
      // TODO(tvm-team) redirect to Text printer once we have a good text format.
      auto* node = static_cast<const PrimFuncNode*>(ref.get());
      p->stream << "PrimFunc(" << node->params << ") ";
      if (node->attrs.defined()) {
        p->stream << "attrs=" << node->attrs;
      }
      p->stream << " {\n";
      p->indent += 2;
      p->Print(node->body);
      p->indent -= 2;
      p->stream << "}\n";
    });

TVM_REGISTER_GLOBAL("tir.PrimFunc")
    .set_body_typed([](Array<tir::Var> params, Stmt body, Type ret_type,
                       Map<tir::Var, Buffer> buffer_map, DictAttrs attrs) {
      return PrimFunc(params, body, ret_type, buffer_map, attrs);
    });

TVM_REGISTER_GLOBAL("tir.TensorIntrin")
    .set_body_typed([](PrimFunc desc_func, PrimFunc intrin_func) {
      return TensorIntrin(desc_func, intrin_func);
    });

// Mutate buffer's declarations, and replace them in the AST
class BufferMutator : public StmtExprMutator {
 public:
  explicit BufferMutator(const std::function<Buffer(const Buffer&)>& fmutate) : fmutate_(fmutate) {}

  PrimFunc MutatePrimFunc(PrimFunc f) {
    PrimFunc new_f = f;
    std::unordered_map<tir::Var, Buffer, ObjectPtrHash, ObjectPtrEqual> new_buffer_map;
    for (auto it : new_f->buffer_map) {
      Buffer new_buffer = fmutate_(it.second);
      new_buffer_map[it.first] = new_buffer;
      if (!new_buffer.same_as(it.second)) {
        buffer_map_[it.second] = new_buffer;
      }
    }
    new_f.CopyOnWrite()->buffer_map = Map<tir::Var, Buffer>(new_buffer_map);
    new_f.CopyOnWrite()->body = VisitStmt(new_f->body);
    return new_f;
  }

  Stmt VisitStmt_(const BlockNode* op) override {
    auto fmutate_buffer_allocate = [this](const BufferAllocate& buffer_allocate) {
      Buffer buf = fmutate_(buffer_allocate->buffer);
      if (buf.same_as(buffer_allocate->buffer)) {
        return buffer_allocate;
      } else {
        buffer_map_[buffer_allocate->buffer] = buf;
        auto n = CopyOnWrite(buffer_allocate.get());
        n->buffer = std::move(buf);
        return BufferAllocate(n);
      }
    };
    auto fmutate_tensor_region = [this](const TensorRegion& tensor_region) {
      auto it = buffer_map_.find(tensor_region->buffer);
      if (it == buffer_map_.end()) {
        return tensor_region;
      } else {
        auto n = CopyOnWrite(tensor_region.get());
        n->buffer = it->second;
        return TensorRegion(n);
      }
    };
    Array<BufferAllocate> allocations = MutateArray(op->allocations, fmutate_buffer_allocate);
    Array<TensorRegion> reads = MutateArray(op->reads, fmutate_tensor_region);
    Array<TensorRegion> writes = MutateArray(op->writes, fmutate_tensor_region);
    Stmt body = VisitStmt(op->body);
    if (allocations.same_as(op->allocations) && reads.same_as(op->reads) &&
        writes.same_as(op->writes) && body.same_as(op->body)) {
      return GetRef<Block>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->allocations = std::move(allocations);
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      n->body = std::move(body);
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) override {
    auto it = buffer_map_.find(op->buffer);
    PrimExpr value = VisitExpr(op->value);
    if (it == buffer_map_.end() && value.same_as(op->value)) {
      return GetRef<BufferStore>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->buffer = it->second;
      n->value = std::move(value);
      return Stmt(n);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) override {
    auto it = buffer_map_.find(op->buffer);
    if (it == buffer_map_.end()) {
      return GetRef<BufferLoad>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->buffer = it->second;
      return PrimExpr(n);
    }
  }

 private:
  /* \brief a closure that mutates a buffer */
  std::function<Buffer(const Buffer&)> fmutate_;
  /* \brief map from old buffer to mutated buffer */
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
};

PrimFunc PrimFunc::bind_free_vars(Map<tir::Var, PrimExpr> values) {
  // check PrimFunc has free_vars attr
  const auto& attrs = this->get()->attrs;
  auto it = attrs->dict.find("free_vars");
  CHECK(it != attrs->dict.end());
  const auto* free_vars = (*it).second.as<ArrayNode>();
  CHECK(free_vars != nullptr);

  PrimFunc new_func = *this;

  std::vector<tir::Var> new_free_vars;
  for (const auto& var_obj : *free_vars) {
    CHECK(var_obj->IsInstance<tir::VarNode>());
    const auto& var = Downcast<tir::Var>(var_obj);
    auto itt = values.find(var);
    if (itt == values.end()) {
      new_free_vars.push_back(var);
    } else {
      auto f = [&](const Var& v) -> PrimExpr {
        if (v.same_as(var)) {
          return (*itt).second;
        } else {
          return v;
        }
      };
      auto f_buffer = [&](const Buffer& buf) -> Buffer {
        Buffer new_buffer = buf;
        new_buffer.CopyOnWrite()->elem_offset = Substitute(new_buffer->elem_offset, f);
        std::vector<PrimExpr> new_shape, new_stride;
        for (const auto& dim : new_buffer->shape) new_shape.push_back(Substitute(dim, f));
        for (const auto& stride : new_buffer->strides) new_shape.push_back(Substitute(stride, f));
        new_buffer.CopyOnWrite()->shape = Array<PrimExpr>(new_shape);
        new_buffer.CopyOnWrite()->strides = Array<PrimExpr>(new_stride);
        return new_buffer;
      };
      // replace vars in buffer
      BufferMutator buffer_mutator(f_buffer);
      new_func = buffer_mutator.MutatePrimFunc(new_func);
      // replace other vars inside body
      new_func.CopyOnWrite()->body = Substitute(new_func->body, f);
    }
  }
  // update free vars list
  if (!new_free_vars.empty()) {
    new_func.CopyOnWrite()->attrs.CopyOnWrite()->dict.CopyOnWrite()->at(String("free_vars")) =
        Array<Var>(new_free_vars);
  } else {
    new_func.CopyOnWrite()->attrs.CopyOnWrite()->dict.CopyOnWrite()->erase(String("free_vars"));
  }
  return new_func;
}

TVM_REGISTER_GLOBAL("tir.BindFreeVars")
    .set_body_typed<PrimFunc(PrimFunc, Map<tir::Var, PrimExpr>)>(
        [](PrimFunc func, Map<tir::Var, PrimExpr> values) {
          return func.bind_free_vars(std::move(values));
        });

}  // namespace tir
}  // namespace tvm
