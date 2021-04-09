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
 * \file flatten_buffer.cc
 */

#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

inline bool StrStartsWith(const String& str, const String& prefix) {
  int n = prefix.size();
  if (static_cast<int>(str.size()) < n) {
    return false;
  }
  const char* data = str.data();
  return std::equal(data, data + n, prefix.data());
}

PrimExpr BufferArea(const Buffer& buffer) {
  PrimExpr area = Integer(1);
  for (const PrimExpr& dim : buffer->shape) {
    area = area * dim;
  }
  return area;
}

bool IsReduceTempBuffer(const Buffer& buffer) {
  return StrStartsWith(buffer->name, "normal_reduce_temp") ||  //
         StrStartsWith(buffer->name, "reduce_temp");
}

/*!
 * \brief Transform multi-dimension BufferLoad/BufferStore into one-dimension Load/Store
 */
class BufferFlattener : public StmtExprMutator {
 public:
  static Stmt Flatten(const PrimFunc& f) { return BufferFlattener().VisitStmt(f->body); }

 private:
  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    ICHECK(realize->iter_values.empty());
    // Step 1. Visit the body
    Block new_block = Downcast<Block>(this->VisitStmt(realize->block));
    PrimExpr predicate = this->VisitExpr(realize->predicate);
    const BlockNode* block = new_block.get();
    // Step 2. Transform the `predicate` to if-then-else
    Stmt body = block->body;
    if (!is_one(predicate)) {
      body = IfThenElse(predicate, body);
    }
    // Step 4. Handle allocations
    for (const Buffer& buffer : block->alloc_buffers) {
      body = MakeAllocStmt(buffer, body);
    }
    return body;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // Step 1. Visit recursively
    PrimExpr min = this->VisitExpr(op->min);
    PrimExpr extent = this->VisitExpr(op->extent);
    Stmt body = this->VisitStmt(op->body);
    // Step 2. Add the for loop accordingly
    if (op->kind == ForKind::kThreadBinding) {
      // Case 1. Thread binding
      ICHECK(op->thread_binding.defined());
      String thread_tag = op->thread_binding.value()->thread_tag;
      body = MakeLaunchThread(min, extent, op->loop_var, thread_tag, body);
    } else if (is_one(extent) && op->annotations.empty()) {
      // Case 2. Handle unit loop
      return body;
    } else {
      // Case 3. An ordinary loop
      body = For(op->loop_var, min, extent, op->kind, body);
    }
    // Step 3. Handle annotations
    for (const auto& annotation : op->annotations) {
      const String& ann_key = annotation.first;
      const ObjectRef& ann_value = annotation.second;
      if (attr::IsPragmaKey(ann_key)) {
        body = AttrStmt(op->loop_var, ann_key, Downcast<PrimExpr>(ann_value), body);
      }
    }
    return body;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    op = store.get();
    return op->buffer.vstore(op->indices, op->value);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    op = load.get();
    return op->buffer.vload(op->indices, op->dtype);
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::get_elem_offset())) {
      // Handle `get_elem_offset`
      ICHECK_EQ(op->args.size(), 1);
      PrimExpr arg = op->args[0];
      ICHECK(arg->IsInstance<BufferLoadNode>());
      arg = this->VisitExpr(arg);
      const auto* load = arg.as<LoadNode>();
      ICHECK(load != nullptr);
      return load->index;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  static Stmt MakeAllocStmt(const Buffer& buffer, Stmt body) {
    if (IsReduceTempBuffer(buffer)) {
      return body;
    }
    String storage_scope = buffer->scope;
    if (storage_scope.empty()) {
      storage_scope = "global";
    }
    PrimExpr area = BufferArea(buffer);
    body = Allocate(buffer->data, buffer->dtype, {area}, const_true(), body);
    body = AttrStmt(buffer->data, attr::storage_scope, StringImm(storage_scope), body);
    return body;
  }

  static Stmt MakeLaunchThread(const PrimExpr& min, const PrimExpr& extent, const Var& var,
                               const String& thread_tag, Stmt body) {
    IterVar iter_var(/*dom=*/Range::FromMinExtent(min, extent),
                     /*var=*/var,
                     /*iter_type=*/IterVarType::kThreadIndex,
                     /*thread_tag=*/thread_tag);
    String attr_key = thread_tag == "vthread" ? attr::virtual_thread : attr::thread_extent;
    body = AttrStmt(iter_var, attr_key, extent, body);
    return body;
  }
};

PrimFunc FlattenBuffer(PrimFunc f) {
  PrimFuncNode* fptr = f.CopyOnWrite();
  fptr->body = BufferFlattener::Flatten(f);
  return f;
}

namespace transform {

Pass FlattenBuffer() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return FlattenBuffer(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.FlattenBuffer", {});
}

TVM_REGISTER_GLOBAL("tir.transform.FlattenBuffer").set_body_typed(FlattenBuffer);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
