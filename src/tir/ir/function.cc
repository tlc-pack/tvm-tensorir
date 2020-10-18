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
#include <tvm/tir/analysis.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <utility>

#include "../schedule/schedule_common.h"
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
  explicit BufferMutator(std::function<Buffer(const Buffer&)> fmutate)
      : fmutate_(std::move(fmutate)) {}

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

using VarMapType = std::unordered_map<tir::Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual>;

class SpecializeConstraintRemover : public StmtMutator {
 public:
  explicit SpecializeConstraintRemover(const Var& param) : param_(param) {}

  Stmt VisitStmt_(const AssertStmtNode* op) override {
    if (auto eq = op->condition.as<EQNode>()) {
      if (eq->a.same_as(param_)) return op->body;
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SeqStmtNode* op) override { return GetRef<SeqStmt>(op); }

 private:
  const tir::Var& param_;
};

PrimFunc RemoveSpecializeConstraint(const PrimFunc& f, const tir::Var& param) {
  CHECK(f->body->IsInstance<BlockRealizeNode>())
      << "ValueError: The body of PrimFunc ought to be block";
  SpecializeConstraintRemover specialize_constraint_remover(param);
  PrimFunc new_f = f;
  BlockRealize new_block = Downcast<BlockRealize>(f->body);
  new_block.CopyOnWrite()->block.CopyOnWrite()->body =
      specialize_constraint_remover(new_block->block->body);
  new_f.CopyOnWrite()->body = new_block;
  return new_f;
}

PrimExpr FetchSpecializeConstraint(const PrimFunc& f, const tir::Var& param) {
  CHECK(f->body->IsInstance<BlockRealizeNode>())
      << "ValueError: The body of PrimFunc ought to be block";
  PrimExpr result(nullptr);
  tir::PreOrderVisit(f->body.as<BlockRealizeNode>()->block->body,
                     [&](const ObjectRef& obj) -> bool {
                       if (obj->IsInstance<AssertStmtNode>()) {
                         if (auto eq = obj.as<AssertStmtNode>()->condition.as<EQNode>()) {
                           if (eq->a.same_as(param)) {
                             result = eq->b;
                           }
                         }
                       } else if (obj->IsInstance<SeqStmtNode>()) {
                         return false;
                       }
                       return true;
                     });
  return result;
}

PrimFunc ExertSpecializeConstraint(const VarMapType& param_var_map, const PrimFunc& f) {
  CHECK(f->body->IsInstance<BlockRealizeNode>())
      << "ValueError: The body of PrimFunc ought to be block";
  Stmt body = f->body.as<BlockRealizeNode>()->block->body;
  for (const auto& it : param_var_map) {
    PrimExpr old_constraint = FetchSpecializeConstraint(f, it.first);
    CHECK(!old_constraint.defined())
        << "ValueError: param " << it.first << "has already been specialized";
  }
  for (const auto& it : param_var_map) {
    body = AssertStmt(EQ(it.first, it.second), StringImm("violate specialize constraint"), body);
  }
  PrimFunc new_f = f;
  BlockRealize new_block = Downcast<BlockRealize>(f->body);
  new_block.CopyOnWrite()->block.CopyOnWrite()->body = body;
  new_f.CopyOnWrite()->body = new_block;
  return new_f;
}

PrimFunc specialize(PrimFunc func, const tir::Var& param, const ObjectRef& instance) {
  // preliminaries
  tir::ExprDeepEqual equal;
  VarMapType internal_var_map, param_var_map, all_var_map;
  auto in_params = [&](const Var& param) {
    auto it = std::find_if(func->params.begin(), func->params.end(),
                           [&](const tir::Var& var) { return var.same_as(param); });
    return it != func->params.end();
  };
  auto in_buffer_map = [&](const Buffer& buf) {
    auto it =
        std::find_if(func->buffer_map.begin(), func->buffer_map.end(),
                     [&](const std::pair<tir::Var, Buffer>& it) { return it.second.same_as(buf); });
    return it != func->buffer_map.end();
  };
  // check param is in PrimFunc's parameters
  CHECK(in_params(param)) << "ValueError: specialize expects param to be in PrimFunc's params";
  if (instance->IsInstance<BufferNode>()) {
    // specialize a param in buffer_map
    Buffer specific_buf = Downcast<Buffer>(instance);
    CHECK_GT(func->buffer_map.count(param), 0)
        << "ValueError: specialize expects param to be in PrimFunc's buffer_map";
    const Buffer& buf_to_specialize = func->buffer_map[param];
    // build var mapping using specific_buf's parameters
    auto build_var_mapping = [&](const PrimExpr& new_expr, const PrimExpr& old_expr) {
      if (!equal(new_expr, old_expr)) {
        CHECK(old_expr->IsInstance<tir::VarNode>());
        const Var& var = Downcast<tir::Var>(old_expr);
        std::unordered_map<tir::Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual>& var_map =
            in_params(var) ? param_var_map : internal_var_map;
        auto it = var_map.find(var);
        if (it != var_map.end()) {
          CHECK(equal(it->second, new_expr));
        } else {
          var_map[var] = new_expr;
          all_var_map[var] = new_expr;
        }
      }
    };
    CHECK_EQ(specific_buf->shape.size(), buf_to_specialize->shape.size());
    for (size_t i = 0; i < specific_buf->shape.size(); ++i) {
      build_var_mapping(specific_buf->shape[i], buf_to_specialize->shape[i]);
    }
    CHECK_EQ(specific_buf->strides.size(), buf_to_specialize->strides.size());
    for (size_t i = 0; i < specific_buf->strides.size(); ++i) {
      build_var_mapping(specific_buf->strides[i], buf_to_specialize->strides[i]);
    }
    build_var_mapping(specific_buf->elem_offset, buf_to_specialize->elem_offset);
  } else if (instance->IsInstance<PrimExprNode>()) {
    // specialize a param not in buffer_map
    CHECK_EQ(func->buffer_map.count(param), 0)
        << "ValueError: specialize expects param to not be in PrimFunc's buffer_map";
    PrimExpr specific_expr = Downcast<PrimExpr>(instance);
    // build var mapping using specific_expr
    CHECK(!FetchSpecializeConstraint(func, param).defined())
        << "ValueError: param already specialized";
    param_var_map[param] = specific_expr;
    all_var_map[param] = specific_expr;
  } else {
    LOG(FATAL) << "TypeError: PrimFunc.specialize expects instance to be Buffer or PrimExpr";
  }
  // buffer replacement function
  auto f_buffer = [&](const Buffer& buf) -> Buffer {
    Buffer new_buffer = buf;
    const auto& var_map = in_buffer_map(buf) ? all_var_map : internal_var_map;
    new_buffer.CopyOnWrite()->elem_offset = Substitute(new_buffer->elem_offset, var_map);
    std::vector<PrimExpr> new_shape, new_stride;
    for (const auto& dim : new_buffer->shape) new_shape.push_back(Substitute(dim, var_map));
    for (const auto& stride : new_buffer->strides) new_shape.push_back(Substitute(stride, var_map));
    new_buffer.CopyOnWrite()->shape = Array<PrimExpr>(new_shape);
    new_buffer.CopyOnWrite()->strides = Array<PrimExpr>(new_stride);
    return new_buffer;
  };
  // replace buffer
  BufferMutator buffer_mutator(f_buffer);
  PrimFunc new_func = buffer_mutator.MutatePrimFunc(func);
  // replace vars in body
  new_func.CopyOnWrite()->body = Substitute(new_func->body, internal_var_map);
  new_func = ExertSpecializeConstraint(param_var_map, new_func);
  return new_func;
}

PrimFunc remove_constant_param(PrimFunc func, const tir::Var& param) {
  // Check param is in params
  auto in_params = [&](const Var& param) {
    auto it = std::find_if(func->params.begin(), func->params.end(),
                           [&](const tir::Var& var) { return var.same_as(param); });
    return it != func->params.end();
  };
  CHECK(in_params(param));
  // Check param is constant
  // Cond 1. Buffer map has no param
  for (const auto& it : func->buffer_map) {
    CHECK(!it.first.same_as(param));
    for (const auto& expr : it.second->shape) CHECK(!StmtExprContainsVar(expr, {param}));
    for (const auto& expr : it.second->strides) CHECK(!StmtExprContainsVar(expr, {param}));
    CHECK(!StmtExprContainsVar(it.second->elem_offset, {param}));
  }
  // Cond 2. body contains no param or param is constantly specialized
  PrimExpr constraint = FetchSpecializeConstraint(func, param);
  auto is_constant = [&](PrimExpr expr) -> bool {
    arith::Analyzer analyzer;
    if (expr.defined()) {
      expr = analyzer.Simplify(expr);
    } else {
      return false;
    }
    return expr->IsInstance<IntImmNode>() || expr->IsInstance<FloatImmNode>();
  };
  bool needs_substitute = StmtExprContainsVar(func->body, param);
  CHECK(is_constant(constraint) || !needs_substitute);
  // Remove
  PrimFunc new_f = func;
  if (constraint.defined()) {
    new_f = RemoveSpecializeConstraint(new_f, param);
    new_f.CopyOnWrite()->body = Substitute(new_f->body, {{param, constraint}});
  }
  std::vector<tir::Var> new_params;
  for (const auto& var : func->params) {
    if (!var.same_as(param)) new_params.push_back(var);
  }
  new_f.CopyOnWrite()->params = new_params;
  return new_f;
}

TVM_REGISTER_GLOBAL("tir.Specialize")
    .set_body_typed<PrimFunc(PrimFunc, Var, ObjectRef)>([](PrimFunc func, Var param,
                                                           ObjectRef instance) {
      return specialize(std::move(func), std::move(param), std::move(instance));
    });

TVM_REGISTER_GLOBAL("tir.RemoveConstParam")
    .set_body_typed<PrimFunc(PrimFunc, Var)>([](PrimFunc func, tir::Var param) {
      return remove_constant_param(std::move(func), std::move(param));
    });

}  // namespace tir
}  // namespace tvm
