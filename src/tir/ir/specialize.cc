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
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <utility>

#include "../schedule/schedule_common.h"
#include "functor_common.h"

namespace tvm {
namespace tir {

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
    auto fmutate_range = [&](const Range& range) {
      PrimExpr min = this->VisitExpr(range->min);
      PrimExpr extent = this->VisitExpr(range->extent);
      if (min.same_as(range->min) && extent.same_as(range->extent)) {
        return range;
      } else {
        auto n = CopyOnWrite(range.get());
        n->min = std::move(min);
        n->extent = std::move(extent);
        return Range(n);
      }
    };
    auto fmutate_buffer_allocate = [&](const BufferAllocate& buffer_allocate) {
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
    auto fmutate_tensor_region = [&](const TensorRegion& tensor_region) {
      auto it = buffer_map_.find(tensor_region->buffer);
      Array<Range> region = MutateArray(tensor_region->region, fmutate_range);
      if (it == buffer_map_.end() && region.same_as(tensor_region->region)) {
        return tensor_region;
      } else {
        auto n = CopyOnWrite(tensor_region.get());
        n->buffer = it->second;
        n->region = std::move(region);
        return TensorRegion(n);
      }
    };
    auto fmutate_iter_var = [&](const IterVar& iter_var) {
      Range range = fmutate_range(iter_var->dom);
      if (range.same_as(iter_var->dom)) {
        return iter_var;
      } else {
        auto n = CopyOnWrite(iter_var.get());
        n->dom = std::move(range);
        return IterVar(n);
      }
    };
    auto fmutate_annotation = [this](const Annotation& annotation) {
      PrimExpr value = this->VisitExpr(annotation->value);
      if (value.same_as(annotation->value)) {
        return annotation;
      } else {
        return Annotation(annotation->attr_key, annotation->value);
      }
    };
    Array<BufferAllocate> allocations = MutateArray(op->allocations, fmutate_buffer_allocate);
    Array<TensorRegion> reads = MutateArray(op->reads, fmutate_tensor_region);
    Array<TensorRegion> writes = MutateArray(op->writes, fmutate_tensor_region);
    Array<IterVar> block_vars = MutateArray(op->iter_vars, fmutate_iter_var);
    Array<Annotation> annotations = MutateArray(op->annotations, fmutate_annotation);
    Stmt body = VisitStmt(op->body);
    if (allocations.same_as(op->allocations) && reads.same_as(op->reads) &&
        writes.same_as(op->writes) && block_vars.same_as(op->iter_vars) && body.same_as(op->body) &&
        annotations.same_as(op->annotations)) {
      return GetRef<Block>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->allocations = std::move(allocations);
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      n->iter_vars = std::move(block_vars);
      n->annotations = std::move(annotations);
      n->body = std::move(body);
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) override {
    auto fmutate = [this](const PrimExpr& e) { return this->VisitExpr(e); };
    auto it = buffer_map_.find(op->buffer);
    PrimExpr value = VisitExpr(op->value);
    Array<PrimExpr> indices = MutateArray(op->indices, fmutate);
    if (it == buffer_map_.end() && value.same_as(op->value) && indices.same_as(op->indices)) {
      return GetRef<BufferStore>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->buffer = it->second;
      n->value = std::move(value);
      n->indices = std::move(indices);
      return Stmt(n);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) override {
    auto fmutate = [this](const PrimExpr& e) { return this->VisitExpr(e); };
    auto it = buffer_map_.find(op->buffer);
    Array<PrimExpr> indices = MutateArray(op->indices, fmutate);
    if (it == buffer_map_.end() && indices.same_as(op->indices)) {
      return GetRef<BufferLoad>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->buffer = it->second;
      n->indices = std::move(indices);
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

bool InParams(const PrimFunc& func, const Var& param) {
  auto it = std::find_if(func->params.begin(), func->params.end(),
                         [&](const tir::Var& var) { return var.same_as(param); });
  return it != func->params.end();
}

bool InBufferMap(const PrimFunc& func, const Buffer& buf) {
  auto it =
      std::find_if(func->buffer_map.begin(), func->buffer_map.end(),
                   [&](const std::pair<tir::Var, Buffer>& it) { return it.second.same_as(buf); });
  return it != func->buffer_map.end();
}

PrimFunc GenerateNewFunc(PrimFunc func, const VarMapType& internal_var_map,
                         const VarMapType& param_var_map, const VarMapType& all_var_map) {
  auto f_buffer = [&](const Buffer& buf) -> Buffer {
    Buffer new_buffer = buf;
    const auto& var_map = InBufferMap(func, buf) ? all_var_map : internal_var_map;
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
  auto &attr_dict = new_func.CopyOnWrite()->attrs.CopyOnWrite()->dict;
  for (auto& kv : attr_dict) {
    if (kv.second.as<PrimExprNode>()) {
      attr_dict.Set(kv.first, Substitute(Downcast<PrimExpr>(kv.second), internal_var_map));
    }
  }
  new_func = ExertSpecializeConstraint(param_var_map, new_func);
  GlobalVar main("main");
  IRModule mod({{main, new_func}});
  new_func = Downcast<PrimFunc>(transform::Simplify()(mod)->Lookup(main));

  return new_func;
}

PrimFunc Specialize(PrimFunc func, const tir::Var& param, const Buffer& specific_buf) {
  tir::ExprDeepEqual equal;
  VarMapType internal_var_map, param_var_map, all_var_map;
  CHECK_GT(func->buffer_map.count(param), 0)
      << "ValueError: specialize expects param to be in PrimFunc's buffer_map";
  const Buffer& buf_to_specialize = func->buffer_map[param];
  // build var mapping using specific_buf's parameters
  auto build_var_mapping = [&](const PrimExpr& new_expr, const PrimExpr& old_expr) {
    if (!equal(new_expr, old_expr)) {
      CHECK(old_expr->IsInstance<tir::VarNode>());
      const Var& var = Downcast<tir::Var>(old_expr);
      std::unordered_map<tir::Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual>& var_map =
          InParams(func, var) ? param_var_map : internal_var_map;
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
  // Generate new function
  return GenerateNewFunc(func, internal_var_map, param_var_map, all_var_map);
}

PrimFunc Specialize(PrimFunc func, const tir::Var& param, const PrimExpr& specific_expr) {
  // preliminaries
  VarMapType internal_var_map, param_var_map, all_var_map;
  // check param is in PrimFunc's parameters
  CHECK(InParams(func, param)) << "ValueError: specialize expects param to be in PrimFunc's params";
  // specialize a param not in buffer_map
  CHECK_EQ(func->buffer_map.count(param), 0)
      << "ValueError: specialize expects param to not be in PrimFunc's buffer_map";
  // build var mapping using specific_expr
  CHECK(!FetchSpecializeConstraint(func, param).defined())
      << "ValueError: param already specialized";
  param_var_map[param] = specific_expr;
  all_var_map[param] = specific_expr;
  // Generate new function
  return GenerateNewFunc(func, internal_var_map, param_var_map, all_var_map);
}

PrimFunc RemoveConstantParam(PrimFunc func, const tir::Var& param) {
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
    new_f = GenerateNewFunc(new_f, {{param, constraint}}, {}, {{param, constraint}});
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
      if (instance->IsInstance<BufferNode>()) {
        return Specialize(std::move(func), std::move(param), Downcast<Buffer>(instance));
      } else if (instance->IsInstance<PrimExprNode>()) {
        return Specialize(std::move(func), std::move(param), Downcast<PrimExpr>(instance));
      } else {
        LOG(FATAL) << "TypeError: specialize expects instance to be Buffer or PrimExpr";
        return NullValue<PrimFunc>();
      }
    });

TVM_REGISTER_GLOBAL("tir.RemoveConstParam")
    .set_body_typed<PrimFunc(PrimFunc, Var)>([](PrimFunc func, tir::Var param) {
      return RemoveConstantParam(std::move(func), std::move(param));
    });

}  // namespace tir
}  // namespace tvm
