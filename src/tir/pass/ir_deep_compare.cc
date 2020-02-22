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
 * \file ir_deep_compare.cc
 */
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace tir {

using ExprComparator = ExprFunctor<void(const PrimExpr& n, const PrimExpr &other)>;
using StmtComparator = StmtFunctor<void(const Stmt& n, const Stmt &other)>;

#define DEFINE_BIOP_EXPR_CMP_(OP)                                 \
  void VisitExpr_(const OP* op, const PrimExpr& other) final {    \
    const OP* rhs = other.as<OP>();                               \
    if (CompareExpr(op->a, rhs->a) != 0) return;                  \
    if (CompareExpr(op->b, rhs->b) != 0) return;                  \
  }

// Deep comparison to check if two IR graph are equivalent
class IRDeepCompare :
      public ExprComparator, public StmtComparator {
 public:
  explicit IRDeepCompare(bool map_free_var, bool assert_mode)
      : map_free_var_(map_free_var), assert_mode_(assert_mode) {}

  // Equality comparison
  bool Equal(const Stmt& lhs, const Stmt& rhs) {
    tie_def_ = true;
    VisitStmt(lhs, rhs);
    return order_ == 0;
  }

  bool Equal(const PrimExpr& lhs, const PrimExpr& rhs) {
    tie_def_ = true;
    VisitExpr(lhs, rhs);
    return order_ == 0;
  }

  int Compare(const PrimExpr& lhs, const PrimExpr& rhs) {
    tie_def_ = false;
    VisitExpr(lhs, rhs);
    return order_;
  }

  void Bind(const VarNode* lhs, const VarNode* rhs) {
    vmap_[lhs] = rhs;
  }

  void VisitExpr(const PrimExpr& n, const PrimExpr& other) override {
    if (order_ != 0) return;
    if (n.same_as(other)) {
      if (const auto* v = n.as<VarNode>()) visited_vars.insert(v);
      return;
    }
    if (CompareValue(n->type_index(), other->type_index()) != 0) return;
    if (CompareType(n.dtype(), other.dtype()) != 0) return;
    ExprComparator::VisitExpr(n, other);
    if (assert_mode_) {
      CHECK_EQ(order_, 0) << n << " is not equal to: " << other;
    }
  }

  void VisitStmt(const Stmt& n, const Stmt& other) override {
    if (order_ != 0) return;
    if (n.same_as(other)) return;
    if (CompareValue(n->type_index(), other->type_index()) != 0) return;
    StmtComparator::VisitStmt(n, other);
    if (assert_mode_) {
      CHECK_EQ(order_, 0) << "\n" << n << "\nis not equal to:\n" << other;
    }
  }
  // Stmt
  void VisitStmt_(const LetStmtNode* op, const Stmt& other) final {
    const LetStmtNode* rhs = other.as<LetStmtNode>();
    if (CompareExpr(op->value, rhs->value) != 0) return;
    if (tie_def_) {
      vmap_[op->var.get()] = rhs->var.get();
    } else {
      if (CompareExpr(op->var, rhs->var) != 0) return;
    }
    if (CompareStmt(op->body, rhs->body) != 0) return;
  }

  void VisitStmt_(const AttrStmtNode* op, const Stmt& other) final {
    const AttrStmtNode* rhs = other.as<AttrStmtNode>();
    if (CompareString(op->attr_key, rhs->attr_key) != 0) return;
    if (CompareStmt(op->body, rhs->body) != 0) return;
    if (CompareNodeRef(op->node, rhs->node) != 0) return;
    if (CompareExpr(op->value, rhs->value) != 0) return;
  }

  void VisitStmt_(const IfThenElseNode* op, const Stmt& other) final {
    const IfThenElseNode* rhs = other.as<IfThenElseNode>();
    if (CompareExpr(op->condition, rhs->condition) != 0) return;
    if (CompareStmt(op->then_case, rhs->then_case) != 0) return;
    if (CompareStmt(op->else_case, rhs->else_case) != 0) return;
  }

  void VisitStmt_(const ForNode* op, const Stmt& other) final {
    const ForNode* rhs = other.as<ForNode>();
    if (CompareExpr(op->min, rhs->min) != 0) return;
    if (CompareExpr(op->extent, rhs->extent) != 0) return;
    if (tie_def_) {
      vmap_[op->loop_var.get()] = rhs->loop_var.get();
    } else {
      if (CompareExpr(op->loop_var, rhs->loop_var) != 0) return;
    }
    if (CompareStmt(op->body, rhs->body) != 0) return;
  }

  void VisitStmt_(const AllocateNode* op, const Stmt& other) final {
    const AllocateNode* rhs = other.as<AllocateNode>();
    if (tie_def_) {
      vmap_[op->buffer_var.get()] = rhs->buffer_var.get();
    } else {
      if (CompareExpr(op->buffer_var, rhs->buffer_var) != 0) return;
    }
    if (CompareType(op->dtype, rhs->dtype) != 0) return;
    if (CompareExprArray(op->extents, rhs->extents) != 0) return;
    if (CompareExpr(op->condition, rhs->condition) != 0) return;
    if (CompareStmt(op->body, rhs->body) != 0) return;
    if (CompareExpr(op->new_expr, rhs->new_expr) != 0) return;
    if (CompareString(op->free_function, rhs->free_function) != 0) return;
  }

  void VisitStmt_(const StoreNode* op, const Stmt& other) final {
    const StoreNode* rhs = other.as<StoreNode>();
    if (CompareExpr(op->buffer_var, rhs->buffer_var) != 0) return;
    if (CompareExpr(op->value, rhs->value) != 0) return;
    if (CompareExpr(op->index, rhs->index) != 0) return;
    if (CompareExpr(op->predicate, rhs->predicate) != 0) return;
  }

  void VisitStmt_(const FreeNode* op, const Stmt& other) final {
    const FreeNode* rhs = other.as<FreeNode>();
    if (CompareExpr(op->buffer_var, rhs->buffer_var) != 0) return;
  }

  void VisitStmt_(const AssertStmtNode* op, const Stmt& other) final {
    const AssertStmtNode* rhs = other.as<AssertStmtNode>();
    if (CompareExpr(op->condition, rhs->condition) != 0) return;
    if (CompareExpr(op->message, rhs->message) != 0) return;
    if (CompareStmt(op->body, rhs->body) != 0) return;
  }

  void VisitStmt_(const ProducerConsumerNode* op, const Stmt& other) final {
    const ProducerConsumerNode* rhs = other.as<ProducerConsumerNode>();
    if (CompareNodeRef(op->func, rhs->func) != 0) return;
    if (CompareValue(op->is_producer, rhs->is_producer) != 0) return;
    if (CompareStmt(op->body, rhs->body) != 0) return;
  }

  void VisitStmt_(const ProvideNode* op, const Stmt& other) final {
    const ProvideNode* rhs = other.as<ProvideNode>();
    if (CompareNodeRef(op->func, rhs->func) != 0) return;
    if (CompareValue(op->value_index, rhs->value_index) != 0) return;
    if (CompareExpr(op->value, rhs->value) != 0) return;
    if (CompareExprArray(op->args, rhs->args) != 0) return;
  }

  void VisitStmt_(const RealizeNode* op, const Stmt& other) final {
    const RealizeNode* rhs = other.as<RealizeNode>();
    if (CompareNodeRef(op->func, rhs->func) != 0) return;
    if (CompareValue(op->value_index, rhs->value_index) != 0) return;
    if (CompareType(op->dtype, rhs->dtype) != 0) return;
    if (CompareRegion(op->bounds, rhs->bounds) != 0) return;
    if (CompareStmt(op->body, rhs->body) != 0) return;
  }

  void VisitStmt_(const PrefetchNode* op, const Stmt& other) final {
    const PrefetchNode* rhs = other.as<PrefetchNode>();
    if (CompareNodeRef(op->func, rhs->func) != 0) return;
    if (CompareValue(op->value_index, rhs->value_index) != 0) return;
    if (CompareType(op->dtype, rhs->dtype) != 0) return;
    if (CompareRegion(op->bounds, rhs->bounds) != 0) return;
  }

  void VisitStmt_(const SeqStmtNode* op, const Stmt& other) final {
    const SeqStmtNode* rhs = other.as<SeqStmtNode>();
    if (CompareValue(op->size(), rhs->size()) != 0) return;
    for (size_t i = 0; i < op->size(); ++i) {
      if (CompareStmt(op->seq[i], rhs->seq[i]) != 0) return;
    }
  }

  void VisitStmt_(const EvaluateNode* op, const Stmt& other) final {
    const EvaluateNode* rhs = other.as<EvaluateNode>();
    CompareExpr(op->value, rhs->value);
  }

  void VisitStmt_(const BlockNode* op, const Stmt& other) final {
    const auto* rhs = other.as<BlockNode>();
    if (tie_def_) {
      if (CompareValue(op->iter_vars.size(), rhs->iter_vars.size()) != 0) return;
      for (size_t i = 0; i < op->iter_vars.size(); ++i) {
        vmap_[op->iter_vars[i]->var.get()] = rhs->iter_vars[i]->var.get();
      }
    } else {
      if (CompareArray(op->iter_vars, op->iter_vars,
                       [this](const IterVar& a, const IterVar& b) {
                         return CompareExpr(a->var, b->var) && CompareRange(a->dom, b->dom);
                       }) != 0) return;
    }
    if (CompareExprArray(op->values, rhs->values) != 0) return;
    if (CompareArray(op->allocations, rhs->allocations,
                     [this](const Stmt& a, const Stmt& b) {
                       return CompareStmt(a, b);
                     }) != 0) return;
    if (CompareArray(op->reads, rhs->reads,
                     [this](const te::TensorRegion& a, const te::TensorRegion& b) {
                       return CompareTensorRegion(a, b);
                     }) != 0) return;
    if (CompareArray(op->writes, rhs->writes,
                     [this](const te::TensorRegion& a, const te::TensorRegion& b) {
                       return CompareTensorRegion(a, b);
                     }) != 0) return;

    if (CompareExpr(op->predicate, rhs->predicate) != 0) return;
    if (CompareArray(op->annotations, rhs->annotations,
                     [this](const Annotation& a, const Annotation& b) {
                       return CompareAnnotation(a, b);
                     }) != 0) return;
    if (CompareStmt(op->body, rhs->body) != 0) return;
    if (CompareString(op->tag, rhs->tag) != 0) return;
  }

  void VisitStmt_(const BufferStoreNode* op, const Stmt& other) final {
    const auto* rhs = other.as<BufferStoreNode>();
    if (CompareExpr(op->buffer->data, rhs->buffer->data) != 0) return;
    if (CompareExprArray(op->indices, rhs->indices) != 0) return;
    if (CompareExpr(op->value, rhs->value) != 0) return;
  }

  void VisitStmt_(const BufferAllocateNode* op, const Stmt& other) final {
    const auto* rhs = other.as<BufferAllocateNode>();
    if (tie_def_) {
      vmap_[op->buffer->data.get()] = rhs->buffer->data.get();
    } else {
      if (CompareExpr(op->buffer->data, rhs->buffer->data) != 0) return;
    }
    if (CompareString(op->scope, rhs->scope) != 0) return;
  }

  void VisitStmt_(const LoopNode* op, const Stmt& other) final {
    const auto* rhs = other.as<LoopNode>();
    if (CompareExpr(op->min, rhs->min) != 0) return;
    if (CompareExpr(op->extent, rhs->extent) != 0) return;
    if (tie_def_) {
      vmap_[op->loop_var.get()] = rhs->loop_var.get();
    } else {
      if (CompareExpr(op->loop_var, rhs->loop_var) != 0) return;
    }
    if (CompareStmt(op->body, rhs->body) != 0) return;
  }

  // Exprs
  void VisitExpr_(const VarNode* op, const PrimExpr& other) final {
    const VarNode* rhs = other.as<VarNode>();
    if (map_free_var_ && !visited_vars.count(op) && !visited_vars.count(rhs)) {
      vmap_[op] = rhs;
    }
    visited_vars.insert(op);
    visited_vars.insert(rhs);
    auto it = vmap_.find(op);
    if (it != vmap_.end()) op = it->second;
    if (op < rhs) {
      order_ = -1;
    } else if (op > rhs) {
      order_ = +1;
    }
  }
  void VisitExpr_(const LoadNode* op, const PrimExpr& other) final {
    const LoadNode* rhs = other.as<LoadNode>();
    if (CompareExpr(op->buffer_var, rhs->buffer_var) != 0) return;
    if (CompareExpr(op->index, rhs->index) != 0) return;
    if (CompareExpr(op->predicate, rhs->predicate) != 0) return;
  }

  void VisitExpr_(const LetNode* op, const PrimExpr& other) final {
    const LetNode* rhs = other.as<LetNode>();
    if (tie_def_) {
      vmap_[op->var.get()] = rhs->var.get();
    } else {
      if (CompareExpr(op->var, rhs->var) != 0) return;
    }
    if (CompareExpr(op->value, rhs->value) != 0) return;
    if (CompareExpr(op->body, rhs->body) != 0) return;
  }

  void VisitExpr_(const CallNode* op, const PrimExpr& other) final {
    const CallNode* rhs = other.as<CallNode>();
    if (CompareString(op->name, rhs->name)) return;
    if (CompareExprArray(op->args, rhs->args)) return;
    if (CompareValue(op->call_type, rhs->call_type) != 0) return;
    if (CompareNodeRef(op->func, rhs->func) != 0) return;
    if (CompareValue(op->value_index, rhs->value_index) != 0) return;
  }

  void VisitExpr_(const ReduceNode *op, const PrimExpr& other) final {
    const ReduceNode* rhs = other.as<ReduceNode>();
    if (CompareCommReducer(op->combiner, rhs->combiner) != 0) return;
    if (CompareValue(op->axis.size(), rhs->axis.size()) != 0) return;
    if (CompareValue(op->value_index, rhs->value_index) != 0) return;
    for (size_t i = 0; i < op->axis.size(); ++i) {
      if (CompareExpr(op->axis[i]->dom->min, rhs->axis[i]->dom->min) != 0) return;
      if (CompareExpr(op->axis[i]->dom->extent, rhs->axis[i]->dom->extent) != 0) return;
      if (tie_def_) {
        vmap_[op->axis[i]->var.get()] = rhs->axis[i]->var.get();
      } else {
        if (CompareExpr(op->axis[i]->var, rhs->axis[i]->var) != 0) return;
      }
    }
    if (CompareExpr(op->condition, rhs->condition) != 0) return;
    if (CompareExprArray(op->source, rhs->source) != 0) return;
  }

  void VisitExpr_(const IntImmNode *op, const PrimExpr& other) final {
    CompareValue(op->value, other.as<IntImmNode>()->value);
  }

  void VisitExpr_(const FloatImmNode *op, const PrimExpr& other) final {
    CompareValue(op->value, other.as<FloatImmNode>()->value);
  }

  void VisitExpr_(const StringImmNode *op, const PrimExpr& other) final {
    CompareString(op->value, other.as<StringImmNode>()->value);
  }

  void VisitExpr_(const CastNode *op, const PrimExpr& other) final {
    CompareExpr(op->value, other.as<CastNode>()->value);
  }

  void VisitExpr_(const NotNode *op, const PrimExpr& other) final {
    CompareExpr(op->a, other.as<NotNode>()->a);
  }

  void VisitExpr_(const SelectNode *op, const PrimExpr& other) final {
    const SelectNode* rhs = other.as<SelectNode>();
    if (CompareExpr(op->condition, rhs->condition) != 0) return;
    if (CompareExpr(op->true_value, rhs->true_value) != 0) return;
    if (CompareExpr(op->false_value, rhs->false_value) != 0) return;
  }

  void VisitExpr_(const RampNode *op, const PrimExpr& other) final {
    const RampNode* rhs = other.as<RampNode>();
    if (CompareExpr(op->base, rhs->base) != 0) return;
    if (CompareExpr(op->stride, rhs->stride) != 0) return;
    if (CompareValue(op->lanes, rhs->lanes) != 0) return;
  }

  void VisitExpr_(const BroadcastNode *op, const PrimExpr& other) final {
    const BroadcastNode* rhs = other.as<BroadcastNode>();
    if (CompareExpr(op->value, rhs->value) != 0) return;
    if (CompareValue(op->lanes, rhs->lanes) != 0) return;
  }

  void VisitExpr_(const ShuffleNode *op, const PrimExpr& other) final {
    const ShuffleNode* rhs = other.as<ShuffleNode>();
    if (CompareExprArray(op->vectors, rhs->vectors) != 0) return;
    if (CompareExprArray(op->indices, rhs->indices) != 0) return;
  }

  void VisitExpr_(const BufferLoadNode* op, const PrimExpr& other) final {
    const auto* rhs = other.as<BufferLoadNode>();
    if (CompareExpr(op->buffer->data, rhs->buffer->data) != 0) return;
    if (CompareExprArray(op->indices, rhs->indices) != 0) return;
    if (CompareType(op->dtype, rhs->dtype) != 0) return;
  }

  DEFINE_BIOP_EXPR_CMP_(AddNode)
  DEFINE_BIOP_EXPR_CMP_(SubNode)
  DEFINE_BIOP_EXPR_CMP_(MulNode)
  DEFINE_BIOP_EXPR_CMP_(DivNode)
  DEFINE_BIOP_EXPR_CMP_(ModNode)
  DEFINE_BIOP_EXPR_CMP_(FloorDivNode)
  DEFINE_BIOP_EXPR_CMP_(FloorModNode)
  DEFINE_BIOP_EXPR_CMP_(MinNode)
  DEFINE_BIOP_EXPR_CMP_(MaxNode)
  DEFINE_BIOP_EXPR_CMP_(EQNode)
  DEFINE_BIOP_EXPR_CMP_(NENode)
  DEFINE_BIOP_EXPR_CMP_(LTNode)
  DEFINE_BIOP_EXPR_CMP_(LENode)
  DEFINE_BIOP_EXPR_CMP_(GTNode)
  DEFINE_BIOP_EXPR_CMP_(GENode)
  DEFINE_BIOP_EXPR_CMP_(AndNode)
  DEFINE_BIOP_EXPR_CMP_(OrNode)

 private:
  int CompareExpr(const PrimExpr& lhs, const PrimExpr& rhs) {
    if (order_ != 0) return order_;
    if (!lhs.defined() && rhs.defined()) {
      order_ = -1; return order_;
    }
    if (!rhs.defined() && lhs.defined()) {
      order_ = +1; return order_;
    }
    VisitExpr(lhs, rhs);
    return order_;
  }

  int CompareStmt(const Stmt& lhs, const Stmt& rhs) {
    if (order_ != 0) return order_;
    if (!lhs.defined() && rhs.defined()) {
      order_ = -1; return order_;
    }
    if (!rhs.defined() && lhs.defined()) {
      order_ = +1; return order_;
    }
    VisitStmt(lhs, rhs);
    return order_;
  }

  template <typename T, typename F>
  int CompareArray(const Array<T>& lhs, const Array<T>& rhs, F comp) {
    if (order_ != 0) return order_;
    if (CompareValue(lhs.size(), rhs.size()) != 0) return order_;
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (comp(lhs[i], rhs[i]) != 0) return order_;
    }
    return order_;
  }

  int CompareExprArray(const Array<PrimExpr>& lhs, const Array<PrimExpr>& rhs) {
    return CompareArray(lhs, rhs, [this](const PrimExpr& a, const PrimExpr& b) {
      return CompareExpr(a, b);
    });
  }

  int CompareRange(const Range& lhs, const Range& rhs) {
    if (order_ != 0) return order_;
    if (CompareExpr(lhs->min, rhs->min) != 0) return order_;
    if (CompareExpr(lhs->extent, rhs->extent) != 0) return order_;
    return order_;
  }

  int CompareRegion(const Region& lhs, const Region& rhs) {
    if (order_ != 0) return order_;
    if (CompareValue(lhs.size(), rhs.size()) != 0) return order_;
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (CompareRange(lhs[i], rhs[i]) != 0) return order_;
    }
    return order_;
  }

  int CompareNodeRef(const ObjectRef& lhs, const ObjectRef& rhs) {
    if (order_ != 0) return order_;
    if (lhs.as<PrimExprNode>() && rhs.as<PrimExprNode>()) {
      return CompareExpr(Downcast<PrimExpr>(lhs), Downcast<PrimExpr>(rhs));
    }
    if (lhs.get() < rhs.get()) {
      order_ = -1; return order_;
    }
    if (lhs.get() > rhs.get()) {
      order_ = +1; return order_;
    }
    return order_;
  }

  int CompareType(const DataType& lhs, const DataType& rhs) {
    if (order_ != 0) return order_;
    if (lhs == rhs) return order_;
    if (CompareValue(lhs.code(), rhs.code()) != 0) return order_;
    if (CompareValue(lhs.bits(), rhs.bits()) != 0) return order_;
    if (CompareValue(lhs.lanes(), rhs.lanes()) != 0) return order_;
    return order_;
  }

  int CompareString(const std::string& lhs, const std::string& rhs) {
    if (order_ != 0) return order_;
    order_ = lhs.compare(rhs);
    return order_;
  }

  template<typename T>
  int CompareValue(const T& lhs, const T& rhs) {
    if (order_ != 0) return order_;
    if (lhs < rhs) {
      order_ = -1; return order_;
    } else if (lhs > rhs) {
      order_ = +1; return order_;
    }
    return order_;
  }

  int CompareCommReducer(const CommReducer& lhs, const CommReducer& rhs) {
    if (order_ != 0) return order_;
    if (lhs == rhs) return order_;
    if (CompareValue(lhs->lhs.size(), rhs->lhs.size()) != 0) return order_;
    if (CompareValue(lhs->rhs.size(), rhs->rhs.size()) != 0) return order_;
    IRDeepCompare cmp(map_free_var_, assert_mode_);
    if (tie_def_) {
      for (size_t i = 0; i < lhs->lhs.size(); ++i) {
        cmp.vmap_[lhs->lhs[i].get()] = rhs->lhs[i].get();
      }
      for (size_t i = 0; i < lhs->rhs.size(); ++i) {
        cmp.vmap_[lhs->rhs[i].get()] = rhs->rhs[i].get();
      }
    } else {
      for (size_t i = 0; i < lhs->lhs.size(); ++i) {
        if (CompareExpr(lhs->lhs[i], rhs->lhs[i]) != 0) return order_;
      }
      for (size_t i = 0; i < lhs->lhs.size(); ++i) {
        if (CompareExpr(lhs->rhs[i], rhs->rhs[i]) != 0) return order_;
      }
    }
    order_ = cmp.CompareExprArray(lhs->result, rhs->result);
    return order_;
  }

  int CompareTensorRegion(const te::TensorRegion& lhs, const te::TensorRegion& rhs) {
    if (order_ != 0) return order_;
    if (CompareExpr(lhs->buffer->data, rhs->buffer->data) != 0) return order_;
    if (CompareRegion(lhs->region, rhs->region) != 0) return order_;
    return order_;
  }

  int CompareAnnotation(const Annotation& lhs, const Annotation& rhs) {
    if (order_ != 0) return order_;
    if (CompareString(lhs->attr_key, rhs->attr_key) != 0) return order_;
    if (CompareExpr(lhs->value, rhs->value) != 0) return order_;
    return order_;
  }
  // The order flag, smaller, -1, bigger: +1, equal: 0
  int order_{0};
  // Whether tie intermediate definitions.
  // This allows use to tie definitions of two variables together.
  // This enables us to assert equal between (let x in x + 1),  (let y in y + 1)
  // However, the comparison is no longer in total order.
  // Only equality/non-equality information is valid.
  bool tie_def_{false};
  // whether to map open terms.
  bool map_free_var_;
  // if in assert mode, must return true, and will throw error otherwise.
  bool assert_mode_;
  // variable remap if any
  std::unordered_map<const VarNode*, const VarNode*> vmap_;
  // all vars which have been visited before
  std::unordered_set<const VarNode*> visited_vars;
};

bool Equal(const Function& lhs,
           const Function& rhs,
           bool remap_free_var,
           bool assert_mode) {
  IRDeepCompare ir_deep_compare(remap_free_var, assert_mode);
  if (lhs->params.size() != rhs->params.size()) return false;
  for (size_t i = 0; i < lhs->params.size(); ++i) {
    const auto* lhs_var = lhs->buffer_map[lhs->params[i]]->data.get();
    const auto* rhs_var = rhs->buffer_map[rhs->params[i]]->data.get();
    ir_deep_compare.Bind(lhs_var, rhs_var);
  }
  return ir_deep_compare.Equal(lhs->body, rhs->body);
}

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

#define REGISTER_MODULE_EQUAL_PASS(PassName, remap_free_var, assert_mode)                    \
  TVM_REGISTER_GLOBAL("ir_pass.Module"#PassName)                                             \
  .set_body_typed([](const IRModule& lhs, const IRModule& rhs) {                             \
        return Equal(lhs, rhs, remap_free_var, assert_mode);                                 \
  });

// Basic equal pass for module
REGISTER_MODULE_EQUAL_PASS(Equal, false, false);
// Basic equal pass with assert mode for module
REGISTER_MODULE_EQUAL_PASS(AssertEqual, false, true);
// Struct equal pass, which can remap free vars for module
REGISTER_MODULE_EQUAL_PASS(StructEqual, true, false);
// Struct equal pass with assert mode for module
REGISTER_MODULE_EQUAL_PASS(AssertStructEqual, true, true);

bool Equal(const Stmt& lhs, const Stmt& rhs, bool remap_free_var, bool assert_mode) {
  return IRDeepCompare(remap_free_var, assert_mode).Equal(lhs, rhs);
}

bool Equal(const PrimExpr& lhs, const PrimExpr& rhs, bool remap_free_var, bool assert_mode) {
  // quick pass for constant expressions.
  if (const int64_t *a = as_const_int(lhs)) {
    if (const int64_t *b = as_const_int(rhs)) {
      return a[0] == b[0];
    }
  }
  if (!lhs.defined()) {
    if (rhs.defined()) return false;
    if (!rhs.defined()) return true;
  } else {
    if (!rhs.defined()) return false;
  }
  // deep comparison.
  return IRDeepCompare(remap_free_var, assert_mode).Equal(lhs, rhs);
}

int Compare(const PrimExpr& lhs, const PrimExpr& rhs) {
  return IRDeepCompare(false, false).Compare(lhs, rhs);
}

}  // namespace tir
}  // namespace tvm
