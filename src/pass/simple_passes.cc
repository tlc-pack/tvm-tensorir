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
 *  Copyright (c) 2016 by Contributors
 * \file simple_passes.cc
 * \brief Implementation of simple passes
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>

namespace tvm {
namespace ir {

class IRSideEffect : public IRVisitor {
 public:
  void Visit(const NodeRef& e) final {
    if (has_side_effect_) return;
    IRVisitor::Visit(e);
  }

  void Visit_(const Call* op) final {
    if (!op->is_pure()) {
      has_side_effect_ = true; return;
    } else {
      IRVisitor::Visit_(op);
    }
  }

  bool has_side_effect_{false};
};

bool HasSideEffect(const Expr& e) {
  IRSideEffect v;
  v.Visit(e);
  return v.has_side_effect_;
}

class IRSubstitue : public IRMutator {
 public:
  explicit IRSubstitue(
      const std::unordered_map<const Variable*, Expr>& smap)
      : smap_(smap) {
  }

  Expr Mutate_(const Variable* op, const Expr& e) final {
    auto it = smap_.find(op);
    if (it != smap_.end()) {
      return it->second;
    } else {
      return e;
    }
  }

 private:
  const std::unordered_map<const Variable*, Expr>& smap_;
};

class IRTensorSubstitue : public IRMutator {
 public:
  explicit IRTensorSubstitue(
      const std::unordered_map<TensorKey, TensorKey>& vmap, bool read, bool write)
      : vmap(vmap), read(read), write(write) {
  }

  Stmt Mutate_(const Provide* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    if (!write) {
      return stmt;
    }
    op = stmt.as<Provide>();
    TensorKey key{op->func, op->value_index};
    auto it = vmap.find(key);
    if (it != vmap.end()) {
      return Provide::make(vmap.at(key).f, vmap.at(key).value_index, op->value, op->args);
    } else {
      return stmt;
    }
  }

  Expr Mutate_(const Call* op, const Expr& e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    if (!read) {
      return expr;
    }
    op = expr.as<Call>();
    TensorKey key{op->func, op->value_index};
    auto it = vmap.find(key);
    if (it != vmap.end()) {
      return Call::make(op->type, vmap.at(key).GetName(), op->args, op->call_type, vmap.at(key).f, vmap.at(key).value_index);
    } else {
      return e;
    }
  }

 private:
  const std::unordered_map<TensorKey, TensorKey>& vmap;
  bool read, write;
};

Stmt Substitute(Stmt stmt,
                const std::unordered_map<const Variable*, Expr>& value_map) {
  if (value_map.size() == 0) return stmt;
  return IRSubstitue(value_map).Mutate(stmt);
}

Expr Substitute(Expr expr,
                const std::unordered_map<const Variable*, Expr>& value_map) {
  if (value_map.size() == 0) return expr;
  return IRSubstitue(value_map).Mutate(expr);
}

Stmt Substitute(Stmt stmt,
                const std::unordered_map<TensorKey, TensorKey>& value_map, bool read, bool write) {
  if (value_map.size() == 0) return stmt;
  return IRTensorSubstitue(value_map, read, write).Mutate(stmt);
}

Stmt Substitute(Stmt stmt, const Map<Var, Expr>& value_map) {
  std::unordered_map<const Variable*, Expr> vmap;
  for (const auto& kv : value_map) {
    vmap[kv.first.get()] = kv.second;
  }
  return Substitute(stmt, vmap);
}

Expr Substitute(Expr expr, const Map<Var, Expr>& value_map) {
  std::unordered_map<const Variable*, Expr> vmap;
  for (const auto& kv : value_map) {
    vmap[kv.first.get()] = kv.second;
  }
  return Substitute(expr, vmap);
}

class VarTouchVisitor : public IRVisitor {
 public:
  void Visit(const NodeRef& e) final {
    if (use_var_) return;
    IRVisitor::Visit(e);
  }

  void Visit_(const Variable* op) final {
    Handle(op);
  }

  void Visit_(const Load* op) final {
    Handle(op->buffer_var.get());
    IRVisitor::Visit_(op);
  }

  virtual void Handle(const Variable* var) = 0;

  bool use_var_{false};
};

class ExprUseVarVisitor : public VarTouchVisitor {
 public:
  explicit ExprUseVarVisitor(const Variable* var)
      : var_(var) {}

  void Handle(const Variable* var) final {
    if (var == var_) use_var_ = true;
  }
 private:
  const Variable* var_;
};

class ExprUseVSetVisitor : public VarTouchVisitor {
 public:
  explicit ExprUseVSetVisitor(
      const std::unordered_set<const Variable*>& vset)
      : vset_(vset) {}

  void Handle(const Variable* var) final {
    if (vset_.count(var)) use_var_ = true;
  }
 private:
  const std::unordered_set<const Variable*>& vset_;
};

bool ExprUseVar(const Expr& e, const Var& v) {
  ExprUseVarVisitor visitor(v.get());
  visitor.Visit(e);
  return visitor.use_var_;
}

bool ExprUseVar(const Expr& e,
                const std::unordered_set<const Variable*>& vset) {
  ExprUseVSetVisitor visitor(vset);
  visitor.Visit(e);
  return visitor.use_var_;
}

}  // namespace ir
}  // namespace tvm
