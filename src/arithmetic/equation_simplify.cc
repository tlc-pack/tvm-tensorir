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
 *  Copyright (c) 2019 by Contributors
 * \file tvm/arithmetic/equation_simplify.cc
 */

#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>

namespace tvm {
namespace arith {

using namespace ir;

bool DoNotContain(const Expr& expr, const Expr& vars) {
  std::set<const Variable*> var_set;

  // gather vars
  PostOrderVisit(vars, [&var_set](const NodeRef &node) {
    if (const Variable *op = node.as<Variable>())
      var_set.insert(op);
  });

  // check
  bool ret = true;
  PostOrderVisit(expr, [&var_set, &ret](const NodeRef &node) {
    if (const Variable *op = node.as<Variable>()) {
      if (var_set.count(op) != 0) {
        ret = false;
      }
    }
  });

  return ret;
}

class MatchingSimplifier : public IRMutator {
 public:
  MatchingSimplifier(const std::unordered_map<Var, Expr, ExprHash, ExprEqual> &var_map,
                     Analyzer* parent) : var_map_(var_map), parent_(parent) {}

  Expr Mutate(Expr expr) override {
    for (auto x : var_map_) {
      Expr diff = parent_->rewrite_simplify(parent_->canonical_simplify(expr - x.second));

      // direct replace
      if (is_zero(diff)) {
        return x.first;
      }

      Expr quet;
      Expr inv_quet;
      if (!is_zero(x.second)) {
        quet = parent_->rewrite_simplify(parent_->canonical_simplify(expr / x.second));
      }
      if (!is_zero(expr)) {
        inv_quet = parent_->rewrite_simplify(parent_->canonical_simplify(x.second / expr));
      }

      // multiplier
      if (quet.defined() && is_const(quet)) {
        return x.first * quet;
      }

      if (inv_quet.defined() && is_const(inv_quet)) {
        return x.first / inv_quet;
      }

      // sub
      if (DoNotContain(diff, x.second)) {
        return Mutate(diff + x.first);
      }
    }
    return IRMutator::Mutate(expr);
  }

 private:
  const std::unordered_map<Var, Expr, ExprHash, ExprEqual> &var_map_;
  Analyzer* parent_;
};


class EquationSimplifier::Impl {
 public:
  explicit Impl(Analyzer* parent) : parent_(parent) {}

  Expr Simplify(const Expr &expr) {
    // step 1, matching simplify
    MatchingSimplifier ms(var_map_, parent_);
    Expr new_expr = ms.Mutate(expr);

    // step 2, linear equation solver


    return new_expr;
  }

  void Update(const Var& var, const Expr& info, bool override) {
    if (!override) {
      CHECK(!var_map_.count(var));
    }
    var_map_[var] = info;
  }

 private:
  Analyzer* parent_;
  std::unordered_map<Var, Expr, ExprHash, ExprEqual> var_map_;
};

Expr EquationSimplifier::operator()(const Expr &expr) {
  return impl_->Simplify(expr);
}

void EquationSimplifier::Update(const Var& var, const Expr& new_expr, bool override) {
  impl_->Update(var, new_expr, override);
}

EquationSimplifier::EquationSimplifier(Analyzer *parent)
    : impl_(new Impl(parent)){
}

EquationSimplifier::~EquationSimplifier() {
  delete impl_;
}

}  // namespace arith
}  // namespace tvm
