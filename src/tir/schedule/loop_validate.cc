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
 * \file ir_validate.cc
 */

#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/schedule.h>
#include <tvm/ir/attrs.h>
#include <tvm/tir/expr_functor.h>
#include "../../arith/pattern_match.h"

namespace tvm {
namespace tir {

class Detector : public ExprVisitor {
 public:
  using ReplaceType = std::function<PrimExpr(PrimExpr)>;

  explicit Detector(std::unordered_map<const VarNode*, PrimExpr>* loop_vars)
      : loop_vars_(loop_vars) {}

  bool IsChanged() { return changed_; }

  void ResetChange() { changed_ = false; }

  ReplaceType& ReplaceMap() { return substitute_; }

 private:
  void VisitExpr(const PrimExpr& n) override {
    if (changed_) return;

    if ((i1_p * c_p + i2_p).Match(n)) {
      // split pattern detected
      i1 = i1_p.Eval();
      i2 = i2_p.Eval();
      c = Simplify(c_p.Eval());
      const auto& it_i1 = loop_vars_->find(i1.get());
      const auto& it_i2 = loop_vars_->find(i2.get());
      if (it_i1 != loop_vars_->end() && it_i2 != loop_vars_->end() && Equal(c, it_i2->second)) {
        changed_ = true;
        k = Var(i1->name_hint + "_" + i2->name_hint + "_fuse");
        (*loop_vars_)[k.get()] = Simplify(it_i1->second * it_i2->second);
        substitute_ = split_substitute_gen();
        loop_vars_->erase(it_i1);
        loop_vars_->erase(it_i2);
      }
    } else {
      bool div = false, mod = false;
      if (floordiv(k_p, c_p).Match(n)) div = true;
      else
        mod = floormod(k_p, c_p).Match(n);
      if (div || mod) {
        // fuse pattern detected
        k = k_p.Eval();
        c = Simplify(c_p.Eval());
        const auto& it_k = loop_vars_->find(k.get());
        if (it_k != loop_vars_->end()) {
          changed_ = true;
          i1 = Var(k->name_hint + "_o");
          i2 = Var(k->name_hint + "_i");
          (*loop_vars_)[i1.get()] = Simplify(floordiv(it_k->second, c));
          (*loop_vars_)[i2.get()] = Simplify(c);
          substitute_ = fuse_substitute_gen();
          loop_vars_->erase(it_k);
        }
      }
    }

    if (!changed_) ExprVisitor::VisitExpr(n);
  }

  ReplaceType split_substitute_gen() {
    return [&](PrimExpr expr) -> PrimExpr {
      arith::PVar<Var> i1_p, i2_p;
      arith::PVar<PrimExpr> c_p;
      if ((i1_p * c_p + i2_p).Match(expr)) {
        if (i1_p.Eval().same_as(i1) && i2_p.Eval().same_as(i2) && Equal(c, c_p.Eval())) {
          return k;
        }
      }
      return expr;
    };
  }

  ReplaceType fuse_substitute_gen() {
    return [&](PrimExpr expr) -> PrimExpr {
      arith::PVar<Var> k_p;
      arith::PVar<PrimExpr> c_p;
      if (floordiv(k_p, c_p).Match(expr)) {
        if (k_p.Eval().same_as(k) && Equal(c, c_p.Eval())) {
          return i1;
        }
      } else if (floormod(k_p, c_p).Match(expr)) {
        if (k_p.Eval().same_as(k) && Equal(c, c_p.Eval())) {
          return i2;
        }
      }
      return expr;
    };
  }

  bool changed_{false};
  std::unordered_map<const VarNode*, PrimExpr>* loop_vars_;
  ReplaceType substitute_;

  arith::PVar<Var> i1_p, i2_p, k_p;
  arith::PVar<PrimExpr> c_p;
  Var i1, i2, k;
  PrimExpr c;
};

class Replacer : public ExprMutator {
 public:
  using ReplacerType = Detector::ReplaceType;
  explicit Replacer(ReplacerType& replace_map) : replace_map_(replace_map) {}

 private:
  PrimExpr VisitExpr(const PrimExpr& n) override {
    return replace_map_(ExprMutator::VisitExpr(n));
  }

  ReplacerType& replace_map_;
};

/*!
 * \brief Validate Tir, now the LoopValidate pass contains the following checks
 *        1) loop binding validation : a set of binding expressions is valid if and only if
 *          1.  vi=i, vj=j, vk=k ... (one loop_var binds exactly one block_var)
 *          2.  if f is a legal binding and g is the binding after we applying `split` on f,
 *          then g is legal
 *          3.  if f is a legal binding and g is the binding after we applying `fuse` on f,
 *          then g is legal
 *        algorithm:
 *          1. detector : pattern match binding expressions
 *              patterns : i1*c + i2, k/c, k%c
 *          2. replacer : substitute pattern detected above with new loop vars
 *        2) region cover check : Suppose B is a RAW predecessor of C, Loop k is the LCA of B and
 *          C, then B's output region covers C's input region under Loop k
 */
class LoopValidator : public StmtMutator {
 public:
  LoopValidator() = default;

  Stmt VisitStmt_(const BlockRealizeNode* op) override {
    auto n = CopyOnWrite(op);
    Stmt block = this->VisitStmt(op->block);
    n->binding_valid = CheckBinding(op->binding_values, op->predicate);
    n->block = Downcast<Block>(block);
    return Stmt(n);
  }

  Stmt VisitStmt_(const LoopNode* op) override {
    surrounding_loops_[op->loop_var.get()] = op->extent;
    Stmt res = StmtMutator::VisitStmt_(op);
    surrounding_loops_.erase(op->loop_var.get());
    return res;
  }

 private:
  // loop binding validation
  bool CheckBinding(const Array<PrimExpr>& bindings_input, const PrimExpr& predicate) {
    std::vector<PrimExpr> bindings;
    std::vector<std::pair<PrimExpr, PrimExpr>> predicates;
    std::unordered_map<const VarNode*, PrimExpr> loop_vars;
    for (const auto& binding : bindings_input)
      bindings.emplace_back(binding);
    for (const auto& loop : surrounding_loops_)
      loop_vars[loop.first] = Simplify(loop.second);
    ProcessPredicate(&predicates, predicate);

    Detector detector(&loop_vars);
    bool changed = true;
    while (changed) {
      changed = false;
      // Detect pattern
      for (auto& binding : bindings) {
        detector.ResetChange();
        detector(binding);
        changed |= detector.IsChanged();
        if (changed) break;
      }
      if (changed) {
        // Substitute pattern
        Replacer replacer(detector.ReplaceMap());
        for (auto& binding : bindings) {
          binding = replacer(binding);
        }
        for (auto it = predicates.begin(); it != predicates.end();) {
          *it = std::make_pair(replacer(it->first), it->second);
          if (it->first->IsInstance<VarNode>()) {
            loop_vars[DowncastPtr<VarNode>(it->first.get())] = Simplify(it->second);
            it = predicates.erase(it);
          } else {
            it++;
          }
        }
      }
    }
    if (!predicates.empty()) return false;
    if (!CheckOneToOneMapping(&bindings)) return false;
    return true;
  }

  static bool CheckOneToOneMapping(std::vector<PrimExpr>* bindings) {
    std::unordered_set<const VarNode*> count;
    for (const auto& binding : *bindings)
      if (!binding->IsInstance<VarNode>()) {
        return false;
      } else {
        const auto* ptr = DowncastPtr<VarNode>(binding.get());
        if (count.count(ptr)) {
          return false;
        } else {
          count.insert(ptr);
        }
      }
    return true;
  }

  static void ProcessPredicate(std::vector<std::pair<PrimExpr, PrimExpr>>* res,
                               const PrimExpr& predicate) {
    arith::PVar<PrimExpr> sub1, extent, sub2;
    PrimExpr rest = predicate;
    for (;;) {
      if (((sub1 < extent) && sub2).Match(rest)) {
        res->emplace_back(sub1.Eval(), extent.Eval());
        rest = sub2.Eval();
      } else if ((sub1 < extent).Match(rest)) {
        res->emplace_back(sub1.Eval(), extent.Eval());
        break;
      } else {
        break;
      }
    }
  }

  std::unordered_map<const VarNode*, PrimExpr> surrounding_loops_;
};

void ScheduleNode::LoopValidate(Function function) {
  LoopValidator loopValidator;
  function->body = loopValidator(function->body);
}

}  // namespace tir
}  // namespace tvm
