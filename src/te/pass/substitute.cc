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
 *  Copyright (c) by Contributors
 * \file substitute.cc
 */

#include <tvm/te/ir.h>
#include <tvm/ir_mutator.h>

namespace tvm {
namespace te {

class IRSubstitutor : public IRMutator {
 public:
  explicit IRSubstitutor(
      const std::unordered_map<const Variable*, Expr>& vmap)
      : vmap_(vmap) {
  }

  Expr Mutate_(const Variable* op, const Expr& e) final {
    auto it = vmap_.find(op);
    if (it != vmap_.end()) {
      return it->second;
    } else {
      return e;
    }
  }

  Stmt Mutate_(const LoopNode* op, const Stmt& s) final {
    Loop loop = GetRef<Loop>(op);
    loop->min = Mutate(loop->min);
    loop->extent = Mutate(loop->min);
    loop->body = Mutate(loop->body);
    return loop;
  }

  Stmt Mutate_(const BlockNode* op, const Stmt& s) final {
    Block block = GetRef<Block>(op);
    for (size_t i = 0; i < block->values.size(); ++i) {
      Expr expr = Mutate(block->values[i]);
      block->values.Set(i, expr);
    }
    return block;
  }

 private:
  const std::unordered_map<const Variable*, Expr>& vmap_;
};

Stmt Substitute(Stmt stmt,
                const std::unordered_map<const Variable*, Expr>& value_map) {
  if (value_map.size() == 0) return stmt;
  return IRSubstitutor(value_map).Mutate(stmt);
}

Expr Substitute(Expr expr,
                const std::unordered_map<const Variable*, Expr>& value_map) {
  if (value_map.size() == 0) return expr;
  return IRSubstitutor(value_map).Mutate(expr);
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

}  // namespace te
}  // namespace tvm

