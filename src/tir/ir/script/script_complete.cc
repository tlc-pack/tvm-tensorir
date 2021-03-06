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
 * \file tir/ir/script/script_complete.cc
 * \brief Used by TVM Script parser to expand incomplete TIR input
 */

#include <tvm/arith/int_set.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <utility>

namespace tvm {
namespace tir {

/*! \brief Generate surrounding loops automatically */
class ScriptCompleter : public StmtMutator {
 public:
  /*! \brief Whether the stmt contains at least one block. */
  bool contains_block = false;

 private:
  Stmt VisitStmt_(const BlockRealizeNode* op) override {
    contains_block = true;
    Stmt body = StmtMutator::VisitStmt_(op);
    if (!op->iter_values.empty() && !op->iter_values[0].dtype().is_int()) {
      auto block_with_binding = CopyOnWrite(Downcast<BlockRealize>(body).get());
      std::vector<PrimExpr> bindings;
      for (size_t i = 0; i < op->iter_values.size(); ++i) {
        bindings.push_back(Var("i" + std::to_string(i)));
      }
      block_with_binding->iter_values = bindings;
      body = BlockRealize(block_with_binding);
      for (int i = op->iter_values.size() - 1; i >= 0; --i) {
        body = For(Downcast<Var>(bindings[i]), op->block->iter_vars[i]->dom->min,
                   op->block->iter_vars[i]->dom->extent, {}, body);
      }
    }
    return body;
  }

  Stmt VisitStmt_(const BlockNode* op) override {
    Block block = Downcast<Block>(StmtMutator::VisitStmt_(op));
    if (block->reads.empty() || block->writes.empty()) {
      auto access_region = GetBlockAccessRegion(block);
      auto n = CopyOnWrite(block.operator->());
      if (!n->reads.defined()) n->reads = access_region[0];
      if (!n->writes.defined()) n->writes = access_region[1];
      return Block(n);
    } else {
      return std::move(block);
    }
  }
};

Stmt ScriptComplete(const Stmt& body, const Array<Buffer>& root_allocates) {
  ScriptCompleter script_completer;
  // generate surrounding loops automatically
  Stmt res = script_completer(body);
  // generate root block automatically
  if (script_completer.contains_block &&
      (!res->IsInstance<BlockRealizeNode>() || !root_allocates.empty())) {
    res = Block({}, {}, {}, "root", res, NullOpt, root_allocates);
    res = BlockRealize({}, Bool(true), Downcast<Block>(res));
  }
  return res;
}

TVM_REGISTER_GLOBAL("script.Complete").set_body_typed(ScriptComplete);

}  // namespace tir
}  // namespace tvm
