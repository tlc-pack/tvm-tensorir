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
 * \file tir/hybrid/auto_complete.cc
 * \brief Used by Hybrid Script parser to expand incomplete TIR input
 */

#include <tvm/tir/stmt_functor.h>
#include <tvm/runtime/registry.h>
#include <utility>

namespace tvm {
namespace tir {
namespace hybrid {

/*! \brief Generate surrounding loops automatically */
class AutoCompleter : public StmtMutator {
 public:
  bool contains_block = false;

 private:
  Stmt VisitStmt_(const BlockRealizeNode* op) override {
    contains_block = true;
    Stmt body = StmtMutator::VisitStmt_(op);
    if (!op->binding_values.empty() && !op->binding_values[0].defined()) {
      auto block_with_binding = CopyOnWrite(Downcast<BlockRealize>(body).get());
      std::vector<PrimExpr> bindings;
      for (size_t i = 0; i < op->binding_values.size(); ++i) {
        bindings.push_back(Var("i" + std::to_string(i)));
      }
      block_with_binding->binding_values = bindings;
      body = BlockRealize(block_with_binding);
      for (int i = op->binding_values.size() - 1; i >= 0; --i) {
        body = Loop(Downcast<Var>(bindings[i]), op->block->iter_vars[i]->dom->min,
                    op->block->iter_vars[i]->dom->extent, {}, body);
      }
    }
    return body;
  }
};

TVM_REGISTER_GLOBAL("hybrid.AutoComplete")
    .set_body_typed<Stmt(Stmt, Array<BufferAllocate>)>([](Stmt body,
                                                          Array<BufferAllocate> root_allocates) {
      AutoCompleter auto_completer;
      // generate surrounding loops automatically
      Stmt res = auto_completer(std::move(body));
      // generate root block automatically
      if (auto_completer.contains_block
          && (!res->IsInstance<BlockRealizeNode>() || !root_allocates.empty())) {
          res = Block({}, {}, {}, res, root_allocates, {}, "root");
          res = BlockRealize({}, Bool(true), Downcast<Block>(res), String(""));
      }
      return res;
    });

}  // namespace hybrid
}  // namespace tir
}  // namespace tvm
