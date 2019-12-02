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

#include <tvm/te/schedule.h>
#include <tvm/ir_mutator.h>
#include "../util.h"
#include "tvm/ir_pass.h"

namespace tvm {
namespace te {

class StatementInliner : public ScheduleMutator {
 public:
  explicit StatementInliner(const BufferStoreNode* op) : op_(op) {
    for (const auto& index : op->indices) {
      const auto* variable = index.as<Variable>();
      CHECK(variable) << "Only support inline direct access block";
      Var var = GetRef<Var>(variable);
      vars_.push_back(var);
    }

    Array<Var> value_vars = GatherVars(value_);
    for (const auto& x : value_vars) {
      CHECK(std::find_if(vars_.begin(), vars_.end(),
                         [=](Var var) -> bool { return var.same_as(x); }) != vars_.end())
        << "Not All variable in value can be replaced by index vars";
    }
  }

  Stmt Mutate_(const BlockNode* op, const Stmt& s) final {
    Stmt stmt = ScheduleMutator::Mutate_(op, s);
    op = stmt.as<BlockNode>();
    CHECK(op);
    Array<IterVar> new_vars;
    Array<Expr> new_values;
    Array<Var> all_vars = GatherVars(op->body);
    Array<TensorRegion> new_reads = CreateInputRegions(op->body);
    for (size_t i = 0; i < op->iter_vars.size(); ++i) {
      Var var = op->iter_vars[i]->var;
      if (std::find_if(vars_.begin(), vars_.end(),
                       [=](Var x) -> bool { return x.same_as(var); }) != all_vars.end()) {
        new_vars.push_back(op->iter_vars[i]);
        new_values.push_back(op->values[i]);
      }
    }
    Block block = GetRef<Block>(op);
    block.Mutable()->iter_vars = new_vars;
    block.Mutable()->values = new_values;
    block.Mutable()->reads = new_reads;
    return std::move(block);
  }

  Expr Mutate_(const BufferLoadNode* op, const Expr& e) final {
    if (op->buffer == op_->buffer) {
      std::unordered_map<Var, Expr, NodeHash, NodeEqual> vmap;
      for (size_t i = 0; i < op->indices.size(); ++i) {
        vmap[vars_[i]] = op->indices[i];
      }
      return Substitute(value_, vmap);
    } else {
      return IRMutator::Mutate_(op, e);
    }
  }

 private:
  const BufferStoreNode* op_;
  Array<Var> vars_;
  Expr value_;
};

void Schedule::compute_inline(Block block) {
  // conditions:
  // 1. only write to one element
  // 2. is terminal block
  // -> The inner stmt is a BufferStore
  CHECK(block->body.as<BufferStoreNode>())
    << "Can only inline single assignment statement";
  CHECK_EQ(block->writes.size(), 1)
    << "Can only inline statement with one output";
  CHECK(IsCompleteBlock(block))
    << "Can only inline a complete block";

  auto write = block->writes[0];
//  RemoveLeaf(block);
  const auto& blocks = operator->()->write_map_.at(write->buffer);
  Array<Block> new_blocks;
  for (const auto& b : blocks) {
    if (!b.same_as(block)) {
      new_blocks.push_back(b);
    }
  }
  Mutable()->write_map_.Set(write->buffer, new_blocks);

  for (const auto& x : operator->()->dependency_graph_->forward_edges.at(block)) {
    Block dst = x->dst;
    StatementInliner(block->body.as<BufferStoreNode>()).Mutate(dst);
  }

  // update in dependency graph
  Mutable()->dependency_graph_.InlineBlock(block);
}

}  // namespace te
}  // namespace tvm