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

class StatementInliner : public IRMutator {
 public:
  explicit StatementInliner(const BufferStoreNode* op) : op_(op) {
    for (const auto& index : op->indices) {
      const auto* variable = index.as<Variable>();
      CHECK(variable) << "Only support inline direct access block";
      Var var = GetRef<Var>(variable);
      vars_.push_back(var);
    }
    value_ = op->value;
    Array<Var> value_vars = GatherVars(value_);
    for (const auto& x : value_vars) {
      CHECK(std::find_if(vars_.begin(), vars_.end(),
                         [=](Var var) -> bool { return var.same_as(x); }) != vars_.end())
        << "Not All variable in value can be replaced by index vars";
    }
  }

  Stmt Mutate_(const BlockNode* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
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
    return Block(new_vars,
                 new_values,
                 new_reads,
                 op->writes,
                 op->body,
                 op->predicate,
                 op->allocations,
                 op->annotations,
                 op->tag);
  }

  Expr Mutate_(const BufferLoadNode* op, const Expr& e) final {
    if (op->buffer == op_->buffer) {
      Map<Var, Expr> vmap;
      for (size_t i = 0; i < op->indices.size(); ++i) {
        vmap.Set(vars_[i], op->indices[i]);
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

void Schedule::compute_inline(BlockTreeNodeRef block) {
  // conditions:
  // 1. only write to one element
  // 2. is terminal block
  // -> The inner stmt is a BufferStore
  // TODO(siyuan): remove useless Loops and BufferAllocate
  Block block_stmt = GetRef<Block>(block->block);
  BlockTreeNodeRef father_block = GetFatherBlock(block);
  CHECK(block_stmt->body.as<BufferStoreNode>())
    << "Can only inline single assignment statement";
  CHECK_EQ(block_stmt->writes.size(), 1)
    << "Can only inline statement with one output";
  CHECK(IsCompleteBlock(block))
    << "Can only inline a complete block";
  CHECK(!operator->()->dependency_graph.at(father_block)->forward_edges.at(block).empty())
    << "Can not inline a output block";

  auto write = block_stmt->writes[0];
  Replace(block, Evaluate::make(0));
  const auto& blocks = operator->()->write_map.at(write->buffer);
  Array<BlockTreeNodeRef> new_blocks;
  for (const auto& b : blocks) {
    if (!b.same_as(block)) {
      new_blocks.push_back(b);
    }
  }

  operator->()->write_map.Set(write->buffer, new_blocks);
  for (const auto& x : operator->()->dependency_graph.at(father_block)->forward_edges.at(block)) {
    Block dst = GetRef<Block>(x->dst->block);
    Replace(x->dst, StatementInliner(block_stmt->body.as<BufferStoreNode>()).Mutate(dst));
  }

  // update in dependency graph
  const auto* graph_ptr = operator->()->dependency_graph[father_block].as<DependencyGraphNode>();
  auto graph = GetRef<DependencyGraph>(graph_ptr);
  graph.InlineBlock(block);
}

}  // namespace te
}  // namespace tvm
