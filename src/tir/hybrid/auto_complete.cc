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

#include <tvm/arith/int_set.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt_functor.h>

#include <utility>

#include "../schedule/schedule_common.h"

namespace tvm {
namespace tir {

Array<TensorRegion> BlockReadWriteCollector::reads() {
  std::vector<TensorRegion> res;
  for (size_t i = 0; i < read_regions_.size(); ++i) {
    std::vector<Range> region;
    for (const auto& range : read_regions_[i])
      region.push_back(range.CoverRange(Range::FromMinExtent(0, 0)));
    res.emplace_back(read_buffers_[i], region);
  }
  return res;
}

Array<TensorRegion> BlockReadWriteCollector::writes() {
  std::vector<TensorRegion> res;
  for (size_t i = 0; i < write_regions_.size(); ++i) {
    std::vector<Range> region;
    for (const auto& range : write_regions_[i])
      region.push_back(range.CoverRange(Range::FromMinExtent(0, 0)));
    res.emplace_back(writes_buffers_[i], region);
  }
  return res;
}

void BlockReadWriteCollector::VisitStmt_(const LoopNode* op) {
  Range range = Range::FromMinExtent(op->min, op->extent);
  dom_map_[op->loop_var.get()] = arith::IntSet::FromRange(range);
  StmtVisitor::VisitStmt_(op);
  dom_map_.erase(op->loop_var.get());
}

void BlockReadWriteCollector::Update(std::vector<Buffer>* buffers,
                                     std::vector<std::vector<arith::IntSet>>* regions,
                                     const Buffer& buffer,
                                     const std::vector<arith::IntSet>& region) {
  if (inner_buffers_.find(buffer.get()) != inner_buffers_.end()) return;
  bool find = false;
  for (size_t i = 0; i < regions->size(); ++i)
    if ((*buffers)[i].same_as(buffer)) {
      find = true;
      CHECK_EQ((*regions)[i].size(), region.size()) << "Inconsistent buffer dimension";
      for (size_t j = 0; j < region.size(); ++j) {
        (*regions)[i][j] = arith::Union({(*regions)[i][j], region[j]});
      }
    }
  if (!find) {
    buffers->push_back(buffer);
    regions->push_back(region);
  }
}

void BlockReadWriteCollector::VisitExpr_(const BufferLoadNode* op) {
  std::vector<arith::IntSet> relaxed_region;
  for (size_t j = 0; j < op->indices.size(); ++j) {
    relaxed_region.push_back(arith::EvalSet(op->indices[j], dom_map_));
  }
  Update(&read_buffers_, &read_regions_, op->buffer, relaxed_region);
  ExprVisitor::VisitExpr_(op);
}

void BlockReadWriteCollector::VisitStmt_(const BufferStoreNode* op) {
  std::vector<arith::IntSet> relaxed_region;
  for (size_t j = 0; j < op->indices.size(); ++j) {
    relaxed_region.push_back(arith::EvalSet(op->indices[j], dom_map_));
  }
  Update(&writes_buffers_, &write_regions_, op->buffer, relaxed_region);
  StmtVisitor::VisitStmt_(op);
}

void BlockReadWriteCollector::VisitStmt_(const ReduceStepNode* op) {
  const auto* buffer_load = op->lhs.as<BufferLoadNode>();
  CHECK(buffer_load != nullptr)
      << "TypeError: 'decompose_reduction' expects the body of the reduce step "
         "is BufferLoad, but get type: "
      << op->lhs->GetTypeKey();
  std::vector<arith::IntSet> relaxed_region;
  for (size_t j = 0; j < buffer_load->indices.size(); ++j) {
    relaxed_region.push_back(arith::EvalSet(buffer_load->indices[j], dom_map_));
  }
  Update(&writes_buffers_, &write_regions_, buffer_load->buffer, relaxed_region);
  StmtVisitor::VisitStmt_(op);
}

void BlockReadWriteCollector::VisitStmt_(const BlockRealizeNode* op) {
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  for (size_t i = 0; i < op->block->iter_vars.size(); ++i) {
    vmap[op->block->iter_vars[i]->var.get()] = op->binding_values[i];
  }
  for (const auto& read : op->block->reads) {
    std::vector<arith::IntSet> relaxed_region;
    for (const auto& range : read->region) {
      relaxed_region.push_back(
          arith::EvalSet(arith::IntSet::FromRange(Range::FromMinExtent(
                             Substitute(range->min, vmap), Substitute(range->extent, vmap))),
                         dom_map_));
    }
    Update(&read_buffers_, &read_regions_, read->buffer, relaxed_region);
  }
  for (const auto& write : op->block->writes) {
    std::vector<arith::IntSet> relaxed_region;
    for (const auto& range : write->region) {
      relaxed_region.push_back(
          arith::EvalSet(arith::IntSet::FromRange(Range::FromMinExtent(
                             Substitute(range->min, vmap), Substitute(range->extent, vmap))),
                         dom_map_));
    }
    Update(&writes_buffers_, &write_regions_, write->buffer, relaxed_region);
  }
}

}  // namespace tir

namespace tir {
namespace script {

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

  Stmt VisitStmt_(const BlockNode* op) override {
    Block block = Downcast<Block>(StmtMutator::VisitStmt_(op));
    if (!block->reads.defined() || !block->writes.defined()) {
      BlockReadWriteCollector block_read_write_collector(block->allocations);
      block_read_write_collector(block->body);
      auto n = CopyOnWrite(block.operator->());
      if (!n->reads.defined()) n->reads = block_read_write_collector.reads();
      if (!n->writes.defined()) n->writes = block_read_write_collector.writes();
      return Block(n);
    } else {
      return std::move(block);
    }
  }
};

TVM_REGISTER_GLOBAL("script.AutoComplete")
    .set_body_typed<Stmt(Stmt, Array<BufferAllocate>)>([](Stmt body,
                                                          Array<BufferAllocate> root_allocates) {
      AutoCompleter auto_completer;
      // generate surrounding loops automatically
      Stmt res = auto_completer(std::move(body));
      // generate root block automatically
      if (auto_completer.contains_block &&
          (!res->IsInstance<BlockRealizeNode>() || !root_allocates.empty())) {
        res = Block({}, {}, {}, res, root_allocates, {}, "root");
        res = BlockRealize({}, Bool(true), Downcast<Block>(res), String(""));
      }
      return res;
    });

}  // namespace script
}  // namespace tir
}  // namespace tvm
