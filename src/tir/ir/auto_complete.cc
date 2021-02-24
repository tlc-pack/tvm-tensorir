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
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <utility>

namespace tvm {
namespace tir {

/* \brief Auto calculate the block read write region */
class BlockReadWriteCollector : public StmtExprVisitor {
 public:
  explicit BlockReadWriteCollector(const Array<Buffer>& allocations) {
    for (const auto& allocate : allocations) inner_buffers_.insert(allocate.get());
  }

  /* \brief Collect read regions of the block*/
  Array<BufferRegion> CollectReads();
  /* \brief Collect write regions of the block*/
  Array<BufferRegion> CollectWrites();

 private:
  std::unordered_map<const VarNode*, arith::IntSet> dom_map_;
  std::vector<Buffer> read_buffers_, writes_buffers_;
  std::vector<std::vector<tvm::arith::IntSet>> read_regions_, write_regions_;
  std::unordered_set<const BufferNode*> inner_buffers_;

  void VisitStmt_(const ForNode* op) override;
  void Update(std::vector<Buffer>* buffers, std::vector<std::vector<arith::IntSet>>* regions,
              const Buffer& buffer, const std::vector<arith::IntSet>& region);
  void VisitExpr_(const BufferLoadNode* op) override;
  void VisitStmt_(const BufferStoreNode* op) override;
  void VisitStmt_(const BlockRealizeNode* op) override;
};

Array<BufferRegion> BlockReadWriteCollector::CollectReads() {
  std::vector<BufferRegion> res;
  for (size_t i = 0; i < read_regions_.size(); ++i) {
    std::vector<Range> region;
    for (const auto& range : read_regions_[i])
      region.push_back(range.CoverRange(Range::FromMinExtent(0, 0)));
    res.emplace_back(read_buffers_[i], region);
  }
  return res;
}

Array<BufferRegion> BlockReadWriteCollector::CollectWrites() {
  std::vector<BufferRegion> res;
  for (size_t i = 0; i < write_regions_.size(); ++i) {
    std::vector<Range> region;
    for (const auto& range : write_regions_[i])
      region.push_back(range.CoverRange(Range::FromMinExtent(0, 0)));
    res.emplace_back(writes_buffers_[i], region);
  }
  return res;
}

void BlockReadWriteCollector::VisitStmt_(const ForNode* op) {
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
      ICHECK_EQ((*regions)[i].size(), region.size()) << "Inconsistent buffer dimension";
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

void BlockReadWriteCollector::VisitStmt_(const BlockRealizeNode* op) {
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  for (size_t i = 0; i < op->block->iter_vars.size(); ++i) {
    vmap[op->block->iter_vars[i]->var.get()] = op->iter_values[i];
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

/*! \brief Generate surrounding loops automatically */
class AutoCompleter : public StmtMutator {
 public:
  /* \brief Whether the stmt contains at least one block. */
  bool contains_block = false;

 private:
  Stmt VisitStmt_(const BlockRealizeNode* op) override {
    contains_block = true;
    Stmt body = StmtMutator::VisitStmt_(op);
    if (!op->iter_values.empty() && !op->iter_values[0].defined()) {
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
    if (!block->reads.defined() || !block->writes.defined()) {
      BlockReadWriteCollector block_read_write_collector(block->alloc_buffers);
      block_read_write_collector(block->body);
      auto n = CopyOnWrite(block.operator->());
      if (!n->reads.defined()) n->reads = block_read_write_collector.CollectReads();
      if (!n->writes.defined()) n->writes = block_read_write_collector.CollectWrites();
      return Block(n);
    } else {
      return std::move(block);
    }
  }
};

Stmt AutoComplete(const Stmt& body, const Array<Buffer>& root_allocates) {
  AutoCompleter auto_completer;
  // generate surrounding loops automatically
  Stmt res = auto_completer(body);
  // generate root block automatically
  if (auto_completer.contains_block &&
      (!res->IsInstance<BlockRealizeNode>() || !root_allocates.empty())) {
    res = Block({}, {}, {}, "root", res, NullOpt, "", root_allocates);
    res = BlockRealize({}, Bool(true), Downcast<Block>(res));
  }
  return res;
}

TVM_REGISTER_GLOBAL("script.AutoComplete").set_body_typed(AutoComplete);

}  // namespace tir
}  // namespace tvm
