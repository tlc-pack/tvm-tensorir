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
 * \file tir/analysis/block_region_detector.cc
 * \brief Detect block read/write regions by visiting its body
 */

#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

void BlockReadWriteDetector::operator()(const Stmt& stmt) {
  ICHECK(stmt.as<BlockNode>() != nullptr) << "Only allow to visit a block";
  StmtExprVisitor::operator()(stmt);
}

Array<BufferRegion> BlockReadWriteDetector::CollectReads() {
  std::vector<BufferRegion> res;
  for (size_t i = 0; i < read_regions_.size(); ++i) {
    std::vector<Range> region;
    for (const auto& range : read_regions_[i]) {
      region.push_back(range.CoverRange(Range::FromMinExtent(0, 0)));
    }
    res.emplace_back(read_buffers_[i], region);
  }
  return res;
}

Array<BufferRegion> BlockReadWriteDetector::CollectWrites() {
  std::vector<BufferRegion> res;
  for (size_t i = 0; i < write_regions_.size(); ++i) {
    std::vector<Range> region;
    for (const auto& range : write_regions_[i]) {
      region.push_back(range.CoverRange(Range::FromMinExtent(0, 0)));
    }
    res.emplace_back(writes_buffers_[i], region);
  }
  return res;
}

void BlockReadWriteDetector::VisitStmt_(const ForNode* op) {
  Range range = Range::FromMinExtent(op->min, op->extent);
  dom_map_[op->loop_var.get()] = arith::IntSet::FromRange(range);
  StmtVisitor::VisitStmt_(op);
  dom_map_.erase(op->loop_var.get());
}

void BlockReadWriteDetector::VisitExpr_(const BufferLoadNode* op) {
  std::vector<arith::IntSet> relaxed_region;
  for (const PrimExpr& index : op->indices) {
    relaxed_region.push_back(arith::EvalSet(index, dom_map_));
  }
  Update(&read_buffers_, &read_regions_, op->buffer, relaxed_region);
  ExprVisitor::VisitExpr_(op);
}

void BlockReadWriteDetector::VisitStmt_(const BufferStoreNode* op) {
  std::vector<arith::IntSet> relaxed_region;
  for (const PrimExpr& index : op->indices) {
    relaxed_region.push_back(arith::EvalSet(index, dom_map_));
  }
  Update(&writes_buffers_, &write_regions_, op->buffer, relaxed_region);
  StmtVisitor::VisitStmt_(op);
}

void BlockReadWriteDetector::VisitStmt_(const BlockRealizeNode* op) {
  /*! \note detector will not visit child block recursively, so that it will stop here */
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

void BlockReadWriteDetector::VisitStmt_(const BlockNode* op) {
  /*! \note Only for the block to be detected, detector will not visit child block recursively */
  for (const auto& alloc_buffer : op->alloc_buffers) {
    inner_buffers_.insert(alloc_buffer.get());
  }
  StmtVisitor::VisitStmt_(op);
}

void BlockReadWriteDetector::Update(std::vector<Buffer>* buffers,
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

}  // namespace tir
}  // namespace tvm