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

/*!
 * \brief Auto detect the block read write region
 *        It will detect the read/write region as an array in order of appearance in AST
 * \note This detector only accepts to visit a block and will not visit child blocks recursively
 */
class BlockReadWriteDetector : public StmtExprVisitor {
 public:
  BlockReadWriteDetector() = default;

  /*! \brief Return read regions of the block */
  Array<BufferRegion> CollectReads();
  /*! \brief Return write regions of the block */
  Array<BufferRegion> CollectWrites();
  /*!
   * \brief Return opaque buffer regions of the block
   * \note The buffer accessed by load/store or call with buffer.data will
   *       be marked as opaque.
   */
  Array<BufferRegion> CollectOpaques();
  /*! \brief overload operator() to make sure it accepts a block node */
  void operator()(const Stmt &stmt);

 private:
  /*! \brief Iteration range for loop_vars */
  std::unordered_map<const VarNode*, arith::IntSet> dom_map_;
  /*! \brief The buffers that the current block reads */
  std::vector<Buffer> read_buffers_;
  /*! \brief The buffers that the current block writes */
  std::vector<Buffer> writes_buffers_;
  /*! \brief The opaque buffer which is access by buffer.data */
  std::vector<Buffer> opaque_buffers_;
  /*! \brief The read regions of the current block */
  std::vector<std::vector<tvm::arith::IntSet>> read_regions_;
  /*! \brief The write regions of the current block */
  std::vector<std::vector<tvm::arith::IntSet>> write_regions_;
  /*! \brief The buffer allocated inside the block, which will not been shown in the reads/writes */
  std::unordered_set<const BufferNode*> inner_buffers_;

  /*!
   * \brief Update read/write buffers and regions with provided buffer and region
   * \param buffers The buffers should be updated
   * \param regions The access regions should be updated
   * \param buffer The provided buffer
   * \param region The provided region
   */
  void Update(std::vector<Buffer>* buffers, std::vector<std::vector<arith::IntSet>>* regions,
              const Buffer& buffer, const std::vector<arith::IntSet>& region);

  void VisitStmt_(const ForNode* op) override;
  void VisitExpr_(const BufferLoadNode* op) override;
  void VisitStmt_(const BufferStoreNode* op) override;
  void VisitStmt_(const BlockRealizeNode* op) override;
  void VisitStmt_(const BlockNode* op) override;
};

void BlockReadWriteDetector::operator()(const Stmt& stmt) {
  ICHECK(stmt.as<BlockNode>() != nullptr) << "Only allow to visit a block";
  StmtExprVisitor::operator()(stmt);
}

Array<BufferRegion> BlockReadWriteDetector::CollectReads() {
  Array<BufferRegion> res;
  for (size_t i = 0; i < read_regions_.size(); ++i) {
    Array<Range> region;
    for (const auto& range : read_regions_[i]) {
      region.push_back(range.CoverRange(Range::FromMinExtent(0, 0)));
    }
    res.push_back(BufferRegion(read_buffers_[i], region));
  }
  return res;
}

Array<BufferRegion> BlockReadWriteDetector::CollectWrites() {
  Array<BufferRegion> res;
  for (size_t i = 0; i < write_regions_.size(); ++i) {
    Array<Range> region;
    for (const auto& range : write_regions_[i]) {
      region.push_back(range.CoverRange(Range::FromMinExtent(0, 0)));
    }
    res.push_back(BufferRegion(writes_buffers_[i], region));
  }
  return res;
}

Array<BufferRegion> CollectOpaques() {

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
  ICHECK_EQ(buffers->size(), regions->size())
      << " Expect the buffer and regions to have the same size ";
  for (size_t i = 0; i < regions->size(); ++i)
    if ((*buffers)[i].same_as(buffer)) {
      ICHECK_EQ((*regions)[i].size(), region.size()) << "Inconsistent buffer dimension";
      for (size_t j = 0; j < region.size(); ++j) {
        (*regions)[i][j] = arith::Union({(*regions)[i][j], region[j]});
      }
      return;
    }
  buffers->push_back(buffer);
  regions->push_back(region);
}

Array<Array<BufferRegion>> GetBlockAccessRegion(const Block& block) {
  BlockReadWriteDetector detector;
  detector(block);
  return {detector.CollectReads(), detector.CollectWrites(), detector.CollectWrites()};
}

}  // namespace tir
}  // namespace tvm
