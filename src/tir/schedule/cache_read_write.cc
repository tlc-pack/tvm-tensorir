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

#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt_functor.h>

#include "../ir/functor_common.h"
#include "schedule_common.h"

namespace tvm {
namespace tir {

/*!
 * \brief Detect the insert position of buffer copy
 * \note See comments in function cache_read and cache_write
 *       for detail rules and information
 * \param offset The offset of insert position from the block
 *        offset 0 for cache_read and 1 for cache_write
 *        e.g the block itself locates at the n-th element of the SeqStmt,
 *        so, the buffer copy stmt should be inserted at n+offset position
 *        (2rd when cache_read or at 3rd when cache_write)
 */

class CachePositionDetector : public StmtVisitor {
 public:
  explicit CachePositionDetector(const ScheduleNode* sch, const StmtSRef& block_sref,
                                 const std::vector<StmtSRef>& related_blocks, const size_t offset)
      : sch_(sch), block_sref_(block_sref), related_blocks_(related_blocks), offset_(offset) {}

  void VisitStmt_(const SeqStmtNode* op) final {
    bool visited_block = false, visited_related = false;
    std::swap(visited_block, visited_block_);
    std::swap(visited_related, visited_related_);
    int pos = -1;
    for (size_t i = 0; i < op->size(); ++i) {
      if (pos_index_ != -1) break;
      VisitStmt((*op)[i]);

      // pos can only be assigned once when we visited the block_sref
      if (visited_block_ && pos == -1) {
        pos = static_cast<int>(i) + offset_;
      }
    }

    std::swap(visited_block, visited_block_);
    std::swap(visited_related, visited_related_);
    visited_block_ |= visited_block;
    visited_related_ |= visited_related;

    // Only we visited the writing block and any one of the related blocks
    // That means that we have found the lowest ancestor
    // of the block and any one of the related ones
    if (visited_block && visited_related) {
      pos_index_ = pos;
    }
  }

  void VisitStmt_(const BlockNode* op) final {
    auto it = sch_->stmt2ref.find(op);
    CHECK(it != sch_->stmt2ref.end());
    StmtSRef sref = it->second;
    // Only visit current scope
    if (block_cnt_++ == 0) {
      StmtVisitor::VisitStmt_(op);
      // handling cache_read for input buffer
      if (visited_block_ && visited_related_ && !pos_sref_.defined()) {
        pos_sref_ = sref;
        if (pos_index_ == -1) pos_index_ = 1;
      }
    } else {
      // Update visiting info
      if (sref.same_as(block_sref_)) {
        visited_block_ = true;
      } else if (std::find(related_blocks_.begin(), related_blocks_.end(), sref) !=
                 related_blocks_.end()) {
        visited_related_ = true;
      }
    }
  }

  void VisitStmt_(const LoopNode* op) final {
    auto it = sch_->stmt2ref.find(op);
    CHECK(it != sch_->stmt2ref.end());
    StmtSRef sref = it->second;
    StmtVisitor::VisitStmt_(op);
    if (visited_block_ && visited_related_ && !pos_sref_.defined() && pos_index_ != -1) {
      pos_sref_ = sref;
    }
  }

 public:
  // The index where the cache copy stmt should be inserted
  int pos_index_{-1};
  // The StmtSRef where the cache copy stmt should be inserted
  StmtSRef pos_sref_{NullValue<StmtSRef>()};

 private:
  const ScheduleNode* sch_;
  // The dominate block which write the buffer
  const StmtSRef& block_sref_;
  // Producer blocks for cache_write and consumer blocks for cache_read
  const std::vector<StmtSRef>& related_blocks_;
  // The flag whether we have visited the dominate block
  bool visited_block_{false};
  // The flag whether we have visited at least one related blocks
  bool visited_related_{false};
  // The block visiting time
  size_t block_cnt_{0};
  size_t offset_;
};

/*!
 * \brief base class for CacheReadRewriter and CacheWriteRewrite
 * \note This Mutator will update buffer allocate and insert the
 *       cache copy stmt into the correct position
 */
class CacheRewriter : public StmtExprMutator {
 public:
  explicit CacheRewriter(
      const std::unordered_map<Buffer, Buffer, ObjectHash, ObjectEqual>& buffer_map,
      const StmtSRef& insert_sref, const size_t insert_pos, const BufferAllocate& cache_allocate,
      const Stmt& stmt)
      : buffer_map_(buffer_map),
        insert_sref_(insert_sref),
        insert_pos_(insert_pos),
        cache_allocate_(cache_allocate),
        stmt_to_be_inserted_(stmt) {}

  Stmt VisitStmt_(const LoopNode* op) final { return VisitSRefStmt(op); }

  Stmt VisitStmt_(const BlockNode* op) override {
    bool is_scope_block = block_visited_cnt_++ == 0;
    Stmt s = VisitSRefStmt(op);
    op = s.as<BlockNode>();
    CHECK(op != nullptr);
    if (is_scope_block) {
      auto n = CopyOnWrite(op);
      n->allocations.push_back(cache_allocate_);
      return Stmt(n);
    } else {
      return GetRef<Stmt>(op);
    }
  }

 protected:
  const std::unordered_map<Buffer, Buffer, ObjectHash, ObjectEqual>& buffer_map_;
  Array<TensorRegion> UpdateBufferViaMap(const Array<TensorRegion>& tensor_regions) {
    auto fmutate = [this](const TensorRegion& tensor_region) {
      auto it = buffer_map_.find(tensor_region->buffer);
      if (it != buffer_map_.end()) {
        auto n = CopyOnWrite(tensor_region.operator->());
        n->buffer = it->second;
        return TensorRegion(n);
      } else {
        return tensor_region;
      }
    };
    return MutateArray(tensor_regions, fmutate, allow_copy_on_write_);
  }

  size_t block_visited_cnt_{0};

 private:
  /*! \brief insert the copy stmt into the correct position into the seq stmt under the sref*/
  template <typename T>
  Stmt VisitSRefStmt(const T* op) {
    bool is_insert_pos = op == insert_sref_->node;
    Stmt s = StmtMutator::VisitStmt_(op);
    op = s.as<T>();
    CHECK(op != nullptr);
    if (is_insert_pos) {
      auto n = CopyOnWrite(op);
      if (const auto* seq = n->body.template as<SeqStmtNode>()) {
        auto seq_node = CopyOnWrite(seq);
        seq_node->seq.insert(seq_node->seq.begin() + insert_pos_, stmt_to_be_inserted_);
        n->body = SeqStmt(seq_node);
      } else {
        if (insert_pos_ == 0) {
          n->body = SeqStmt({stmt_to_be_inserted_, n->body});
        } else {
          CHECK_EQ(insert_pos_, 1);
          n->body = SeqStmt({n->body, stmt_to_be_inserted_});
        }
      }
      return Stmt(n);
    } else {
      return GetRef<Stmt>(op);
    }
  }
  // The StmtSRef where the cache copy stmt should be inserted
  const StmtSRef& insert_sref_;
  // The index where the cache copy stmt should be inserted
  const size_t insert_pos_;
  // The BufferAllocate for cache buffer
  const BufferAllocate& cache_allocate_;
  // The Stmt for copying between the buffer and the cache
  const Stmt& stmt_to_be_inserted_;
};

/*! \brief Mutater for CacheRead */
class CacheReadRewriter : public CacheRewriter {
 public:
  explicit CacheReadRewriter(
      const std::unordered_map<Buffer, Buffer, ObjectHash, ObjectEqual>& buffer_map,
      const StmtSRef& insert_sref, const size_t insert_pos, const BufferAllocate& cache_allocate,
      const Stmt& stmt)
      : CacheRewriter(buffer_map, insert_sref, insert_pos, cache_allocate, stmt) {}

  Stmt VisitStmt_(const BlockNode* op) final {
    bool is_scope_block = block_visited_cnt_ == 0;
    Block old_block = GetRef<Block>(op);
    Stmt s = CacheRewriter::VisitStmt_(op);
    op = s.as<BlockNode>();
    CHECK(op != nullptr);
    Block ret;
    if (is_scope_block) {
      ret = GetRef<Block>(op);
    } else {
      auto reads = UpdateBufferViaMap(op->reads);
      if (reads.same_as(op->reads)) {
        ret = GetRef<Block>(op);
      } else {
        auto n = CopyOnWrite(op);
        n->reads = std::move(reads);
        ret = Block(n);
      }
    }
    block_sref_map_.Set(ret, old_block);
    return Stmt(ret);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto it = buffer_map_.find(op->buffer);
    if (it != buffer_map_.end()) {
      auto n = CopyOnWrite(op);
      n->buffer = it->second;
      return PrimExpr(n);
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  using CacheRewriter::VisitStmt_;

 public:
  Map<Block, Block> block_sref_map_;
};

/*! \brief Mutater for CacheWrite */
class CacheWriteRewriter : public CacheRewriter {
 public:
  explicit CacheWriteRewriter(
      const std::unordered_map<Buffer, Buffer, ObjectHash, ObjectEqual>& buffer_map,
      const StmtSRef& insert_sref, const size_t insert_pos, const BufferAllocate& cache_allocate,
      const Stmt& stmt)
      : CacheRewriter(buffer_map, insert_sref, insert_pos, cache_allocate, stmt) {}

  Stmt VisitStmt_(const BlockNode* op) final {
    bool is_scope_block = block_visited_cnt_ == 0;
    Block old_block = GetRef<Block>(op);
    Stmt s = CacheRewriter::VisitStmt_(op);
    op = s.as<BlockNode>();
    CHECK(op != nullptr);
    Block ret;
    if (is_scope_block) {
      ret = GetRef<Block>(op);
    } else {
      // Since cache_write changes the block, we need to update the buffer it writes
      auto writes = UpdateBufferViaMap(op->writes);
      auto reads = UpdateBufferViaMap(op->reads);
      if (writes.same_as(op->writes) && reads.same_as(op->reads)) {
        ret = GetRef<Block>(op);
      } else {
        auto n = CopyOnWrite(op);
        n->writes = std::move(writes);
        n->reads = std::move(reads);
        ret = Block(n);
      }
    }
    block_sref_map_.Set(ret, old_block);
    return Stmt(ret);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto it = buffer_map_.find(op->buffer);
    if (it != buffer_map_.end()) {
      auto n = CopyOnWrite(op);
      n->buffer = it->second;
      return Stmt(n);
    } else {
      return GetRef<Stmt>(op);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto it = buffer_map_.find(op->buffer);
    if (it != buffer_map_.end()) {
      auto n = CopyOnWrite(op);
      n->buffer = it->second;
      return PrimExpr(n);
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  using CacheRewriter::VisitStmt_;

 public:
  Map<Block, Block> block_sref_map_;
};

/*! \brief Create cache buffer */
Buffer CreateCacheBuffer(const Buffer& buffer, const std::string& scope) {
  auto n = make_object<BufferNode>(*(buffer.operator->()));
  n->data = buffer->data.copy_with_suffix("_" + scope);
  n->name = buffer->name + "_" + scope;
  n->scope = scope;
  return Buffer(n);
}

/*!
 * \brief Create Stmt for copying from read buffer to write buffer
 * \param read_buffer The read buffer
 * \param write_buffer The write buffer
 * \param relaxed_region The copy region
 * \returns Stmt The Whole Stmt for buffer copying with loop nesting
 *          Block The Copy block without loop nesting
 * */
std::pair<Stmt, Block> GenerateCopyStmt(const Buffer& read_buffer, const Buffer& write_buffer,
                                        const TensorRegion& relaxed_region) {
  // Generate copy nested loops and block
  Array<Var> loop_vars;
  Array<PrimExpr> binding_value;

  Region access_region;
  for (size_t i = 0; i < relaxed_region->region.size(); ++i) {
    Var loop_var(Var("ax" + std::to_string(i)));
    loop_vars.push_back(loop_var);
    binding_value.push_back(loop_var + relaxed_region->region[i]->min);
  }

  Array<IterVar> block_vars;
  Array<PrimExpr> indices;
  const auto& shape = relaxed_region->buffer->shape;
  for (size_t i = 0; i < shape.size(); ++i) {
    IterVar var = IterVarNode::make(Range::make_by_min_extent(0, shape[i]),
                                    Var("v" + std::to_string(i)), kDataPar);
    block_vars.push_back(var);
    indices.push_back(var);
    access_region.push_back(Range::make_by_min_extent(var, 1));
  }
  Stmt body =
      BufferStore(write_buffer, BufferLoad(read_buffer, indices), indices);
  Block block(block_vars, {TensorRegion(read_buffer, access_region)},
              {TensorRegion(write_buffer, access_region)}, body, Array<BufferAllocate>(),
              Array<Annotation>(), "");
  BlockRealize block_realize(binding_value, IntImm(DataType::Bool(), 1), block);
  body = block_realize;
  for (size_t i = loop_vars.size(); i > 0; --i) {
    const Range& range = relaxed_region->region[i - 1];
    body = Loop(loop_vars[i - 1], 0, range->extent, Array<Annotation>(), body);
  }

  return std::make_pair(body, block);
}

/*!
 * \brief Get the innermost block who write the buffer
 * \note  This function will check whether the block is dominate
 */
StmtSRef GetInnermostBlock(const ScheduleNode* sch, const Buffer& buffer) {
  StmtSRef sref = sch->root;
  Scope scope = sch->scopes_.at(sref);
  // return nullptr when the buffer is an input buffer
  auto it = scope->write_map.find((buffer));
  if (it == scope->write_map.end()) {
    return NullValue<StmtSRef>();
  }
  do {
    const auto& write_blocks = it->second;
    CHECK_EQ(write_blocks.size(), 1)
        << "Can only cache_read or cache_write a dominate block (only producer)";
    sref = write_blocks[0];
    scope = sch->scopes_.at(sref);
    it = scope->write_map.find((buffer));
  } while (it != scope->write_map.end());
  return sref;
}

StmtSRef ScheduleNode::cache_read(const Buffer& buffer, const std::string& storage_scope) {
  /*!
   * Check:
   *   - check the buffer has only one writing block
   *   - check the buffer is not a output buffer
   *
   * Mutate:
   *   - allocate new cache buffer under the current scope.
   *   - find the lowest ancestor of the block and ANY ONE of the consumers blocks.
   *   - Copy the buffer with the necessary region.
   */
  StmtSRef block_sref = GetInnermostBlock(this, buffer);
  StmtSRef scope_sref;
  const BlockNode* scope_block = nullptr;
  StmtSRef insert_sref;
  size_t insert_pos;
  TensorRegion cache_region;
  if (block_sref.defined()) {
    const auto* block = DowncastPtr<BlockNode>(block_sref->node);
    CHECK(block != nullptr) << buffer << "is not a block sref";
    scope_sref = GetScope(block_sref);

    const Scope& scope = scopes_.at(scope_sref);
    scope_block = DowncastPtr<BlockNode>(scope_sref->node);

    // Check the block is not a output block
    std::unordered_set<Buffer, ObjectHash, ObjectEqual> seen_buffer;
    for (const auto& x : block->writes) {
      for (const auto& output_buffer : scope_block->writes)
        CHECK(!x->buffer.same_as(output_buffer->buffer)) << "Can not cache_read an output block";
    }

    // Check there is only one output buffer
    CHECK_EQ(block->writes.size(), 1);

    std::vector<StmtSRef> consumers;
    for (const auto& x : scope.GetSuccessors(block_sref)) {
      if (x->type == DepType::kRAW) consumers.push_back(x->dst);
    }
    CHECK(!consumers.empty());

    // Detector insert position
    CachePositionDetector detector(this, block_sref, consumers, 0);
    detector(GetRef<Stmt>(scope_block));

    insert_sref = detector.pos_sref_;
    insert_pos = detector.pos_index_;
    cache_region = RelaxRegion(block_sref, scope_sref, block->writes[0]);
  } else {
    scope_sref = root;
    scope_block = DowncastPtr<BlockNode>(scope_sref->node);
    insert_sref = root;
    insert_pos = 0;
    Region region;
    for (const auto& shape : buffer->shape) {
      region.push_back(Range::make_by_min_extent(0, shape));
    }
    cache_region = TensorRegion(buffer, region);
  }

  // Generate cache buffer
  Buffer cache_buffer = CreateCacheBuffer(buffer, storage_scope);

  auto x = GenerateCopyStmt(buffer, cache_buffer, cache_region);
  Stmt stmt = x.first;
  Block cache_block = x.second;

  BufferAllocate cache_allocate(cache_buffer, storage_scope);
  std::unordered_map<Buffer, Buffer, ObjectHash, ObjectEqual> buffer_map;
  buffer_map[buffer] = cache_buffer;

  CacheReadRewriter rewriter(buffer_map, insert_sref, insert_pos, cache_allocate, stmt);
  Stmt s = rewriter(GetRef<Stmt>(scope_block));
  this->Replace(scope_sref, s, rewriter.block_sref_map_);
  return stmt2ref.at(cache_block.operator->());
}

StmtSRef ScheduleNode::cache_write(const Buffer& buffer, const std::string& storage_scope) {
  /*!
   * Check:
   *   - check the buffer has only one writing block
   *   - check the buffer is not a input buffer
   *
   * Mutate:
   *   - allocate new cache buffer under the current scope.
   *   - find the lowest ancestor of the block and ANY ONE of the producer blocks.
   *   - Copy the buffer with the necessary region.
   */
  StmtSRef block_sref = GetInnermostBlock(this, buffer);
  CHECK(block_sref.defined()) << "Cannot cache_write an input buffer";

  const auto* block = DowncastPtr<BlockNode>(block_sref->node);
  CHECK(block != nullptr) << buffer << "is not a block sref";

  const StmtSRef& scope_sref = GetScope(block_sref);
  const Scope& scope = scopes_.at(scope_sref);
  const auto* scope_block = DowncastPtr<BlockNode>(scope_sref->node);

  // Check there is only one output buffer
  CHECK_EQ(block->writes.size(), 1);

  std::vector<StmtSRef> producers;
  for (const auto& x : scope.GetPredecessors(block_sref)) {
    if (x->type == DepType::kRAW) producers.push_back(x->dst);
  }
  CHECK(!producers.empty());

  // Detector insert position
  CachePositionDetector detector(this, block_sref, producers, 1);
  detector(GetRef<Stmt>(scope_block));

  // Generate cache buffer
  Buffer cache_buffer = CreateCacheBuffer(buffer, storage_scope);

  TensorRegion cache_region = RelaxRegion(block_sref, scope_sref, block->writes[0]);

  auto x = GenerateCopyStmt(cache_buffer, buffer, cache_region);
  Stmt stmt = x.first;
  Block cache_block = x.second;

  BufferAllocate cache_allocate(cache_buffer, storage_scope);
  std::unordered_map<Buffer, Buffer, ObjectHash, ObjectEqual> buffer_map{};
  buffer_map[buffer] = cache_buffer;

  CacheWriteRewriter rewriter(buffer_map, detector.pos_sref_, detector.pos_index_, cache_allocate,
                              stmt);
  Stmt s = rewriter(GetRef<Stmt>(scope_block));

  // Handling block remapping
  Map<Block, Block> block_map = rewriter.block_sref_map_;

  Block replaced_block;
  for (const auto& mapping : block_map) {
    if (mapping.second.operator->() == block) {
      replaced_block = mapping.first;
      break;
    }
  }
  CHECK(replaced_block.defined());
  block_map.Set(cache_block, GetRef<Block>(block));
  block_map.erase(replaced_block);
  this->Replace(scope_sref, s, block_map);
  return stmt2ref.at(replaced_block.operator->());
}

}  // namespace tir
}  // namespace tvm
