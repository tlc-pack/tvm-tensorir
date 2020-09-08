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
#include "./schedule_common.h"

namespace tvm {
namespace tir {

/*! \brief Kind of the cache stage */
enum class CacheKind : int {
  /*! \brief Indicating that the cache stage is cache_read */
  kCacheRead = 0,
  /*! \brief Indicating that the cache stage is cache_write */
  kCacheWrite = 1,
};

/*! \brief The auxilary info used for the insertion point and content of the cache stage */
struct CacheStageInfo {
  /*! \brief Kind of the cache stage */
  CacheKind kind;
  /*! \brief The buffer to be read */
  Buffer read_buffer;
  /*! \brief The buffer to be written */
  Buffer write_buffer;
  /*! \brief The buffer allocation statement to be inserted */
  BufferAllocate alloc;
  /*! \brief The AST node whose body is where the cache stage should be inserted */
  StmtSRef loc_sref;
  /*! \brief The index to insert the cache_read/cache_write stage */
  int loc_pos;
  /*! \brief The cache_read/cache_write stage to be inserted */
  Stmt cache_stage;
  /*! \brief The map used for ScheduleNode::Replace */
  std::unordered_map<Block, Block, ObjectPtrHash, ObjectPtrEqual> block_map;
};

/*!
 * \brief Create a loop nest that represents cache copy (cache_read / cache_write) from read buffer
 * to write buffer
 * \param cache_region The copy region
 * \param info Put the cache stage created into the info
 * \returns A block indicating the body of the loop nesting.
 * */
Block MakeCacheStage(const TensorRegion& cache_region, CacheStageInfo* info) {
  // loop variables
  std::vector<Var> loop_vars;
  // bindings in block realize
  std::vector<PrimExpr> binding_values;
  // Create loop vars and block vars' binding_value
  for (const Range& axis : cache_region->region) {
    Var loop_var("ax" + std::to_string(loop_vars.size()));
    loop_vars.push_back(loop_var);
    binding_values.push_back(loop_var + axis->min);
  }
  // block variables
  std::vector<IterVar> block_vars;
  // block access region for read/write buffers
  std::vector<Range> access_region;
  // indices used in function body
  std::vector<PrimExpr> copy_indices;
  // Create block vars, block's accessed region and accessing indices
  for (const PrimExpr& dim : cache_region->buffer->shape) {
    Var var("v" + std::to_string(copy_indices.size()));
    block_vars.emplace_back(/*dom=*/Range::FromMinExtent(0, dim),
                            /*var=*/var,
                            /*IterVarType=*/kDataPar);
    copy_indices.push_back(var);
    access_region.push_back(Range::FromMinExtent(var, 1));
  }
  // Create the body block:
  //   reads = [read_buffer[access_region]]
  //   writes = [write_buffer[access_region]]
  //     write_buffer[copy_indices] = read_buffer[copy_indices]
  Block block(
      /*iter_vars=*/block_vars,
      /*reads=*/{TensorRegion(info->read_buffer, access_region)},
      /*writes=*/{TensorRegion(info->write_buffer, access_region)},
      /*body=*/
      BufferStore(info->write_buffer, BufferLoad(info->read_buffer, copy_indices), copy_indices),
      /*allocations=*/{},
      /*annotations=*/{},
      /*tag=*/"");
  // Create the block realize node
  Stmt body = BlockRealize(/*binding_values=*/binding_values,
                           /*predicate=*/Bool(true),
                           /*block=*/block,
                           /*exe_scope=*/"");
  // Create surrounding loops
  for (int i = static_cast<int>(loop_vars.size()) - 1; i >= 0; --i) {
    body = Loop(/*loop_var=*/loop_vars[i],
                /*min=*/0,
                /*extent=*/cache_region->region[i]->extent,
                /*annotations=*/{},
                /*body=*/body);
  }
  info->cache_stage = std::move(body);
  return block;
}

/*!
 * \brief Get the only write region of the block
 * \param block_sref The block to be queried
 * \return The only region the block writes
 */
TensorRegion GetOnlyWriteRegion(const StmtSRef& block_sref) {
  const auto* block = block_sref->GetStmt<BlockNode>();
  CHECK(block != nullptr) << "TypeError: Expect a block, but gets: "
                          << block_sref->stmt->GetTypeKey();
  CHECK_EQ(block->writes.size(), 1) << "ValueError: Only one write buffer is allowed in the block";
  return block->writes[0];
}

/*!
 * \brief Checks if the block is an output block
 * \param block_sref The block to be checked
 * \param scope_sref The parent scope the block is in
 * \return A boolean indicating if the block is an output block
 */
bool IsOutputBlock(const StmtSRef& block_sref, const StmtSRef& scope_sref) {
  const auto* block = block_sref->GetStmt<BlockNode>();
  CHECK(block != nullptr) << "TypeError: Expect a block, but gets: "
                          << block_sref->stmt->GetTypeKey();
  const auto* scope = scope_sref->GetStmt<BlockNode>();
  CHECK(scope != nullptr) << "TypeError: Expect a block, but gets: "
                          << scope_sref->stmt->GetTypeKey();
  for (const TensorRegion& x : block->writes) {
    for (const TensorRegion& y : scope->writes) {
      if (x->buffer.same_as(y->buffer)) {
        return true;
      }
    }
  }
  return false;
}

/*!
 * \brief Insert the cache_read/cache_write stage into the specific position into to get the SeqStmt
 * \param stmts A sequence of statements, or a single statement, to be inserted into
 * \param pos The position the cache stage to be put
 * \param stage The stage to be inserted
 * \return A SeqStmt, the result after insertion
 */
SeqStmt InsertCacheStage(const Stmt& stmts, int pos, const Stmt& stage) {
  if (const auto* seq_stmt = stmts.as<SeqStmtNode>()) {
    ObjectPtr<SeqStmtNode> result = make_object<SeqStmtNode>(*seq_stmt);
    result->seq.insert(result->seq.begin() + pos, stage);
    return SeqStmt(result);
  }
  if (pos == 0) {
    return SeqStmt({stage, stmts});
  }
  CHECK_EQ(pos, 1);
  return SeqStmt({stmts, stage});
}

/*!
 * \brief Replaces the buffer within the specific sequence of regions
 * \param regions The regions whose buffers are to be replaced
 * \param from The buffer to be replaced
 * \param to The buffer to be replaced to
 * \return The new sequence of regions after replacement
 */
Array<TensorRegion> ReplaceBuffer(const Array<TensorRegion>& regions, const Buffer& from,
                                  const Buffer& to) {
  Array<TensorRegion> copy = regions;
  copy.MutateByApply([&from, &to](TensorRegion region) -> TensorRegion {
    if (region->buffer.same_as(from)) {
      ObjectPtr<TensorRegionNode> n = make_object<TensorRegionNode>(*region.get());
      n->buffer = to;
      return TensorRegion(n);
    }
    return region;
  });
  return copy;
}

/*!
 * \brief Get the innermost block who writes the buffer
 * \param sch The schedule class
 * \param buffer The buffer to be retrieved
 * \return Schedule root if the block has no writer, or the innermost the sref to the writer block
 * \note This method also checks whether the block is dominate. If not, an exception will be thrown
 */
StmtSRef GetInnermostWriterBlock(const ScheduleNode* sch, const Buffer& buffer) {
  StmtSRef sref = sch->root;
  for (;;) {
    Scope scope = sch->scopes.at(sref);
    auto it = scope->buffer_writers.find(buffer);
    if (it == scope->buffer_writers.end()) {
      break;
    }
    const Array<StmtSRef>& write_blocks = it->second;
    CHECK_EQ(write_blocks.size(), 1) << "ValueError: Can only apply `cache_read` or `cache_write` "
                                        "on a dominate block (only producer)";
    sref = write_blocks[0];
  }
  return sref;
}

/*! \brief Detect the insert position of buffer copy */
class CacheLocDetector : public StmtVisitor {
 public:
  /*!
   * \brief Constructor
   * \param sch The schedule class
   * \param block_sref The dominate block which write the buffer
   * \param scope_sref The parent scope of the dominate block
   * \param related_blocks Producer blocks for cache_write and consumer blocks for cache_read
   * \param kind Kind of insertion: for cache_read or cache_write
   */
  CacheLocDetector(const ScheduleNode* sch, const StmtSRef& block_sref, const StmtSRef& scope_sref,
                   const std::vector<StmtSRef>& related_blocks, CacheKind kind)
      : sch(sch),
        block_sref(block_sref),
        scope_sref(scope_sref),
        related_blocks(related_blocks),
        kind(kind),
        visited_block(false),
        visited_related(false),
        loc_sref(nullptr),
        loc_pos(-1) {}

  void VisitStmt_(const SeqStmtNode* seq_stmt) final {
    bool previous_visited_block = visited_block;
    bool previous_visited_related = visited_related;
    visited_block = visited_related = false;
    // TODO(@junrushao1994): I don't understand the logic here. Revisit later.
    int pos = -1;
    for (size_t i = 0; i < seq_stmt->size(); ++i) {
      if (loc_pos != -1) {
        break;
      }
      VisitStmt(seq_stmt->seq[i]);
      // pos can only be assigned once when we visited the block_sref
      if (visited_block && visited_related && pos == -1) {
        // The offset of insert position from the block
        // offset 0 for cache_read and 1 for cache_write
        // e.g the block itself locates at the n-th element of the SeqStmt,
        // so, the buffer copy stmt should be inserted at n+offset position
        // (2rd when cache_read or at 3rd when cache_write)
        if (kind == CacheKind::kCacheRead) {
          pos = i;
        } else {
          pos = i + 1;
        }
      }
    }
    visited_block = visited_block || previous_visited_block;
    visited_related = visited_related || previous_visited_related;
    // Only we visited the writing block and any one of the related blocks
    // That means that we have found the lowest ancestor
    // of the block and any one of the related ones
    if (visited_block && visited_related) {
      loc_pos = pos;
    }
  }

  void VisitStmt_(const BlockNode* block) final {
    // Only visit the current scope under buffer writer's parent block
    if (block == scope_sref->stmt) {
      // The block vistied is the current parent scope
      StmtVisitor::VisitStmt_(block);
      // Handling cache_read for input buffer
      if (visited_block && visited_related && !loc_sref.defined()) {
        loc_sref = sch->stmt2ref.at(block);
        if (loc_pos == -1) {
          loc_pos = 1;
        }
      }
      return;
    }
    // Update `visited_block`
    if (block_sref->stmt == block) {
      visited_block = true;
      return;
    }
    // Update `visited_related`
    for (const StmtSRef& related_block : related_blocks) {
      if (related_block->stmt == block) {
        visited_related = true;
        return;
      }
    }
  }

  void VisitStmt_(const LoopNode* loop) final {
    StmtVisitor::VisitStmt_(loop);
    if (visited_block && visited_related && !loc_sref.defined() && loc_pos != -1) {
      loc_sref = sch->stmt2ref.at(loop);
    }
  }

  /*!
   * \brief Detect the position for adding the cache stage
   * \param sch The schedule class
   * \param block_sref The innermost writer block
   * \param scope_sref The parent scope of the block
   * \param kind Kind of the cache stage, i.e. cache_read or cache_write
   * \return The location to insert the cache stage
   */
  static void Detect(const ScheduleNode* sch, const StmtSRef& block_sref,
                     const StmtSRef& scope_sref, CacheStageInfo* info) {
    std::vector<StmtSRef> related_blocks;
    // cache_read => consumer
    // cache_write => producer
    for (const DepEdge& x : (info->kind == CacheKind::kCacheRead)
                                ? sch->scopes.at(scope_sref).GetSuccessors(block_sref)
                                : sch->scopes.at(scope_sref).GetPredecessors(block_sref)) {
      if (x->type == DepType::kRAW) {
        related_blocks.push_back(x->dst);
      }
    }
    CHECK(!related_blocks.empty());
    CacheLocDetector detector(sch, block_sref, scope_sref, related_blocks, info->kind);
    detector(GetRef<Stmt>(scope_sref->stmt));
    info->loc_sref = detector.loc_sref;
    info->loc_pos = detector.loc_pos;
  }

  /*! \brief The schedule class */
  const ScheduleNode* sch;
  /*! \brief The dominate block which write the buffer */
  const StmtSRef& block_sref;
  /*! \brief The parent scope of the dominate block */
  const StmtSRef& scope_sref;
  /*! \brief Producer blocks for cache_write and consumer blocks for cache_read */
  const std::vector<StmtSRef>& related_blocks;
  /*! \brief Kind of insertion: for cache_read or cache_write */
  CacheKind kind;
  /*! \brief The flag whether we have visited the dominate block */
  bool visited_block;
  /*! \brief The flag whether we have visited at least one related blocks */
  bool visited_related;
  /*! \brief The AST node whose body is where the cache stage should be inserted */
  StmtSRef loc_sref;
  /*! \brief The index to insert the cache_read/cache_write stage */
  int loc_pos;
};

/*! \brief Mutator for CacheRead */
class CacheReadRewriter : public StmtExprMutator {
 public:
  /*!
   * \brief Constructor
   * \param scope_sref The parent scope where the mutator is working on
   * \param info The necessary info for inserting cache stage
   */
  explicit CacheReadRewriter(const StmtSRef& scope_sref, CacheStageInfo* info)
      : scope_sref(scope_sref), info(info) {}

  Stmt VisitStmt_(const LoopNode* loop) override {
    Stmt stmt = StmtMutator::VisitStmt_(loop);
    // Check the insertion point
    if (loop == info->loc_sref->stmt) {
      // Insert cache stage into the loop if it is the right place
      ObjectPtr<LoopNode> n = make_object<LoopNode>(*stmt.as<LoopNode>());
      n->body = InsertCacheStage(n->body, info->loc_pos, info->cache_stage);
      stmt = Stmt(n);
    }
    return stmt;
  }

  Stmt VisitStmt_(const BlockNode* block) override {
    Block old_stmt = GetRef<Block>(block);
    Block stmt = Downcast<Block>(StmtMutator::VisitStmt_(block));
    // Check the insertion point
    if (block == info->loc_sref->stmt) {
      // Insert cache stage into the block if it is the right place
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->body = InsertCacheStage(n->body, info->loc_pos, info->cache_stage);
      stmt = Block(n);
    }
    // Check if it is the block correpsonding to the parent scope
    if (block == scope_sref->stmt) {
      // If so, put buffer allocation on the parent scope
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->allocations.push_back(info->alloc);
      stmt = Block(n);
    } else {
      // Otherwise, update read/write regions
      auto writes = ReplaceBuffer(block->writes, info->read_buffer, info->write_buffer);
      auto reads = ReplaceBuffer(block->reads, info->read_buffer, info->write_buffer);
      if (!writes.same_as(block->writes) || !reads.same_as(block->reads)) {
        ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
        n->writes = std::move(writes);
        n->reads = std::move(reads);
        stmt = Block(n);
      }
    }
    info->block_map[stmt] = old_stmt;
    return std::move(stmt);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* load) override {
    if (load->buffer.same_as(info->read_buffer)) {
      ObjectPtr<BufferLoadNode> n = CopyOnWrite(load);
      n->buffer = info->write_buffer;
      return PrimExpr(n);
    }
    return ExprMutator::VisitExpr_(load);
  }

  /*!
   * \brief Rewrite the AST and add a cache_read stage with the information provided
   * \param scope_sref The parent scope of this mutation
   * \param info The cache stage information
   * \return The new AST rooting at the original parent scope
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, CacheStageInfo* info) {
    CacheReadRewriter rewriter(scope_sref, info);
    return rewriter(GetRef<Stmt>(scope_sref->stmt));
  }

  /*! \brief The parent scope of the insertion */
  const StmtSRef& scope_sref;
  /*! \brief The info for inserting cache stage */
  CacheStageInfo* info;
};

/*! \brief Mutator for CacheWrite */
class CacheWriteRewriter : public StmtExprMutator {
 public:
  /*!
   * \brief Constructor
   * \param scope_sref The parent scope where the mutator is working on
   * \param info The necessary info for inserting cache stage
   */
  explicit CacheWriteRewriter(const StmtSRef& scope_sref, CacheStageInfo* info)
      : scope_sref(scope_sref), info(info) {}

  Stmt VisitStmt_(const LoopNode* loop) override {
    Stmt stmt = StmtMutator::VisitStmt_(loop);
    // Check the insertion point
    if (loop == info->loc_sref->stmt) {
      // Insert cache stage into the loop if it is the right place
      ObjectPtr<LoopNode> n = make_object<LoopNode>(*stmt.as<LoopNode>());
      n->body = InsertCacheStage(n->body, info->loc_pos, info->cache_stage);
      stmt = Stmt(n);
    }
    return stmt;
  }

  Stmt VisitStmt_(const BlockNode* block) override {
    Block old_stmt = GetRef<Block>(block);
    Block stmt = Downcast<Block>(StmtMutator::VisitStmt_(block));
    // Find the insertion point
    if (block == info->loc_sref->stmt) {
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->body = InsertCacheStage(n->body, info->loc_pos, info->cache_stage);
      stmt = Block(n);
    }
    // Put buffer allocation on the parent scope
    if (block == scope_sref->stmt) {
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->allocations.push_back(info->alloc);
      stmt = Block(n);
    } else {
      // Since cache_write changes the block, we need to update the buffer it writes
      auto writes = ReplaceBuffer(block->writes, info->write_buffer, info->read_buffer);
      auto reads = ReplaceBuffer(block->reads, info->write_buffer, info->read_buffer);
      if (!writes.same_as(block->writes) || !reads.same_as(block->reads)) {
        ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
        n->writes = std::move(writes);
        n->reads = std::move(reads);
        stmt = Block(n);
      }
    }
    info->block_map[stmt] = old_stmt;
    return std::move(stmt);
  }

  Stmt VisitStmt_(const BufferStoreNode* store) final {
    if (store->buffer.same_as(info->write_buffer)) {
      auto n = CopyOnWrite(store);
      n->buffer = info->read_buffer;
      return Stmt(n);
    }
    return StmtMutator::VisitStmt_(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* load) final {
    if (load->buffer.same_as(info->write_buffer)) {
      auto n = CopyOnWrite(load);
      n->buffer = info->read_buffer;
      return PrimExpr(n);
    }
    return ExprMutator::VisitExpr_(load);
  }

  /*! \brief The parent scope of the insertion */
  const StmtSRef& scope_sref;
  /*! \brief The info for inserting cache stage */
  CacheStageInfo* info;

  /*!
   * \brief Rewrite the AST and add a cache_write stage with the information provided
   * \param scope_sref The parent scope of this mutation
   * \param info The cache stage information
   * \return The new AST rooting at the original parent scope
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, CacheStageInfo* info) {
    CacheWriteRewriter rewriter(scope_sref, info);
    return rewriter(GetRef<Stmt>(scope_sref->stmt));
  }
};

StmtSRef ScheduleNode::cache_read(const Buffer& read_buffer, const std::string& storage_scope) {
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
  CacheStageInfo info;
  info.kind = CacheKind::kCacheRead;
  info.read_buffer = read_buffer;
  // Create corresponding the buffer to be written, i.e. result of cache_read
  info.write_buffer = read_buffer->WithScope(storage_scope);
  // Create the corresponding buffer allocation
  info.alloc = BufferAllocate(info.write_buffer, storage_scope);
  // Find the innermost writer to the read buffer
  StmtSRef block_sref = GetInnermostWriterBlock(this, read_buffer);

  StmtSRef scope_sref{nullptr};
  TensorRegion cache_region(nullptr);
  if (!block_sref.same_as(this->root)) {
    // Check the block is not a output block
    CHECK(!IsOutputBlock(block_sref, scope_sref));
    // Find the parent scope
    scope_sref = GetParentBlockSRef(block_sref);
    // Find the region to be cache_read
    cache_region = RelaxRegion(block_sref, scope_sref, GetOnlyWriteRegion(block_sref));
    // Detector insert position
    CacheLocDetector::Detect(this, block_sref, scope_sref, &info);
  } else {
    info.loc_sref = this->root;
    info.loc_pos = 0;
    scope_sref = this->root;
    cache_region = TensorRegion(read_buffer);
  }
  Block cache_read_stage = MakeCacheStage(/*cache_region=*/cache_region, /*info=*/&info);
  Stmt new_scope = CacheReadRewriter::Rewrite(/*scope_sref=*/scope_sref, /*info=*/&info);
  this->Replace(scope_sref, new_scope, info.block_map);
  return stmt2ref.at(cache_read_stage.get());
}

StmtSRef ScheduleNode::cache_write(const Buffer& write_buffer, const std::string& storage_scope) {
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
  CacheStageInfo info;
  info.kind = CacheKind::kCacheWrite;
  info.write_buffer = write_buffer;
  // Create corresponding the buffer to be read, i.e. result of cache_write
  info.read_buffer = write_buffer->WithScope(storage_scope);
  // Create the corresponding buffer allocation
  info.alloc = BufferAllocate(info.read_buffer, storage_scope);
  // Find the innermost writer to the write buffer
  StmtSRef block_sref = GetInnermostWriterBlock(this, write_buffer);
  CHECK(!block_sref.same_as(this->root))
      << "ValueError: `cache_write` cannot be applied to an input buffer";
  // Find the parent scope
  StmtSRef scope_sref = GetParentBlockSRef(block_sref);
  CacheLocDetector::Detect(this, block_sref, scope_sref, &info);
  // Generate cache buffer
  Block cache_write_stage = MakeCacheStage(
      /*cache_region=*/RelaxRegion(block_sref, scope_sref, GetOnlyWriteRegion(block_sref)),
      /*info=*/&info);
  Stmt new_scope = CacheWriteRewriter::Rewrite(/*scope_sref=*/scope_sref, /*info=*/&info);
  // Handling block remapping
  std::unordered_map<Block, Block, ObjectPtrHash, ObjectPtrEqual>& block_map = info.block_map;
  for (const auto& mapping : block_map) {
    const Block& new_block = mapping.first;
    const Block& old_block = mapping.second;
    if (old_block.get() == block_sref->stmt) {
      // It is okay to mutate inside iteration, because it is going to break anyways
      block_map[cache_write_stage] = old_block;
      cache_write_stage = new_block;
      block_map.erase(new_block);
      break;
    }
  }
  this->Replace(scope_sref, new_scope, block_map);
  return stmt2ref.at(cache_write_stage.get());
}

}  // namespace tir
}  // namespace tvm
