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
#include "../utils.h"

namespace tvm {
namespace tir {

/*! \brief The auxilary info used for the insertion point and content of the cache stage */
struct CacheStageInfo {
  /*! \brief The buffer to be read */
  Buffer read_buffer;
  /*! \brief The buffer to be written */
  Buffer write_buffer;
  /*! \brief The buffer allocation statement to be inserted */
  Buffer alloc;
  /*! \brief The AST node whose body is where the cache stage should be inserted */
  StmtSRef loc_sref;
  /*! \brief The index to insert the cache_read/cache_write stage */
  int loc_pos;
  /*! \brief The cache_read/cache_write stage to be inserted */
  Stmt cache_stage;
  /*! \brief The map used for ScheduleStateNode::Replace */
  std::unordered_map<Block, Block, ObjectPtrHash, ObjectPtrEqual> block_map;
};

/*!
 * \brief Create a loop nest that represents cache copy (cache_read / cache_write) from read buffer
 * to write buffer
 * \param cache_region The copy region
 * \param info Put the cache stage created into the info
 * \returns A block indicating the body of the loop nesting.
 * */
Block MakeCacheStage(const BufferRegion& cache_region, CacheStageInfo* info,
                     const String& storage_scope) {
  // loop variables
  std::vector<Var> loop_vars;
  // bindings in block realize
  std::vector<PrimExpr> iter_values;
  // Create loop vars and block vars' binding_value
  for (const Range& axis : cache_region->region) {
    Var loop_var("ax" + std::to_string(loop_vars.size()));
    loop_vars.push_back(loop_var);
    iter_values.push_back(loop_var + axis->min);
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
      /*reads=*/{BufferRegion(info->read_buffer, access_region)},
      /*writes=*/{BufferRegion(info->write_buffer, access_region)},
      /*name_hint=*/cache_region->buffer->name + "_" + storage_scope,
      /*body=*/
      BufferStore(info->write_buffer, BufferLoad(info->read_buffer, copy_indices), copy_indices),
      /*init=*/NullOpt,
      /*alloc_buffers=*/{},
      /*match_buffers=*/{},
      /*annotations=*/{});
  // Create the block realize node
  Stmt body = BlockRealize(/*values=*/iter_values,
                           /*predicate=*/Bool(true),
                           /*block=*/block);
  // Create surrounding loops
  for (int i = static_cast<int>(loop_vars.size()) - 1; i >= 0; --i) {
    body = For(/*loop_var=*/loop_vars[i],
               /*min=*/0,
               /*extent=*/cache_region->region[i]->extent,
               /*kind=*/ForKind::kSerial,
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
BufferRegion GetOnlyWriteRegion(const StmtSRef& block_sref) {
  const auto* block = block_sref->StmtAs<BlockNode>();
  ICHECK(block != nullptr) << "TypeError: Expect a block, but gets: "
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
  const auto* block = block_sref->StmtAs<BlockNode>();
  ICHECK(block != nullptr) << "TypeError: Expect a block, but gets: "
                           << block_sref->stmt->GetTypeKey();
  const auto* scope = scope_sref->StmtAs<BlockNode>();
  ICHECK(scope != nullptr) << "TypeError: Expect a block, but gets: "
                           << scope_sref->stmt->GetTypeKey();
  for (const BufferRegion& x : block->writes) {
    for (const BufferRegion& y : scope->writes) {
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
  ICHECK_EQ(pos, 1);
  return SeqStmt({stmts, stage});
}

/*!
 * \brief Replaces the buffer within the specific sequence of regions
 * \param regions The regions whose buffers are to be replaced
 * \param from The buffer to be replaced
 * \param to The buffer to be replaced to
 * \return The new sequence of regions after replacement
 */
Array<BufferRegion> ReplaceBuffer(const Array<BufferRegion>& regions, const Buffer& from,
                                  const Buffer& to) {
  Array<BufferRegion> copy = regions;
  copy.MutateByApply([&from, &to](BufferRegion region) -> BufferRegion {
    if (region->buffer.same_as(from)) {
      ObjectPtr<BufferRegionNode> n = make_object<BufferRegionNode>(*region.get());
      n->buffer = to;
      return BufferRegion(n);
    }
    return region;
  });
  return copy;
}

/*!
 * \brief Get the innermost block who writes the buffer
 * \param sch The schedule class
 * \param buffer The buffer to be retrieved
 * \param sref The root of the sref
 * \return Schedule root if the block has no writer, or the innermost the sref to the writer block
 * \note This method also checks whether the block is dominate. If not, an exception will be thrown
 */
StmtSRef GetInnermostWriterBlock(const ScheduleState self, const Buffer& buffer, StmtSRef sref) {
  for (;;) {
    BlockScope scope = self->GetBlockScope(sref);
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
   * \param self The schedule class
   * \param block_sref The dominate block which write the buffer
   * \param scope_sref The parent scope of the dominate block
   * \param related_blocks Producer blocks for cache_write and consumer blocks for cache_read
   * \param kind Kind of insertion: for cache_read or cache_write
   */
  CacheLocDetector(const ScheduleState self, const StmtSRef& block_sref, const StmtSRef& scope_sref,
                   const std::vector<StmtSRef>& related_blocks)
      : self_(self),
        block_sref_(block_sref),
        scope_sref_(scope_sref),
        related_blocks_(related_blocks) {}

  void VisitStmt_(const SeqStmtNode* seq_stmt) final {
    bool previous_visited_block = visited_block_;
    bool previous_visited_related = visited_related_;
    visited_block_ = visited_related_ = false;
    // TODO(@junrushao1994): I don't understand the logic here. Revisit later.
    int pos = -1;
    for (size_t i = 0; i < seq_stmt->size(); ++i) {
      if (loc_pos_ != -1) {
        break;
      }
      VisitStmt(seq_stmt->seq[i]);
      // pos can only be assigned once when we visited the block_sref
      if (visited_block_ && visited_related_ && pos == -1) {
        // The offset of insert position from the block
        pos = i;
      }
    }
    visited_block_ = visited_block_ || previous_visited_block;
    visited_related_ = visited_related_ || previous_visited_related;
    // Only we visited the writing block and any one of the related blocks
    // That means that we have found the lowest ancestor
    // of the block and any one of the related ones
    if (visited_block_ && visited_related_ && loc_pos_ == -1) {
      loc_pos_ = pos;
    }
  }

  void VisitStmt_(const BlockNode* block) final {
    // Only visit the current scope under buffer writer's parent block
    if (block == scope_sref_->stmt) {
      // The block vistied is the current parent scope
      StmtVisitor::VisitStmt_(block);
      // Handling cache_read for input buffer
      if (visited_block_ && visited_related_ && !loc_sref_.defined()) {
        loc_sref_ = self_->stmt2ref.at(block);
        if (loc_pos_ == -1) {
          loc_pos_ = 1;
        }
      }
      return;
    }
    // Update `visited_block`
    if (block_sref_->stmt == block) {
      visited_block_ = true;
      return;
    }
    // Update `visited_related`
    for (const StmtSRef& related_block : related_blocks_) {
      if (related_block->stmt == block) {
        visited_related_ = true;
        return;
      }
    }
  }

  void VisitStmt_(const ForNode* loop) final {
    StmtVisitor::VisitStmt_(loop);
    if (visited_block_ && visited_related_ && !loc_sref_.defined() && loc_pos_ != -1) {
      loc_sref_ = self_->stmt2ref.at(loop);
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
  static void Detect(const ScheduleState self, const StmtSRef& block_sref,
                     const StmtSRef& scope_sref, CacheStageInfo* info) {
    std::vector<StmtSRef> related_blocks;
    for (const Dependency& x : self->GetBlockScope(scope_sref)->GetDepsBySrc(block_sref)) {
      if (x->kind == DepKind::kRAW) {
        related_blocks.push_back(x->dst);
      }
    }
    if (!related_blocks.empty()) {
      CacheLocDetector detector(self, block_sref, scope_sref, related_blocks);
      detector(GetRef<Stmt>(scope_sref->stmt));
      info->loc_sref = detector.loc_sref_;
      info->loc_pos = detector.loc_pos_;
    } else {
      info->loc_sref = scope_sref;
      const auto* body = scope_sref->StmtAs<BlockNode>()->body.as<SeqStmtNode>();
      info->loc_pos = body == nullptr ? 1 : body->size();
    }
  }

 private:
  /*! \brief The schedule class */
  const ScheduleState self_;
  /*! \brief The dominate block which write the buffer */
  const StmtSRef& block_sref_;
  /*! \brief The parent scope of the dominate block */
  const StmtSRef& scope_sref_;
  /*! \brief Producer blocks for cache_write and consumer blocks for cache_read */
  const std::vector<StmtSRef>& related_blocks_;
  /*! \brief The flag whether we have visited the dominate block */
  bool visited_block_{false};
  /*! \brief The flag whether we have visited at least one related blocks */
  bool visited_related_{false};
  /*! \brief The AST node whose body is where the cache stage should be inserted */
  StmtSRef loc_sref_{nullptr};
  /*! \brief The index to insert the cache_read/cache_write stage */
  int loc_pos_{-1};
};

bool RelatedWithBuffer(const Array<BufferRegion>& buffer_regions, const Buffer& buffer) {
  for (const auto& region : buffer_regions) {
    if (region->buffer.same_as(buffer)) return true;
  }
  return false;
}

/*! \brief Mutator for CacheRead */
class CacheReadRewriter : public StmtExprMutator {
 public:
  /*!
   * \brief Constructor
   * \param scope_sref The parent scope where the mutator is working on
   * \param info The necessary info for inserting cache stage
   */
  explicit CacheReadRewriter(const StmtSRef& scope_sref, CacheStageInfo* info)
      : scope_sref_(scope_sref), info_(info) {}

  Stmt VisitStmt_(const ForNode* loop) override {
    Stmt stmt = StmtMutator::VisitStmt_(loop);
    // Check the insertion point
    if (loop == info_->loc_sref->stmt) {
      // Insert cache stage into the loop if it is the right place
      ObjectPtr<ForNode> n = make_object<ForNode>(*stmt.as<ForNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Stmt(n);
    }
    return stmt;
  }

  Stmt VisitStmt_(const BlockNode* block) override {
    Block old_stmt = GetRef<Block>(block);
    // We don't mutate the block which generates info->read_buffer
    if (RelatedWithBuffer(block->writes, info_->read_buffer)) {
      return std::move(old_stmt);
    }
    // Mutate the body
    Block stmt = Downcast<Block>(StmtMutator::VisitStmt_(block));
    // Check the insertion point
    if (block == info_->loc_sref->stmt) {
      // Insert cache stage into the block if it is the right place
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Block(n);
    }
    // Check if it is the block corresponding to the parent scope
    if (block == scope_sref_->stmt) {
      // If so, put buffer allocation on the parent scope
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->alloc_buffers.push_back(info_->alloc);
      stmt = Block(n);
    } else {
      // Otherwise, update read/write regions
      auto reads = ReplaceBuffer(block->reads, info_->read_buffer, info_->write_buffer);
      if (!reads.same_as(block->reads)) {
        ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
        n->reads = std::move(reads);
        stmt = Block(n);
      }
    }
    info_->block_map[old_stmt] = stmt;
    return std::move(stmt);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* load) override {
    if (load->buffer.same_as(info_->read_buffer)) {
      ObjectPtr<BufferLoadNode> n = make_object<BufferLoadNode>(*load);
      n->buffer = info_->write_buffer;
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

 private:
  /*! \brief The parent scope of the insertion */
  const StmtSRef& scope_sref_;
  /*! \brief The info for inserting cache stage */
  CacheStageInfo* info_;
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
      : scope_sref_(scope_sref), info_(info) {}

  Stmt VisitStmt_(const ForNode* loop) override {
    Stmt stmt = StmtMutator::VisitStmt_(loop);
    // Check the insertion point
    if (loop == info_->loc_sref->stmt) {
      // Insert cache stage into the loop if it is the right place
      ObjectPtr<ForNode> n = make_object<ForNode>(*stmt.as<ForNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Stmt(n);
    }
    return stmt;
  }

  Stmt VisitStmt_(const BlockNode* block) override {
    Block old_stmt = GetRef<Block>(block);
    // We only mutate the block which generates info->write_buffer
    if (!RelatedWithBuffer(block->writes, info_->write_buffer) && block != scope_sref_->stmt) {
      return std::move(old_stmt);
    }
    // Mutate the body
    Block stmt = Downcast<Block>(StmtMutator::VisitStmt_(block));
    // Find the insertion point
    if (block == info_->loc_sref->stmt) {
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Block(n);
    }
    // Put buffer allocation on the parent scope
    if (block == scope_sref_->stmt) {
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->alloc_buffers.push_back(info_->alloc);
      stmt = Block(n);
    } else {
      // Since cache_write changes the block, we need to update the buffer it writes
      auto writes = ReplaceBuffer(block->writes, info_->write_buffer, info_->read_buffer);
      auto reads = ReplaceBuffer(block->reads, info_->write_buffer, info_->read_buffer);
      if (!writes.same_as(block->writes) || !reads.same_as(block->reads)) {
        ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
        n->writes = std::move(writes);
        n->reads = std::move(reads);
        stmt = Block(n);
      }
    }
    info_->block_map[old_stmt] = stmt;
    return std::move(stmt);
  }

  Stmt VisitStmt_(const BufferStoreNode* store) final {
    BufferStore stmt = Downcast<BufferStore>(StmtMutator::VisitStmt_(store));
    if (stmt->buffer.same_as(info_->write_buffer)) {
      auto n = CopyOnWrite(stmt.get());
      n->buffer = info_->read_buffer;
      return Stmt(n);
    } else {
      return std::move(stmt);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* load) final {
    if (load->buffer.same_as(info_->write_buffer)) {
      auto n = make_object<BufferLoadNode>(*load);
      n->buffer = info_->read_buffer;
      return PrimExpr(n);
    }
    return ExprMutator::VisitExpr_(load);
  }

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

 private:
  /*! \brief The parent scope of the insertion */
  const StmtSRef& scope_sref_;
  /*! \brief The info for inserting cache stage */
  CacheStageInfo* info_;
};

StmtSRef CacheRead(ScheduleState self, const StmtSRef& _block_sref, int i,
                   const String& storage_scope) {
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
  Buffer read_buffer{nullptr};
  {
    const auto* block = _block_sref->StmtAs<BlockNode>();
    CHECK(block) << "ValueError: `cache_read` expects a block as the first argument";
    CHECK_LT(i, block->reads.size()) << "ValueError: index out of range";
    read_buffer = block->reads[i]->buffer;
  }
  StmtSRef root = GetSRefTreeRoot(_block_sref);
  // TODO(@junrushao1994): change it
  StmtSRef block_sref = GetInnermostWriterBlock(self, read_buffer, root);
  CacheStageInfo info;
  info.read_buffer = read_buffer;
  // Create corresponding the buffer to be written, i.e. result of cache_read
  info.write_buffer = read_buffer->WithScope(storage_scope);
  // Create the corresponding buffer allocation
  info.alloc = info.write_buffer;
  // Find the innermost writer to the read buffer
  StmtSRef scope_sref{nullptr};
  BufferRegion cache_region(nullptr);
  if (!block_sref.same_as(root)) {
    // Find the parent scope
    scope_sref = GetScopeRoot(block_sref).value();
    // Check the block is not a output block
    ICHECK(!IsOutputBlock(block_sref, scope_sref));
    // Find the region to be cache_read
    cache_region = RelaxRegion(block_sref, scope_sref, GetOnlyWriteRegion(block_sref));
    // Detect insert position
    CacheLocDetector::Detect(self, block_sref, scope_sref, &info);
  } else {
    info.loc_sref = root;
    info.loc_pos = 0;
    scope_sref = root;
    cache_region = BufferRegion::FullRegion(read_buffer);
  }
  Block cache_read_stage = MakeCacheStage(/*cache_region=*/cache_region, /*info=*/&info,
                                          /*storage_scope=*/storage_scope);
  Stmt new_scope = CacheReadRewriter::Rewrite(/*scope_sref=*/scope_sref, /*info=*/&info);
  self->Replace(scope_sref, new_scope, info.block_map);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_read_stage.get());
  UpdateAffineFlag(self, result_block_sref);
  return result_block_sref;
}

StmtSRef CacheWrite(ScheduleState self, const StmtSRef& block_sref, int i,
                    const String& storage_scope) {
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
  const auto* block = block_sref->StmtAs<BlockNode>();
  CHECK(block) << "ValueError: `cache_write` expects a block as the first argument";
  CHECK_LT(i, block->writes.size()) << "ValueError: index out of range";
  Buffer write_buffer = block->writes[i]->buffer;
  CacheStageInfo info;
  info.write_buffer = write_buffer;
  // Create corresponding the buffer to be read, i.e. result of cache_write
  info.read_buffer = write_buffer->WithScope(storage_scope);
  // Create the corresponding buffer allocation
  info.alloc = info.read_buffer;
  ICHECK(block_sref->parent != nullptr)
      << "ValueError: `cache_write` cannot be applied to an input buffer";
  // Find the parent scope
  StmtSRef scope_sref = GetScopeRoot(block_sref).value();
  CacheLocDetector::Detect(self, block_sref, scope_sref, &info);
  // Generate cache buffer
  Block cache_write_stage = MakeCacheStage(
      /*cache_region=*/RelaxRegion(block_sref, scope_sref, GetOnlyWriteRegion(block_sref)),
      /*info=*/&info, /*storage_scope=*/storage_scope);
  Stmt new_scope = CacheWriteRewriter::Rewrite(/*scope_sref=*/scope_sref, /*info=*/&info);
  // Handling block remapping
  std::unordered_map<Block, Block, ObjectPtrHash, ObjectPtrEqual>& block_map = info.block_map;
  {
    auto it = block_map.find(GetRef<Block>(block));
    ICHECK(it != block_map.end());
    std::swap(it->second, cache_write_stage);
  }
  self->Replace(scope_sref, new_scope, block_map);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_write_stage.get());
  UpdateAffineFlag(self, result_block_sref);
  return result_block_sref;
}

struct CacheReadTraits : public UnpackedInstTraits<CacheReadTraits> {
  static constexpr const char* kName = "CacheRead";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block, Integer i,
                                         String storage_scope) {
    return sch->CacheRead(block, i->value, storage_scope);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Integer i,
                                 String storage_scope) {
    PythonAPICall py("cache_read");
    py.Input("block", block);
    py.Input("i", i->value);
    py.Input("storage_scope", storage_scope);
    py.SingleOutput(outputs);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

struct CacheWriteTraits : public UnpackedInstTraits<CacheWriteTraits> {
  static constexpr const char* kName = "CacheWrite";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block, Integer i,
                                         String storage_scope) {
    return sch->CacheWrite(block, i->value, storage_scope);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Integer i,
                                 String storage_scope) {
    PythonAPICall py("cache_write");
    py.Input("block", block);
    py.Input("i", i->value);
    py.Input("storage_scope", storage_scope);
    py.SingleOutput(outputs);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(CacheReadTraits);
TVM_REGISTER_INST_KIND_TRAITS(CacheWriteTraits);

}  // namespace tir
}  // namespace tvm
