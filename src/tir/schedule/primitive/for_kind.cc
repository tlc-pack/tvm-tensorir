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

/*!
 * \brief Checks if a loop variable is parallelizable.
 * \param loop_var The loop variable
 * \param block_realize The block realize node under the loop. It is possible that there are
 * multiple blocks, and in this case, we should invoke this function multiple times.
 * \param schedule The schedule object
 * \param anno_value The annotation anno_value
 * \return A boolean indicating if the loop var is parallelizable
 */
bool IsLoopVarParallelizable(const ScheduleState self, const Var& loop_var,
                             const Stmt& block_realize, const Optional<IterVar>& thread_binding) {
  const BlockRealizeNode* realize = block_realize.as<BlockRealizeNode>();
  ICHECK(realize != nullptr)
      << "InternalError: in IsLoopVarParallelizable, expect BlockRealize, but get type: "
      << block_realize->GetTypeKey();
  const BlockNode* block = realize->block.get();
  // Cond 1. Binding is validated
  // TODO(auto-tir-team): affine
  // if (!self->stmt2ref.at(block)->affine) {
  //   return false;
  // }
  CHECK_EQ(realize->iter_values.size(), block->iter_vars.size())
      << "InternalError: BlockRealize is inconsistent with its Block";
  int n = realize->iter_values.size();
  // Cond 2. For each iter var that is not data parallel, the binding does not involve loop_var
  std::string thread_tag = thread_binding.defined() ? thread_binding.value()->thread_tag : "";
  for (int i = 0; i < n; ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    const PrimExpr& binding = realize->iter_values[i];
    bool contains = StmtExprContainsVar(binding, loop_var);
    if (contains && iter_var->iter_type != kDataPar && iter_var->iter_type != kCommReduce) {
      return false;
    }
    if (contains && iter_var->iter_type == kCommReduce && thread_tag.substr(0, 9) != "threadIdx") {
      return false;
    }
  }
  return true;
}

/*!
 * \brief Create a new loop with the given annotation added
 * \param loop The loop with original annotation
 * \param attr_key The annotation key to be added
 * \param attr_value The annotation value to be added
 * \return A new loop with the given annotation as its last annotation
 */
For WithAnnotation(const ForNode* loop, const String& attr_key, const ObjectRef& attr_value) {
  Map<String, ObjectRef> annotations = loop->annotations;
  annotations.Set(attr_key, attr_value);
  ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loop);
  new_loop->annotations = std::move(annotations);
  return For(new_loop);
}

/*!
 * \brief Create a new block with the given annotation added
 * \param block The block with original annotation
 * \param attr_key The annotation key to be added
 * \param attr_value The annotation value to be added
 * \return A new block with the given annotation as its last annotation
 */
Block WithAnnotation(const BlockNode* block, const String& attr_key, const ObjectRef& attr_value) {
  Map<String, ObjectRef> annotations = block->annotations;
  annotations.Set(attr_key, attr_value);
  ObjectPtr<BlockNode> new_block = make_object<BlockNode>(*block);
  new_block->annotations = std::move(annotations);
  return Block(new_block);
}

void ParallelCompute(ScheduleState self, const StmtSRef& loop_sref, const ForKind& for_kind,
                     const Optional<IterVar>& thread_binding) {
  /*!
   * Check:
   * - 1. check the block under is complete block or reduction block
   * - 2. check `input_loop` is bound and only bound to `data_par` block_vars
   * - 3. check the loops of reduction blocks are validatable
   * Mutate:
   * - 4. set Annotation on the loop
   * Proof:
   * We prove by showing that there are no data flows between `input_loop=i` and`input_loop=j`,
   * and we show this by induction on the number of blocks.
   *
   * If there is only one block below
   * - The block is complete. All the instances are independent of each other.
   * - The block is reduction. `input_loop` bound and only bound to `data_par` blocks + loops of
   * reduction blocks are validatable => instances of `input_loop=i` will write different positions
   * with instances of `input_loop=j`, hence they are independent.
   *
   * If there's a new block coming in. Consider its instances under `input_loop=i`.
   * - If the producer is complete. Producer instances under `input_loop=j` may write the positions
   * that new instances under `input_loop=i`  may read, but it will read the same value produced by
   * the producer under `input_loop=i` since it's complete.
   * - If the producer is reduction. Producer instances under `input_loop=j` will never write the
   * positions that new instances under `input_loop=j` may read. Hence no data flow.
   */
  const auto* loop = loop_sref->StmtAs<ForNode>();
  CHECK(loop != nullptr) << "TypeError: Parallel compute applies only to a loop, but get: "
                         << loop_sref->stmt->GetTypeKey();
  // Now only support:
  //   1. All the blocks are complete below
  //   2. A single block below the loop
  StmtSRef scope_root = GetScopeRoot(loop_sref).value();
  bool is_compact_dataflow = IsCompactDataFlow(self, scope_root, GetChildBlocks(self, loop_sref));
  if (!is_compact_dataflow) {
    Array<Stmt> single_child = GetChildren(GetRef<Stmt>(loop), true);
    // TODO(@junrushao1994): I am not super convinced by the checks here, revisit later
    CHECK(single_child.size() == 1)
        << "ValueError: loop with variable \"" << loop->loop_var << "\" cannot be parallelized, "
        << "because it does not satisfy one-way fine-grained dataflow "
           "condition, and has more than 1 child block";
    const auto* realize = single_child[0].as<BlockRealizeNode>();
    CHECK(realize != nullptr) << "TypeError: Expects 'BlockRealizeNode', but gets: "
                              << single_child[0]->GetTypeKey();
    CHECK(IsLoopVarParallelizable(self, loop->loop_var, GetRef<Stmt>(realize), thread_binding))
        << "ValueError: loop with variable \"" << loop->loop_var
        << "\" cannot be parallelized because of block:\n"
        << GetRef<Stmt>(realize);
  } else {
    PreOrderVisit(GetRef<Stmt>(loop), [self, &loop, thread_binding](const ObjectRef& node) {
      if (const auto* realize = node.as<BlockRealizeNode>()) {
        CHECK(IsLoopVarParallelizable(self, loop->loop_var, GetRef<Stmt>(realize), thread_binding))
            << "ValueError: loop with variable \"" << loop->loop_var
            << "\" cannot be parallelized because of block:\n"
            << GetRef<Stmt>(realize);
        return false;
      }
      return true;
    });
  }
  ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loop);
  new_loop->kind = for_kind;
  if (thread_binding.defined()) {
    new_loop->thread_binding = thread_binding;
  }
  self->Replace(loop_sref, For(new_loop), {});
}

/*!
 * \brief A helper mutator which recursively mutates the old buffer's storage scope and collects
 *         the block sref reuse information for the following replacement.
 */
class StorageScopeMutator : StmtExprMutator {
 public:
  /*!
   * \param allocate_site The block where `old_buffer` was allocated.
   * \param old_buffer The old buffer
   * \param storage_scope The storage scope to be set
   * \param block_sref_reuse The block sref reuse map to be updated
   * \return The new block after the mutation
   */
  static Block Mutate(const Block& allocate_site, const Buffer& old_buffer,
                      const String& storage_scope, Map<Block, Block>* block_sref_reuse) {
    Buffer new_buffer = old_buffer->WithScope(storage_scope);
    StorageScopeMutator mutator(old_buffer, new_buffer, storage_scope, block_sref_reuse);
    Stmt new_block = mutator.VisitStmt(allocate_site);
    return Downcast<Block>(new_block);
  }

 private:
  StorageScopeMutator(const Buffer& old_buffer, Buffer new_buffer, String storage_scope,
                      Map<Block, Block>* block_sref_reuse)
      : storage_scope(std::move(storage_scope)), block_sref_reuse_(block_sref_reuse) {
    buffer_map_[old_buffer.get()] = std::move(new_buffer);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr res = ExprMutator::VisitExpr_(op);
    op = res.as<BufferLoadNode>();
    ICHECK(op);
    auto it = buffer_map_.find(op->buffer.get());
    if (it != buffer_map_.end()) {
      ObjectPtr<BufferLoadNode> ptr = make_object<BufferLoadNode>(*op);
      ptr->buffer = it->second;
      return PrimExpr(ptr);
    } else {
      return res;
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt res = StmtMutator::VisitStmt_(op);
    auto it = buffer_map_.find(op->buffer.get());
    if (it != buffer_map_.end()) {
      ObjectPtr<BufferStoreNode> ptr = CopyOnWrite(res.as<BufferStoreNode>());
      ptr->buffer = it->second;
      return Stmt(ptr);
    } else {
      return res;
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    // To reduce the number of blocks in block sref reuse map, we check whether the block is really
    // mutated (i.e., the old buffer appears in the block). If so, we return the block after
    // mutation. Otherwise we just return the original block.
    bool changed = false;
    // Step 1. Mutate the read region.
    Array<BufferRegion> reads;
    for (const BufferRegion& read : op->reads) {
      auto it = buffer_map_.find(read->buffer.get());
      if (it != buffer_map_.end()) {
        changed = true;
        reads.push_back(BufferRegion(it->second, read->region));
      } else {
        reads.push_back(read);
      }
    }
    // Step 2. Mutate the write region.
    Array<BufferRegion> writes;
    for (const BufferRegion& write : op->writes) {
      auto it = buffer_map_.find(write->buffer.get());
      if (it != buffer_map_.end()) {
        changed = true;
        writes.push_back(BufferRegion(it->second, write->region));
      } else {
        writes.push_back(write);
      }
    }
    // Step 3. Mutate `alloc_buffers` for the old buffer allocated in this block.
    Array<Buffer> alloc_buffers;
    for (const Buffer& buffer : op->alloc_buffers) {
      auto it = buffer_map_.find(buffer.get());
      if (it != buffer_map_.end()) {
        changed = true;
        alloc_buffers.push_back(it->second);
      } else {
        alloc_buffers.push_back(buffer);
      }
    }
    // Step 4. Mutate `match_buffers`. If an old buffer appears as a source of MatchBufferRegion,
    // the storage scope of the target buffer also needs to be set.
    Array<MatchBufferRegion> match_buffers;
    for (const MatchBufferRegion& match_buffer : op->match_buffers) {
      auto it = buffer_map_.find(match_buffer->source->buffer.get());
      if (it != buffer_map_.end()) {
        changed = true;
        Buffer new_target_buffer = match_buffer->buffer->WithScope(storage_scope);
        buffer_map_[match_buffer->buffer.get()] = new_target_buffer;
        match_buffers.push_back(MatchBufferRegion(
            new_target_buffer, BufferRegion(it->second, match_buffer->source->region)));
      } else {
        match_buffers.push_back(match_buffer);
      }
    }
    // Step 5. Recursively mutate the block.
    Stmt res = StmtMutator::VisitStmt_(op);
    if (res.get() != op) {
      changed = true;
    }

    if (changed) {
      ObjectPtr<BlockNode> block = CopyOnWrite(res.as<BlockNode>());
      block->reads = std::move(reads);
      block->writes = std::move(writes);
      block->alloc_buffers = std::move(alloc_buffers);
      block->match_buffers = std::move(match_buffers);
      block_sref_reuse_->Set(GetRef<Block>(op), Block(block));
      return Stmt(block);
    } else {
      return GetRef<Block>(op);
    }
  }

  /*! \brief The storage scope to be set. */
  String storage_scope;
  /*! \brief A mapping which maps old buffers to new buffers, including the buffers defined in
   *         MatchBufferRegion.*/
  std::unordered_map<const BufferNode*, Buffer> buffer_map_;
  /*! \brief The block sref reuse map for the following replacement */
  Map<Block, Block>* block_sref_reuse_;
};

void Vectorize(ScheduleState self, const StmtSRef& loop_sref) {
  if (is_one(loop_sref->StmtAs<ForNode>()->extent)) {
    return;
  }
  ParallelCompute(self, loop_sref, ForKind::kVectorized, NullOpt);
}

void Parallel(ScheduleState self, const StmtSRef& loop_sref) {
  ParallelCompute(self, loop_sref, ForKind::kParallel, NullOpt);
}

void Unroll(ScheduleState self, const StmtSRef& loop_sref) {
  const auto* loop = loop_sref->StmtAs<ForNode>();
  CHECK(loop != nullptr) << "TypeError: Unroll expects a loop, but get type: "
                         << loop_sref->stmt->GetTypeKey();
  ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loop);
  new_loop->kind = ForKind::kUnrolled;
  self->Replace(loop_sref, For(new_loop), {});
}

void Bind(ScheduleState self, const StmtSRef& loop_sref, const IterVar& thread) {
  const auto* loop = loop_sref->StmtAs<ForNode>();
  CHECK(loop != nullptr) << "Parallel-like compute expect a loop";
  if (thread->dom.defined()) {
    CHECK(ExprDeepEqual()(loop->extent, thread->dom->extent))
        << "Thread axis extent and loop extent mismatch";
  }
  ParallelCompute(self, loop_sref, ForKind::kThreadBinding, thread);
}

void DoubleBuffer(ScheduleState self, const StmtSRef& block_sref) {
  const auto* block_ptr = block_sref->StmtAs<BlockNode>();
  CHECK(block_ptr) << "TypeError: double_buffer expects 'block' as its argument";
  StmtSRef parent_block_sref = GetScopeRoot(block_sref).value();
  const auto* parent_block = parent_block_sref->StmtAs<BlockNode>();
  CHECK(CompleteBlock(self, block_sref, parent_block_sref))
      << "ValueError: 'double_buffer' expects 'block' to be a complete block";
  for (const BufferRegion& parent_write : parent_block->writes) {
    for (const BufferRegion& write : block_ptr->writes) {
      CHECK_NE(write->buffer.get(), parent_write->buffer.get())
          << "ValueError: 'double_buffer' does not work on an output block";
    }
  }
  CHECK_EQ(block_ptr->writes.size(), 1)
      << "ValueError: 'double_buffer' expects 'block' with only one write buffer";
  Block new_block =
      WithAnnotation(block_ptr, tir::attr::double_buffer_scope, IntImm(DataType::Int(32), 1));
  self->Replace(block_sref, new_block, {{GetRef<Block>(block_ptr), new_block}});
}

/*!
 * \brief Find the defining site of the buffer in the given block and its ancestors
 * \param block_sref The block sref
 * \param buffer The buffer
 * \return The defining site of the buffer and whether the buffer is allocated (otherwise the
 *         buffer is from match_buffer).
 */
std::pair<StmtSRef, bool> GetBufferDefiningSite(const StmtSRef& block_sref, const Buffer& buffer) {
  // Climb up along the sref tree, and find the block where `buffer` is in alloc_buffers or
  // match_buffers.
  const StmtSRefNode* defining_site_sref = block_sref.get();
  while (defining_site_sref != nullptr) {
    const auto* block = defining_site_sref->StmtAs<BlockNode>();
    // If this sref is not a block sref, skip it.
    if (block == nullptr) {
      defining_site_sref = defining_site_sref->parent;
      continue;
    }
    // Try to find the buffer in `allloc_buffers`
    for (const Buffer& alloc_buffer : block->alloc_buffers) {
      if (buffer.same_as(alloc_buffer)) {
        return {GetRef<StmtSRef>(defining_site_sref), true};
      }
    }
    // We do not allow the buffer being defined in `match_buffer`.
    for (const MatchBufferRegion match_buffer : block->match_buffers) {
      if (buffer.same_as(match_buffer)) {
        return {GetRef<StmtSRef>(defining_site_sref), false};
      }
    }
    defining_site_sref = defining_site_sref->parent;
  }
  // If we cannot find the defining site block, it means that the buffer must be in the function's
  // buffer_map, which isn't an intermediate buffer. In this case we should report error.
  LOG(FATAL)
      << "ValueError: The buffer is expected to be an intermediate buffer defined in some block";
  throw;
}

void SetScope(ScheduleState self, const StmtSRef& block_sref, int i, const String& storage_scope) {
  const auto* block_ptr = block_sref->StmtAs<BlockNode>();
  CHECK(block_ptr) << "TypeError: set_scope expects a block as its first argument";
  CHECK_GE(i, 0) << "ValueError: index out of range";
  CHECK_LT(i, block_ptr->writes.size()) << "ValueError: index out of range";
  Buffer buffer = block_ptr->writes[i]->buffer;
  // If the `storage_scope` equals the original storage scope of the buffer, just return.
  if (buffer.scope() == storage_scope) {
    return;
  }
  StmtSRef allocate_site_sref;
  bool is_alloc;
  std::tie(allocate_site_sref, is_alloc) = GetBufferDefiningSite(block_sref, buffer);
  // We do not allow the buffer being defined in `match_buffer`.
  CHECK(is_alloc) << "ValueError: Set the storage scope of a buffer defined in MatchBufferRegion is"
                     " not allowed. You might want to set the storage scope of its source buffer if"
                     " you really want to change its storage scope.";
  const auto* allocate_site = allocate_site_sref->StmtAs<BlockNode>();
  // The allocate site must be a block.
  ICHECK(allocate_site != nullptr);
  // Recursively replace the old buffer to a new buffer, where the new buffer has the given storage
  // scope. In the meanwhile, collect the block sref reuse information.
  Map<Block, Block> block_reuse_map;
  Block new_block = StorageScopeMutator::Mutate(GetRef<Block>(allocate_site), buffer, storage_scope,
                                                &block_reuse_map);
  self->Replace(allocate_site_sref, new_block, block_reuse_map);
}

void StorageAlign(ScheduleState self, const StmtSRef& block_sref, int buffer_index, int axis,
                  int factor, int offset) {
  const auto* block_ptr = block_sref->StmtAs<BlockNode>();
  CHECK_GE(buffer_index, 0) << "ValueError: index out of range";
  CHECK_LT(buffer_index, block_ptr->writes.size()) << "ValueError: Index out of range";
  CHECK_GT(factor, 0) << "ValueError: The factor of storage align should be positive.";
  Buffer buffer = block_ptr->writes[buffer_index]->buffer;
  if (axis < 0) {
    axis += buffer->shape.size();
  }
  CHECK(0 <= axis && axis < static_cast<int>(buffer->shape.size()))
      << "ValueError: Axis exceeds the dimension of the buffer.";

  // Step 0: Check the buffer allocation site exists
  StmtSRef allocate_site_sref;
  bool is_alloc;
  std::tie(allocate_site_sref, is_alloc) = GetBufferDefiningSite(block_sref, buffer);
  // We do not allow the buffer being defined in `match_buffer`.
  CHECK(is_alloc) << "ValueError: Set the storage alignment of a buffer defined in "
                     "MatchBufferRegion is not allowed.";

  // Step 1: Get existing or create new annotation value.
  auto annotation = block_ptr->annotations.Get(attr::buffer_dim_align);

  // Use an array to store the storage alignment information for each output tensor.
  // For each output tensor, we use an array of tuples (axis, factor, offset) to specify storage
  // alignment for each dimension.
  Array<Array<Array<Integer>>> storage_align;

  if (annotation.defined()) {
    storage_align = Downcast<Array<Array<Array<Integer>>>>(annotation.value());
    ICHECK(storage_align.size() == block_ptr->writes.size());
  } else {
    storage_align.resize(block_ptr->writes.size());
  }

  // Step 2: Update the annotation value
  Array<Array<Integer>> dim_aligns = storage_align[buffer_index];
  bool found = false;
  for (size_t j = 0; j < dim_aligns.size(); ++j) {
    ICHECK(dim_aligns[j].size() == 3);
    if (dim_aligns[j][0] == axis) {
      dim_aligns.Set(j, {Integer(axis), Integer(factor), Integer(offset)});
      found = true;
      break;
    }
  }
  if (!found) {
    dim_aligns.push_back({Integer(axis), Integer(factor), Integer(offset)});
  }
  storage_align.Set(buffer_index, std::move(dim_aligns));

  // Step 3: Replace the block with the new annotation
  Block new_block = WithAnnotation(block_ptr, attr::buffer_dim_align, storage_align);
  self->Replace(block_sref, new_block, {{GetRef<Block>(block_ptr), new_block}});
}

struct VectorizeTraits : public UnpackedInstTraits<VectorizeTraits> {
  static constexpr const char* kName = "Vectorize";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv) {
    return sch->Vectorize(loop_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv) {
    PythonAPICall py("vectorize");
    py.Input("loop", loop_rv);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

struct ParallelTraits : public UnpackedInstTraits<ParallelTraits> {
  static constexpr const char* kName = "Parallel";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv) {
    return sch->Parallel(loop_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv) {
    PythonAPICall py("parallel");
    py.Input("loop", loop_rv);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

struct UnrollTraits : public UnpackedInstTraits<UnrollTraits> {
  static constexpr const char* kName = "Unroll";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv) { return sch->Unroll(loop_rv); }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv) {
    PythonAPICall py("unroll");
    py.Input("loop", loop_rv);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

struct BindTraits : public UnpackedInstTraits<BindTraits> {
  static constexpr const char* kName = "Bind";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv, String thread) {
    return sch->Bind(loop_rv, thread);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv, String thread) {
    PythonAPICall py("bind");
    py.Input("loop", loop_rv);
    py.Input("thread", thread);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

struct DoubleBufferTraits : public UnpackedInstTraits<DoubleBufferTraits> {
  static constexpr const char* kName = "DoubleBuffer";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv) {
    return sch->DoubleBuffer(block_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv) {
    PythonAPICall py("double_buffer");
    py.Input("block", block_rv);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

struct SetScopeTraits : public UnpackedInstTraits<SetScopeTraits> {
  static constexpr const char* kName = "SetScope";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, Integer i,
                                      String storage_scope) {
    return sch->SetScope(block_rv, i->value, storage_scope);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, Integer i,
                                 String storage_scope) {
    PythonAPICall py("set_scope");
    py.Input("block", block_rv);
    py.Input("i", i->value);
    py.Input("storage_scope", storage_scope);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

struct StorageAlignTraits : public UnpackedInstTraits<StorageAlignTraits> {
  static constexpr const char* kName = "StorageAlign";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 4;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, Integer buffer_index,
                                      Integer axis, Integer factor, Integer offset) {
    return sch->StorageAlign(block_rv, buffer_index->value, axis->value, factor->value,
                             offset->value);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, Integer buffer_index,
                                 Integer axis, Integer factor, Integer offset) {
    PythonAPICall py("storage_align");
    py.Input("block", block_rv);
    py.Input("buffer_index", buffer_index->value);
    py.Input("axis", axis->value);
    py.Input("factor", factor->value);
    py.Input("offset", offset->value);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(VectorizeTraits);
TVM_REGISTER_INST_KIND_TRAITS(ParallelTraits);
TVM_REGISTER_INST_KIND_TRAITS(UnrollTraits);
TVM_REGISTER_INST_KIND_TRAITS(BindTraits);
TVM_REGISTER_INST_KIND_TRAITS(DoubleBufferTraits);
TVM_REGISTER_INST_KIND_TRAITS(SetScopeTraits);
TVM_REGISTER_INST_KIND_TRAITS(StorageAlignTraits);

}  // namespace tir
}  // namespace tvm
