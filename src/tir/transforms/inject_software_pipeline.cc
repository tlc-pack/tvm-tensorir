/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file inject_software_pipeline.cc
 * \brief Transform annotated loops into pipelined one that parallelize producers and consumers
 */
#include <tvm/target/target.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>

#include "../../support/utils.h"
#include "../schedule/utils.h"
#include "./ir_utils.h"

namespace tvm {
namespace tir {

namespace software_pipeline {

Block MakeBlock(const Stmt& body, const Map<Var, Buffer>& buffer_data_to_buffer) {
  Block block = Block({}, {}, {}, "", body);
  auto access = GetBlockReadWriteRegion(block, buffer_data_to_buffer);
  auto* n = block.CopyOnWrite();
  n->reads = access[0];
  n->writes = access[1];
  return block;
}

struct PipelineStageOrder {
  int stage;
  int order;
  PipelineStageOrder(int stage, int order) : stage(stage), order(order) {}
};

using PipelineInfo = std::unordered_map<Block, PipelineStageOrder, ObjectPtrHash, ObjectPtrEqual>;

struct BufferAccessInfo {
  int def;  // the defining stage of the buffer
  int use;  // the last using stage of the buffer
  BufferAccessInfo(int def = -1, int use = -1) : def(def), use(use){};
};

/*!
 * \brief Rewriter for the body of the software pipeline. This pass inserts `floormod` to indices
 * of accessing to remapped buffer to select the version corresponding to the pipeline stage.
 */
class PipelineBodyRewriter : public StmtExprMutator {
 public:
  /*!
   * \brief Constructor of PipelineBodyRewriter.
   * \param buffer_data_to_buffer The map from buffer data to buffer.
   * \param buffer_remap The map from original buffer to the buffer with updated shape for
   *        multi-versioning in the sofeware pipeline.
   * \param pipeline_loop The original loop to be software pipelined.
   * \param access_all_versions Whether all versions the the buffers in the software pipeline are
   *        accessed. This will be used to update block access region. In the prologue and epilogue
   *        of a two-stage software pipeline, only one version of these buffers are accessed.
   */
  PipelineBodyRewriter(const Map<Var, Buffer>& buffer_data_to_buffer,
                       const Map<Buffer, Buffer>& buffer_remap, For pipeline_loop,
                       bool access_all_versions)
      : buffer_data_to_buffer_(buffer_data_to_buffer),
        buffer_remap_(buffer_remap),
        pipeline_loop_(pipeline_loop),
        access_all_versions_(access_all_versions) {}

 private:
  BufferRegion RewritePipelineBufferRegion(const BufferRegion& buffer_region) const {
    auto it = buffer_remap_.find(buffer_region->buffer);
    if (it != buffer_remap_.end()) {
      Region new_region = buffer_region->region;
      const Buffer& new_buffer = (*it).second;
      // For pipeline buffers, always relax the access region of the first dimension to full extent
      Range accessed_version =
          access_all_versions_
              ? Range::FromMinExtent(0, new_buffer->shape[0])
              : Range::FromMinExtent(floormod((pipeline_loop_->loop_var - pipeline_loop_->min),
                                              new_buffer->shape[0]),
                                     Integer(1));
      new_region.insert(new_region.begin(), accessed_version);
      return BufferRegion(new_buffer, new_region);
    }
    return buffer_region;
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    for (const Buffer& alloc_buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(alloc_buffer->data, alloc_buffer);
    }
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    BlockNode* n = block.CopyOnWrite();
    n->reads.MutateByApply(
        std::bind(&PipelineBodyRewriter::RewritePipelineBufferRegion, this, std::placeholders::_1));
    n->writes.MutateByApply(
        std::bind(&PipelineBodyRewriter::RewritePipelineBufferRegion, this, std::placeholders::_1));
    for (const Buffer& alloc_buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(alloc_buffer->data);
    }
    return block;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = buffer_remap_.find(store->buffer);
    if (it == buffer_remap_.end()) {
      return std::move(store);
    }
    const Buffer& new_buffer = (*it).second;
    auto* n = store.CopyOnWrite();
    n->buffer = new_buffer;
    PrimExpr version =
        floormod((pipeline_loop_->loop_var - pipeline_loop_->min), new_buffer->shape[0]);
    n->indices.insert(n->indices.begin(), version);
    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = buffer_remap_.find(load->buffer);
    if (it == buffer_remap_.end()) {
      return std::move(load);
    }
    const Buffer& new_buffer = (*it).second;
    auto* n = load.CopyOnWrite();
    n->buffer = new_buffer;
    PrimExpr version =
        floormod((pipeline_loop_->loop_var - pipeline_loop_->min), new_buffer->shape[0]);
    n->indices.insert(n->indices.begin(), version);
    return std::move(load);
  }

  PrimExpr RewriteWmmaFragmentIndex(const Buffer& old_buffer, const Buffer& new_buffer,
                                    const PrimExpr& old_index) {
    PrimExpr new_buffer_offset = old_index;

    const int fragment_size = 256;
    PrimExpr offset =
        floordiv(foldl([](PrimExpr a, PrimExpr b, Span span) { return mul(a, b, span); },
                       make_const(DataType::Int(32), 1), old_buffer->shape),
                 fragment_size);
    new_buffer_offset +=
        floormod(pipeline_loop_->loop_var - pipeline_loop_->min, new_buffer->shape[0]) * offset;
    return new_buffer_offset;
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    // Intrinsic calls should be handled explicitly here as they are opaque accesses to
    // buffer.
    static const auto& load_matrix_sync = builtin::tvm_load_matrix_sync();
    static const auto& store_matrix_sync = builtin::tvm_store_matrix_sync();
    static const auto& mma_sync = builtin::tvm_mma_sync();
    static const auto& access_ptr = builtin::tvm_access_ptr();
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (call->op.same_as(load_matrix_sync) || call->op.same_as(store_matrix_sync)) {
      const Buffer& buffer = buffer_data_to_buffer_.at(Downcast<Var>(call->args[0]));
      auto it = buffer_remap_.find(buffer);
      if (it != buffer_remap_.end()) {
        Array<PrimExpr> new_args = call->args;
        const Buffer& new_buffer = (*it).second;
        new_args.Set(4, RewriteWmmaFragmentIndex(buffer, new_buffer, call->args[4]));
        return Call(call->dtype, call->op, new_args, call->span);
      }
    } else if (call->op.same_as(mma_sync)) {
      Array<PrimExpr> new_args = call->args;
      for (int i = 0; i < 4; i++) {
        const Var& buffer_var = Downcast<Var>(call->args[i * 2]);
        const PrimExpr& index = call->args[i * 2 + 1];
        const Buffer& buffer = buffer_data_to_buffer_.at(buffer_var);
        auto it = buffer_remap_.find(buffer);
        if (it != buffer_remap_.end()) {
          PrimExpr new_index = RewriteWmmaFragmentIndex(buffer, (*it).second, index);
          new_args.Set(i * 2 + 1, new_index);
        }
      }
      return Call(call->dtype, call->op, new_args, call->span);
    } else if (call->op.same_as(access_ptr)) {
      const Buffer& buffer = buffer_data_to_buffer_.at(Downcast<Var>(call->args[1]));
      auto it = buffer_remap_.find(buffer);
      if (it != buffer_remap_.end()) {
        Array<PrimExpr> new_args = call->args;
        const Buffer& new_buffer = (*it).second;
        const PrimExpr& old_index = call->args[2];
        PrimExpr offset;
        if (new_buffer->strides.empty()) {
          offset = foldl([](PrimExpr a, PrimExpr b, Span span) { return mul(a, b, span); },
                         make_const(DataType::Int(32), 1), buffer->shape);
        } else {
          offset = new_buffer->strides[0];
        }
        PrimExpr new_index = old_index + floormod(pipeline_loop_->loop_var, 2) * offset;
        new_args.Set(2, new_index);
        return Call(call->dtype, call->op, new_args, call->span);
      }
    }
    return std::move(call);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Buffer> buffer_remap_;
  For pipeline_loop_;
  bool access_all_versions_;
};

class PipelineRewriter : public StmtExprMutator {
 public:
  static Stmt Rewrite(
      Map<Var, Buffer> buffer_data_to_buffer,
      const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>& double_buffers,
      const Array<Buffer> pipeline_allocs, const For& pipeline_loop,
      const PipelineInfo& pipeline_info) {
    PipelineRewriter rewriter(buffer_data_to_buffer, double_buffers, pipeline_allocs, pipeline_loop,
                              pipeline_info);
    return rewriter.BuildPipeline();
  }

 private:
  PipelineRewriter(Map<Var, Buffer> buffer_data_to_buffer,
                   const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>& double_buffers,
                   const Array<Buffer>& pipeline_allocs, const For& pipeline_loop,
                   const PipelineInfo& pipeline_info)

      : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)),
        double_buffers_(double_buffers),
        pipeline_allocs_(pipeline_allocs),
        pipeline_loop_(pipeline_loop),
        pipeline_info_(pipeline_info) {}

  Stmt BuildPipeline() {
    // Step 1: Analyze accesses to the buffers in the pipeline and compute the number of versions
    // need to maintain for each buffer.
    RemapPipelineBuffers(pipeline_allocs_);

    ordered_stmts_.resize(pipeline_info_.size());
    for (const auto& pair : pipeline_info_) {
      const Block& block = pair.first;
      int order = pair.second.order;
      ordered_stmts_.Set(order, block);
    }

    // Step 2: Emit the pipeline prologue, body and epilogue.
    Stmt prologue = EmitImpl(pipeline_loop_->min, pipeline_loop_->min + max_stage_, true);
    Stmt body = EmitImpl(pipeline_loop_->min + max_stage_,
                         pipeline_loop_->min + pipeline_loop_->extent, false);
    Stmt epilogue = EmitImpl(pipeline_loop_->min + pipeline_loop_->extent,
                             pipeline_loop_->min + pipeline_loop_->extent + max_stage_, true);

    SeqStmt stmt = SeqStmt({prologue, body, epilogue});

    // Step 3: Add annotations of nested software pipeline (if appliable)
    stmt = AnnotateNestedPipeline(stmt);

    // Step 4: Make a new block that contains new buffer allocations after pipeline rewriting.
    Array<Buffer> alloc_buffers;
    for (const auto& alloc : pipeline_allocs_) {
      auto it = buffer_remap_.find(alloc);
      if (it != buffer_remap_.end()) {
        alloc_buffers.push_back((*it).second);
      } else {
        alloc_buffers.push_back(alloc);
      }
      buffer_data_to_buffer_.erase(alloc->data);
    }
    Block block = MakeBlock(stmt, buffer_data_to_buffer_);
    auto* n = block.CopyOnWrite();
    n->alloc_buffers = std::move(alloc_buffers);
    return BlockRealize({}, Bool(true), block);
  }

 private:
  /*!
   * \brief Annotate the result of software pipeline rewriting with user-provided annotations.
   *
   * When there are nested software pipelines, after rewriting the inner software pipeline,
   * it is required to add annotations to the result of the inner software pipeline to specify
   * the rewriting behavior of the outer software pipeline.
   * This method expects the annotations `attr::nested_software_pipeline_order`, and
   * `attr::nested_software_pipeline_stage` are present on the inner software pipeline loop.
   *
   * \param pipeline_seq The sequence of statements after pipeline rewriting, which consists of
   * three BlockRealize that represents the prologue, the body, and the epilogue of the software
   * pipeline.
   * \return The sequence of the statements that consists of the annotated software pipeline.
   */
  SeqStmt AnnotateNestedPipeline(const SeqStmt& pipeline_seq) {
    auto it = pipeline_loop_->annotations.find(attr::nested_software_pipeline_stage);
    if (it == pipeline_loop_->annotations.end()) {
      return pipeline_seq;
    }
    Array<Integer> nested_stage = Downcast<Array<Integer>>((*it).second);
    CHECK(pipeline_loop_->annotations.count(attr::nested_software_pipeline_order))
        << "ValueError: Annotation for the order of the nested software pipeline is missing.";
    Array<Integer> nested_order = Downcast<Array<Integer>>(
        pipeline_loop_->annotations.at(attr::nested_software_pipeline_order));
    CHECK_EQ(nested_stage.size(), 3) << "ValueError: Annotation for the stage of the nested "
                                        "software pipeline should be a 3-tuple";
    CHECK_EQ(nested_order.size(), 3) << "ValueError: Annotation for the order of the nested "
                                        "software pipeline should be a 3-tuple";
    Array<Stmt> new_seq;
    new_seq.reserve(pipeline_seq->seq.size());
    for (size_t i = 0; i < pipeline_seq->seq.size(); i++) {
      BlockRealize block_realize = Downcast<BlockRealize>(pipeline_seq->seq[i]);
      auto* block = block_realize.CopyOnWrite()->block.CopyOnWrite();
      block->annotations.Set(attr::software_pipeline_stage, nested_stage[i]);
      block->annotations.Set(attr::software_pipeline_order, nested_order[i]);
      new_seq.push_back(std::move(block_realize));
    }
    return SeqStmt(std::move(new_seq));
  }

  /*!
   * \brief Analyze accesses to the buffers in the software pipeline.
   *
   * This method check the 'define' and 'use' stage of the buffers in the software pipeline, which
   * can be used to compute the number of versions needed to maintain after rewriting.
   */
  std::unordered_map<Buffer, BufferAccessInfo, ObjectPtrHash, ObjectPtrEqual>
  GetBufferAccessInfo() {
    std::unordered_map<Buffer, BufferAccessInfo, ObjectPtrHash, ObjectPtrEqual> infos;
    for (const auto& pair : pipeline_info_) {
      const Block& block = pair.first;
      int stage = pair.second.stage;
      max_stage_ = std::max(max_stage_, stage);

      for (const BufferRegion& write : block->writes) {
        if (!infos.count(write->buffer)) {
          infos.emplace(write->buffer, BufferAccessInfo{});
        }
        auto& info = infos.at(write->buffer);
        if (info.def == -1) {
          info.def = stage;
        }
      }

      for (const BufferRegion& read : block->reads) {
        if (!infos.count(read->buffer)) {
          infos.emplace(read->buffer, BufferAccessInfo{});
        }
        auto& info = infos.at(read->buffer);
        info.use = std::max(info.use, stage);
      }
    }
    return infos;
  }

  /*!
   * \brief Check whether two regions have intersections.
   * \param region1 The first region.
   * \param region2 The second region.
   * \return Whether region1 and region2 have intersections.
   */
  bool MayConflict(Region region1, Region region2) {
    ICHECK(region1.size() == region2.size());
    for (size_t i = 0; i < region1.size(); i++) {
      Range dim1 = region1[i];
      Range dim2 = region2[i];
      auto int_set1 = arith::IntSet::FromRange(dim1);
      auto int_set2 = arith::IntSet::FromRange(dim2);
      if (arith::Intersect({int_set1, int_set2}).IsNothing()) {
        return false;
      }
    }
    return true;
  }

  /*!
   * \brief Compute the number of versions need to maintain for buffer accessed in the software
   * pipeline.
   *
   * This method applies liveness analysis to the target buffer to compute the number of versions
   * need to maintain during the software pipeline.
   * Annotation `attr::double_buffer_scope` is handled here which provides a way to override the
   * result of the analysis. Additional double buffering in the software pipeline can be useful
   * to eliminate synchonizations in GPU devices.
   *
   * \param buffer The target buffer
   * \param buffer_info The access information of the target buffer.
   * \return The number of versions required for the target buffer.
   */
  int ComputeBufferVersions(const Buffer& buffer, const BufferAccessInfo& buffer_info) {
    if (buffer_info.def == -1) {
      // Keep the original number of versions as buffers defined outside the software pipeline
      // should not be mutated.
      return 1;
    }

    // `use - def + 1` is a upper bound of the needed versions
    // We optimize a few case where the number of versions can be smaller than the upper bound
    int num_versions = buffer_info.use - buffer_info.def + 1;
    if (num_versions == 2) {
      // A special case when `use - def + 1 == 2`. Double buffering is only needed in this case when
      // these exists a reader block_i and a writer block_j such that
      // order(block_i) < order(block_j) and stage(block_i) < stage(block_j) and the access regions
      // of block_i and block_j overlap.
      bool need_multi_version = false;
      for (const auto& pair1 : pipeline_info_) {
        const Block& writer_block = pair1.first;
        const auto& writer_info = pair1.second;

        auto it1 = std::find_if(writer_block->writes.begin(), writer_block->writes.end(),
                                [&](const BufferRegion& buffer_region) {
                                  return buffer_region->buffer.same_as(buffer);
                                });
        if (it1 == writer_block->writes.end()) {
          continue;
        }

        for (const auto& pair2 : pipeline_info_) {
          const Block& reader_block = pair2.first;
          const auto& reader_info = pair2.second;
          auto it2 = std::find_if(reader_block->reads.begin(), reader_block->reads.end(),
                                  [&](const BufferRegion& buffer_region) {
                                    return buffer_region->buffer.same_as(buffer);
                                  });
          if (it2 == reader_block->reads.end()) {
            continue;
          }
          if (writer_info.order < reader_info.order && writer_info.stage < reader_info.stage &&
              MayConflict((*it1)->region, (*it2)->region)) {
            need_multi_version = true;
            break;
          }
        }
      }
      if (!need_multi_version) {
        num_versions = 1;
      }
    }
    if (num_versions == 1 && double_buffers_.count(buffer)) {
      num_versions = 2;
    }
    return num_versions;
  }

  /*!
   * \brief Rewrite buffer allocations to create new buffers with new shapes according to
   * the software pipeline.
   * \param pipeline_allocs The buffer allocations inside the software pipeline scope.
   */
  void RemapPipelineBuffers(Array<Buffer> pipeline_allocs) {
    std::unordered_map<Buffer, BufferAccessInfo, ObjectPtrHash, ObjectPtrEqual> infos =
        GetBufferAccessInfo();
    for (const auto& pair : infos) {
      const Buffer& buffer = pair.first;
      const BufferAccessInfo& buffer_info = pair.second;
      int num_versions = ComputeBufferVersions(buffer, buffer_info);
      if (num_versions > 1) {
        Buffer new_buffer = RewriteAllocBuffer(buffer, num_versions);
        CHECK(std::find(pipeline_allocs.begin(), pipeline_allocs.end(), buffer) !=
              pipeline_allocs.end());
        buffer_remap_.Set(pair.first, new_buffer);
      }
    }
  }

  /*!
   * \brief Rewrite buffer allocation to keep multiple versions of original buffer for pipelined
   * accesses.
   * \param buffer The buffer to be resized.
   * \param num_versions The number of versions to keep.
   * \return The resized buffer.
   */
  Buffer RewriteAllocBuffer(const Buffer& buffer, int num_versions) {
    ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*(buffer.get()));
    new_buffer->shape.insert(new_buffer->shape.begin(), num_versions);
    if (new_buffer->strides.size()) {
      ICHECK(new_buffer->strides.size() + 1 == new_buffer->shape.size());
      PrimExpr stride_0 = new_buffer->strides[0] * new_buffer->shape[1];
      new_buffer->strides.insert(new_buffer->strides.begin(), stride_0);
    }
    return Buffer(new_buffer);
  }

  Stmt EmitImpl(PrimExpr start, PrimExpr end, bool unroll_loop) {
    Array<Stmt> stmts;
    PrimExpr new_loop_var;
    bool is_unit_loop = analyzer_.CanProveEqual(start + 1, end);
    if (is_unit_loop) {
      new_loop_var = start;
    } else {
      new_loop_var = pipeline_loop_->loop_var.copy_with_suffix("");
      analyzer_.Bind(Downcast<Var>(new_loop_var), Range(start, end), true);
    }

    for (const Block block : ordered_stmts_) {
      int stage = pipeline_info_.at(block).stage;
      PrimExpr skewed_loop_var = new_loop_var - stage;
      PrimExpr inbound = (skewed_loop_var >= pipeline_loop_->min) &&
                         (skewed_loop_var < pipeline_loop_->min + pipeline_loop_->extent);
      inbound = analyzer_.Simplify(inbound);
      if (analyzer_.CanProve(!inbound)) {
        continue;
      }
      Block new_block = Downcast<Block>(PipelineBodyRewriter(
          buffer_data_to_buffer_, buffer_remap_, pipeline_loop_, max_stage_ != 1)(block));
      Map<Var, PrimExpr> subst_map;
      if (is_unit_loop) {
        subst_map.Set(pipeline_loop_->loop_var, skewed_loop_var);
      } else {
        // normalize loop range
        subst_map.Set(pipeline_loop_->loop_var, skewed_loop_var + (start - pipeline_loop_->min));
      }
      new_block = Downcast<Block>(Substitute(new_block, subst_map));
      stmts.push_back(BlockRealize({}, inbound, new_block));
    }

    Stmt stmt;
    if (is_unit_loop) {
      stmt = stmts.size() == 1 ? stmts[0] : SeqStmt(stmts);
    } else {
      stmt = For(Downcast<Var>(new_loop_var), pipeline_loop_->min, end - start,
                 unroll_loop ? ForKind::kUnrolled : pipeline_loop_->kind, SeqStmt(stmts));
    }
    if (stmt->IsInstance<BlockRealizeNode>()) {
      return stmt;
    }
    return BlockRealize({}, Bool(true), MakeBlock(stmt, buffer_data_to_buffer_));
  }

  arith::Analyzer analyzer_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>& double_buffers_;
  Array<Buffer> pipeline_allocs_;
  For pipeline_loop_;
  PipelineInfo pipeline_info_;
  int max_stage_ = -1;
  Map<Buffer, Buffer> buffer_remap_;
  Array<Block> ordered_stmts_;
};

class PipelineInjector : private StmtExprMutator {
 public:
  static Stmt Inject(const PrimFunc& func) {
    PipelineInjector injector;
    for (const auto& kv : func->buffer_map) {
      const Buffer& buffer = kv.second;
      injector.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    return injector(func->body);
  }

 private:
  PipelineInjector() = default;

  PipelineStageOrder CheckAndRemovePipelineAnnotation(Map<String, ObjectRef>* annotations) const {
    CHECK(annotations->count(attr::software_pipeline_stage))
        << "ValueError: Stage of the statement in the software pipeline is not defined.";
    CHECK(annotations->count(attr::software_pipeline_order))
        << "ValueError: Order of the statement in the software pipeline is not defined.";
    Integer stage = Downcast<Integer>(annotations->at(attr::software_pipeline_stage));
    Integer order = Downcast<Integer>(annotations->at(attr::software_pipeline_order));
    annotations->erase(attr::software_pipeline_stage);
    annotations->erase(attr::software_pipeline_order);
    return {static_cast<int>(stage->value), static_cast<int>(order->value)};
  }

  /*!
   * \brief Check the pipeline satisfies the following conditions:
   * 1) No conflicting order: The order of each statement should be unique.
   * 2) No reordering with the same stage: Statements in the same stage are not allowed to be
   * reordered.
   */
  void ValidatePipelineBody(const PipelineInfo& pipeline_info, const Array<Block>& original_order) {
    std::unordered_set<int> used_orders;
    std::unordered_map<int, int> stage_max_order;
    for (const Block& block : original_order) {
      const auto& stmt_info = pipeline_info.at(block);
      int stage = stmt_info.stage;
      int order = stmt_info.order;
      CHECK(!used_orders.count(order))
          << "ValueError: Two statements in the software pipeline cannot have the same order";
      used_orders.insert(order);
      CHECK(!stage_max_order.count(stage) || stage_max_order[stage] < order)
          << "ValueError: Statements in the same stage of the software pipeline must have "
             "increasing order.";
      stage_max_order[stage] = order;
    }
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // Step 1: Recursively rewrite the children first.
    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    bool is_pipeline = HasPipelineAnnotation(op);
    if (!is_pipeline) {
      return std::move(for_node);
    }
    // Step 2: Find the body of the pipeline. It can be direct child of the for-loop. If the
    // for-loop as BlockRealize as its child, the pipeline body will be the child of the block.
    Stmt pipeline_body;
    Array<Buffer> pipeline_allocs;
    if (const auto* realize = for_node->body.as<BlockRealizeNode>()) {
      const auto& block = realize->block;
      for (const auto& buffer : block->alloc_buffers) {
        ICHECK(buffer->IsInstance<BufferNode>());
        buffer_data_to_buffer_.Set(buffer->data, buffer);
      }
      pipeline_body = block->body;
      pipeline_allocs = block->alloc_buffers;
    } else {
      pipeline_body = for_node->body;
    }

    const SeqStmtNode* pipeline_body_seq = pipeline_body.as<SeqStmtNode>();
    CHECK(pipeline_body_seq)
        << "ValueError: The body of the software pipeline should be SeqStmt, got "
        << pipeline_body->GetTypeKey();
    const SeqStmtNode* original_seq =
        op->body->IsInstance<BlockRealizeNode>()
            ? op->body.as<BlockRealizeNode>()->block->body.as<SeqStmtNode>()
            : op->body.as<SeqStmtNode>();
    ICHECK(original_seq);

    // Step 3: Blockize the components of the pipeline. Each child of the pipelined loop should
    // be converted into a block.
    PipelineInfo pipeline_info;
    Array<Block> original_order;

    auto f_add_child = [&](const Stmt& child) {
      const auto* block_realize = child.as<BlockRealizeNode>();
      Block block = (block_realize && is_one(block_realize->predicate))
                        ? block_realize->block
                        : MakeBlock(child, buffer_data_to_buffer_);
      original_order.push_back(block);
    };
    for (size_t i = 0; i < pipeline_body_seq->seq.size(); i++) {
      const auto* nested_block_realize = pipeline_body_seq->seq[i].as<BlockRealizeNode>();
      if (nested_block_realize && is_one(nested_block_realize->predicate) &&
          nested_block_realize->block->body->IsInstance<SeqStmtNode>()) {
        const Block& nested_pipeline_block = nested_block_realize->block;
        ICHECK(
            nested_pipeline_block->match_buffers.empty());  // match_buffer should have been lowered
        for (const auto& buffer : nested_pipeline_block->alloc_buffers) {
          pipeline_allocs.push_back(buffer);
          buffer_data_to_buffer_.Set(buffer->data, buffer);
        }
        const auto* nested_seq = nested_pipeline_block->body.as<SeqStmtNode>();
        for (size_t j = 0; j < nested_seq->seq.size(); j++) {
          f_add_child(nested_seq->seq[j]);
        }
      } else {
        f_add_child(pipeline_body_seq->seq[i]);
      }
    }

    auto pipeline_stages =
        Downcast<Array<Integer>>(op->annotations.at(attr::software_pipeline_stage));
    auto pipeline_orders =
        Downcast<Array<Integer>>(op->annotations.at(attr::software_pipeline_order));
    CHECK_EQ(pipeline_stages.size(), original_order.size());
    CHECK_EQ(pipeline_orders.size(), original_order.size());
    for (size_t i = 0; i < pipeline_stages.size(); i++) {
      PipelineStageOrder stage_order(pipeline_stages[i]->value, pipeline_orders[i]->value);
      pipeline_info.emplace(original_order[i], stage_order);
    }
    // ValidatePipelineBody(pipeline_info, original_order);

    // Step 4: Rewrite the pipeline body.
    Stmt pipeline = PipelineRewriter::Rewrite(buffer_data_to_buffer_, double_buffers,
                                              pipeline_allocs, GetRef<For>(op), pipeline_info);

    if (const auto* realize = op->body.as<BlockRealizeNode>()) {
      const auto& block = realize->block;
      for (const auto& buffer : block->alloc_buffers) {
        buffer_data_to_buffer_.erase(buffer->data);
      }
    }
    return pipeline;
  }

  /*!
   * \brief Add buffer allocations to a block and update the write region of the block.
   * \param n The block pointer to which the buffer allocations are added.
   * \param alloc_buffers The buffer allocations to be added.
   */
  void AddAllocBuffers(BlockNode* n, const Array<Buffer> alloc_buffers) {
    for (const Buffer& alloc_buffer : alloc_buffers) {
      n->alloc_buffers.push_back(alloc_buffer);
      Region region;
      region.reserve(alloc_buffer->shape.size());
      for (const PrimExpr& dim : alloc_buffer->shape) {
        region.push_back(Range::FromMinExtent(0, dim));
      }
      n->writes.push_back(BufferRegion(alloc_buffer, region));
    }
  }

  /*!
   * \brief Flatten nested SeqStmt while passing through BlockRealize / Block.
   * \param block The block which has SeqStmt body to rewrite.
   * \return The new block that contains flattened SeqStmt as its body.
   */
  Block FlattenNestedBlocks(Block block) {
    const SeqStmtNode* seq = block->body.as<SeqStmtNode>();
    auto* n = block.CopyOnWrite();
    Array<Stmt> new_seq;
    new_seq.reserve(seq->seq.size());
    bool changed = false;
    for (size_t i = 0; i < seq->seq.size(); i++) {
      const auto* nested_block_realize = seq->seq[i].as<BlockRealizeNode>();
      if (!nested_block_realize || !is_one(nested_block_realize->predicate) ||
          !nested_block_realize->block->body->IsInstance<SeqStmtNode>()) {
        new_seq.push_back(seq->seq[i]);
        continue;
      }
      AddAllocBuffers(n, nested_block_realize->block->alloc_buffers);
      const auto* nested_seq = nested_block_realize->block->body.as<SeqStmtNode>();
      new_seq.reserve(new_seq.size() + nested_seq->seq.size());
      for (const auto& nested_seq_body : nested_seq->seq) {
        new_seq.push_back(nested_seq_body);
      }
      changed = true;
    }
    if (changed) {
      n->body = SeqStmt(new_seq);
    }
    return block;
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    for (const auto& buffer : op->alloc_buffers) {
      ICHECK(buffer->IsInstance<BufferNode>());
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }

    auto it = op->annotations.find(attr::double_buffer_scope);
    if (it != op->annotations.end()) {
      int buffer_index = Downcast<Integer>((*it).second);
      CHECK(buffer_index >= 0 && static_cast<size_t>(buffer_index) < op->writes.size())
          << "ValueError: Index of the buffer exceeds the size of the write regions of the block. ("
          << buffer_index << " vs. " << op->writes.size() << ")";
      double_buffers.insert(op->writes[buffer_index]->buffer);
    }
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));

    // if (block->body->IsInstance<SeqStmtNode>()) {
    //   // Rewriting for software pipelining will produce nested SeqStmt. These statements need to
    //   be
    //   // flattened for rewriting outer software pipeline (if nested software pipelines are
    //   present). block = FlattenNestedBlocks(block);
    // }

    for (const auto& buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
    }
    return block;
  }

  bool HasPipelineAnnotation(const ForNode* op) const {
    auto it1 = op->annotations.find(attr::software_pipeline_stage);
    auto it2 = op->annotations.find(attr::software_pipeline_order);
    bool has_stage = it1 != op->annotations.end();
    bool has_order = it2 != op->annotations.end();
    if (has_stage && has_order) {
      return true;
    }
    if (has_stage) {
      LOG(FATAL) << "ValueError: Order of the software pipeline is not defined.";
    }
    if (has_order) {
      LOG(FATAL) << "ValueError: Stage of the software pipeline is not defined.";
    }
    return false;
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> double_buffers;
};

}  // namespace software_pipeline

namespace transform {

/*!
 * \brief Transform annotated loops into pipelined one that parallelize producers and consumers.
 * \return The IR transform pass.
 */
Pass InjectSoftwarePipeline() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* fptr = f.CopyOnWrite();
    fptr->body = software_pipeline::PipelineInjector::Inject(f);
    fptr->body = ConvertSSA(std::move(fptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectSoftwarePipeline", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InjectSoftwarePipeline").set_body_typed(InjectSoftwarePipeline);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
