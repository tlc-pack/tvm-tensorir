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

#include "../schedule/utils.h"
#include "./ir_utils.h"

namespace tvm {
namespace tir {

struct InjectSoftwarePipelineConfigNode : public tvm::AttrsNode<InjectSoftwarePipelineConfigNode> {
  bool use_native_pipeline;

  TVM_DECLARE_ATTRS(InjectSoftwarePipelineConfigNode,
                    "tir.transform.InjectSoftwarePipelineConfig") {
    TVM_ATTR_FIELD(use_native_pipeline)
        .describe("Whether to use native pipeline APIs if available")
        .set_default(true);
  }
};

class InjectSoftwarePipelineConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(InjectSoftwarePipelineConfig, Attrs,
                                            InjectSoftwarePipelineConfigNode);
};

TVM_REGISTER_NODE_TYPE(InjectSoftwarePipelineConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.InjectSoftwarePipeline", InjectSoftwarePipelineConfig);

namespace inject_software_pipeline {

// STL map that takes Object as its key
template <class K, class V>
using SMap = std::unordered_map<K, V, ObjectPtrHash, ObjectPtrEqual>;
// STL set that takes Object as its element
template <class K>
using SSet = std::unordered_set<K, ObjectPtrHash, ObjectPtrEqual>;

struct BufferAccess {
  // Buffer variables being written.
  SSet<Var> writes;
  // Buffer variables being read.
  SSet<Var> reads;
};

/*!
 * \brief Get buffer access information of a statement and its children.
 */
BufferAccess GetBufferAccess(const Stmt& stmt) {
  BufferAccess access;
  PreOrderVisit(stmt, [&access](const ObjectRef& obj) {
    if (const auto* block = obj.as<BlockNode>()) {
      for (const auto& read : block->reads) {
        access.reads.insert(read->buffer->data);
      }
      for (const auto& write : block->writes) {
        access.writes.insert(write->buffer->data);
      }
    }
    return true;
  });
  return access;
}

struct PipelineBufferInfo {
  Buffer new_buffer;
  Var loop_var;
  PipelineBufferInfo(Buffer new_buffer, Var loop_var)
      : new_buffer(std::move(new_buffer)), loop_var(std::move(loop_var)) {}
};

/*!
 * \brief Use the pipeline information produced by PipelineDetector to transform the IR.
 *
 * Given a for-loop annotated with pipeline_scope, this pass does the following transformation.
 *
 * Input:
 * \code
 * for ax in range(min, min + extent, annotations={pipeline_scope: num_stages}):
 *   buffer allocations;
 *   producers(ax);  // producers(ax) denotes ax-th iteration of the producers
 *   consumers(ax);  // consumers(ax) denotes ax-th iteration of the consumers
 * \endcode
 *
 * Output:
 * \code
 *
 * buffer allocations;
 *
 * // prologue
 * for ax in range(min, min + shift_extent):
 *   producers(ax);
 *
 * // main loop
 * for ax in range(min, min + extent + shift_extent, annotations={pipeline_scope: 1}):
 *   producers(ax + shift_extent);
 *   consumers(ax);
 *
 * // epilogue
 * for ax in range(min, min + shift_extent):
 *   consumers(ax + extent - shift_extent);
 *
 * where shift_extent = num_stages - 1
 * \endcode
 *
 * Synchronizatons and native pipeline API calls are inserted if needed. The main loop is annotated
 * with AttrStmt so that `ThreadStorageSync` pass will skip this loop which prevents unnecessary
 * synchronizations being inserted.
 *
 * Since producers are executed ahead of the consumers by `shift_extent` iterations, buffers written
 * by the producers need to be enlarged by `num_stages` times. During iterations, results of the
 * producers up to `num_stages` iterations will be kept in the buffer. This reduces synchronizations
 * needed between the producers and the consumers so that they can be executed concurrently.
 */
class PipelineInjector : public StmtExprMutator {
 public:
  static Stmt Inject(bool use_native_pipeline, const PrimFunc& func) {
    // detector(stmt);
    PipelineInjector injector(use_native_pipeline, func);
    Stmt new_stmt = injector(func->body);
    return ConvertSSA(new_stmt);
  }

  PipelineInjector(bool use_native_pipeline, const PrimFunc& func) : use_native_pipeline_(use_native_pipeline) {
    DetectNativePipeline();
    for (const auto& kv : func->buffer_map) {
      const Buffer& buffer = kv.second;
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
  }

 private:
  /*!
   * \brief Build the dependency graph among each direct child of the SeqStmt.
   * \param[in] seq The SeqStmt
   * \param[out] buffer_access A map to store buffer access info of each direct child of `seq`.
   * \param[out] dep_src2dst A map to store dependency edges from the source to the destination.
   * \param[out] dep_dst2src A map to store dependency edges from the destination to the source.
   */
  void BuildDependencyGraph(const SeqStmtNode* seq, SMap<Stmt, BufferAccess>* buffer_access,
                            SMap<Stmt, Array<Stmt>>* dep_src2dst,
                            SMap<Stmt, Array<Stmt>>* dep_dst2src) {
    SMap<Var, Array<Stmt>> buffer_writers;
    for (const Stmt& stmt : seq->seq) {
      BufferAccess access = GetBufferAccess(stmt);
      buffer_access->emplace(stmt, access);
      for (const Var& read : access.reads) {
        auto it = buffer_writers.find(read);
        if (it != buffer_writers.end()) {
          for (const Stmt& writer : it->second) {
            (*dep_src2dst)[writer].push_back(stmt);
            (*dep_dst2src)[stmt].push_back(writer);
          }
        }
      }
      for (const Var& write : access.writes) {
        buffer_writers[write].push_back(stmt);
      }
    }
  }

  std::pair<Array<Stmt>, Array<Stmt>> GetPipelineProducerConsumers(const SeqStmt& seq) {
    // Build the dependency graph from buffer accesses.
    // A map from a Stmt to its buffer access info.
    SMap<Stmt, BufferAccess> buffer_access;
    // A map from a Stmt to its dependants.
    SMap<Stmt, Array<Stmt>> dep_src2dst;
    // A map from a Stmt to its dependencies.
    SMap<Stmt, Array<Stmt>> dep_dst2src;
    BuildDependencyGraph(seq.get(), &buffer_access, &dep_src2dst, &dep_dst2src);

    // analyze dependencies among direct children of the pipeline loop
    Array<Stmt> producers, consumers;
    for (const auto& stmt : seq->seq) {
      if (!dep_src2dst.count(stmt)) {
        consumers.push_back(stmt);
      } else {
        producers.push_back(stmt);
      }
    }
    return {producers, consumers};
  }

  Buffer RewriteAllocBuffer(const Buffer& buffer, int num_stages) {
    ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*(buffer.get()));
    new_buffer->shape.insert(new_buffer->shape.begin(), num_stages);
    if (new_buffer->strides.size()) {
      PrimExpr stride_0 = foldl([](PrimExpr a, PrimExpr b, Span span) { return mul(a, b, span); },
                                make_const(DataType::Int(32), 1), new_buffer->strides);
      new_buffer->strides.insert(new_buffer->strides.begin(), stride_0);
    }
    return Buffer(new_buffer);
  }

  Stmt RewritePipelineBody(Stmt stmt, const For& pipeline_loop, int num_stages,
                           const String& scope) {
    Array<Stmt> producers, consumers;
    CHECK(stmt->IsInstance<SeqStmtNode>())
        << "ValueError: The body of the pipeline should be SeqStmt.";
    std::tie(producers, consumers) = GetPipelineProducerConsumers(Downcast<SeqStmt>(stmt));
    CHECK(!producers.empty()) << "ValueError: Producer not found in the pipeline.";
    CHECK(!consumers.empty()) << "ValueError: Consumer not found in the pipeline.";
    PrimExpr shift_extent = Integer(num_stages - 1);

    // Step 1: Initialize pipeline_var for native pipeline, which will be used in the native
    // pipeline API calls
    bool use_native_pipeline = use_native_pipeline_ && scope == "shared";
    if (use_native_pipeline) {
      CHECK(!pipeline_var_.defined()) << "ValueError: Nested native pipeline not supported.";
      pipeline_var_ = Var("pipeline", PrimType(DataType::Handle()));
    }

    // Step 2: Mutate children to rewrite pipeline buffer access.
    producers.MutateByApply(std::bind(&PipelineInjector::VisitStmt, this, std::placeholders::_1));
    consumers.MutateByApply(std::bind(&PipelineInjector::VisitStmt, this, std::placeholders::_1));

    // Step 3: Build each part of the pipeline
    Stmt prologue = BuildPrologue(producers, pipeline_loop, shift_extent, use_native_pipeline);
    Stmt epilogue =
        BuildEpilogue(consumers, pipeline_loop, shift_extent, scope, use_native_pipeline);
    Stmt main_loop = BuildMainLoop(producers, consumers, pipeline_loop, shift_extent, num_stages,
                                   scope, use_native_pipeline);

    Array<Stmt> pipeline_seq;
    if (use_native_pipeline) {
      pipeline_seq = {prologue, main_loop, epilogue};
    } else {
      pipeline_seq = {prologue, GetPipelineSync(scope), main_loop, epilogue};
    }
    Stmt pipeline = SeqStmt::Flatten(pipeline_seq);

    // Step 4: Create the native pipeline object if necessary
    if (use_native_pipeline) {
      PrimExpr create_pipeline = Call(DataType::Handle(), builtin::tvm_create_pipeline(), {});
      pipeline = LetStmt(pipeline_var_.value(), create_pipeline, pipeline);
      pipeline_var_ = NullOpt;
    }

    return pipeline;
  }

  String GetPipelineScope(const Array<Buffer>& producer_buffers) {
    CHECK(producer_buffers.size()) << "ValueError: Cannot find producer buffers.";
    String scope = GetPtrStorageScope(producer_buffers[0]->data);
    for (size_t i = 1; i < producer_buffers.size(); i++) {
      String new_scope = GetPtrStorageScope(producer_buffers[i]->data);
      CHECK_EQ(scope, new_scope) << "ValueError: Inconsistent storage scopes of producer buffers "
                                    "of the software pipeline ("
                                 << scope << " vs. " << new_scope << ").";
    }
    return scope;
  }

  Stmt InjectPipeline(const ForNode* op) {
    // Get and check annotation
    Integer num_stages = Downcast<Integer>(op->annotations.Get(attr::pipeline_scope).value());
    CHECK_GE(num_stages->value, 2) << "ValueError: Pipeline should have at least two stages.";

    // Clear the pipeline annotation
    For pipeline_loop = GetRef<For>(op);
    auto* pipeline_loop_node = pipeline_loop.CopyOnWrite();
    pipeline_loop_node->annotations.erase(attr::pipeline_scope);

    // Resize producer buffers for pipelined accesses
    CHECK(pipeline_loop->body->IsInstance<BlockRealizeNode>())
        << "ValueError: Cannot find buffer allocations inside the pipeline scope.";

    BlockRealize block_realize = Downcast<BlockRealize>(pipeline_loop->body);
    String scope = GetPipelineScope(block_realize->block->alloc_buffers);
    Array<Buffer> new_alloc_buffers;
    for (const Buffer& alloc_buffer : block_realize->block->alloc_buffers) {
      Buffer new_buffer = RewriteAllocBuffer(alloc_buffer, num_stages);
      new_alloc_buffers.push_back(new_buffer);
      buffer_map_.emplace(alloc_buffer, PipelineBufferInfo(new_buffer, op->loop_var));
      // buffer_data_to_buffer_.Set(new_buffer->data, new_buffer);
    }

    CHECK(is_one(block_realize->predicate))
        << "ValueError: The body block of the software pipeline can not have predicates.";
    CHECK(block_realize->block->match_buffers.empty()) << "ValueError: Pipeline body with match_buffer is not supported.";

    // Rewrite pipeline body
    Stmt pipeline_body =
        RewritePipelineBody(block_realize->block->body, pipeline_loop, num_stages, scope);

    auto new_block = Block({}, {}, {}, "", pipeline_body, NullOpt, new_alloc_buffers);
    auto access = GetBlockReadWriteRegion(new_block, buffer_data_to_buffer_);
    auto* new_block_ptr = new_block.CopyOnWrite();
    new_block_ptr->reads = access[0];
    new_block_ptr->writes = access[1];
    return BlockRealize({}, Bool(true), std::move(new_block));
  }

  Stmt GetPipelineSync(String scope) {
    return Evaluate(
        Call(DataType::Int(32), builtin::tvm_storage_sync(), Array<PrimExpr>{StringImm(scope)}));
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  std::unordered_map<Buffer, PipelineBufferInfo, ObjectPtrHash, ObjectPtrEqual> buffer_map_;

  /*!
   * \brief Wrap a producer statement with native pipeline API calls.
   *
   * This function does the following transformation:
   *
   * Input:
   * \code
   *   producer;
   * \endcode
   *
   * Output:
   * \code
   *   tvm_pipeline_producer_acquire(pipeline);
   *   producer;
   *   tvm_pipeline_producer_commit(pipeline);
   * \endcode
   */
  Stmt WrapNativeProducer(const Stmt& producer) {
    ICHECK(use_native_pipeline_);
    ICHECK(pipeline_var_.defined());
    Stmt producer_acquire = Evaluate(
        Call(DataType::Handle(), builtin::tvm_pipeline_producer_acquire(), {pipeline_var_.value()}));
    Stmt producer_commit = Evaluate(
        Call(DataType::Handle(), builtin::tvm_pipeline_producer_commit(), {pipeline_var_.value()}));
    return SeqStmt::Flatten(producer_acquire, producer, producer_commit);
  }

  /*!
   * \brief Wrap a producer statement with native pipeline API calls.
   *
   * This function does the following transformation:
   *
   * Input:
   * \code
   *   consumer;
   * \endcode
   *
   * Output:
   * \code
   *   tvm_pipeline_consumer_wait(pipeline);
   *   tvm_storage_sync(pipeline_scope);
   *   consumer;
   *   tvm_pipeline_consumer_commit(pipeline);
   * \endcode
   */
  Stmt WrapNativeConsumer(const Stmt& consumer, const String& scope) {
    ICHECK(use_native_pipeline_);
    ICHECK(pipeline_var_.defined());
    Stmt consumer_wait = Evaluate(
        Call(DataType::Handle(), builtin::tvm_pipeline_consumer_wait(), {pipeline_var_.value()}));
    Stmt consumer_release = Evaluate(
        Call(DataType::Handle(), builtin::tvm_pipeline_consumer_release(), {pipeline_var_.value()}));
    Stmt storage_sync = GetPipelineSync(scope);
    return SeqStmt::Flatten(consumer_wait, storage_sync, consumer, consumer_release);
  }

  Stmt BuildPrologue(const Array<Stmt>& producers, For pipeline_loop, const PrimExpr& shift_extent,
                     bool use_native_pipeline) {
    Stmt producer = SeqStmt::Flatten(producers);
    if (use_native_pipeline) {
      producer = WrapNativeProducer(producer);
    }
    PrimExpr new_loop_var =
        is_one(shift_extent) ? pipeline_loop->min : pipeline_loop->loop_var.copy_with_suffix("");
    Map<Var, PrimExpr> subst_map{{pipeline_loop->loop_var, new_loop_var}};
    producer = Substitute(producer, subst_map);
    if (is_one(shift_extent)) {
      return producer;
    } else {
      ForNode* prologue = pipeline_loop.CopyOnWrite();
      prologue->loop_var = Downcast<Var>(new_loop_var);
      prologue->extent = shift_extent;
      prologue->body = producer;
      return pipeline_loop;
    }
  }

  Stmt BuildEpilogue(const Array<Stmt>& consumers, For pipeline_loop, const PrimExpr& shift_extent,
                     const String& scope, bool use_native_pipeline) {
    Stmt consumer = SeqStmt::Flatten(consumers);
    if (use_native_pipeline) {
      consumer = WrapNativeConsumer(consumer, scope);
    }
    PrimExpr new_loop_var =
        is_one(shift_extent) ? pipeline_loop->min : pipeline_loop->loop_var.copy_with_suffix("");
    Map<Var, PrimExpr> subst_map{
        {pipeline_loop->loop_var, new_loop_var + pipeline_loop->extent - shift_extent}};
    consumer = Substitute(consumer, subst_map);
    if (is_one(shift_extent)) {
      return consumer;
    } else {
      ForNode* epilogue = pipeline_loop.CopyOnWrite();
      epilogue->loop_var = Downcast<Var>(new_loop_var);
      epilogue->extent = shift_extent;
      epilogue->body = consumer;
      return pipeline_loop;
    }
  }

  Stmt ScheduleMainLoop(const Array<Stmt>& producers, const Array<Stmt>& consumers, int num_stages,
                        const String& scope, bool use_native_pipeline) {
    // Schedule the execution of producers and consumers. Producers and consumers are assumed to be
    // independant and can be executed concurrently. The schedule can be target-dependant.
    Stmt storage_sync =
        Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(), {StringImm(scope)}));
    // default case: run producers and consumers sequentially.
    Stmt producer = SeqStmt::Flatten(producers);
    Stmt consumer = SeqStmt::Flatten(consumers);
    if (use_native_pipeline) {
      producer = WrapNativeProducer(producer);
      consumer = WrapNativeConsumer(consumer, scope);
    }
    if (!use_native_pipeline_ || num_stages == 2) {
      return SeqStmt::Flatten(producer, consumer, storage_sync);
    } else {
      return SeqStmt::Flatten(producer, consumer);
    }
  }

  Stmt BuildMainLoop(const Array<Stmt>& producers, const Array<Stmt>& consumers, For pipeline_loop,
                     const PrimExpr& shift_extent, int num_stages, const String& scope,
                     bool use_native_pipeline) {
    ForNode* main_loop = pipeline_loop.CopyOnWrite();
    main_loop->extent -= shift_extent;

    // Shift the producers
    Array<Stmt> shifted_producers;
    shifted_producers.reserve(producers.size());
    Map<Var, PrimExpr> subst_map{{pipeline_loop->loop_var, pipeline_loop->loop_var + shift_extent}};
    for (const Stmt& producer : producers) {
      Stmt shifted_producer = Substitute(producer, subst_map);
      shifted_producers.push_back(shifted_producer);
    }
    main_loop->body =
        ScheduleMainLoop(shifted_producers, consumers, num_stages, scope, use_native_pipeline);
    // Annotate the main loop so that thread_storage_sync will skip this part
    main_loop->annotations.Set(attr::pipeline_scope, Integer(1));
    return pipeline_loop;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = buffer_map_.find(op->buffer);
    if (it != buffer_map_.end()) {
      auto* n = store.CopyOnWrite();
      n->buffer = (*it).second.new_buffer;
      n->indices.insert(n->indices.begin(),
                        indexmod(buffer_map_.at(op->buffer).loop_var, n->buffer->shape[0]));
    }
    return store;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = buffer_map_.find(op->buffer);
    if (it != buffer_map_.end()) {
      auto* n = load.CopyOnWrite();
      n->buffer = (*it).second.new_buffer;
      n->indices.insert(n->indices.begin(),
                        indexmod(buffer_map_.at(op->buffer).loop_var, n->buffer->shape[0]));
    }
    return load;
  }

  BufferRegion RewritePipelineBufferRegion(const BufferRegion& buffer_region) {
    auto it = buffer_map_.find(buffer_region->buffer);
    if (it != buffer_map_.end()) {
      Region new_region = buffer_region->region;
      new_region.insert(new_region.begin(),
                        Range::FromMinExtent(0, (*it).second.new_buffer->shape[0]));
      return BufferRegion((*it).second.new_buffer, new_region);
    }
    return buffer_region;
  }

  Stmt VisitStmt_(const BlockNode* op) {
    for (const auto& buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    for (const auto& buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
    }
    auto* n = block.CopyOnWrite();
    n->reads.MutateByApply(
        std::bind(&PipelineInjector::RewritePipelineBufferRegion, this, std::placeholders::_1));
    n->writes.MutateByApply(
        std::bind(&PipelineInjector::RewritePipelineBufferRegion, this, std::placeholders::_1));

    return std::move(block);
  }

  PrimExpr VisitExpr_(const CallNode* op) {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (call->op.same_as(builtin::tvm_get_pipeline())) {
      CHECK(pipeline_var_.defined())
          << "ValueError: intrinsic tvm_get_pipeline can only be called inside the pipeline scope.";
      return pipeline_var_.value();
    }
    return call;
  }

  Stmt VisitStmt_(const ForNode* op) {
    auto it = op->annotations.find(attr::pipeline_scope);
    if (it != op->annotations.end()) {
      return InjectPipeline(op);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  void DetectNativePipeline() {
    if (!use_native_pipeline_) {
      return;
    }
    // Detect whether the runtime has native pipeline support. Currently, the pipeline APIs on
    // CUDA sm_8x devices are supported.
    use_native_pipeline_ = false;
    const Target& target = Target::Current();
    if (!target.defined()) {
      return;
    }
    if (target->kind->name == "cuda") {
      Optional<String> arch = target->GetAttr<String>("arch");
      if (arch.defined() && StartsWith(arch.value(), "sm_8")) {
        use_native_pipeline_ = true;
      }
    }
  }

  // Whether the native pipeline is enabled.
  bool use_native_pipeline_;
  // The pipeline object if native pipeline is enabled.
  Optional<Var> pipeline_var_;
};

}  // namespace inject_software_pipeline

namespace transform {

/*!
 * \brief Transform annotated loops into pipelined one that parallelize producers and consumers.
 * \return The IR transform pass.
 */
Pass InjectSoftwarePipeline() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* fptr = f.CopyOnWrite();
    auto cfg = ctx->GetConfig<InjectSoftwarePipelineConfig>("tir.InjectSoftwarePipeline");
    if (!cfg.defined()) {
      cfg = AttrsWithDefaultValues<InjectSoftwarePipelineConfig>();
    }
    fptr->body = inject_software_pipeline::PipelineInjector::Inject(
        cfg.value()->use_native_pipeline, f);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectSoftwarePipeline", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InjectSoftwarePipeline").set_body_typed(InjectSoftwarePipeline);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
