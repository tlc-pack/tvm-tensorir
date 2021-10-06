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

using support::StartsWith;

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

  PipelineInjector(bool use_native_pipeline, const PrimFunc& func)
      : use_native_pipeline_(use_native_pipeline) {
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
      PrimExpr stride_0 = new_buffer->strides[0] * new_buffer->shape[1];
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
      // CHECK_EQ(scope, new_scope) << "ValueError: Inconsistent storage scopes of producer buffers
      // "
      //                               "of the software pipeline ("
      //                            << scope << " vs. " << new_scope << ").";
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
    CHECK(block_realize->block->match_buffers.empty())
        << "ValueError: Pipeline body with match_buffer is not supported.";

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
    Stmt producer_acquire = Evaluate(Call(
        DataType::Handle(), builtin::tvm_pipeline_producer_acquire(), {pipeline_var_.value()}));
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
    Stmt consumer_release = Evaluate(Call(
        DataType::Handle(), builtin::tvm_pipeline_consumer_release(), {pipeline_var_.value()}));
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
      if (arch.defined() && support::StartsWith(arch.value(), "sm_8")) {
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

namespace pipeline_v2 {

Block MakeBlock(const Stmt& body, const Map<Var, Buffer>& buffer_data_to_buffer) {
  Block block = Block({}, {}, {}, "", body);
  auto access = GetBlockReadWriteRegion(block, buffer_data_to_buffer);
  auto* n = block.CopyOnWrite();
  n->reads = access[0];
  n->writes = access[1];
  return block;
}

bool FindInAST(const Stmt& stmt, std::function<bool(const ObjectRef&)> pred) {
  bool result = false;
  PreOrderVisit(stmt, [&](const ObjectRef& obj) {
    if (pred(obj)) {
      result = true;
      return false;
    }
    return true;
  });
  return result;
}

/*!
 * \brief Annotate statements for software pipeline rewriting.
 * This function adds `pipeline_stage` and `pipeline_order` to block annotations which will be
 * used for software pipeline rewriting.
 * The value of these annotations are determined by a few heuristic-based rules.
 */
Array<Block> AnnotatePipelineBody(const Array<Block>& stmts) {
  Array<Block> new_stmts;

  std::unordered_map<Stmt, int, ObjectPtrHash, ObjectPtrEqual> stmt_stages;
  std::unordered_map<Stmt, int, ObjectPtrHash, ObjectPtrEqual> stmt_orders;

  std::map<int, Array<Stmt>> priority_groups;

  for (size_t i = 0; i < stmts.size(); i++) {
    const Block& block = stmts[i];
    if (FindInAST(block, [](const ObjectRef& obj) {
          if (const BufferLoadNode* buffer_load = obj.as<BufferLoadNode>()) {
            return buffer_load->buffer.scope() == "global";
          }
          return false;
        })) {
      stmt_stages.emplace(block, 0);
      priority_groups[0].push_back(block);
      continue;
    }
    if (block->annotations.count("pipeline_prologue")) {
      stmt_stages.emplace(block, 0);
      priority_groups[3].push_back(block);
      continue;
    }
    if (block->annotations.count("pipeline_body")) {
      stmt_stages.emplace(block, 1);
      priority_groups[1].push_back(block);
      continue;
    }
    if (block->annotations.count("pipeline_epilogue")) {
      stmt_stages.emplace(block, 1);
      priority_groups[3].push_back(block);
      continue;
    }
    if (FindInAST(block, [](const ObjectRef& obj) {
          if (const BufferStoreNode* buffer_store = obj.as<BufferStoreNode>()) {
            auto scope = buffer_store->buffer.scope();
            return StartsWith(scope, "shared");
          }
          return false;
        })) {
      stmt_stages.emplace(block, 0);
      priority_groups[2].push_back(block);
      continue;
    }
    // default rule
    if (i + 1 < stmts.size()) {
      stmt_stages.emplace(block, 0);
      priority_groups[1].push_back(block);
    } else {
      stmt_stages.emplace(block, 1);
      priority_groups[1].push_back(block);
    }
  }
  int i = 0;
  for (const auto& pair : priority_groups) {
    const auto& ordered_subset = pair.second;
    for (const auto& stmt : ordered_subset) {
      stmt_orders.emplace(stmt, i++);
    }
  }

  for (Block block : stmts) {
    int stage = stmt_stages.at(block);
    int order = stmt_orders.at(block);
    auto* n = block.CopyOnWrite();
    n->annotations.Set(attr::pipeline_stage, Integer(stage));
    n->annotations.Set(attr::pipeline_order, Integer(order));
    new_stmts.push_back(block);
  }
  return new_stmts;
}


struct BufferInfo {
  int def;  // the defining stage of the buffer
  int use;  // the last using stage of the buffer
  BufferInfo(int def = -1, int use = -1) : def(def), use(use){};
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
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    BlockNode* n = block.CopyOnWrite();
    n->reads.MutateByApply(
        std::bind(&PipelineBodyRewriter::RewritePipelineBufferRegion, this, std::placeholders::_1));
    n->writes.MutateByApply(
        std::bind(&PipelineBodyRewriter::RewritePipelineBufferRegion, this, std::placeholders::_1));
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
  PipelineRewriter(Map<Var, Buffer> buffer_data_to_buffer, const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>& double_buffers)
      : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)), double_buffers_(double_buffers) {}

  Stmt BuildPipeline(const Array<Block>& blocks, Array<Buffer> pipeline_allocs, For pipeline_loop) {
    pipeline_loop_ = pipeline_loop;
    RemapPipelineBuffers(blocks, pipeline_allocs);

    ordered_stmts_.resize(blocks.size());
    for (const Block& block : blocks) {
      int order =
          static_cast<int>(block->annotations.at(attr::pipeline_order).as<IntImmNode>()->value);
      ordered_stmts_.Set(order, block);
    }

    Stmt prologue =
        EmitImpl(pipeline_loop_->min, pipeline_loop_->min + max_stage_, true, "pipeline_prologue");
    Stmt body = EmitImpl(pipeline_loop_->min + max_stage_,
                         pipeline_loop_->min + pipeline_loop_->extent, false, "pipeline_body");
    Stmt epilogue = EmitImpl(pipeline_loop_->min + pipeline_loop_->extent,
                             pipeline_loop_->min + pipeline_loop_->extent + max_stage_, true,
                             "pipeline_epilogue");

    Stmt stmt = SeqStmt({prologue, body, epilogue});
    Block block = MakeBlock(stmt, buffer_data_to_buffer_);
    auto* n = block.CopyOnWrite();
    for (const auto& alloc : pipeline_allocs) {
      auto it = buffer_remap_.find(alloc);
      if (it != buffer_remap_.end()) {
        n->alloc_buffers.push_back((*it).second);
      } else {
        n->alloc_buffers.push_back(alloc);
      }
      ICHECK(n->alloc_buffers.back()->IsInstance<BufferNode>());
    }
    return BlockRealize({}, Bool(true), block);
  }

 private:
  std::pair<int, int> GetBlockStageOrder(const Block& block) {
    int stage = block->annotations.at(attr::pipeline_stage).as<IntImmNode>()->value;
    int order = block->annotations.at(attr::pipeline_order).as<IntImmNode>()->value;
    return {stage, order};
  }

  std::unordered_map<Buffer, BufferInfo, ObjectPtrHash, ObjectPtrEqual> GetBufferInfo(
      const Array<Block>& blocks) {
    std::unordered_map<Buffer, BufferInfo, ObjectPtrHash, ObjectPtrEqual> infos;
    for (const Block& block : blocks) {
      int stage = GetBlockStageOrder(block).first;
      max_stage_ = std::max(max_stage_, stage);

      for (const BufferRegion& write : block->writes) {
        if (!infos.count(write->buffer)) {
          infos.emplace(write->buffer, BufferInfo{});
        }
        auto& info = infos.at(write->buffer);
        if (info.def == -1) {
          info.def = stage;
        }
      }

      for (const BufferRegion& read : block->reads) {
        if (!infos.count(read->buffer)) {
          infos.emplace(read->buffer, BufferInfo{});
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

  int ComputeBufferVersions(const Array<Block>& blocks, const Buffer& buffer,
                            const BufferInfo& buffer_info) {
    if (buffer_info.def == -1) {
      // Keep the original number of versions as buffers defined outside the software pipeline
      // should not be mutated.
      return 1;
    }

    // `use - def + 1` is a upper bound of the needed versions
    // We optimize a few case where the number of versions can be smaller than the upper bound
    int num_versions = buffer_info.use - buffer_info.def + 1;
    // TODO handle double buffer
    if (num_versions == 2) {
      // A special case when `use - def + 1 == 2`. Double buffering is only needed in this case when
      // these exists a reader block_i and a writer block_j such that
      // order(block_i) < order(block_j) and stage(block_i) < stage(block_j) and the access regions
      // of block_i and block_j overlap.
      bool need_multi_version = false;
      for (const Block& writer_block : blocks) {
        auto it1 = std::find_if(writer_block->writes.begin(), writer_block->writes.end(),
                                [&](const BufferRegion& buffer_region) {
                                  return buffer_region->buffer.same_as(buffer);
                                });
        if (it1 == writer_block->writes.end()) {
          continue;
        }
        int writer_stage, writer_order;
        std::tie(writer_stage, writer_order) = GetBlockStageOrder(writer_block);

        for (const Block& reader_block : blocks) {
          auto it2 = std::find_if(reader_block->reads.begin(), reader_block->reads.end(),
                                  [&](const BufferRegion& buffer_region) {
                                    return buffer_region->buffer.same_as(buffer);
                                  });
          if (it2 == reader_block->reads.end()) {
            continue;
          }
          int reader_stage, reader_order;
          std::tie(reader_stage, reader_order) = GetBlockStageOrder(reader_block);
          if (writer_order < reader_order && writer_stage < reader_stage &&
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

  void RemapPipelineBuffers(const Array<Block>& blocks, Array<Buffer> pipeline_allocs) {
    std::unordered_map<Buffer, BufferInfo, ObjectPtrHash, ObjectPtrEqual> infos =
        GetBufferInfo(blocks);
    for (const auto& pair : infos) {
      const Buffer& buffer = pair.first;
      const BufferInfo& buffer_info = pair.second;
      int num_versions = ComputeBufferVersions(blocks, buffer, buffer_info);
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

  Stmt EmitImpl(PrimExpr start, PrimExpr end, bool unroll_loop, const String& tag) {
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
      int stage =
          static_cast<int>(block->annotations.at(attr::pipeline_stage).as<IntImmNode>()->value);
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
      stmt = SeqStmt(stmts);
    } else {
      stmt = For(Downcast<Var>(new_loop_var), pipeline_loop_->min, end - start,
                 unroll_loop ? ForKind::kUnrolled : pipeline_loop_->kind, SeqStmt(stmts));
    }
    Block block = MakeBlock(stmt, buffer_data_to_buffer_);
    block.CopyOnWrite()->annotations.Set(tag, Bool(1));
    return BlockRealize({}, Bool(true), block);
  }

  arith::Analyzer analyzer_;
  For pipeline_loop_;
  int max_stage_ = -1;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Buffer> buffer_remap_;
  Array<Block> ordered_stmts_;
  const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>& double_buffers_;
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

  Array<Block> BlockizePipelineBody(const Array<Stmt>& stmts) {
    Array<Block> blocks;
    for (const Stmt& stmt : stmts) {
      const auto* block_realize = stmt.as<BlockRealizeNode>();
      if (block_realize && is_one(block_realize->predicate)) {
        blocks.push_back(block_realize->block);
      } else {
        blocks.push_back(MakeBlock(stmt, buffer_data_to_buffer_));
      }
    }
    return blocks;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // Step 1: Recursively rewrite the children first.
    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    auto it = for_node->annotations.find(attr::pipeline_scope);
    if (it == for_node->annotations.end()) {
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
        LOG(INFO) << "Push " << buffer->data;
        buffer_data_to_buffer_.Set(buffer->data, buffer);
      }
      pipeline_body = block->body;
      pipeline_allocs = block->alloc_buffers;
    } else {
      pipeline_body = for_node->body;
    }
    CHECK(pipeline_body->IsInstance<SeqStmtNode>())
        << "ValueError: The body of the software pipeline should be SeqStmt, got "
        << pipeline_body->GetTypeKey();
    Array<Block> blocks = BlockizePipelineBody(pipeline_body.as<SeqStmtNode>()->seq);
    Array<Block> annotated_body = AnnotatePipelineBody(blocks);
    PipelineRewriter rewriter(buffer_data_to_buffer_, double_buffers);
    Stmt pipeline = rewriter.BuildPipeline(annotated_body, pipeline_allocs, GetRef<For>(op));

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

    if (op->annotations.count("pragma_double_buffer")) {
      for (const auto& write : op->writes) {
        double_buffers.insert(write->buffer);
      }
    }
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));

    if (block->body->IsInstance<SeqStmtNode>()) {
      // Rewriting for software pipelining will produce nested SeqStmt. These statements need to be
      // flattened for rewriting outer software pipeline (if nested software pipelines are present).
      block = FlattenNestedBlocks(block);
    }

    for (const auto& buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
    }
    return block;
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> double_buffers;
};

}  // namespace pipeline_v2

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
    // fptr->body = inject_software_pipeline::PipelineInjector::Inject(
    //     cfg.value()->use_native_pipeline, f);
    fptr->body = pipeline_v2::PipelineInjector::Inject(f);
    fptr->body = ConvertSSA(std::move(fptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectSoftwarePipeline", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InjectSoftwarePipeline").set_body_typed(InjectSoftwarePipeline);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
