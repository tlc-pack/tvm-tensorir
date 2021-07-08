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

/*! \brief Information about a software pipeline that will be used in the transformation */
struct PipelineInfo {
  // Buffers written by the producers. These buffers can only be read by the consumers.
  SSet<Var> producer_buffers;
  // Producers of the pipeline.
  Array<Stmt> producers;
  // Consumers of the pipeline.
  Array<Stmt> consumers;
  // Storage scope of the pipeline. The scope is the same as the storage scope of the producer
  // buffers. Producer buffers are required to have the same storage scope.
  String scope;
  // Number of stages of the pipeline.
  Integer num_stages;
  // The loop variable of the pipelined loop.
  Var loop_var;
  // Buffer allocations that need to be relocated outside of the pipeline after the transformation.
  Array<Var> buffer_allocs;

  PipelineInfo(const SSet<Var>& producer_buffers, const Array<Stmt>& producers,
               const Array<Stmt>& consumers, const String& scope, const Integer& num_stages,
               const Var& loop_var, const Array<Var>& buffer_allocs)
      : producer_buffers(producer_buffers),
        producers(producers),
        consumers(consumers),
        scope(scope),
        num_stages(num_stages),
        loop_var(loop_var),
        buffer_allocs(buffer_allocs) {}
};

/* \brief Information about a buffer allocation.
 * \note In TIR, a buffer allocation is consist of one or more AttrStmt followed by Allocate.
 * This structure holds reference of these statements so that it can be used to rebuild the buffer
 * allocation during the software pipeline transformaion.
 */
struct BufferInfo {
  // The first AttrStmt related to the buffer.
  Stmt annotation;
  // The Allocate statement of the buffer.
  Allocate allocate;
  // The storage scope of the buffer.
  String scope;
  BufferInfo(const Stmt& annotation, const Allocate& allocate, const String& scope)
      : annotation(annotation), allocate(allocate), scope(scope) {}
};

/*!
 * \brief Strips AttrStmt of the buffer and get the closest nested Allocate.
 * \param attr_node The AttrStmt related to the buffer.
 */
static Allocate GetBufferAllocate(const AttrStmtNode* attr_node) {
  while (attr_node) {
    ICHECK(attr_node->attr_key == tir::attr::storage_scope ||
           attr_node->attr_key == tir::attr::double_buffer_scope);
    if (attr_node->body.as<AllocateNode>()) {
      return Downcast<Allocate>(attr_node->body);
    }
    attr_node = attr_node->body.as<AttrStmtNode>();
  }
  ICHECK(false) << "unreachable";
  throw;
}

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
    if (const auto* store = obj.as<StoreNode>()) {
      access.writes.insert(store->buffer_var);
    } else if (const auto* load = obj.as<LoadNode>()) {
      access.reads.insert(load->buffer_var);
    } else if (const auto* call = obj.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_access_ptr())) {
        ICHECK(call->args.size() == 5U);
        Var buffer_var = Downcast<Var>(call->args[1]);
        int64_t rw_mask = Downcast<Integer>(call->args[4])->value;
        if (rw_mask & 1) {
          access.reads.insert(buffer_var);
        }
        if (rw_mask & 2) {
          access.writes.insert(buffer_var);
        }
        return false;
      }
    }
    return true;
  });
  return access;
}

/*!
 * \brief Detect the annotated pipeline loop and generate information that will be used for the
 * software pipeline transformation later.
 */
class PipelineDetector : public StmtVisitor {
 public:
  SMap<Var, BufferInfo> buffer_info_;
  SMap<AttrStmt, PipelineInfo> pipeline_info_;

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

  /*!
   * \brief Make the plan for the pipeline transformation for a AST subtree.
   * \param pipeline_scope The AttrStmt that annotates the for-loop for software pipelining.
   *
   * This function analyzes the dependencies among the children of the software pipelined for-loop,
   * generates and stores the information of the pipeline in `pipeline_info_`.
   */

  void PlanPipeline(const AttrStmtNode* pipeline_scope) {
    CHECK(current_pipeline_scope_ == nullptr) << "ValueError: Nested pipeline is not allowed.";
    current_pipeline_scope_ = pipeline_scope;
    StmtVisitor::VisitStmt_(pipeline_scope);
    current_pipeline_scope_ = nullptr;

    Integer num_stages = Downcast<Integer>(pipeline_scope->value);
    CHECK_GE(num_stages->value, 2) << "ValueError: Pipeline should have at least two stages.";

    const ForNode* op = TVM_TYPE_AS(op, pipeline_scope->body, ForNode);
    // The body of the annotated pipeline for-loop should be optional buffer allocations followed by
    // SeqStmt.
    Array<Var> buffer_allocs;
    Stmt stmt = GetRef<Stmt>(op);
    const auto* attr_node = op->body.as<AttrStmtNode>();
    while (attr_node) {
      Allocate alloc = GetBufferAllocate(attr_node);
      buffer_allocs.push_back(alloc->buffer_var);
      stmt = alloc->body;
      attr_node = stmt.as<AttrStmtNode>();
    }
    const SeqStmtNode* body = stmt.as<SeqStmtNode>();
    CHECK(body) << "ValueError: The body of the pipeline should be SeqStmt.";

    // Build the dependency graph from buffer accesses.

    // A map from a Stmt to its buffer access info.
    SMap<Stmt, BufferAccess> buffer_access;
    // A map from a Stmt to its dependants.
    SMap<Stmt, Array<Stmt>> dep_src2dst;
    // A map from a Stmt to its dependencies.
    SMap<Stmt, Array<Stmt>> dep_dst2src;
    BuildDependencyGraph(body, &buffer_access, &dep_src2dst, &dep_dst2src);

    // analyze dependencies among direct children of the pipeline loop
    Array<Stmt> producers, consumers;
    for (const auto& stmt : body->seq) {
      if (!dep_src2dst.count(stmt)) {
        consumers.push_back(stmt);
      } else {
        producers.push_back(stmt);
      }
    }
    // Find buffers that are written by producers and read by consumers.
    // These buffers need to be resized.
    SSet<Var> producer_buffers;
    for (const Stmt& consumer : consumers) {
      for (const Stmt& dependency : dep_dst2src[consumer]) {
        for (const Var& read : buffer_access.at(consumer).reads) {
          if (buffer_access.at(dependency).writes.count(read)) {
            producer_buffers.insert(read);
          }
        }
      }
    }

    CHECK(!producers.empty()) << "ValueError: Producer not found in the pipeline.";
    CHECK(!consumers.empty()) << "ValueError: Consumer not found in the pipeline.";
    CHECK(!producer_buffers.empty()) << "ValueError: Producer buffer not found in the pipeline.";

    // Check the consistency of pipeline scope.
    String scope = buffer_info_.at(*producer_buffers.begin()).scope;
    for (const Var& buffer : producer_buffers) {
      CHECK_EQ(buffer_info_.at(buffer).scope, scope) << "ValueError: Inconsistent scopes among "
                                                        "buffers of pipeline producers";
    }
    pipeline_info_.emplace(GetRef<AttrStmt>(pipeline_scope),
                           PipelineInfo{producer_buffers, producers, consumers, scope, num_stages,
                                        op->loop_var, buffer_allocs});
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::pipeline_scope) {
      PlanPipeline(op);
      return;
    }

    StmtVisitor::VisitStmt_(op);
    AttrStmt attr_stmt = GetRef<AttrStmt>(op);
    if (op->attr_key == tir::attr::storage_scope) {
      Allocate allocate = Downcast<Allocate>(op->body);
      buffer_info_.emplace(allocate->buffer_var,
                           BufferInfo{attr_stmt, allocate, Downcast<StringImm>(op->value)->value});
    } else if (op->attr_key == tir::attr::double_buffer_scope) {
      buffer_info_.at(Downcast<Var>(op->node)).annotation = attr_stmt;
    }
  }

  const AttrStmtNode* current_pipeline_scope_ = nullptr;
};

/*!
 * \brief Use the pipeline information produced by PipelineDetector to transform the IR.
 *
 * Given a for-loop annotated with pipeline_scope, this pass does the following transformation.
 *
 * Input:
 * \code
 * AttrStmt(pipeline_scope, num_stages)
 * for ax in range(min, min + extent):
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
 * AttrStmt(pipeline_scope, 1)
 * for ax in range(min, min + extent + shift_extent):
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
  static Stmt Inject(bool use_native_pipeline, const Stmt& stmt) {
    PipelineDetector detector;
    detector(stmt);
    PipelineInjector injector(use_native_pipeline, detector.pipeline_info_, detector.buffer_info_);
    Stmt new_stmt = injector(stmt);
    return ConvertSSA(new_stmt);
  }

  PipelineInjector(bool use_native_pipeline, const SMap<AttrStmt, PipelineInfo>& pipeline_info,
                   const SMap<Var, BufferInfo>& buffer_info)
      : pipeline_info_(pipeline_info),
        buffer_info_(buffer_info),
        use_native_pipeline_(use_native_pipeline) {
    DetectNativePipeline();
    for (const auto& kv : pipeline_info_) {
      for (const auto& buffer : kv.second.producer_buffers) {
        skip_allocs.emplace(buffer_info_.at(buffer).annotation);
      }
    }
  }

 private:
  Stmt BuildPipeline(const AttrStmt& pipeline_scope) {
    const PipelineInfo* pipeline_info = &pipeline_info_.at(pipeline_scope);
    std::swap(pipeline_info, current_pipeline_);

    For pipeline_loop = Downcast<For>(pipeline_scope->body);
    PrimExpr shift_extent = Integer(current_pipeline_->num_stages->value - 1);

    // Step 1: Initialize pipeline_var for native pipeline, which will be used in the native
    // pipeline API calls
    if (use_native_pipeline_) {
      pipeline_var_ = Var("pipeline", DataType::Handle());
    }

    // Step 2: Mutate children to rewrite pipeline buffer access.
    Array<Stmt> producers, consumers;
    for (const auto& stmt : current_pipeline_->producers) {
      producers.push_back(VisitStmt(stmt));
    }
    for (const auto& stmt : current_pipeline_->consumers) {
      consumers.push_back(VisitStmt(stmt));
    }

    // Step 3: Build each part of the pipeline
    Stmt prologue = BuildPrologue(producers, pipeline_loop, shift_extent);
    Stmt epilogue = BuildEpilogue(consumers, pipeline_loop, shift_extent);
    Stmt main_loop = BuildMainLoop(producers, consumers, pipeline_loop, shift_extent);
    // Annotate the main loop so that thread_storage_sync will skip this part
    main_loop = AttrStmt(Stmt(), tir::attr::pipeline_scope, Integer(1), main_loop);

    Array<Stmt> pipeline_seq;
    if (use_native_pipeline_) {
      pipeline_seq = {prologue, main_loop, epilogue};
    } else {
      pipeline_seq = {prologue, GetPipelineSync(), main_loop, epilogue};
    }
    Stmt pipeline = SeqStmt(pipeline_seq);

    // Step 4: Create the native pipeline object if necessary
    if (use_native_pipeline_) {
      PrimExpr create_pipeline = Call(DataType::Handle(), builtin::tvm_create_pipeline(), {});
      pipeline = LetStmt(pipeline_var_.value(), create_pipeline, pipeline);
    }

    // Step 5: Add buffer allocation
    std::vector<Stmt> allocs;
    Stmt no_op = Evaluate(0);
    for (const Var& buffer_var : current_pipeline_->buffer_allocs) {
      Stmt stmt = buffer_info_.at(buffer_var).annotation;
      while (const auto* attr_node = stmt.as<AttrStmtNode>()) {
        allocs.push_back(AttrStmt(attr_node->node, attr_node->attr_key, attr_node->value, no_op));
        stmt = attr_node->body;
      }
      const auto* alloc_node = TVM_TYPE_AS(alloc_node, stmt, AllocateNode);
      if (current_pipeline_->producer_buffers.count(buffer_var)) {
        ICHECK(alloc_node->extents.size() == 1U);
        PrimExpr new_extent = alloc_node->extents[0] * current_pipeline_->num_stages;
        Allocate new_alloc(alloc_node->buffer_var, alloc_node->dtype, {new_extent},
                           alloc_node->condition, no_op);
        allocs.push_back(new_alloc);
      } else {
        Allocate new_alloc(alloc_node->buffer_var, alloc_node->dtype, alloc_node->extents,
                           alloc_node->condition, no_op);
        allocs.push_back(new_alloc);
      }
    }

    std::swap(pipeline_info, current_pipeline_);
    pipeline_var_ = NullOpt;
    return MergeNest(allocs, pipeline);
  }

  Stmt GetPipelineSync() {
    return Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(),
                         Array<PrimExpr>{StringImm(current_pipeline_->scope)}));
  }

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
  Stmt WrapNativeConsumer(const Stmt& consumer) {
    ICHECK(use_native_pipeline_);
    ICHECK(pipeline_var_.defined());
    Stmt consumer_wait = Evaluate(
        Call(DataType::Handle(), builtin::tvm_pipeline_consumer_wait(), {pipeline_var_.value()}));
    Stmt consumer_release = Evaluate(
        Call(DataType::Handle(), builtin::tvm_pipeline_consumer_release(), {pipeline_var_.value()}));
    Stmt storage_sync = GetPipelineSync();
    return SeqStmt::Flatten(consumer_wait, storage_sync, consumer, consumer_release);
  }

  Stmt BuildPrologue(const Array<Stmt>& producers, For pipeline_loop,
                     const PrimExpr& shift_extent) {
    Stmt producer = SeqStmt::Flatten(producers);
    if (use_native_pipeline_) {
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

  Stmt BuildEpilogue(const Array<Stmt>& consumers, For pipeline_loop,
                     const PrimExpr& shift_extent) {
    Stmt consumer = SeqStmt::Flatten(consumers);
    if (use_native_pipeline_) {
      consumer = WrapNativeConsumer(consumer);
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

  Stmt ScheduleMainLoop(const Array<Stmt>& producers, const Array<Stmt>& consumers) {
    // Schedule the execution of producers and consumers. Producers and consumers are assumed to be
    // independant and can be executed concurrently. The schedule can be target-dependant.
    Stmt storage_sync = Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(),
                                      {StringImm(current_pipeline_->scope)}));
    // default case: run producers and consumers sequentially.
    Stmt producer = SeqStmt::Flatten(producers);
    Stmt consumer = SeqStmt::Flatten(consumers);
    if (use_native_pipeline_) {
      producer = WrapNativeProducer(producer);
      consumer = WrapNativeConsumer(consumer);
    }
    if (!use_native_pipeline_ || current_pipeline_->num_stages->value == 2) {
      return SeqStmt::Flatten(producer, consumer, storage_sync);
    } else {
      return SeqStmt::Flatten(producer, consumer);
    }
  }

  Stmt BuildMainLoop(const Array<Stmt>& producers, const Array<Stmt>& consumers, For pipeline_loop,
                     const PrimExpr& shift_extent) {
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
    main_loop->body = ScheduleMainLoop(shifted_producers, consumers);
    return pipeline_loop;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) {
    // Skip allocate of pipeline buffers in the original TensorIR AST. These buffers should be
    // allocated later outside the pipeline scope.
    if (skip_allocs.count(GetRef<Stmt>(op))) {
      Allocate alloc = GetBufferAllocate(op);
      return VisitStmt(alloc->body);
    }
    AttrStmt attr_stmt = GetRef<AttrStmt>(op);
    if (pipeline_info_.count(attr_stmt)) {
      Stmt new_stmt = BuildPipeline(attr_stmt);
      return new_stmt;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  /*!
   * \brief Rewrite accesses to the producer buffers after they are resized for the pipeline.
   * \param buffer_var The buffer variable.
   * \param index The index of he buffer access.
   * \return The updated index for accessing the resized buffer.
   */
  PrimExpr RewriteProducerBufferAccess(const Var& buffer_var, const PrimExpr& index) {
    const auto& extents = buffer_info_.at(buffer_var).allocate->extents;
    ICHECK(extents.size() == 1U);
    PrimExpr stride = extents[0];
    return indexmod(current_pipeline_->loop_var, current_pipeline_->num_stages) * stride + index;
  }

  Stmt VisitStmt_(const StoreNode* op) {
    Store store = Downcast<Store>(StmtExprMutator::VisitStmt_(op));
    if (current_pipeline_ && current_pipeline_->producer_buffers.count(store->buffer_var)) {
      PrimExpr new_index = RewriteProducerBufferAccess(store->buffer_var, store->index);
      store = Store(store->buffer_var, store->value, new_index, store->predicate);
    }
    return store;
  }

  PrimExpr VisitExpr_(const LoadNode* op) {
    Load load = Downcast<Load>(StmtExprMutator::VisitExpr_(op));
    if (current_pipeline_ && current_pipeline_->producer_buffers.count(load->buffer_var)) {
      PrimExpr new_index = RewriteProducerBufferAccess(load->buffer_var, load->index);
      load = Load(load->dtype, load->buffer_var, new_index, load->predicate);
    }
    return load;
  }

  PrimExpr VisitExpr_(const CallNode* op) {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (call->op.same_as(builtin::tvm_get_pipeline())) {
      CHECK(pipeline_var_.defined())
          << "ValueError: intrinsic tvm_get_pipeline can only be called inside the pipeline scope.";
      return pipeline_var_.value();
    } else if (call->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK(call->args.size() == 5U);
      Var buffer_var = Downcast<Var>(call->args[1]);
      if (current_pipeline_ && current_pipeline_->producer_buffers.count(buffer_var)) {
        PrimExpr elem_offset = call->args[2];
        elem_offset = RewriteProducerBufferAccess(buffer_var, elem_offset);
        Array<PrimExpr> new_args(call->args);
        new_args.Set(2, elem_offset);
        return Call(call->dtype, call->op, new_args);
      }
    }
    return call;
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

  // Information of the current pipeline.
  const PipelineInfo* current_pipeline_ = nullptr;
  // A map from annotated pipeline statements to the information for the transformation.
  const SMap<AttrStmt, PipelineInfo>& pipeline_info_;
  // A map from buffer variables to their information.
  const SMap<Var, BufferInfo>& buffer_info_;
  // Buffer allocations that need to be skipped as they will be regenerated by the pipeline
  // transformation.
  SSet<Stmt> skip_allocs;
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
        cfg.value()->use_native_pipeline, std::move(fptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectSoftwarePipeline", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InjectSoftwarePipeline").set_body_typed(InjectSoftwarePipeline);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
