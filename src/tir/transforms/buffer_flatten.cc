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
 * \file buffer_flatten.cc
 */
#include <tvm/arith/int_set.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>

#include "../../support/utils.h"
#include "../schedule/utils.h"

namespace tvm {
namespace tir {

using namespace arith;

template <class K, class V>
using SMap = std::unordered_map<K, V, ObjectPtrHash, ObjectPtrEqual>;
template <class K>
using SSet = std::unordered_set<K, ObjectPtrHash, ObjectPtrEqual>;

Array<Range> NDIntSet2Region(const NDIntSet& nd_int_set) {
  Integer one(1);
  Array<Range> result;
  result.reserve(nd_int_set.size());
  for (const arith::IntSet& int_set : nd_int_set) {
    PrimExpr min = int_set.min();
    PrimExpr max = int_set.max();
    result.push_back(Range(/*begin=*/min, /*end=*/max + one));
  }
  return result;
}

bool IsThreadBound(const For& loop) {
  if (loop->kind != ForKind::kThreadBinding) {
    return false;
  }
  ICHECK(loop->thread_binding.defined());
  IterVar binding = loop->thread_binding.value();
  if (support::StartsWith(binding->thread_tag, "threadIdx")) {
    return true;
  }
  if (support::StartsWith(binding->thread_tag, "vthread")) {
    return true;
  }
  return false;
}

bool IsReduceTempBuffer(const Buffer& buffer) {
  return support::StartsWith(buffer->name, "normal_reduce_temp") ||  //
         support::StartsWith(buffer->name, "reduce_temp");
}

PrimExpr BufferArea(const Buffer& buffer);

Stmt RealizeInitBlock(const Stmt& init, const Array<IterVar>& iter_vars) {
  std::vector<PrimExpr> conditions;
  for (const IterVar& var : iter_vars) {
    if (var->iter_type == IterVarType::kCommReduce) {
      conditions.push_back(equal(var->var, var->dom->min));
    }
  }
  int n = conditions.size();
  // Handle the case where there is no condition
  if (n == 0) {
    return init;
  }
  // Concate the conditions with logical and (&&)
  PrimExpr cond = conditions[0];
  for (int i = 1; i < n; ++i) {
    cond = logical_and(cond, conditions[i]);
  }
  return IfThenElse(cond, init);
}

class ReductionTransformer : public StmtMutator {
 public:
  static Stmt Transform(const PrimFunc& f) { return ReductionTransformer().VisitStmt(f->body); }

 private:
  Stmt VisitStmt_(const BlockNode* block) override {
    if (!block->init.defined()) {
      return StmtMutator::VisitStmt_(block);
    }
    Stmt init = RealizeInitBlock(block->init.value(), block->iter_vars);
    Stmt body = VisitStmt(block->body);
    ObjectPtr<BlockNode> new_block = make_object<BlockNode>(*block);
    new_block->init = NullOpt;
    new_block->body = SeqStmt::Flatten(init, body);
    return Stmt(std::move(new_block));
  }
};

/*!
 * \brief Detecting the LCA of buffer access points of
 *        buffers for calculating the realize region
 */
class LCACollector : public StmtExprVisitor {
 public:
  static Map<Buffer, Optional<For>> Detect(const PrimFunc& func) {
    LCACollector detector;
    // Buffers, who appear as arguments, do not have allocation sites
    for (const auto& kv : func->buffer_map) {
      const Buffer& buffer = kv.second;
      detector.buffer_lca_.emplace(buffer.get(), nullptr);
    }
    detector(func->body);
    // Prepare the return
    Map<Buffer, Optional<For>> buffer_lca;
    for (const auto& kv : detector.buffer_lca_) {
      buffer_lca.Set(GetRef<Buffer>(kv.first), GetRef<Optional<For>>(kv.second));
    }
    return buffer_lca;
  }

 private:
  void VisitStmt_(const ForNode* op) final {
    int n = ancestor_loops_.size();
    for_info_.emplace(op, ForInfo{ancestor_loops_.back(), n});
    ancestor_loops_.push_back(op);
    StmtExprVisitor::VisitStmt_(op);
    ancestor_loops_.pop_back();
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    CalcBufferLCA(op->buffer.get());
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    CalcBufferLCA(op->buffer.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  void CalcBufferLCA(const BufferNode* buffer) {
    // We use `nullptr` as lca to represent the case that the lca is the root block.
    const auto it = buffer_lca_.find(buffer);
    if (it == buffer_lca_.end()) {
      // If the buffer is visited for the first time, the lca is the most recent For loop.
      buffer_lca_.emplace(buffer, ancestor_loops_.back());
      return;
    }
    const ForNode*& lca = buffer_lca_[buffer];
    lca = LowestCommonAncestor(lca, ancestor_loops_.back());
  }

  const ForNode* LowestCommonAncestor(const ForNode* lhs, const ForNode* rhs) const {
    if (lhs == nullptr || rhs == nullptr) {
      // The lca is the root block.
      return nullptr;
    }
    while (lhs != rhs) {
      auto it_l = for_info_.find(lhs);
      auto it_r = for_info_.find(rhs);
      ICHECK(it_l != for_info_.end());
      ICHECK(it_r != for_info_.end());
      const ForInfo& l = it_l->second;
      const ForInfo& r = it_r->second;
      if (l.depth == r.depth) {
        lhs = l.parent_loop;
        rhs = r.parent_loop;
      } else if (l.depth < r.depth) {
        rhs = r.parent_loop;
      } else {
        lhs = l.parent_loop;
      }
    }
    ICHECK_EQ(lhs, rhs);
    return lhs;
  }

  /*! \brief The AST node information for querying LCA */
  struct ForInfo {
    // The parent loop node
    const ForNode* parent_loop;
    // The scope depth in the AST
    int depth;
  };

  /*! \brief The current scope initializing with Null */
  std::vector<const ForNode*> ancestor_loops_ = {nullptr};
  /*! \brief The parent and depth info of each Loop/BufferLoad/BufferStore Node */
  std::unordered_map<const ForNode*, ForInfo> for_info_ = {};
  /*! \brief The map from Buffer to its LCA Stmt/Expr */
  std::unordered_map<const BufferNode*, const ForNode*> buffer_lca_ = {};
};

class BufferRewriter : public StmtExprMutator {
 public:
  using FRewriteBufferAccess = std::function<void(Buffer* buffer, Array<PrimExpr>* indices)>;

  static Stmt Rewrite(Stmt stmt, const FRewriteBufferAccess& f_rewrite) {
    BufferRewriter rewriter(f_rewrite);
    return rewriter.VisitStmt(stmt);
  }

 private:
  explicit BufferRewriter(const FRewriteBufferAccess& f_rewrite) : f_rewrite_(f_rewrite) {}

  Stmt VisitStmt_(const BufferStoreNode* _op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_op));
    BufferStoreNode* op = store.CopyOnWrite();
    f_rewrite_(&op->buffer, &op->indices);
    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* _op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_op));
    BufferLoadNode* op = load.CopyOnWrite();
    f_rewrite_(&op->buffer, &op->indices);
    return std::move(load);
  }

  Stmt VisitStmt_(const BlockNode* _op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(_op));
    BlockNode* op = block.CopyOnWrite();
    ArrayNode* reads = op->reads.CopyOnWrite();
    for (int i = 0, n = reads->size(); i < n; ++i) {
      BufferRegion buffer_region = Downcast<BufferRegion>(reads->at(i));
      BufferRegionNode* p = buffer_region.CopyOnWrite();
      f_rewrite_(&p->buffer, nullptr);
    }
    ArrayNode* writes = op->writes.CopyOnWrite();
    for (int i = 0, n = writes->size(); i < n; ++i) {
      BufferRegion buffer_region = Downcast<BufferRegion>(writes->at(i));
      BufferRegionNode* p = buffer_region.CopyOnWrite();
      f_rewrite_(&p->buffer, nullptr);
    }
    return std::move(block);
  }

  const FRewriteBufferAccess& f_rewrite_;
};

/*!
 * \brief Alloc the used region of each buffers.
 */
class BufferAllocator : public StmtExprMutator {
 public:
  static Stmt Alloc(const PrimFunc& f) {
    Map<Buffer, Optional<For>> buffer_lca = LCACollector::Detect(f);
    SMap<Buffer, BufferInfo> buffer_info;
    SMap<Optional<For>, Array<Buffer>> loop_allocs;
    buffer_info.reserve(buffer_lca.size());
    loop_allocs.reserve(buffer_lca.size());
    for (const auto& kv : buffer_lca) {
      const Buffer& buffer = kv.first;
      const Optional<For>& alloc_site = kv.second;
      int ndim = buffer->shape.size();
      buffer_info.emplace(buffer, BufferInfo(ndim, alloc_site));
      loop_allocs[alloc_site].push_back(buffer);
    }
    for (const auto& kv : f->buffer_map) {
      const Buffer& buffer = kv.second;
      ICHECK(buffer_info.count(buffer));
      BufferInfo& info = buffer_info.at(buffer);
      info.accessed_region = NDIntSetFromShape(buffer->shape);
      info.alloc_site = NullOpt;
      info.region = NDIntSet2Region(info.accessed_region);
      info.new_buffer = buffer;
      info.is_arg = true;
    }
    BufferAllocator alloc(std::move(buffer_info), std::move(loop_allocs));
    Stmt stmt = alloc.VisitStmt(f->body);
    stmt = BufferRewriter::Rewrite(
        /*stmt=*/std::move(stmt),
        /*f_rewrite=*/std::bind(&BufferAllocator::RewriteBufferAccess,
                                &alloc,                 //
                                std::placeholders::_1,  //
                                std::placeholders::_2));
    return stmt;
  }

 private:
  struct BufferInfo {
    NDIntSet accessed_region;
    Optional<For> alloc_site;
    Array<Range> region;
    Buffer new_buffer;
    bool is_arg;

    explicit BufferInfo(int ndim, Optional<For> alloc_site)
        : accessed_region(NDIntSetEmpty(ndim)),
          alloc_site(std::move(alloc_site)),
          region{nullptr},
          new_buffer{nullptr},
          is_arg(false) {}
  };

  explicit BufferAllocator(SMap<Buffer, BufferInfo> buffer_info,
                           SMap<Optional<For>, Array<Buffer>> loop_allocs)
      : block_nest_depth_(0),
        buffer_info_(std::move(buffer_info)),
        loop_allocs_(std::move(loop_allocs)),
        ancestor_loops_{},
        var_substitutes_{},
        reduction_loop_vars_{} {}

  Stmt VisitStmt_(const ForNode* loop) final {
    // Step 1. Handle block vars in `min` and `extent`
    PrimExpr min = this->VisitExpr(loop->min);
    PrimExpr extent = this->VisitExpr(loop->extent);
    // Step 2. Handle unit loops
    if (is_one(extent)) {
      var_substitutes_[loop->loop_var] = min;
    }
    // Step 3. Visit recursively
    ancestor_loops_.push_back(GetRef<For>(loop));
    Stmt body = this->VisitStmt(loop->body);
    ancestor_loops_.pop_back();
    // Step 4. Add allocation
    Array<Buffer> alloc_buffers = AllocBufferUnderLoop(GetRef<For>(loop));
    if (!alloc_buffers.empty()) {
      body = BlockRealize(/*iter_values=*/{},
                          /*predicate=*/const_true(),
                          /*block=*/
                          Block(/*iter_vars=*/{},                            //
                                /*reads=*/{},                                //
                                /*writes=*/{},                               //
                                /*name_hint=*/"alloc",                       //
                                /*body=*/std::move(body),                    //
                                /*init=*/NullOpt,                            //
                                /*alloc_buffers=*/std::move(alloc_buffers),  //
                                /*match_buffers=*/{},                        //
                                /*annotations=*/{}                           //
                                ));
    }
    // Step 5. Make the new loop
    if (loop->kind == ForKind::kThreadBinding && reduction_loop_vars_.count(loop->loop_var)) {
      // do nothing, because the loop is going to be removed
    } else {
      body = For(/*loop_var=*/loop->loop_var,
                 /*min=*/min,
                 /*extent=*/extent,
                 /*kind=*/loop->kind,
                 /*body=*/std::move(body),
                 /*thread_binding=*/loop->thread_binding,
                 /*annotations=*/loop->annotations);
    }
    return body;
  }

  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    const auto* block = realize->block.get();
    ICHECK(!block->init.defined());
    // Step 1. Update "block vars => loop vars" for substitution, add reduction loop vars
    ICHECK_EQ(block->iter_vars.size(), realize->iter_values.size());
    for (int i = 0, n = block->iter_vars.size(); i < n; ++i) {
      IterVar block_var = block->iter_vars[i];
      PrimExpr v = this->VisitExpr(realize->iter_values[i]);
      var_substitutes_[block_var->var] = v;
      if (block_var->iter_type == kCommReduce) {
        for (const Var& var : UndefinedVars(v)) {
          this->reduction_loop_vars_.insert(var);
        }
      }
    }
    // Step 2. Visit recursively
    ++block_nest_depth_;
    Stmt body = this->VisitStmt(block->body);
    --block_nest_depth_;
    // Step 3. Update the read/write buffer regions
    Array<BufferRegion> reads = VisitBufferRegions(block->reads);
    Array<BufferRegion> writes = VisitBufferRegions(block->writes);
    // Step 4. Handle predicate
    PrimExpr predicate = this->VisitExpr(realize->predicate);
    // Step 5. Root allocation
    Array<Buffer> alloc_buffers =
        (block_nest_depth_ == 0) ? AllocBufferUnderLoop(NullOpt) : Array<Buffer>{};
    // Step 6. Create new blocks
    return BlockRealize(/*iter_values=*/{},
                        /*predicate=*/std::move(predicate),
                        /*block=*/
                        Block(/*iter_vars=*/{},                            //
                              /*reads=*/std::move(reads),                  //
                              /*writes=*/std::move(writes),                //
                              /*name_hint=*/block->name_hint,              //
                              /*body=*/std::move(body),                    //
                              /*init=*/NullOpt,                            //
                              /*alloc_buffers=*/std::move(alloc_buffers),  //
                              /*match_buffers=*/block->match_buffers,      //
                              /*annotations=*/block->annotations           //
                              ));
  }

  PrimExpr VisitExpr_(const VarNode* var) final {
    auto it = var_substitutes_.find(GetRef<Var>(var));
    if (it != var_substitutes_.end()) {
      return it->second;
    }
    return GetRef<Var>(var);
  }

  Array<BufferRegion> VisitBufferRegions(const Array<BufferRegion>& buffer_regions) {
    // Calculate `new_buffer_regions` by recursively visiting min/extent of each range
    Array<BufferRegion> new_buffer_regions;
    new_buffer_regions.reserve(buffer_regions.size());
    for (const BufferRegion& buffer_region : buffer_regions) {
      const Buffer& buffer = buffer_region->buffer;
      const Array<Range>& region = buffer_region->region;
      Array<Range> new_region;
      new_region.reserve(region.size());
      for (const Range& range : region) {
        new_region.push_back(Range::FromMinExtent(/*min=*/this->VisitExpr(range->min),
                                                  /*extent=*/this->VisitExpr(range->extent)));
      }
      new_buffer_regions.push_back(BufferRegion(buffer, new_region));
    }
    // Calculate `info.accessed_region`
    for (const BufferRegion& buffer_region : new_buffer_regions) {
      const Buffer& buffer = buffer_region->buffer;
      ICHECK(buffer_info_.count(buffer));
      BufferInfo& info = buffer_info_.at(buffer);
      if (info.is_arg) {
        continue;
      }
      std::unordered_map<const VarNode*, arith::IntSet> dom_map;
      {
        const Object* alloc_site = info.alloc_site.get();
        // Every loop will be relaxed if the lca is the root
        bool need_relax = (alloc_site == nullptr);
        for (const For& loop : this->ancestor_loops_) {
          const VarNode* loop_var = loop->loop_var.get();
          if (need_relax || (buffer.scope() == "shared" && IsThreadBound(loop))) {
            // TODO
            dom_map[loop_var] = IntSetFromMinExtent(loop->min, loop->extent);
          }
          if (loop.get() == alloc_site) {
            need_relax = true;
          }
        }
      }
      NDIntSet int_set;
      int_set.reserve(buffer_region->region.size());
      for (const Range& range : buffer_region->region) {
        int_set.push_back(arith::EvalSet(range, dom_map));
      }
      NDIntSetUnionWith(&info.accessed_region, int_set);
    }
    return new_buffer_regions;
  }

  Array<Buffer> AllocBufferUnderLoop(const Optional<For>& loop) {
    auto it = loop_allocs_.find(loop);
    if (it == loop_allocs_.end()) {
      return {};
    }
    const Array<Buffer>& buffers = it->second;
    Array<Buffer> result;
    result.reserve(buffers.size());
    for (const Buffer& buffer : buffers) {
      ICHECK(buffer_info_.count(buffer));
      BufferInfo& info = buffer_info_.at(buffer);
      if (info.is_arg) {
        ICHECK(info.region.defined());
        ICHECK(info.new_buffer.defined());
        continue;
      } else {
        ICHECK(!info.region.defined());
        ICHECK(!info.new_buffer.defined());
      }
      // Calculate `info.region`
      info.region = NDIntSet2Region(info.accessed_region);
      // Calculate `info.new_buffer`
      Array<PrimExpr> shape;
      shape.reserve(info.region.size());
      for (const Range& range : info.region) {
        shape.push_back(range->extent);
      }
      ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*buffer.get());
      new_buffer->shape = std::move(shape);
      info.new_buffer = Buffer(std::move(new_buffer));
      result.push_back(info.new_buffer);
    }
    return result;
  }

  void RewriteBufferAccess(Buffer* buffer, Array<PrimExpr>* indices) const {
    ICHECK(buffer_info_.count(*buffer));
    const BufferInfo& info = buffer_info_.at(*buffer);
    ICHECK(info.new_buffer.defined());
    if (indices == nullptr) {
      *buffer = info.new_buffer;
      return;
    }
    ICHECK(info.region.defined());
    // TODO: the ndim could be changed by tensorize, more investigation is needed
    ICHECK_GE(indices->size(), info.region.size());
    int ndim = info.region.size();
    Array<PrimExpr> new_indices;
    new_indices.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      new_indices.push_back((*indices)[i] - info.region[i]->min);
    }
    *buffer = info.new_buffer;
    *indices = std::move(new_indices);
  }

  /*! \brief Number of blocks nested in the ancestor during visiting */
  int block_nest_depth_;
  /*! \brief Collective information about each buffer */
  SMap<Buffer, BufferInfo> buffer_info_;
  /*! \brief Buffers allocated at each for loop */
  SMap<Optional<For>, Array<Buffer>> loop_allocs_;
  /*! \brief The loops from the current node up to the root */
  std::vector<For> ancestor_loops_;
  /*! \brief The map from block vars to the expr value */
  SMap<Var, PrimExpr> var_substitutes_;
  /*! \brief Loop variables that are bound to reduction block vars */
  SSet<Var> reduction_loop_vars_;
};

/*!
 * \brief Transform multi-dimension BufferLoad/BufferStore into one-dimension Load/Store
 */
class Flattener : public StmtExprMutator {
 public:
  static Stmt Flatten(const PrimFunc& f) { return Flattener().VisitStmt(f->body); }

 private:
  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    ICHECK(realize->iter_values.empty());
    // Step 1. Visit the body
    Block new_block = Downcast<Block>(this->VisitStmt(realize->block));
    PrimExpr predicate = this->VisitExpr(realize->predicate);
    const BlockNode* block = new_block.get();
    // Step 2. Transform the `predicate` to if-then-else
    Stmt body = block->body;
    if (!is_one(predicate)) {
      body = IfThenElse(predicate, body);
    }
    // Step 3. Pick out blocks that writes with double buffering
    if (IsDoubleBufferScope(block->annotations)) {
      ICHECK_EQ(block->writes.size(), 1);
      const Buffer& write = block->writes[0]->buffer;
      double_buffered_.insert(write);
    }
    // Step 4. Handle allocations
    for (const Buffer& buffer : block->alloc_buffers) {
      body = MakeAllocStmt(buffer, body, double_buffered_.count(buffer));
    }
    return body;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // Step 1. Visit recursively
    PrimExpr min = this->VisitExpr(op->min);
    PrimExpr extent = this->VisitExpr(op->extent);
    Stmt body = this->VisitStmt(op->body);
    // Step 2. Add the for loop accordingly
    if (op->kind == ForKind::kThreadBinding) {
      // Case 1. Thread binding
      ICHECK(op->thread_binding.defined());
      String thread_tag = op->thread_binding.value()->thread_tag;
      body = MakeLaunchThread(min, extent, op->loop_var, thread_tag, body);
    } else if (is_one(extent) && op->annotations.empty()) {
      // Case 2. Handle unit loop
      return body;
    } else {
      // Case 3. An ordinary loop
      body = For(op->loop_var, min, extent, op->kind, body);
    }
    // Step 3. Handle annotations
    for (const auto& annotation : op->annotations) {
      const String& ann_key = annotation.first;
      const ObjectRef& ann_value = annotation.second;
      if (attr::IsPragmaKey(ann_key) || ann_key == attr::pipeline_scope) {
        body = AttrStmt(op->loop_var, ann_key, Downcast<PrimExpr>(ann_value), body);
      }
    }
    return body;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    op = store.get();
    return op->buffer.vstore(op->indices, op->value);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    op = load.get();
    return op->buffer.vload(op->indices, op->dtype);
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::get_elem_offset())) {
      // Handle `get_elem_offset`
      ICHECK_EQ(op->args.size(), 1);
      PrimExpr arg = op->args[0];
      ICHECK(arg->IsInstance<BufferLoadNode>());
      arg = this->VisitExpr(arg);
      const auto* load = TVM_TYPE_AS(load, arg, LoadNode);
      return load->index;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt MakeAllocStmt(const Buffer& buffer, Stmt body, bool is_double_buffer) {
    if (IsReduceTempBuffer(buffer)) {
      return body;
    }
    String storage_scope = buffer.scope();
    if (storage_scope.empty()) {
      storage_scope = "global";
    }
    PrimExpr area = analyzer_.Simplify(BufferArea(buffer));
    body = Allocate(buffer->data, buffer->dtype, {area}, const_true(), body);
    body = AttrStmt(buffer->data, attr::storage_scope, StringImm(storage_scope), body);
    if (is_double_buffer) {
      body = AttrStmt(buffer->data, attr::double_buffer_scope, Integer(1), body);
    }
    return body;
  }

  static Stmt MakeLaunchThread(const PrimExpr& min, const PrimExpr& extent, const Var& var,
                               const String& thread_tag, Stmt body) {
    IterVar iter_var(/*dom=*/Range::FromMinExtent(min, extent),
                     /*var=*/var,
                     /*iter_type=*/IterVarType::kThreadIndex,
                     /*thread_tag=*/thread_tag);
    String attr_key = thread_tag == "vthread" ? attr::virtual_thread : attr::thread_extent;
    body = AttrStmt(iter_var, attr_key, extent, body);
    return body;
  }

  static bool IsDoubleBufferScope(const Map<String, ObjectRef>& annotations) {
    if (Optional<ObjectRef> ann_value = annotations.Get(attr::double_buffer_scope)) {
      const auto* value = TVM_TYPE_AS(value, ann_value, PrimExprNode);
      if (is_one(GetRef<PrimExpr>(value))) {
        return true;
      }
    }
    return false;
  }

  SSet<Buffer> double_buffered_;
  arith::Analyzer analyzer_;
};

PrimFunc BufferFlatten(PrimFunc f) {
  PrimFuncNode* fptr = f.CopyOnWrite();
  // Step 1. Transform the reduction calls to BufferStore
  fptr->body = ReductionTransformer::Transform(f);
  // Step 2. Recalculate the buffer region
  fptr->body = BufferAllocator::Alloc(f);
  // Step 3. Transform BufferLoad/BufferStore into Load/Store
  fptr->body = Flattener::Flatten(f);
  return f;
}

namespace transform {

Pass BufferFlatten() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return BufferFlatten(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.BufferFlatten", {});
}

TVM_REGISTER_GLOBAL("tir.transform.BufferFlatten").set_body_typed(BufferFlatten);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
