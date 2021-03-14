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
#include <tvm/ir/attrs.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../schedule/utils.h"

namespace tvm {
namespace tir {

using NDIntSet = std::vector<arith::IntSet>;

arith::IntSet IntSetFromMinExtent(const PrimExpr& min, const PrimExpr& extent) {
  return arith::IntSet::FromRange(Range::FromMinExtent(min, extent));
}

void UnionWith(NDIntSet* lhs, const NDIntSet& rhs) {
  ICHECK_EQ(lhs->size(), rhs.size());
  int ndim = rhs.size();
  for (int i = 0; i < ndim; ++i) {
    arith::IntSet& int_set = lhs->at(i);
    int_set = arith::Union({int_set, rhs.at(i)});
  }
}

PrimExpr NDIntSetArea(const NDIntSet& nd_int_set) {
  PrimExpr area = 1;
  for (const arith::IntSet& int_set : nd_int_set) {
    area = area * (int_set.max() - int_set.min() + 1);
  }
  return area;
}

Buffer NDIntSet2Buffer(const BufferNode* buffer, const NDIntSet& nd_int_set) {
  Integer one(1);
  Array<PrimExpr> shape;
  shape.reserve(nd_int_set.size());
  for (const arith::IntSet& int_set : nd_int_set) {
    PrimExpr extent = int_set.max() - int_set.min() + one;
    shape.push_back(extent);
  }
  ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*buffer);
  new_buffer->shape = std::move(shape);
  return Buffer(std::move(new_buffer));
}

NDIntSet NDIntSetFromShape(const Array<PrimExpr>& shape) {
  NDIntSet result;
  for (const PrimExpr& extent : shape) {
    result.push_back(IntSetFromMinExtent(Integer(0), extent));
  }
  return result;
}

bool IsThreadBound(const ForNode* loop) {
  if (loop->kind != ForKind::kThreadBinding) {
    return false;
  }
  ICHECK(loop->thread_binding.defined());
  std::string thread_tag = loop->thread_binding.value()->thread_tag;
  if (StartsWith(thread_tag, "threadIdx")) {
    return true;
  }
  if (StartsWith(thread_tag, "vthread")) {
    return true;
  }
  return false;
}

bool IsReduceTempBuffer(const Buffer& buffer) {
  return StartsWith(buffer->name, "normal_reduce_temp") ||  //
         StartsWith(buffer->name, "reduce_temp");
}

String NormalizeStorageScope(const String& s) {
  if (s.empty()) {
    return "global";
  }
  return s;
}

Stmt MakeAllocStmt(const Buffer& buffer, const PrimExpr& area, Stmt body) {
  body = Allocate(buffer->data, buffer->dtype, {area}, const_true(), body);
  body = AttrStmt(buffer->data, attr::storage_scope,
                  StringImm(NormalizeStorageScope(buffer->scope)), body);
  return body;
}

Stmt MakeLaunchThread(const PrimExpr& min, const PrimExpr& extent, const Var& var,
                      const String& thread_tag, Stmt body) {
  IterVar iter_var(/*dom=*/Range::FromMinExtent(min, extent),
                   /*var=*/var,
                   /*iter_type=*/IterVarType::kThreadIndex,
                   /*thread_tag=*/thread_tag);
  String attr_key = thread_tag == "vthread" ? attr::virtual_thread : attr::thread_extent;
  body = AttrStmt(iter_var, attr_key, extent, body);
  return body;
}

PrimExpr BufferArea(const Buffer& buffer) {
  PrimExpr area = Integer(1);
  for (const PrimExpr& dim : buffer->shape) {
    area = area * dim;
  }
  return area;
}

class ReductionTransformer : public StmtMutator {
 public:
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
class LCADetector : public StmtExprVisitor {
 public:
  static Map<Buffer, Optional<For>> Detect(const PrimFunc& func) {
    LCADetector detector;
    // Buffers, who appear as arguments, do not have allocation sites
    for (const auto& kv : func->buffer_map) {
      const Buffer& buffer = kv.second;
      detector.buffers_lca_.emplace(buffer.get(), nullptr);
    }
    detector(func->body);
    // Prepare the return
    Map<Buffer, Optional<For>> buffer_lca;
    for (const auto& kv : detector.buffers_lca_) {
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
    const ForNode*& lca = buffers_lca_[buffer];
    lca = LowestCommonAncestor(lca, ancestor_loops_.back());
  }

  const ForNode* LowestCommonAncestor(const ForNode* lhs, const ForNode* rhs) const {
    while (lhs != nullptr && rhs != nullptr && lhs != rhs) {
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
    if (lhs == nullptr) {
      return rhs;
    }
    if (rhs == nullptr) {
      return lhs;
    }
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
  std::unordered_map<const BufferNode*, const ForNode*> buffers_lca_ = {};
};

class BufferAccessUpdater : public StmtExprMutator {
 public:
  static Stmt Update(
      const std::unordered_map<const BufferNode*, std::vector<PrimExpr>>& buffers_offsets,
      const std::unordered_map<const BufferNode*, const BufferNode*>& buffer_allocated, Stmt body) {
    BufferAccessUpdater updater(buffers_offsets, buffer_allocated);
    return updater.VisitStmt(body);
  }

 private:
  explicit BufferAccessUpdater(
      const std::unordered_map<const BufferNode*, std::vector<PrimExpr>>& buffers_offsets,
      const std::unordered_map<const BufferNode*, const BufferNode*>& buffer_allocated)
      : buffers_offsets_(buffers_offsets), buffer_allocated_(buffer_allocated) {}

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    op = store.get();
    const BufferNode* old_buffer = op->buffer.get();
    const BufferNode* new_buffer = FindNewBuffer(old_buffer);
    Array<PrimExpr> begins = ComputeRelativeIndices(old_buffer, op->indices);
    return GetRef<Buffer>(new_buffer).vstore(begins, op->value);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    op = load.get();
    const BufferNode* old_buffer = op->buffer.get();
    const BufferNode* new_buffer = FindNewBuffer(old_buffer);
    Array<PrimExpr> begins = ComputeRelativeIndices(old_buffer, op->indices);
    return GetRef<Buffer>(new_buffer).vload(begins, op->dtype);
  }

  const BufferNode* FindNewBuffer(const BufferNode* buffer) const {
    auto it = buffer_allocated_.find(buffer);
    ICHECK(it != buffer_allocated_.end());
    return it->second;
  }

  Array<PrimExpr> ComputeRelativeIndices(const BufferNode* buffer, const Array<PrimExpr>& indices) {
    auto it = buffers_offsets_.find(buffer);
    ICHECK(it != buffers_offsets_.end());
    const std::vector<PrimExpr>& offsets = it->second;
    ICHECK_EQ(offsets.size(), indices.size());
    int ndim = offsets.size();
    Array<PrimExpr> result;
    result.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      result.push_back(indices[i] - offsets[i]);
    }
    return result;
  }

  const std::unordered_map<const BufferNode*, std::vector<PrimExpr>>& buffers_offsets_;
  const std::unordered_map<const BufferNode*, const BufferNode*>& buffer_allocated_;
};

/*!
 * \brief Gather the used region of each buffers.
 */
class RegionGatherer : public StmtExprMutator {
  using VarDomain = std::unordered_map<const VarNode*, arith::IntSet>;

 public:
  Stmt Gather(const Map<Buffer, Optional<For>>& buffers_lca, const PrimFunc& f) {
    for (const auto& kv : buffers_lca) {
      const BufferNode* buffer = kv.first.get();
      const ForNode* loop = static_cast<const ForNode*>(kv.second.get());
      this->buffers_lca_.emplace(buffer, loop);
      this->buffer_alloc_[loop].push_back(buffer);
    }
    for (const auto& arg : f->buffer_map) {
      const BufferNode* buffer = arg.second.get();
      int ndim = buffer->shape.size();
      buffers_region_.emplace(buffer, NDIntSetFromShape(buffer->shape));
      buffer_allocated_.emplace(buffer, buffer);
      buffer_offsets_.emplace(buffer, std::vector<PrimExpr>(ndim, Integer(0)));
    }
    return BufferAccessUpdater::Update(buffer_offsets_, buffer_allocated_,
                                       this->VisitStmt(f->body));
  }

 public:
  /*! \brief The used region of each Buffer */
  std::unordered_map<const BufferNode*, NDIntSet> buffers_region_;
  std::unordered_map<const BufferNode*, const BufferNode*> buffer_allocated_;
  std::unordered_map<const BufferNode*, std::vector<PrimExpr>> buffer_offsets_;

 private:
  Array<Buffer> AllocBufers(const ForNode* loop) {
    auto it = buffer_alloc_.find(loop);
    if (it == buffer_alloc_.end()) {
      return {};
    }
    const std::vector<const BufferNode*>& buffers = it->second;
    Array<Buffer> result;
    result.reserve(buffers.size());
    for (const BufferNode* buffer : buffers) {
      auto it = buffers_region_.find(buffer);
      ICHECK(it != buffers_region_.end());
      const NDIntSet& nd_int_set = it->second;
      Buffer allocated = NDIntSet2Buffer(buffer, nd_int_set);
      buffer_allocated_.emplace(buffer, allocated.get());
      result.push_back(allocated);
      std::vector<PrimExpr> offsets;
      offsets.reserve(nd_int_set.size());
      for (const arith::IntSet& int_set : nd_int_set) {
        offsets.push_back(int_set.min());
      }
      buffer_offsets_.emplace(buffer, std::move(offsets));
    }
    return result;
  }

  Stmt VisitStmt_(const ForNode* loop) final {
    // Step 1. Handle block vars in `min` and `extent`
    PrimExpr min = this->VisitExpr(loop->min);
    PrimExpr extent = this->VisitExpr(loop->extent);
    // Step 2. Handle unit loops
    if (is_one(extent)) {
      var_substitutes_[loop->loop_var.get()] = min;
    }
    // Step 3. Visit recursively
    ancestor_loops_.push_back(loop);
    Stmt body = this->VisitStmt(loop->body);
    ancestor_loops_.pop_back();
    // Step 4. Add allocation
    Array<Buffer> alloc_buffers = AllocBufers(loop);
    if (!alloc_buffers.empty()) {
      body = BlockRealize(/*binding_values=*/{},
                          /*predicate=*/const_true(),
                          /*block=*/
                          Block(/*iter_vars=*/{},                            //
                                /*reads=*/{},                                //
                                /*writes=*/{},                               //
                                /*alloc_buffers=*/std::move(alloc_buffers),  //
                                /*annotations=*/{},                          //
                                /*match_buffers=*/{},                        //
                                /*exec_scope=*/"",                           //
                                /*name_hint=*/"alloc",                       //
                                /*body=*/std::move(body),                    //
                                /*init=*/NullOpt));
    }
    // Step 5. Make the new loop
    if (loop->kind == ForKind::kThreadBinding && reduction_loop_vars_.count(loop->loop_var.get())) {
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
    ICHECK_EQ(block->iter_vars.size(), realize->binding_values.size());
    for (int i = 0, n = block->iter_vars.size(); i < n; ++i) {
      const IterVar& block_var = block->iter_vars[i];
      PrimExpr v = this->VisitExpr(realize->binding_values[i]);
      var_substitutes_.emplace(block_var->var.get(), v);
      if (block_var->iter_type == kCommReduce) {
        for (const VarNode* var : Vars(v)) {
          this->reduction_loop_vars_.insert(var);
        }
      }
    }
    // Step 2. Initialize the buffer region with empty region
    for (const Buffer& buffer : block->alloc_buffers) {
      buffers_region_.emplace(buffer.get(),
                              NDIntSet(buffer->shape.size(), arith::IntSet::Nothing()));
    }
    // Step 3. Visit recursively
    ++block_nest_depth_;
    Stmt body = this->VisitStmt(block->body);
    --block_nest_depth_;
    // Step 4. Update the read/write buffer regions
    Array<BufferRegion> reads = VisitBufferRegions(block->reads);
    Array<BufferRegion> writes = VisitBufferRegions(block->writes);
    // Step 5. Handle predicate
    PrimExpr predicate = this->VisitExpr(realize->predicate);
    // Step 6. Root allocation
    Array<Buffer> alloc_buffers = (block_nest_depth_ == 0) ? AllocBufers(nullptr) : Array<Buffer>{};
    // Step 7. Create new blocks
    return BlockRealize(/*binding_values=*/{},
                        /*predicate=*/std::move(predicate),
                        /*block=*/
                        Block(/*iter_vars=*/{},                            //
                              /*reads=*/std::move(reads),                  //
                              /*writes=*/std::move(writes),                //
                              /*alloc_buffers=*/std::move(alloc_buffers),  //
                              /*annotations=*/block->annotations,          //
                              /*match_buffers=*/block->match_buffers,      //
                              /*exec_scope=*/block->exec_scope,            //
                              /*name_hint=*/block->name_hint,              //
                              /*body=*/std::move(body),                    //
                              /*init=*/NullOpt));
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

  PrimExpr VisitExpr_(const VarNode* var) final {
    auto it = var_substitutes_.find(var);
    if (it != var_substitutes_.end()) {
      return it->second;
    }
    return GetRef<Var>(var);
  }

  Array<BufferRegion> VisitBufferRegions(const Array<BufferRegion>& buffer_regions) {
    Array<BufferRegion> result;
    result.reserve(buffer_regions.size());
    for (const BufferRegion& buffer_region : buffer_regions) {
      const Buffer& buffer = buffer_region->buffer;
      VarDomain dom_map = LoopVarDomain(buffer);
      int ndim = buffer_region->region.size();
      Array<Range> region;
      NDIntSet int_set;
      region.reserve(ndim);
      int_set.reserve(ndim);
      for (const Range& range : buffer_region->region) {
        Range new_range =
            Range::FromMinExtent(this->VisitExpr(range->min), this->VisitExpr(range->extent));
        region.push_back(new_range);
        int_set.push_back(arith::EvalSet(new_range, dom_map));
      }
      auto it = buffers_region_.find(buffer.get());
      ICHECK(it != buffers_region_.end());
      NDIntSet& alloc_region = it->second;
      UnionWith(&alloc_region, int_set);
      result.push_back(BufferRegion(buffer_region->buffer, region));
    }
    return result;
  }

  VarDomain LoopVarDomain(const Buffer& buffer) const {
    auto it = this->buffers_lca_.find(buffer.get());
    ICHECK(it != this->buffers_lca_.end());
    const ForNode* lca = it->second;
    // Every loop will be relaxed if the lca is the root
    VarDomain dom_map;
    bool need_relax = (lca == nullptr);
    for (const ForNode* loop : this->ancestor_loops_) {
      const VarNode* loop_var = loop->loop_var.get();
      // TODO
      if (need_relax || (buffer->scope == "shared" && IsThreadBound(loop))) {
        dom_map[loop_var] = IntSetFromMinExtent(loop->min, loop->extent);
      }
      if (loop == lca) {
        need_relax = true;
      }
    }
    return dom_map;
  }

  int block_nest_depth_ = 0;
  /*! \brief The map from Buffer to its LCA Stmt/Expr */
  std::unordered_map<const BufferNode*, const ForNode*> buffers_lca_;
  std::unordered_map<const ForNode*, std::vector<const BufferNode*>> buffer_alloc_;
  std::unordered_set<const VarNode*> reduction_loop_vars_;
  /*! \brief The loops from the current node up to the root */
  std::vector<const ForNode*> ancestor_loops_;
  /*! \brief The map from block vars to the expr value */
  std::unordered_map<const VarNode*, PrimExpr> var_substitutes_;
};

/*!
 * \brief Transform multi-dimension BufferLoad/BufferStore into one-dimension Load/Store
 */
class BufferFlattener : public StmtExprMutator {
 private:
  Stmt VisitStmt_(const SeqStmtNode* op) final {
    Array<Stmt> seq;
    seq.reserve(op->seq.size());
    for (const Stmt& stmt : op->seq) {
      std::unordered_set<const BufferNode*> double_buffer;
      std::swap(double_buffer, double_buffer_);
      Stmt body = VisitStmt(stmt);
      std::swap(double_buffer, double_buffer_);
      const ForNode* loop = ancestor_loops_.back();
      for (const BufferNode* buffer : double_buffer) {
        // TODO
        // const Object* lca = buffers_lca_.at(GetRef<Buffer>(buffer)).get();
        // if (lca != nullptr && loop == lca) {
        //   body = AttrStmt(buffer->data, attr::double_buffer_scope, 1, body);
        // } else {
        //   double_buffer_.insert(buffer);
        // }
      }
      seq.push_back(body);
    }
    return SeqStmt(seq);
  }

  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    // Step 3. Visit the body
    Block new_block = Downcast<Block>(this->VisitStmt(realize->block));
    const BlockNode* block = new_block.get();
    // Step 4. Transform the `predicate` to if-then-else
    Stmt body = block->body;
    if (!is_one(realize->predicate)) {
      body = IfThenElse(realize->predicate, body);
    }
    // Step 5. Pick out blocks that writes with double buffering
    for (const auto& ann : block->annotations) {
      const String& ann_key = ann.first;
      const ObjectRef& ann_value = ann.second;
      if (ann_key == attr::double_buffer_scope) {
        if (is_one(Downcast<PrimExpr>(ann_value))) {
          ICHECK_EQ(block->writes.size(), 1);
          const BufferRegion& write = block->writes[0];
          double_buffer_.insert(write->buffer.get());
        }
      }
    }
    // Step 6. Handle allocations
    for (const Buffer& buffer : block->alloc_buffers) {
      body = MakeAllocStmt(buffer, BufferArea(buffer), body);
    }
    return body;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // Step 2. Visit recursively
    ancestor_loops_.push_back(op);
    Stmt body = this->VisitStmt(op->body);
    PrimExpr min = this->VisitExpr(op->min);
    PrimExpr extent = this->VisitExpr(op->extent);
    ancestor_loops_.pop_back();
    // Step 4. Add the for loop accordingly
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
    // Step 5. Handle annotations
    for (const auto& annotation : op->annotations) {
      const String& ann_key = annotation.first;
      const ObjectRef& ann_value = annotation.second;
      if (attr::IsPragmaKey(ann_key)) {
        body = AttrStmt(op->loop_var, ann_key, Downcast<PrimExpr>(ann_value), body);
      }
    }
    return body;
  }

  std::unordered_set<const BufferNode*> double_buffer_;
  std::vector<const ForNode*> ancestor_loops_;
};

PrimFunc BufferFlatten(PrimFunc f) {
  tvm::tir::PrimFuncNode* fptr = f.CopyOnWrite();
  // Step 0. Check memory and execution hierarchy
  VerifyExecScope(f);
  // Step 1.Transform the reduction calls to BufferStore
  ReductionTransformer reduction_transformer;
  fptr->body = reduction_transformer(fptr->body);
  // Step 2. Recalculate the buffer region
  Map<Buffer, Optional<For>> buffer_lca = LCADetector::Detect(f);
  RegionGatherer region_gatherer;
  fptr->body = region_gatherer.Gather(buffer_lca, f);
  // Step 3. Transform BufferLoad/BufferStore into Load/Store
  BufferFlattener flattener;
  fptr->body = flattener(fptr->body);
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
