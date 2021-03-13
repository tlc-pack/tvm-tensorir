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

NDIntSet NDIntSetFromShape(const Array<PrimExpr>& shape) {
  NDIntSet result;
  for (const PrimExpr& extent : shape) {
    result.push_back(IntSetFromMinExtent(Integer(0), extent));
  }
  return result;
}

bool IsThreadBinded(const ForNode* loop) {
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

/*!
 * \brief Gather the used region of each buffers.
 */
class RegionGatherer : public StmtVisitor {
  using VarDomain = std::unordered_map<const VarNode*, arith::IntSet>;

 public:
  RegionGatherer(const Map<Buffer, Optional<For>>& buffers_lca, const PrimFunc& f)
      : buffers_lca_(buffers_lca) {
    for (const auto& arg : f->buffer_map) {
      const Buffer& buffer = arg.second;
      buffers_region_[buffer] = NDIntSetFromShape(buffer->shape);
    }
  }

  void VisitStmt_(const ForNode* op) final {
    ancestor_loops_.push_back(op);
    if (!op->thread_binding.defined() && op->annotations.empty() && is_one(op->extent)) {
      unit_loops_[op->loop_var.get()] = op->min;
    }
    StmtVisitor::VisitStmt_(op);
    ancestor_loops_.pop_back();
  }

  void VisitStmt_(const BlockRealizeNode* realize) final {
    const auto* block = realize->block.as<BlockNode>();
    CHECK(!block->init.defined());
    // Update the mapping from block vars to loop vars so that we can substitute them
    CHECK_EQ(block->iter_vars.size(), realize->binding_values.size());
    int n_block_vars = block->iter_vars.size();
    for (int i = 0; i < n_block_vars; ++i) {
      const IterVar& iter = block->iter_vars[i];
      const PrimExpr& v = realize->binding_values[i];
      block_var_[iter->var.get()] = ReplaceBlockVar(v);
    }
    for (const BufferRegion& read_region : block->reads) {
      const Buffer& buffer = read_region->buffer;
      VarDomain dom_map = LoopVarDomain(buffer);
      NDIntSet region = AsRegion(read_region->region, dom_map);
      NDIntSet& alloc_region = buffers_region_.at(buffer);
      UnionWith(&alloc_region, region);
    }
    for (const BufferRegion& write_region : block->writes) {
      const Buffer& buffer = write_region->buffer;
      VarDomain dom_map = LoopVarDomain(buffer);
      NDIntSet region = AsRegion(write_region->region, dom_map);
      NDIntSet& alloc_region = buffers_region_.at(buffer);
      UnionWith(&alloc_region, region);
    }
    for (const Buffer& buffer : block->alloc_buffers) {
      // Initialize the buffer region with empty region.
      buffers_region_[buffer] = NDIntSet(buffer->shape.size(), arith::IntSet::Nothing());
    }
    VisitStmt(block->body);
  }

  /*! \brief The used region of each Buffer */
  std::unordered_map<Buffer, NDIntSet, ObjectPtrHash, ObjectPtrEqual> buffers_region_;
  /*! \brief The map from block vars to the expr value */
  std::unordered_map<const VarNode*, PrimExpr> block_var_;
  /*! \brief The map from unit loop vars to the expr value */
  std::unordered_map<const VarNode*, PrimExpr> unit_loops_;

 private:
  PrimExpr ReplaceBlockVar(const PrimExpr& expr) const {
    return Substitute(Substitute(expr, block_var_), unit_loops_);
  }

  VarDomain LoopVarDomain(const Buffer& buffer) const {
    VarDomain dom_map;
    const Optional<For>& lca = this->buffers_lca_.at(buffer);
    // Every loop will be relaxed if the lca is the root
    bool need_relax = !lca.defined();
    for (const ForNode* loop : this->ancestor_loops_) {
      const VarNode* loop_var = loop->loop_var.get();
      // TODO
      if (need_relax || (buffer->scope == "shared" && IsThreadBinded(loop))) {
        dom_map[loop_var] = IntSetFromMinExtent(loop->min, loop->extent);
      }
      if (loop == lca.get()) {
        need_relax = true;
      }
    }
    return dom_map;
  }

  NDIntSet AsRegion(const Array<Range>& buffer_region, const VarDomain& dom_map) const {
    NDIntSet region;
    region.reserve(buffer_region.size());
    for (const Range& range : buffer_region) {
      PrimExpr min = ReplaceBlockVar(range->min);
      PrimExpr extent = ReplaceBlockVar(range->extent);
      region.push_back(arith::EvalSet(Range::FromMinExtent(min, extent), dom_map));
    }
    return region;
  }

  /*! \brief The map from Buffer to its LCA Stmt/Expr */
  const Map<Buffer, Optional<For>>& buffers_lca_;
  /*! \brief The loops from the current node up to the root */
  std::vector<const ForNode*> ancestor_loops_;
};

/*!
 * \brief Transform multi-dimension BufferLoad/BufferStore into one-dimension Load/Store
 */
class BufferFlattener : public StmtExprMutator {
 public:
  explicit BufferFlattener(
      const std::unordered_map<const VarNode*, PrimExpr>& block_var,
      const std::unordered_map<const VarNode*, PrimExpr>& unit_loops,
      const std::unordered_map<Buffer, NDIntSet, ObjectPtrHash, ObjectPtrEqual>& buffers_region,
      const Map<Buffer, Optional<For>>& buffers_lca, const PrimFunc& func)
      : buffers_region_(buffers_region),
        block_var_(block_var),
        unit_loops_(unit_loops),
        buffers_lca_(buffers_lca),
        arg_buffers_{} {
    arg_buffers_.reserve(func->buffer_map.size());
    for (const auto& kv : func->buffer_map) {
      const Buffer& buffer = kv.second;
      arg_buffers_.insert(buffer.get());
    }
  }

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    Array<Stmt> seq;
    seq.reserve(op->seq.size());
    for (const Stmt& stmt : op->seq) {
      std::unordered_set<const BufferNode*> double_buffer;
      std::swap(double_buffer, double_buffer_);
      Stmt body = VisitStmt(stmt);
      std::swap(double_buffer, double_buffer_);
      const StmtNode* parent_scope = parent_scopes_.back();
      for (const BufferNode* buffer : double_buffer) {
        const Object* lca = buffers_lca_.at(GetRef<Buffer>(buffer)).get();
        if (lca != nullptr && lca == parent_scope) {
          body = AttrStmt(buffer->data, attr::double_buffer_scope, 1, body);
        } else {
          double_buffer_.insert(buffer);
        }
      }
      seq.push_back(body);
    }
    return SeqStmt(seq);
  }

  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    // Handle allocations
    const auto* block = realize->block.get();
    // Step 1. Add non-root block allocations into `pending_allocate_`
    for (const Buffer& buffer : block->alloc_buffers) {
      if (IsReduceTempBuffer(buffer)) {
        continue;
      }
      if (buffers_lca_.at(buffer).defined()) {
        pending_allocate_.insert(buffer.get());
      }
    }
    // Step 2. Add reduction loop vars
    CHECK_EQ(block->iter_vars.size(), realize->binding_values.size());
    int n_block_vars = block->iter_vars.size();
    for (int i = 0; i < n_block_vars; ++i) {
      const IterVar& block_var = block->iter_vars[i];
      const PrimExpr& binding_value = realize->binding_values[i];
      if (block_var->iter_type == kCommReduce) {
        std::unordered_set<const VarNode*> vars = Vars(binding_value);
        for (const VarNode* var : vars) {
          this->reduction_loop_vars_.insert(var);
        }
      }
    }
    // Step 3. Visit the body
    parent_scopes_.push_back(realize->block.get());
    Block new_block = Downcast<Block>(this->VisitStmt(realize->block));
    block = new_block.get();
    parent_scopes_.pop_back();
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
    // Step 6. Add root block allocations
    for (const Buffer& buffer : block->alloc_buffers) {
      if (IsReduceTempBuffer(buffer)) {
        continue;
      }
      if (!buffers_lca_.at(buffer).defined()) {
        const NDIntSet& region = buffers_region_.at(buffer);
        body = MakeAllocStmt(buffer, NDIntSetArea(region), body);
      }
    }
    return body;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // Step 1. Find the buffer that can be allocated under the current loop
    std::vector<const BufferNode*> alloc_buffers;
    for (const BufferNode* buffer : pending_allocate_) {
      const Optional<For> alloc_site = buffers_lca_.at(GetRef<Buffer>(buffer));
      if (op == alloc_site.get()) {
        alloc_buffers.push_back(buffer);
      }
    }
    // Step 2. Visit recursively
    parent_scopes_.push_back(op);
    Stmt body = this->VisitStmt(op->body);
    PrimExpr min = this->VisitExpr(op->min);
    PrimExpr extent = this->VisitExpr(op->extent);
    parent_scopes_.pop_back();
    // Step 3. Add buffer allocation
    for (const BufferNode* buffer : alloc_buffers) {
      const NDIntSet& region = buffers_region_.at(GetRef<Buffer>(buffer));
      body = MakeAllocStmt(GetRef<Buffer>(buffer), NDIntSetArea(region), body);
      pending_allocate_.erase(buffer);
    }
    // Step 4. Add the for loop accordingly
    if (op->kind == ForKind::kThreadBinding) {
      // Case 1. Thread binding
      ICHECK(op->thread_binding.defined());
      String thread_tag = op->thread_binding.value()->thread_tag;
      if (!reduction_loop_vars_.count(op->loop_var.get())) {
        IterVar iter_var(/*dom=*/Range(min, extent),
                         /*var=*/op->loop_var,
                         /*iter_type=*/IterVarType::kThreadIndex,
                         /*thread_tag=*/thread_tag);
        String attr_key = thread_tag == "vthread" ? attr::virtual_thread : attr::thread_extent;
        body = AttrStmt(iter_var, attr_key, extent, body);
      }
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

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    const Buffer& buffer = op->buffer;
    std::vector<PrimExpr> indices = VisitIndices(op->indices);
    PrimExpr value = this->VisitExpr(op->value);
    std::vector<PrimExpr> begins = ComputeRelativeIndices(buffer, indices);
    Buffer new_buffer = ReshapeBuffer(buffer, this->buffers_region_.at(buffer));
    return new_buffer.vstore(begins, value);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    const Buffer& buffer = op->buffer;
    std::vector<PrimExpr> indices = VisitIndices(op->indices);
    std::vector<PrimExpr> begins = ComputeRelativeIndices(buffer, indices);
    Buffer new_buffer = ReshapeBuffer(buffer, this->buffers_region_.at(buffer));
    return new_buffer.vload(begins, op->dtype);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    // Replace the block var with its value
    auto it = block_var_.find(op);
    if (it != block_var_.end()) {
      return Substitute(it->second, unit_loops_);
    } else {
      return Substitute(GetRef<PrimExpr>(op), unit_loops_);
    }
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::get_elem_offset())) {
      // Handle `get_elem_offset`
      ICHECK_EQ(op->args.size(), 1);
      const PrimExpr& arg = op->args[0];
      ICHECK(arg->IsInstance<BufferLoadNode>());
      Load load = Downcast<Load>(VisitExpr(arg));
      return load->index;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

 private:
  const std::unordered_map<Buffer, NDIntSet, ObjectPtrHash, ObjectPtrEqual>& buffers_region_;
  const std::unordered_map<const VarNode*, PrimExpr>& block_var_;
  const std::unordered_map<const VarNode*, PrimExpr>& unit_loops_;
  const Map<Buffer, Optional<For>>& buffers_lca_;
  std::unordered_set<const BufferNode*> arg_buffers_;
  std::unordered_set<const BufferNode*> pending_allocate_;
  std::unordered_set<const VarNode*> reduction_loop_vars_;
  std::unordered_set<const BufferNode*> double_buffer_;
  std::vector<const StmtNode*> parent_scopes_;

  /*!
   * \brief Create a buffer with alternative shape
   */
  Buffer ReshapeBuffer(const Buffer& buffer, const NDIntSet& region) {
    if (arg_buffers_.count(buffer.get())) {
      return buffer;
    }
    Array<PrimExpr> shape;
    for (const arith::IntSet& i : region) {
      shape.push_back(i.max() - i.min() + 1);
    }
    ObjectPtr<BufferNode> n = make_object<BufferNode>(*buffer.get());
    n->shape = std::move(shape);
    return Buffer(std::move(n));
  }

  /*!
   * \brief Transform indices from the absolute indices to relative indices
   * \note T can be BufferLoad or BufferStore
   */
  std::vector<PrimExpr> ComputeRelativeIndices(const Buffer& buffer,
                                               const Array<PrimExpr>& indices) {
    const NDIntSet& region = buffers_region_.at(buffer);
    std::vector<PrimExpr> new_indices;
    for (size_t i = 0; i < region.size(); ++i) {
      if (arg_buffers_.count(buffer.get())) {
        new_indices.push_back(indices[i]);
      } else {
        new_indices.push_back(indices[i] - region[i].min());
      }
    }
    return new_indices;
  }

  std::vector<PrimExpr> VisitIndices(const Array<PrimExpr>& indices) {
    std::vector<PrimExpr> result;
    result.reserve(indices.size());
    for (const PrimExpr& index : indices) {
      result.push_back(this->VisitExpr(index));
    }
    return result;
  }
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
  RegionGatherer region_gatherer(buffer_lca, f);
  region_gatherer(fptr->body);
  // Step 3. Transform BufferLoad/BufferStore into Load/Store
  BufferFlattener flattener(region_gatherer.block_var_, region_gatherer.unit_loops_,
                            region_gatherer.buffers_region_, buffer_lca, f);
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
