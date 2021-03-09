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
  explicit LCADetector(const std::unordered_set<const BufferNode*>& arg_buffers)
      : arg_buffers_(arg_buffers) {
    for (const BufferNode* buffer : arg_buffers) {
      buffers_lca_.emplace(GetRef<Buffer>(buffer), ObjectRef(nullptr));
    }
  }

  void VisitStmt_(const ForNode* op) final {
    Stmt n = GetRef<Stmt>(op);
    ast_scopes_info_[n] = MakeScope();
    ancestor_loops_.push_back(op);
    StmtExprVisitor::VisitStmt_(op);
    ancestor_loops_.pop_back();
  }

  void VisitBuffer(const Buffer& buffer, const ObjectRef& n) {
    ast_scopes_info_[n] = MakeScope();
    if (arg_buffers_.count(buffer.get())) {
      return;
    }
    if (buffers_lca_.count(buffer)) {
      buffers_lca_[buffer] = LowestCommonAncestor(n, buffers_lca_[buffer]);
    } else {
      buffers_lca_[buffer] = n;
    }
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    VisitBuffer(op->buffer, GetRef<ObjectRef>(op));
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    VisitBuffer(op->buffer, GetRef<ObjectRef>(op));
    StmtExprVisitor::VisitStmt_(op);
  }

  /*! \brief The map from Buffer to its LCA Stmt/Expr */
  std::unordered_map<Buffer, ObjectRef, ObjectPtrHash, ObjectPtrEqual> buffers_lca_;

 private:
  /*! \brief The AST node information for querying LCA */
  struct ScopeInfo {
    // The parent loop node
    const StmtNode* parent_scope;
    // The scope depth in the AST
    int depth;
  };

  ScopeInfo MakeScope() {
    int n = ancestor_loops_.size();
    return (n == 0) ? ScopeInfo{nullptr, 0} : ScopeInfo{ancestor_loops_.back(), n};
  }

  /*! \brief The Buffer in function args */
  const std::unordered_set<const BufferNode*>& arg_buffers_;
  /*! \brief The current scope initializing with Null */
  std::vector<const ForNode*> ancestor_loops_;
  /*! \brief The parent and depth info of each Loop/BufferLoad/BufferStore Node */
  std::unordered_map<ObjectRef, ScopeInfo, ObjectPtrHash, ObjectPtrEqual> ast_scopes_info_;

  ObjectRef LowestCommonAncestor(ObjectRef lhs, ObjectRef rhs) {
    ICHECK(lhs.defined());
    ICHECK(rhs.defined());
    ICHECK(ast_scopes_info_.count(lhs));
    ICHECK(ast_scopes_info_.count(rhs));
    while (ast_scopes_info_[lhs].depth > ast_scopes_info_[rhs].depth) {
      lhs = GetRef<ObjectRef>(ast_scopes_info_[lhs].parent_scope);
    }
    while (ast_scopes_info_[lhs].depth < ast_scopes_info_[rhs].depth) {
      rhs = GetRef<ObjectRef>(ast_scopes_info_[rhs].parent_scope);
    }
    while (!lhs.same_as(rhs)) {
      lhs = GetRef<ObjectRef>(ast_scopes_info_[lhs].parent_scope);
      rhs = GetRef<ObjectRef>(ast_scopes_info_[rhs].parent_scope);
    }
    return lhs;
  }
};

/*!
 * \brief Gather the used region of each buffers.
 */
class RegionGatherer : public StmtVisitor {
 public:
  RegionGatherer(
      const std::unordered_map<Buffer, ObjectRef, ObjectPtrHash, ObjectPtrEqual>& buffers_lca,
      const Map<Var, Buffer>& func_args)
      : buffers_lca_(buffers_lca) {
    for (const auto& arg : func_args) {
      const Buffer& buffer = arg.second;
      std::vector<arith::IntSet> region;
      for (const PrimExpr& size : buffer->shape) {
        region.push_back(IntSetFromMinExtent(0, size));
      }
      buffers_region_[buffer] = region;
    }
  }

  void VisitStmt_(const ForNode* op) final {
    For loop = GetRef<For>(op);
    loop_stack_.push_back(loop);
    if (!op->thread_binding.defined() && op->annotations.empty() && is_one(op->extent)) {
      unit_loops_[op->loop_var.get()] = op->min;
    }
    StmtVisitor::VisitStmt_(op);
    loop_stack_.pop_back();
  }

  void VisitStmt_(const BlockRealizeNode* realize) final {
    const auto* block = realize->block.as<BlockNode>();
    CHECK(!block->init.defined());
    for (size_t i = 0; i < block->iter_vars.size(); ++i) {
      const IterVar& iter = block->iter_vars[i];
      const PrimExpr& v = realize->binding_values[i];
      block_var_[iter->var.get()] = ReplaceBlockVar(v);
    }
    for (const BufferRegion& buffer_region : block->reads) {
      UnionWith(&buffers_region_.at(buffer_region->buffer), GatherRegion(buffer_region));
    }
    for (const BufferRegion& buffer_region : block->writes) {
      UnionWith(&buffers_region_.at(buffer_region->buffer), GatherRegion(buffer_region));
    }
    for (const Buffer& alloc_buf : block->alloc_buffers) {
      // Initialize the buffer region with empty region.
      buffers_region_[alloc_buf] =
          std::vector<arith::IntSet>(alloc_buf->shape.size(), arith::IntSet::Nothing());
    }
    VisitStmt(block->body);
  }

  /*! \brief The used region of each Buffer */
  std::unordered_map<Buffer, std::vector<arith::IntSet>, ObjectPtrHash, ObjectPtrEqual>
      buffers_region_;
  /*! \brief The map from block vars to the expr value */
  std::unordered_map<const VarNode*, PrimExpr> block_var_;
  /*! \brief The map from unit loop vars to the expr value */
  std::unordered_map<const VarNode*, PrimExpr> unit_loops_;

 private:
  const std::unordered_map<Buffer, ObjectRef, ObjectPtrHash, ObjectPtrEqual>& buffers_lca_;

  /*! \brief The loops from the current node up to the root */
  std::vector<For> loop_stack_;

  static void UnionWith(std::vector<arith::IntSet>* buffer_new_region,
                        const std::vector<arith::IntSet>& region) {
    ICHECK_EQ(buffer_new_region->size(), region.size());
    for (size_t i = 0; i < region.size(); ++i) {
      arith::IntSet& int_set = buffer_new_region->at(i);
      int_set = arith::Union({int_set, region[i]});
    }
  }

  /*!
   * \brief Gather used buffer region
   */
  std::vector<arith::IntSet> GatherRegion(const BufferRegion& buffer_region) {
    std::unordered_map<const VarNode*, arith::IntSet> dom_map;
    const ObjectRef& lca = buffers_lca_.at(buffer_region->buffer);
    // Every loop will be relaxed if the lca is the root
    bool need_relax = !lca.defined();
    for (const For& loop : loop_stack_) {
      const VarNode* var = loop->loop_var.get();
      if (need_relax || (buffer_region->buffer->scope == "shared" && IsThreadBinded(loop))) {
        dom_map[var] = IntSetFromMinExtent(loop->min, loop->extent);
      }
      if (loop.same_as(lca)) {
        need_relax = true;
      }
    }
    std::vector<arith::IntSet> region;
    for (const Range& range : buffer_region->region) {
      PrimExpr min = ReplaceBlockVar(range->min);
      PrimExpr extent = ReplaceBlockVar(range->extent);
      region.push_back(arith::EvalSet(Range::FromMinExtent(min, extent), dom_map));
    }
    return region;
  }

  PrimExpr ReplaceBlockVar(const PrimExpr& expr) const {
    return Substitute(Substitute(expr, block_var_), unit_loops_);
  }

  static arith::IntSet IntSetFromMinExtent(const PrimExpr& min, const PrimExpr& extent) {
    return arith::IntSet::FromRange(Range::FromMinExtent(min, extent));
  }

  static bool IsThreadBinded(const For& loop) {
    if (loop->kind != ForKind::kThreadBinding || !loop->thread_binding.defined()) return false;
    std::string thread_tag = loop->thread_binding.value()->thread_tag;
    return (thread_tag.substr(0, 9) == "threadIdx" || thread_tag.substr(0, 7) == "vthread");
  }
};

/*!
 * \brief Transform multi-dimension BufferLoad/BufferStore into one-dimension Load/Store
 */
class BufferFlattener : public StmtExprMutator {
 public:
  BufferFlattener(
      const std::unordered_map<const VarNode*, PrimExpr>& block_var,
      const std::unordered_map<const VarNode*, PrimExpr>& unit_loops,
      const std::unordered_map<Buffer, std::vector<arith::IntSet>, ObjectPtrHash, ObjectPtrEqual>&
          buffers_region,
      const std::unordered_map<Buffer, ObjectRef, ObjectPtrHash, ObjectPtrEqual>& buffers_lca,
      const std::unordered_set<const BufferNode*>& arg_buffers)
      : buffers_region_(buffers_region),
        block_var_(block_var),
        unit_loops_(unit_loops),
        buffers_lca_(buffers_lca),
        arg_buffers_(arg_buffers) {}

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    Array<Stmt> seq;
    for (const Stmt& stmt : op->seq) {
      std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> double_buffer;
      std::swap(double_buffer, double_buffer_);
      Stmt body = VisitStmt(stmt);
      std::swap(double_buffer, double_buffer_);

      for (const Buffer& buffer : double_buffer) {
        ObjectRef lca = buffers_lca_.at(buffer);
        if (lca.defined() && lca.same_as(parent_scope_)) {
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
    Block old_block = realize->block;
    int n_alloc_buffer = block->alloc_buffers.size();
    int n_iter_var = block->iter_vars.size();
    // Step 1. Figure out `pending_allocate_`
    for (int i = n_alloc_buffer - 1; i >= 0; --i) {
      // Why the order
      const Buffer& buffer = block->alloc_buffers[i];
      if (StartsWith(buffer->name, "normal_reduce_temp") ||
          StartsWith(buffer->name, "reduce_temp")) {
        continue;
      }
      if (buffers_lca_.at(buffer).defined()) {
        pending_allocate_[buffer] = buffer;
      }
    }
    for (int i = 0; i < n_iter_var; ++i) {
      const IterVar& block_var = block->iter_vars[i];
      const PrimExpr& binding_value = realize->binding_values[i];
      if (block_var->iter_type != kCommReduce) {
        continue;
      }
      std::unordered_set<const VarNode*> vars = Vars(binding_value);
      for (const VarNode* var : vars) {
        this->reduction_relative_.insert(GetRef<Var>(var));
      }
    }
    // Step 2. Visit the body
    Stmt parent_scope = realize->block;
    std::swap(parent_scope, parent_scope_);
    BlockRealize new_stmt = Downcast<BlockRealize>(StmtExprMutator::VisitStmt_(realize));
    std::swap(parent_scope, parent_scope_);
    // Reset `realize` and `block`
    realize = new_stmt.get();
    block = realize->block.get();
    // Step 3. Transform the `predicate` to if-then-else
    Stmt body = block->body;
    if (!is_one(realize->predicate)) {
      body = IfThenElse(realize->predicate, body);
    }
    // Step 4. Pick out blocks that writes with double buffering
    for (const auto& ann : block->annotations) {
      const String& ann_key = ann.first;
      const ObjectRef& ann_value = ann.second;
      if (ann_key == attr::double_buffer_scope) {
        if (is_one(Downcast<PrimExpr>(ann_value))) {
          ICHECK_EQ(block->writes.size(), 1);
          double_buffer_.insert(block->writes[0]->buffer);
        }
      }
    }
    // Step 5. Add allocation and storage scope
    for (int i = n_alloc_buffer - 1; i >= 0; --i) {
      const Buffer& alloc_buf = block->alloc_buffers[i];
      if (StartsWith(alloc_buf->name, "normal_reduce_temp") ||
          StartsWith(alloc_buf->name, "reduce_temp")) {
        continue;
      }
      if (!buffers_lca_.at(alloc_buf).defined() || buffers_lca_.at(alloc_buf).same_as(old_block)) {
        PrimExpr extents = 1;
        for (const arith::IntSet& extent : buffers_region_.at(alloc_buf)) {
          extents *= extent.max() - extent.min() + 1;
        }
        body = Allocate(alloc_buf->data, alloc_buf->dtype, {extents}, const_true(), body);
        // Change empty scope into global
        String scope = alloc_buf->scope;
        if (scope.empty()) {
          scope = "global";
        }
        body = AttrStmt(alloc_buf->data, attr::storage_scope, StringImm(scope), body);
      }
    }

    return body;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    Stmt old_stmt = GetRef<Stmt>(op);
    std::swap(old_stmt, parent_scope_);
    For stmt = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    std::swap(old_stmt, parent_scope_);
    op = stmt.get();

    std::vector<Buffer> removed_buffers;
    // Add buffer allocation
    Stmt body = op->body;
    for (auto it = pending_allocate_.begin(); it != pending_allocate_.end();) {
      const Buffer& alloc_buf = it->first;
      if (old_stmt.same_as(buffers_lca_.at(alloc_buf))) {
        PrimExpr extents = 1;
        for (const arith::IntSet& extent : buffers_region_.at(alloc_buf)) {
          extents *= extent.max() - extent.min() + 1;
        }
        body = Allocate(alloc_buf->data, alloc_buf->dtype, {extents}, const_true(), body);
        // Change empty scope into global
        String scope = alloc_buf->scope.empty() ? "global" : alloc_buf->scope;
        body = AttrStmt(alloc_buf->data, attr::storage_scope, StringImm(scope), body);
        removed_buffers.push_back(alloc_buf);
        ++it;
      } else {
        ++it;
      }
    }
    for (const Buffer& buffer : removed_buffers) {
      pending_allocate_.erase(buffer);
    }

    if (op->kind == ForKind::kThreadBinding) {
      ICHECK(op->thread_binding.defined());
      String thread_tag = op->thread_binding.value()->thread_tag;
      if (!reduction_relative_.count(op->loop_var)) {
        IterVar iter_var(/*dom=*/Range(op->min, op->extent),
                         /*var=*/op->loop_var,
                         /*iter_type=*/IterVarType::kThreadIndex,
                         /*thread_tag=*/thread_tag);
        String attr_key = thread_tag == "vthread" ? attr::virtual_thread : attr::thread_extent;
        body = AttrStmt(iter_var, attr_key, op->extent, body);
      }
    } else if (is_one(op->extent) && op->annotations.empty()) {
      return body;
    } else {
      body = For(op->loop_var, op->min, op->extent, op->kind, body);
    }
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
    BufferStore stmt = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    op = stmt.get();
    std::vector<PrimExpr> begins = ComputeRelativeIndices(op->buffer, op->indices);
    Buffer new_buffer = ReshapeBuffer(op->buffer, this->buffers_region_.at(op->buffer));
    return new_buffer.vstore(begins, op->value);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad expr = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    op = expr.get();
    std::vector<PrimExpr> begins = ComputeRelativeIndices(op->buffer, op->indices);
    Buffer new_buffer = ReshapeBuffer(op->buffer, this->buffers_region_.at(op->buffer));
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
      ICHECK_EQ(op->args.size(), 1);
      const auto* buffer_load = op->args[0].as<BufferLoadNode>();
      ICHECK(buffer_load != nullptr);
      Load load = Downcast<Load>(VisitExpr(op->args[0]));
      return load->index;
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

 private:
  const std::unordered_map<Buffer, std::vector<arith::IntSet>, ObjectPtrHash, ObjectPtrEqual>&
      buffers_region_;
  const std::unordered_map<const VarNode*, PrimExpr>& block_var_;
  const std::unordered_map<const VarNode*, PrimExpr>& unit_loops_;
  const std::unordered_map<Buffer, ObjectRef, ObjectPtrHash, ObjectPtrEqual>& buffers_lca_;
  const std::unordered_set<const BufferNode*>& arg_buffers_;

  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> pending_allocate_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> reduction_relative_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> double_buffer_;
  Stmt parent_scope_;

  /*!
   * \brief Create a buffer with alternative shape
   */
  Buffer ReshapeBuffer(const Buffer& buffer, const std::vector<arith::IntSet>& region) {
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
    const std::vector<arith::IntSet>& region = buffers_region_.at(buffer);
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
};

PrimFunc BufferFlatten(PrimFunc f) {
  auto fptr = f.CopyOnWrite();

  // Check memory and execution hierarchy
  VerifyExecScope(f);

  // Transform the reduction calls to BufferStore
  ReductionTransformer reduction_transformer;
  fptr->body = reduction_transformer(fptr->body);

  std::unordered_set<const BufferNode*> arg_buffers;
  for (const auto& kv : fptr->buffer_map) {
    const Buffer& buffer = kv.second;
    arg_buffers.insert(buffer.get());
  }

  // Find the LCA of each Buffer access
  LCADetector lca_detector(arg_buffers);
  lca_detector(fptr->body);

  // Recalculate the buffer region
  RegionGatherer region_gatherer(lca_detector.buffers_lca_, fptr->buffer_map);
  region_gatherer(fptr->body);

  // Transform BufferLoad/BufferStore into Load/Store
  BufferFlattener flattener(region_gatherer.block_var_, region_gatherer.unit_loops_,
                            region_gatherer.buffers_region_, lca_detector.buffers_lca_,
                            arg_buffers);
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
