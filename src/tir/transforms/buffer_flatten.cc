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
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

/*!
 * \brief Transform block with init into actual computation
 */
class ReductionTransformer : public StmtExprMutator {
 public:
  ReductionTransformer() = default;

  Stmt VisitStmt_(const BlockNode* op) override {
    Block res = Downcast<Block>(StmtMutator::VisitStmt_(op));
    if (op->init) {
      PrimExpr condition = Bool(true);
      for (const auto& var : res->iter_vars) {
        if (var->iter_type == IterVarType::kCommReduce) {
          condition = And(condition, EQ(var, var->dom->min));
        }
      }
      Stmt init = op->init.value();
      if (!is_one(condition)) init = IfThenElse(condition, init);
      res.CopyOnWrite()->body = SeqStmt::Flatten(init, op->body);
      res.CopyOnWrite()->init = NullOpt;
    }
    return std::move(res);
  }
};

/*!
 * \brief Detecting the LCA of buffer access points of
 *        buffers for calculating the realize region
 */
class LCADetector : public StmtExprVisitor {
 public:
  explicit LCADetector(const Map<Var, Buffer>& func_args) {
    for (const auto& x : func_args) {
      arg_buffers_.insert(x.second);
      buffers_lca_[x.second] = NullValue<ObjectRef>();
    }
  }

  // Update parent and depth information for each AST node

  void VisitStmt_(const ForNode* op) final {
    Stmt n = GetRef<Stmt>(op);
    ast_scopes_info_[n] = ScopeInfo{scope_, depth_};
    ++depth_;
    std::swap(scope_, n);
    StmtExprVisitor::VisitStmt_(op);
    std::swap(scope_, n);
    --depth_;
  }

  // Update LCA when visiting BufferLoad and BufferStore
  template <typename T>
  void VisitBuffer(T op) {
    Buffer buffer = op->buffer;
    ObjectRef n = GetRef<ObjectRef>(op);
    ast_scopes_info_[n] = ScopeInfo{scope_, depth_};
    // No need to update LCA if the buffer is in the func args (function input/output buffer)
    if (arg_buffers_.count(buffer)) return;
    if (buffers_lca_.count(buffer)) {
      buffers_lca_[buffer] = LowestCommonAncestor(GetRef<ObjectRef>(op), buffers_lca_[buffer]);
    } else {
      buffers_lca_[buffer] = GetRef<ObjectRef>(op);
    }
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    VisitBuffer(op);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode* op) final {
    VisitBuffer(op);
    StmtExprVisitor::VisitStmt_(op);
  }

  /*! \brief The map from Buffer to its LCA Stmt/Expr */
  std::unordered_map<Buffer, ObjectRef, ObjectPtrHash, ObjectPtrEqual> buffers_lca_;
  /*! \brief The Buffer in function args */
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> arg_buffers_;

 private:
  /*! \brief The AST node information for querying LCA */
  struct ScopeInfo {
    // The parent loop node
    Stmt parent_scope;
    // The scope depth in the AST
    size_t depth;
  };

  /*! \brief The current scope initializing with Null */
  Stmt scope_{NullValue<Stmt>()};
  /*! \brief The current DFS depth */
  size_t depth_{0};
  /*! \brief The parent and depth info of each Loop/BufferLoad/BufferStore Node */
  std::unordered_map<ObjectRef, ScopeInfo, ObjectPtrHash, ObjectPtrEqual> ast_scopes_info_;

  ObjectRef LowestCommonAncestor(ObjectRef lhs, ObjectRef rhs) {
    if (!lhs.defined() || !rhs.defined()) return NullValue<ObjectRef>();
    ICHECK(ast_scopes_info_.count(lhs));
    ICHECK(ast_scopes_info_.count(rhs));
    while (ast_scopes_info_[lhs].depth > ast_scopes_info_[rhs].depth) {
      lhs = ast_scopes_info_[lhs].parent_scope;
    }
    while (ast_scopes_info_[lhs].depth < ast_scopes_info_[rhs].depth) {
      rhs = ast_scopes_info_[rhs].parent_scope;
    }
    while (!lhs.same_as(rhs)) {
      lhs = ast_scopes_info_[lhs].parent_scope;
      rhs = ast_scopes_info_[rhs].parent_scope;
    }
    return lhs;
  }
};

/*!
 * \brief Gather the used region of each buffers.
 */
class RegionGatherer : public StmtExprVisitor {
 public:
  RegionGatherer(
      const std::unordered_map<Buffer, ObjectRef, ObjectPtrHash, ObjectPtrEqual>& buffers_lca,
      const Map<Var, Buffer>& func_args)
      : buffers_lca_(buffers_lca) {
    for (const auto& arg : func_args) {
      std::vector<arith::IntSet> region;
      for (const auto& size : arg.second->shape) {
        region.push_back(arith::IntSet::FromRange(Range::FromMinExtent(0, size)));
      }
      buffers_region_[arg.second] = region;
    }
  }

  void VisitStmt_(const ForNode* op) final {
    auto loop = GetRef<For>(op);
    loop_stack_.push_back(loop);
    if (op->annotations.empty() && is_one(op->extent)) {
      unit_loops_[op->loop_var.get()] = op->min;
    }
    StmtExprVisitor::VisitStmt_(op);
    loop_stack_.pop_back();
  }

  void VisitStmt_(const BlockRealizeNode* op) final {
    const auto* block_op = op->block.as<BlockNode>();
    for (size_t i = 0; i < block_op->iter_vars.size(); ++i) {
      const auto& iter = block_op->iter_vars[i];
      const auto& v = op->binding_values[i];
      block_var_[iter->var.get()] = Substitute(Substitute(v, block_var_), unit_loops_);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BlockNode* op) final {
    for (const auto& buffer_region : op->reads) {
      VisitBufferRegion(buffer_region);
    }
    for (const auto& buffer_region : op->writes) {
      VisitBufferRegion(buffer_region);
    }
    for (const auto& alloc_buf : op->alloc_buffers) {
      std::vector<arith::IntSet> empty_region(alloc_buf->shape.size(), arith::IntSet::Nothing());
      // Initialize the buffer region with empty region.
      buffers_region_[alloc_buf] = empty_region;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  /*! \brief The used region of each Buffer */
  std::unordered_map<Buffer, std::vector<arith::IntSet>, ObjectPtrHash, ObjectPtrEqual>
      buffers_region_;
  /*! \brief The map from block vars to the expr value */
  std::unordered_map<const VarNode*, PrimExpr> block_var_;
  /*! \brief The map from unit lopo vars to the expr value */
  std::unordered_map<const VarNode*, PrimExpr> unit_loops_;

 private:
  const std::unordered_map<Buffer, ObjectRef, ObjectPtrHash, ObjectPtrEqual>& buffers_lca_;

  /*! \brief The loops from the current node up to the root */
  std::vector<For> loop_stack_;

  void VisitBufferRegion(const BufferRegion& buffer_region) {
    auto it = buffers_region_.find(buffer_region->buffer);
    ICHECK(it != buffers_region_.end());
    const auto& region = GatherRegion(buffer_region);
    auto& buffer_new_region = it->second;
    ICHECK_EQ(buffer_new_region.size(), region.size());
    for (size_t i = 0; i < region.size(); ++i) {
      buffer_new_region[i] = arith::Union({buffer_new_region[i], region[i]});
    }
  }

  /*!
   * \brief Gather used buffer region
   */
  std::vector<arith::IntSet> GatherRegion(const BufferRegion& buffer_region) {
    std::unordered_map<const VarNode*, arith::IntSet> dom_map;
    auto it = buffers_lca_.find(buffer_region->buffer);
    ICHECK(it != buffers_lca_.end());
    const auto& lca = it->second;
    // Every loop will be relaxed if the lca is the root
    bool need_relax = !lca.defined();
    for (size_t i = 0; i < loop_stack_.size(); ++i) {
      const For& loop = loop_stack_[i];
      const VarNode* var = loop->loop_var.get();
      if (need_relax || (buffer_region->buffer->scope == "shared" && IsThreadBinded(loop))) {
        dom_map[var] = arith::IntSet::FromRange(Range::FromMinExtent(loop->min, loop->extent));
      }
      if (loop.same_as(lca)) need_relax = true;
    }
    std::vector<arith::IntSet> region;
    for (const auto& range : buffer_region->region) {
      Range r = Range::FromMinExtent(Substitute(Substitute(range->min, block_var_), unit_loops_),
                                     Substitute(Substitute(range->extent, block_var_), unit_loops_));
      region.push_back(arith::EvalSet(r, dom_map));
    }
    return region;
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
      const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>& arg_buffers)
      : buffers_region_(buffers_region),
        block_var_(block_var),
        unit_loops_(unit_loops),
        buffers_lca_(buffers_lca),
        arg_buffers_(arg_buffers) {}

  Stmt VisitStmt(const Stmt& stmt) override {
    Stmt body = StmtMutator::VisitStmt(stmt);
    return body;
  }

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
          body = AttrStmt(buffer->data, tir::attr::double_buffer_scope, 1, body);
        } else {
          double_buffer_.insert(buffer);
        }
      }

      seq.push_back(body);
    }

    return SeqStmt(seq);
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    // Handle allocations
    const auto* block_op = op->block.as<BlockNode>();
    Stmt old_stmt = GetRef<Stmt>(block_op);
    ICHECK(block_op != nullptr);
    for (size_t i = block_op->alloc_buffers.size(); i > 0; --i) {
      const auto& buffer = block_op->alloc_buffers[i - 1];
      const std::string name = std::string(buffer->name);
      if (name.substr(0, 18) == "normal_reduce_temp" || name.substr(0, 11) == "reduce_temp") {
        continue;
      }
      if (buffers_lca_.at(buffer).defined()) {
        pending_allocate_[buffer] = block_op->alloc_buffers[i - 1];
      }
    }
    for (size_t i = 0; i < block_op->iter_vars.size(); ++i) {
      const IterVar& block_var = block_op->iter_vars[i];
      const PrimExpr& binding_value = op->binding_values[i];
      ICHECK(block_var.as<IterVarNode>());
      ICHECK(binding_value.as<PrimExprNode>());

      if (block_var->iter_type == kCommReduce) {
        PreOrderVisit(binding_value, [this] (const ObjectRef& node) {
          if (const auto* var = node.as<VarNode>()) {
            this->reduction_relative_.insert(GetRef<Var>(var));
            return false;
          }
          return true;
        });
      }
    }
    // visit body
    Stmt parent_scope = op->block;
    std::swap(parent_scope, parent_scope_);
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    std::swap(parent_scope, parent_scope_);
    op = stmt.as<BlockRealizeNode>();
    ICHECK(op != nullptr);
    block_op = op->block.as<BlockNode>();
    ICHECK(block_op != nullptr);
    Stmt body = block_op->body;
    // Handle block predicate
    if (!is_one(op->predicate)) {
      body = IfThenElse(op->predicate, body);
    }

    for (const auto& anno : block_op->annotations) {
      if (anno.first == tir::attr::double_buffer_scope && is_one(Downcast<PrimExpr>(anno.second))) {
        ICHECK_EQ(block_op->writes.size(), 1);
        double_buffer_.insert(block_op->writes[0]->buffer);
      }
    }

    for (size_t i = block_op->alloc_buffers.size(); i > 0; --i) {
      const auto& alloc_buf = block_op->alloc_buffers[i - 1];
      const std::string name = std::string(alloc_buf->name);
      if (name.substr(0, 18) == "normal_reduce_temp" || name.substr(0, 11) == "reduce_temp") {
        continue;
      }
      if (!buffers_lca_.at(alloc_buf).defined() || buffers_lca_.at(alloc_buf).same_as(old_stmt)) {
        PrimExpr extents = 1;
        for (const auto& extent : buffers_region_.at(alloc_buf)) {
          extents *= extent.max() - extent.min() + 1;
        }
        body = Allocate(alloc_buf->data, alloc_buf->dtype, {extents}, const_true(), body);

        // Change empty scope into global
        std::string scope = alloc_buf->scope.empty() ? "global" : alloc_buf->scope;
        body = AttrStmt(alloc_buf->data, attr::storage_scope, StringImm(scope), body);
      }
    }

    return body;
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

  Stmt VisitStmt_(const ForNode* op) final {
    Stmt old_stmt = GetRef<Stmt>(op);
    std::swap(old_stmt, parent_scope_);
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    std::swap(old_stmt, parent_scope_);

    op = stmt.as<ForNode>();
    ICHECK(op != nullptr);

    ForKind kind = op->kind;
    if (op->kind == ForKind::kThreadBinding)
      kind = ForKind::kSerial;

    Stmt body = op->body;
    for (auto it = pending_allocate_.begin(); it != pending_allocate_.end();) {
      if (old_stmt.same_as(buffers_lca_.at(it->first))) {
        PrimExpr extents = 1;
        const auto& alloc_buf = it->second;
        for (const auto& extent : buffers_region_.at(alloc_buf)) {
          extents *= extent.max() - extent.min() + 1;
        }
        body = Allocate(alloc_buf->data, alloc_buf->dtype, {extents}, const_true(), body);
        // Change empty scope into global
        std::string scope = alloc_buf->scope.empty() ? "global" : alloc_buf->scope;
        body = AttrStmt(alloc_buf->data, attr::storage_scope, StringImm(scope), body);
        pending_allocate_.erase(it++);
      } else {
        it++;
      }
    }

    Stmt for_stmt;
    if (op->kind == ForKind::kThreadBinding) {
      ICHECK(op->thread_binding.defined());
      String thread_tag = op->thread_binding.value()->thread_tag;
      if (!reduction_relative_.count(op->loop_var)) {
        for_stmt = AttrStmt(IterVar(Range(op->min, op->extent), op->loop_var,
                                    IterVarType::kThreadIndex, thread_tag),
                            thread_tag == "vthread" ? attr::virtual_thread : attr::thread_extent,
                            op->extent, body);
      } else {
        for_stmt = body;
      }
    } else if (is_one(op->extent) && op->annotations.empty()) {
      return body;
    } else {
      for_stmt = For(op->loop_var, op->min, op->extent, op->kind, body);
    }

    for (const auto& annotation : op->annotations) {
      if (attr::IsPragmaKey(annotation.first)) {
        for_stmt = AttrStmt(op->loop_var, annotation.first, Downcast<PrimExpr>(annotation.second),
                            for_stmt);
      }
    }

    return for_stmt;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();
    ICHECK(op != nullptr);
    auto begins = ComputeRelativeIndices(op->buffer, op->indices);
    Buffer new_buffer = ReshapeBuffer(op->buffer, this->buffers_region_.at(op->buffer));
    return new_buffer.vstore(begins, op->value);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();
    auto begins = ComputeRelativeIndices(op->buffer, op->indices);
    Buffer new_buffer = ReshapeBuffer(op->buffer, this->buffers_region_.at(op->buffer));
    return new_buffer.vload(begins, op->dtype);
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
  const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>& arg_buffers_;

  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> pending_allocate_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> reduction_relative_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> double_buffer_;
  Stmt parent_scope_;

  /*!
   * \brief Create a buffer with alternative shape
   */
  Buffer ReshapeBuffer(const Buffer& buffer, const std::vector<arith::IntSet>& region) {
    if (arg_buffers_.count(buffer)) return buffer;
    auto n = runtime::make_object<BufferNode>(*(buffer.operator->()));
    Array<PrimExpr> shape;
    for (const auto& i : region) {
      shape.push_back(i.max() - i.min() + 1);
    }
    n->shape = std::move(shape);
    return Buffer(n);
  }

  /*!
   * \brief Transform indices from the absolute indices to relative indices
   * \note T can be BufferLoad or BufferStore
   */
  std::vector<PrimExpr> ComputeRelativeIndices(const Buffer& buffer,
                                               const Array<PrimExpr>& indices) {
    auto it = buffers_region_.find(buffer);
    ICHECK(it != buffers_region_.end());
    const auto& region = it->second;
    std::vector<PrimExpr> new_indices;
    for (size_t i = 0; i < region.size(); ++i) {
      if (arg_buffers_.count(buffer)) {
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
  ScheduleNode::ValidateHierarchy(f);

  // Transform the reduction calls to BufferStore
  ReductionTransformer reduction_transformer;
  fptr->body = reduction_transformer(fptr->body);

  // Find the LCA of each Buffer access
  LCADetector lca_detector(fptr->buffer_map);
  lca_detector(fptr->body);

  // Recalculate the buffer region
  RegionGatherer region_gatherer(lca_detector.buffers_lca_, fptr->buffer_map);
  region_gatherer(fptr->body);

  // Transform BufferLoad/BufferStore into Load/Store
  BufferFlattener flattener(region_gatherer.block_var_, region_gatherer.unit_loops_,
                            region_gatherer.buffers_region_, lca_detector.buffers_lca_,
                            lca_detector.arg_buffers_);
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
