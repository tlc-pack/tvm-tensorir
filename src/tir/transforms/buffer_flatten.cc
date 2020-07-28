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
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

/*!
 * \brief Transform reduction call into actual computation
 */
class ReductionTransformer : public StmtExprMutator {
 public:
  ReductionTransformer() = default;

  Stmt VisitStmt_(const BlockNode* op) override {
    const BlockNode* block = op;
    std::swap(current_block_, block);
    Stmt res = StmtMutator::VisitStmt_(op);
    std::swap(current_block_, block);
    return res;
  }

  Stmt VisitStmt_(const ReduceStepNode* op) override {
    const auto& init = op->comm_reducer->identity_element[0];
    const auto* lhs = op->lhs.as<BufferLoadNode>();
    PrimExpr cond = make_const(DataType::Bool(1), true);
    for (const auto& iter_var : current_block_->iter_vars)
      if (iter_var->iter_type == IterVarType::kCommReduce) {
        cond = cond && (iter_var == 0);
      }
    return BufferStore(lhs->buffer,
                       if_then_else(cond, op->ApplyCombiner(init, op->rhs), op->ApplyCombiner()),
                       lhs->indices);
  }

 private:
  const BlockNode* current_block_{nullptr};
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

  void VisitStmt_(const LoopNode* op) final {
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
  /*! \brief The Buffer in function args */
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> arg_buffers_;

  ObjectRef LowestCommonAncestor(ObjectRef lhs, ObjectRef rhs) {
    CHECK(ast_scopes_info_.count(lhs));
    CHECK(ast_scopes_info_.count(rhs));
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

  void VisitStmt_(const LoopNode* op) final {
    Loop loop = GetRef<Loop>(op);
    loop_stack_.push_back(loop);
    StmtExprVisitor::VisitStmt_(op);
    loop_stack_.pop_back();
  }

  void VisitStmt_(const BlockRealizeNode* op) final {
    const auto* block_op = op->block.as<BlockNode>();
    for (size_t i = 0; i < block_op->iter_vars.size(); ++i) {
      const auto& iter = block_op->iter_vars[i];
      const auto& v = op->binding_values[i];
      block_var_[iter->var.get()] = v;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferAllocateNode* op) final {
    std::vector<arith::IntSet> empty_region(op->buffer->shape.size(), arith::IntSet::Nothing());
    // Initialize the buffer region with empty region.
    buffers_region_[op->buffer] = empty_region;
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    VisitBuffer(op);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    VisitBuffer(op);
    StmtExprVisitor::VisitExpr_(op);
  }

  /*! \brief The used region of each Buffer */
  std::unordered_map<Buffer, std::vector<arith::IntSet>, ObjectPtrHash, ObjectPtrEqual>
      buffers_region_;
  /*! \brief The map from block vars to the expr value */
  std::unordered_map<const VarNode*, PrimExpr> block_var_;

 private:
  const std::unordered_map<Buffer, ObjectRef, ObjectPtrHash, ObjectPtrEqual>& buffers_lca_;

  /*! \brief The loops from the current node up to the root */
  std::vector<Loop> loop_stack_;

  /*! \note T can be BufferLoad or BufferStore */
  template <typename T>
  void VisitBuffer(const T* op) {
    auto it = buffers_region_.find(op->buffer);
    CHECK(it != buffers_region_.end());
    const auto& region = GatherRegion(op);
    auto& buffer_region = it->second;
    CHECK_EQ(buffer_region.size(), region.size());
    for (size_t i = 0; i < region.size(); ++i) {
      buffer_region[i] = arith::Union({buffer_region[i], region[i]});
    }
  }

  /*!
   * \brief Gather used buffer region
   * \note T can be BufferLoad or BufferStore
   */
  template <typename T>
  std::vector<arith::IntSet> GatherRegion(const T* op) {
    std::unordered_map<const VarNode*, arith::IntSet> dom_map;
    auto it = buffers_lca_.find(op->buffer);
    CHECK(it != buffers_lca_.end());
    const auto& lca = it->second;
    // Every loop will be relaxed if the lca is the root
    bool need_relax = !lca.defined();
    for (size_t i = 0; i < loop_stack_.size(); ++i) {
      const Loop& loop = loop_stack_[i];
      const VarNode* var = loop->loop_var.get();
      if (need_relax) {
        dom_map[var] = arith::IntSet::FromRange(Range::FromMinExtent(loop->min, loop->extent));
      }
      if (loop.same_as(lca)) need_relax = true;
    }
    std::vector<arith::IntSet> region;
    for (const auto& e : op->indices) {
      region.push_back(arith::EvalSet(Substitute(e, block_var_), dom_map));
    }
    return region;
  }
};

/*!
 * \brief Transform multi-dimension BufferLoad/BufferStore into one-dimension Load/Store
 */
class BufferFlattener : public StmtExprMutator {
 public:
  BufferFlattener(
      const std::unordered_map<const VarNode*, PrimExpr>& block_var,
      const std::unordered_map<Buffer, std::vector<arith::IntSet>, ObjectPtrHash, ObjectPtrEqual>&
          buffers_region,
      const std::unordered_map<Buffer, ObjectRef, ObjectPtrHash, ObjectPtrEqual>& buffers_lca)
      : buffers_region_(buffers_region), block_var_(block_var), buffers_lca_(buffers_lca) {}

  Stmt VisitStmt(const Stmt& stmt) override {
    Stmt body = StmtMutator::VisitStmt(stmt);
    return body;
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    // Handle allocations
    const auto* block_op = op->block.as<BlockNode>();
    Stmt old_stmt = GetRef<Stmt>(block_op);
    CHECK(block_op != nullptr);
    for (size_t i = block_op->allocations.size(); i > 0; --i) {
      const auto& buffer = block_op->allocations[i - 1]->buffer;
      if (buffers_lca_.at(buffer).defined()) {
        pending_allocate_[buffer] = block_op->allocations[i - 1];
      }
    }
    // visit body
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BlockRealizeNode>();
    CHECK(op != nullptr);
    block_op = op->block.as<BlockNode>();
    CHECK(block_op != nullptr);
    Stmt body = block_op->body;
    // Handle block predicate
    if (!is_one(op->predicate)) {
      body = IfThenElse(op->predicate, body);
    }

    for (size_t i = block_op->allocations.size(); i > 0; --i) {
      const auto& n = block_op->allocations[i - 1];
      if (!buffers_lca_.at(n->buffer).defined() || buffers_lca_.at(n->buffer).same_as(old_stmt)) {
        PrimExpr extents = 1;
        for (const auto& extent : buffers_region_.at(n->buffer)) {
          extents *= extent.max() - extent.min() + 1;
        }
        body = Allocate(n->buffer->data, n->buffer->dtype, {extents}, const_true(), body);

        // Change empty scope into global
        std::string scope = n->scope.empty() ? "global" : n->scope;
        body = AttrStmt(n->buffer->data, attr::storage_scope, StringImm(scope), body);
      }
    }

    return body;
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    // Replace the block var with its value
    auto it = block_var_.find(op);
    if (it != block_var_.end()) {
      return it->second;
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  Stmt VisitStmt_(const LoopNode* op) final {
    Stmt old_stmt = GetRef<Stmt>(op);

    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<LoopNode>();
    CHECK(op != nullptr);
    // todo(@siyuan): add support for loops with annotations

    ForType for_type = ForType::Serial;
    for (const auto& annotation : op->annotations)
      if (annotation->attr_key == tir::attr::loop_type) {
        std::string type = Downcast<StringImm>(annotation->value)->value;
        if (type == "unroll")
          for_type = ForType::Unrolled;
        else if (type == "vectorize")
          for_type = ForType::Vectorized;
        else if (type == "parallel")
          for_type = ForType::Parallel;
      }

    Stmt body = op->body;
    for (const auto& it : pending_allocate_)
      if (old_stmt.same_as(buffers_lca_.at(it.first))) {
        PrimExpr extents = 1;
        const auto& n = it.second;
        for (const auto& extent : buffers_region_.at(n->buffer)) {
          extents *= extent.max() - extent.min() + 1;
        }
        body = Allocate(n->buffer->data, n->buffer->dtype, {extents}, const_true(), body);

        // Change empty scope into global
        std::string scope = n->scope.empty() ? "global" : n->scope;
        body = AttrStmt(n->buffer->data, attr::storage_scope, StringImm(scope), body);
      }

    return For(op->loop_var, op->min, op->extent, for_type, DeviceAPI::None, body);
  }

  // TODO(Siyuan): add support for For and AttrStmt
  Stmt VisitStmt_(const ForNode* op) final {
    LOG(FATAL) << "For is not allowed in TIR schedule for now.";
    return Stmt();
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    LOG(FATAL) << "AttrStmt is not allowed in TIR schedule for now.";
    return Stmt();
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();
    CHECK(op != nullptr);
    auto begins = ComputeRelativeIndices(op);
    Buffer new_buffer = ReshapeBuffer(op->buffer, this->buffers_region_.at(op->buffer));
    return new_buffer.vstore(begins, op->value);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();
    auto begins = ComputeRelativeIndices(op);
    Buffer new_buffer = ReshapeBuffer(op->buffer, this->buffers_region_.at(op->buffer));
    return new_buffer.vload(begins, op->dtype);
  }

 private:
  const std::unordered_map<Buffer, std::vector<arith::IntSet>, ObjectPtrHash, ObjectPtrEqual>&
      buffers_region_;
  const std::unordered_map<const VarNode*, PrimExpr>& block_var_;
  const std::unordered_map<Buffer, ObjectRef, ObjectPtrHash, ObjectPtrEqual>& buffers_lca_;
  std::unordered_map<Buffer, BufferAllocate, ObjectPtrHash, ObjectPtrEqual> pending_allocate_;

  /*!
   * \brief Create a buffer with alternative shape
   */
  static Buffer ReshapeBuffer(const Buffer& buffer, const std::vector<arith::IntSet>& region) {
    auto n = runtime::make_object<BufferNode>(*(buffer.operator->()));
    std::vector<PrimExpr> shape;
    for (const auto& i : region) {
      shape.push_back(i.max() - i.min() + 1);
    }
    n->shape = shape;
    return Buffer(n);
  }

  /*!
   * \brief Transform indices from the absolute indices to relative indices
   * \note T can be BufferLoad or BufferStore
   */
  template <typename T>
  std::vector<PrimExpr> ComputeRelativeIndices(const T* op) {
    auto it = buffers_region_.find(op->buffer);
    CHECK(it != buffers_region_.end());
    const auto& region = it->second;
    std::vector<PrimExpr> indices;
    for (size_t i = 0; i < region.size(); ++i) {
      indices.push_back(op->indices[i] - region[i].min());
    }
    return indices;
  }
};

PrimFunc BufferFlatten(PrimFunc f) {
  auto fptr = f.CopyOnWrite();

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
  BufferFlattener flattener(region_gatherer.block_var_, region_gatherer.buffers_region_,
                            lca_detector.buffers_lca_);
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
