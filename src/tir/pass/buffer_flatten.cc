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

#include <tvm/tir/ir.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/ir/attrs.h>
#include <tvm/arith/int_set.h>

namespace tvm {
namespace tir {

/*!
 * \brief Remove Block in TIR and replace BufferAllocate with Allocate
 * \note After flattening, the TIR can not be scheduled anymore
 */
class BlockFlattener : public StmtExprMutator {
 public:
  Stmt VisitStmt_(const BlockNode* op) final {
    for (size_t i = 0; i < op->iter_vars.size(); ++i) {
      const auto& iter = op->iter_vars[i];
      const auto& v = op->values[i];
      block_var_[iter->var.get()] = v;
    }
    Stmt block = StmtExprMutator::VisitStmt_(op);
    op = block.as<BlockNode>();
    CHECK(op != nullptr);
    Stmt body = op->body;

    // Handle block predicate
    if (!is_one(op->predicate)) {
      body = IfThenElseNode::make(op->predicate, body);
    }

    // Handle block allocations
    for (size_t i = op->allocations.size(); i > 0; --i) {
      const auto& n = op->allocations[i - 1];
      buffer_map_[n->buffer->data.get()] = n->buffer;
      body = AllocateNode::make(n->buffer->data,
                                n->buffer->dtype,
                                n->buffer->shape,
                                const_true(),
                                body);

      // Change empty scope into global
      std::string scope = n->scope.empty() ? "global" : n->scope;
      body = AttrStmtNode::make(n->buffer->data,
                                attr::storage_scope,
                                StringImmNode::make(scope),
                                body);
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

  /*! \brief The map from buffer_var to buffer */
  std::unordered_map<const VarNode*, Buffer> buffer_map_;
 private:
  /*! \brief The map from block vars to the expr value */
  std::unordered_map<const VarNode*, PrimExpr> block_var_;
};

/*!
 * \brief Detecting the LCA of buffers for calculating the realize region
 */
class LCADetector : public StmtExprVisitor {
 public:
  explicit LCADetector(const Map<Var, Buffer>& func_args) {
    for (const auto& x : func_args) {
      arg_buffers_.insert(x.second);
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
  std::unordered_map<Buffer, ObjectRef, ObjectHash, ObjectEqual> buffers_lca_;

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
  std::unordered_map<ObjectRef, ScopeInfo, ObjectHash, ObjectEqual> ast_scopes_info_;
  /*! \brief The Buffer in function args */
  std::unordered_set<Buffer, ObjectHash, ObjectEqual> arg_buffers_;

  ObjectRef LowestCommonAncestor(ObjectRef lhs, ObjectRef rhs) {
    CHECK(ast_scopes_info_.count(lhs));
    CHECK(ast_scopes_info_.count(rhs));
    while (ast_scopes_info_[lhs].depth > ast_scopes_info_[rhs].depth) {
      lhs = ast_scopes_info_[lhs].parent_scope;
    }
    while (ast_scopes_info_[lhs].depth < ast_scopes_info_[rhs].depth) {
      rhs = ast_scopes_info_[rhs].parent_scope;
    }
    while (lhs != rhs) {
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
  RegionGatherer(const std::unordered_map<Buffer, ObjectRef, ObjectHash, ObjectEqual>& buffers_lca,
                 const std::unordered_map<const VarNode*, Buffer>& buffer_map,
                 const Map<Var, Buffer>& func_args)
      : buffers_lca_(buffers_lca), buffer_map_(buffer_map) {
    for (const auto& arg : func_args) {
      std::vector<arith::IntSet> region;
      for (const auto& shape : arg.second->shape) {
        region.push_back(arith::IntSet::range(Range::make_by_min_extent(0, shape)));
      }
      buffers_region_[arg.second] = region;
      buffers_lca_pos_[arg.second] = 0;
    }
    for (const auto& buffer_lca : buffers_lca_) {
      // The LCA is the root loop
      if (!buffer_lca.second.defined()) {
        buffers_lca_pos_[buffer_lca.first] = 0;
      }
    }
  }

  void VisitStmt_(const LoopNode* op) final {
    Loop loop = GetRef<Loop>(op);
    loop_stack_.push_back(loop);
    for (const auto& buffer_lca : buffers_lca_) {
      if (buffer_lca.second == loop) {
        buffers_lca_pos_[buffer_lca.first] = loop_stack_.size();
      }
    }
    StmtExprVisitor::VisitStmt_(op);
    loop_stack_.pop_back();
  }

  void VisitStmt_(const AllocateNode* op) final {
    std::vector<arith::IntSet> empty_region(op->extents.size(), arith::IntSet::nothing());
    CHECK(buffer_map_.count(op->buffer_var.get()));
    const Buffer& buffer = buffer_map_.at(op->buffer_var.get());
    // Initialize the buffer region with empty region.
    buffers_region_[buffer] = empty_region;
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
  std::unordered_map<Buffer, std::vector<arith::IntSet>, ObjectHash, ObjectEqual> buffers_region_;

 private:
  const std::unordered_map<Buffer, ObjectRef, ObjectHash, ObjectEqual>& buffers_lca_;
  const std::unordered_map<const VarNode*, Buffer>& buffer_map_;

  /*!
   * \brief The buffer's LCA loop position at the loop stack
   * \note Only loops are interested since the buffer index will contain loop_vars
   */
  std::unordered_map<Buffer, size_t, ObjectHash, ObjectEqual> buffers_lca_pos_;
  /*! \brief The loops from the current node up to the root */
  std::vector<Loop> loop_stack_;

  template <typename T>
  void VisitBuffer(T op) {
    CHECK(buffers_region_.count(op->buffer));
    std::vector<arith::IntSet> region = GatherRegion(op);
    const std::vector<arith::IntSet>& buffer_region = buffers_region_[op->buffer];
    CHECK_EQ(buffer_region.size(), region.size());
    for (size_t i = 0; i < region.size(); ++i) {
      region[i] = arith::Union({buffer_region[i], region[i]});
    }
    buffers_region_[op->buffer] = region;
  }

  /*! \brief Gather used buffer region */
  template <typename T>
  std::vector<arith::IntSet> GatherRegion(T op) {
    std::unordered_map<const VarNode*, arith::IntSet> dom_map;
    CHECK(buffers_lca_pos_.count(op->buffer));
    size_t pos = buffers_lca_pos_[op->buffer];
    for (size_t i = pos; i < loop_stack_.size(); ++i) {
      const Loop& loop = loop_stack_[i];
      const VarNode* var = loop->loop_var.get();
      dom_map[var] = arith::IntSet::range(Range::make_by_min_extent(loop->min, loop->extent));
    }
    std::vector<arith::IntSet> region;
    for (const auto& e : op->indices) {
      region.push_back(arith::EvalSet(e, dom_map));
    }
    return region;
  }
};

/*!
 * \brief Transform multi-dimension BufferLoad/BufferStore into one-dimension Load/Store
 */
class BufferFlattener : public StmtExprMutator {
 public:
  BufferFlattener(const std::unordered_map<const VarNode*, Buffer>& buffer_map,
                  const std::unordered_map<Buffer, std::vector<arith::IntSet>,
                                     ObjectHash, ObjectEqual>& buffers_region)
      : buffer_map_(buffer_map), buffers_region_(buffers_region) {}

  Stmt VisitStmt_(const AllocateNode* op) final {
    const Buffer& buffer = buffer_map_.at(op->buffer_var.get());
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<AllocateNode>();
    CHECK(op != nullptr);
    PrimExpr extents = 1;
    for (const auto& extent : buffers_region_.at(buffer)) {
      extents *= extent.max() - extent.min() + 1;
    }
    auto o = make_object<AllocateNode>(*op);
    o->extents = {extents};
    return Stmt(o);
  }

  Stmt VisitStmt_(const LoopNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<LoopNode>();
    CHECK(op != nullptr);
    // todo(@siyuan): add support for loops with annotations
    return ForNode::make(op->loop_var,
                         op->min,
                         op->extent,
                         ForType::Serial,
                         DeviceAPI::None,
                         op->body);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto begins = GetIndices(op);
    return op->buffer.vstore(begins, VisitExpr(op->value));
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto begins = GetIndices(op);
    return op->buffer.vload(begins, op->dtype);
  }

 private:
  const std::unordered_map<const VarNode*, Buffer>& buffer_map_;
  const std::unordered_map<Buffer, std::vector<arith::IntSet>,
                           ObjectHash, ObjectEqual>& buffers_region_;

  /*! \brief Transform indeces from the absolute indices to relative indices */
  template <typename T>
  std::vector<PrimExpr> GetIndices(T op) {
    CHECK(buffers_region_.count(op->buffer));
    std::vector<arith::IntSet> region = buffers_region_.at(op->buffer);
    std::vector<PrimExpr> indices;
    for (size_t i = 0; i < region.size(); ++i) {
      indices.push_back(op->indices[i] - region[i].min());
    }
    return indices;
  }
};

Function BufferFlatten(Function func) {
  // Remove block and transfer BufferAllocate to Allocate
  BlockFlattener block_flattener;
  Stmt stmt = block_flattener(func->body);

  // Find the LCA of each Buffer access
  LCADetector lca_detector(func->buffer_map);
  lca_detector(stmt);

  // Recalculate the buffer region
  RegionGatherer region_gatherer(lca_detector.buffers_lca_,
                                 block_flattener.buffer_map_,
                                 func->buffer_map);
  region_gatherer(stmt);

  // Transform BufferLoad/BufferStore into Load/Store
  BufferFlattener flattener(block_flattener.buffer_map_, region_gatherer.buffers_region_);
  auto new_func = make_object<FunctionNode>(*func.operator->());
  new_func->body = flattener(stmt);
  return Function(new_func);
}

}  // namespace tir
}  // namespace tvm
