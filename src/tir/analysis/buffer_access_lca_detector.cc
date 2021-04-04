/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
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
 * \file tir/analysis/buffer_access_lca_detector.cc
 * \brief Detect LCA of buffer access
 */

#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

/*!
 * \brief Detect the LCA position of Buffer access.
 * \note Only consider BlockNode and ForNode to be the LCA nodes.
 */
class LCADetector : public StmtExprVisitor {
 public:
  static Map<Buffer, Stmt> Detect(const PrimFunc& func) {
    LCADetector detector;
    for (const auto& kv : func->buffer_map) {
      const Buffer& buffer = kv.second;
      detector.buffer_var_map_.emplace(buffer->data.get(), buffer.get());
    }
    detector.root_ = Downcast<BlockRealize>(func->body)->block.get();
    detector(func->body);
    // Prepare the return
    Map<Buffer, Stmt> buffer_lca;
    for (const auto& kv : detector.buffer_lca_) {
      buffer_lca.Set(GetRef<Buffer>(kv.first), GetRef<Stmt>(kv.second));
    }
    return buffer_lca;
  }

 private:
  LCADetector() = default;

  void VisitStmt_(const ForNode* op) final {
    int n = ancestor_scopes_.size();
    scope_info_.emplace(op, ScopeInfo{ancestor_scopes_.back(), n});
    ancestor_scopes_.push_back(op);
    StmtExprVisitor::VisitStmt_(op);
    ancestor_scopes_.pop_back();
  }

  void VisitStmt_(const BlockNode* op) final {
    int n = ancestor_scopes_.size();
    for (const Buffer& buf : op->alloc_buffers) {
      buffer_var_map_.emplace(buf->data.get(), buf.get());
    }
    scope_info_.emplace(op, ScopeInfo{ancestor_scopes_.back(), n});
    ancestor_scopes_.push_back(op);
    StmtExprVisitor::VisitStmt_(op);
    ancestor_scopes_.pop_back();
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    UpdateBufferLCA(op->buffer.get());
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    UpdateBufferLCA(op->buffer.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferRealizeNode* op) final {
    buffer_var_map_.emplace(op->buffer->data.get(), op->buffer.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  // Works for Load/Store and opaque access.
  void VisitExpr_(const VarNode* op) final {
    auto it = buffer_var_map_.find(op);
    if (it != buffer_var_map_.end()) {
      UpdateBufferLCA(it->second);
    }
  }

  // Explict to visit buffer data in Load and Store node.
  void VisitExpr_(const LoadNode* op) final {
    ExprVisitor::VisitExpr_(op);
    VisitExpr(op->buffer_var);
  }

  void VisitStmt_(const StoreNode* op) final {
    StmtVisitor::VisitStmt_(op);
    VisitExpr(op->buffer_var);
  }

  void UpdateBufferLCA(const BufferNode* buffer) {
    const StmtNode*& lca = buffer_lca_[buffer];
    lca = LowestCommonAncestor(lca, ancestor_scopes_.back());
  }

  const StmtNode* LowestCommonAncestor(const StmtNode* lhs, const StmtNode* rhs) const {
    if (lhs == nullptr) return rhs;
    if (rhs == nullptr) return lhs;
    while (lhs != root_ && rhs != root_ && lhs != rhs) {
      auto it_l = scope_info_.find(lhs);
      auto it_r = scope_info_.find(rhs);
      ICHECK(it_l != scope_info_.end());
      ICHECK(it_r != scope_info_.end());
      const ScopeInfo& l = it_l->second;
      const ScopeInfo& r = it_r->second;
      if (l.depth == r.depth) {
        lhs = l.parent_scope;
        rhs = r.parent_scope;
      } else if (l.depth < r.depth) {
        rhs = r.parent_scope;
      } else {
        lhs = l.parent_scope;
      }
    }
    if (lhs == root_ || rhs == root_) {
      return root_;
    }
    ICHECK(lhs == rhs);
    return lhs;
  }

  /*!
   * \brief The AST node information for querying LCA.
   * \note Only BlockNode and ForNode are considered, since they are the only position where the
   *       body can be a SeqStmt (The LCA of buffer access) in TensorIR.
   */
  struct ScopeInfo {
    ScopeInfo(const StmtNode* parent, int depth) : parent_scope(parent), depth(depth) {}
    // The parent scope node
    const StmtNode* parent_scope;
    // The scope depth in the AST
    int depth;
  };

  /*! \brief The ancestor scope stacks (Block and For), initialized with Null. */
  std::vector<const StmtNode*> ancestor_scopes_ = {nullptr};
  /*! \brief The parent and depth info of each for Node. */
  std::unordered_map<const StmtNode*, ScopeInfo> scope_info_ = {};
  /*! \brief The map from Buffer to its LCA for nodes. */
  std::unordered_map<const BufferNode*, const StmtNode*> buffer_lca_ = {};
  /*! \brief The map from Buffer data to the Buffer. */
  std::unordered_map<const VarNode*, const BufferNode*> buffer_var_map_ = {};
  /*! \brief The root block of the func. */
  const BlockNode* root_ = nullptr;
};

Map<Buffer, Stmt> DetectBufferAccessLCA(const PrimFunc& func) { return LCADetector::Detect(func); }

TVM_REGISTER_GLOBAL("tir.analysis.detect_buffer_access_lca").set_body_typed(DetectBufferAccessLCA);
}  // namespace tir
}  // namespace tvm
