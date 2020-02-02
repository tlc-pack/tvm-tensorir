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

struct ASTInfo {
  ObjectRef parent;
  size_t depth;
};

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
    Array<Stmt> stmts;
    CHECK(op != nullptr);

    if (is_one(op->predicate)) {
      stmts.push_back(op->body);
    } else {
      stmts.push_back(IfThenElseNode::make(op->predicate, op->body));
    }
    Stmt body = SeqStmt::Flatten(stmts);
    for (size_t i = op->allocations.size(); i > 0; --i) {
      const auto& n = op->allocations[i - 1];
      buffer_map_[n->buffer->data.get()] = n->buffer;
      body = AllocateNode::make(n->buffer->data,
                                n->buffer->dtype,
                                n->buffer->shape,
                                const_true(),
                                body);

      std::string scope = n->scope == "" ? "global" : n->scope;
      body = AttrStmtNode::make(n->buffer->data,
                                attr::storage_scope,
                                StringImmNode::make(scope),
                                body);
    }
    return body;
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = block_var_.find(op);
    if (it != block_var_.end()) {
      return it->second;
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  std::unordered_map<const VarNode*, Buffer> buffer_map_;

 private:
  std::unordered_map<const VarNode*, PrimExpr> block_var_;
};

class LCADetector : public StmtExprVisitor {
 public:
  explicit LCADetector(const Map<Var, Buffer>& func_args) {
    for (const auto& x : func_args) {
      arg_buffers_.insert(x.second);
    }
  }

  void VisitExpr(const PrimExpr& e) final {
    ObjectRef current_node = ObjectRef(e);
    ast_nodes_info_[e] = ASTInfo{parent_, depth_};
    ++depth_;
    std::swap(parent_, current_node);
    StmtExprVisitor::VisitExpr(e);
    std::swap(parent_, current_node);
    --depth_;
  }

  void VisitStmt(const Stmt& s) final {
    ObjectRef current_node = ObjectRef(s);
    ast_nodes_info_[s] = ASTInfo{parent_, depth_};
    ++depth_;
    std::swap(parent_, current_node);
    StmtExprVisitor::VisitStmt(s);
    std::swap(parent_, current_node);
    --depth_;
  }

  template <typename T>
  void VisitBuffer(T op) {
    Buffer buffer = op->buffer;
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
  std::unordered_map<Buffer, ObjectRef, ObjectHash, ObjectEqual> buffers_lca_;

 private:
  ObjectRef parent_{NullValue<ObjectRef>()};
  size_t depth_{0};
  std::unordered_map<ObjectRef, ASTInfo, ObjectHash, ObjectEqual> ast_nodes_info_;
  std::unordered_set<Buffer, ObjectHash, ObjectEqual> arg_buffers_;

  ObjectRef LowestCommonAncestor(ObjectRef lhs, ObjectRef rhs) {
    CHECK(ast_nodes_info_.count(lhs));
    CHECK(ast_nodes_info_.count(rhs));
    while (ast_nodes_info_[lhs].depth > ast_nodes_info_[rhs].depth) {
      lhs = ast_nodes_info_[lhs].parent;
    }
    while (ast_nodes_info_[lhs].depth < ast_nodes_info_[rhs].depth) {
      rhs = ast_nodes_info_[rhs].parent;
    }
    while (lhs != rhs) {
      lhs = ast_nodes_info_[lhs].parent;
      rhs = ast_nodes_info_[rhs].parent;
    }
    return lhs;
  }
};

class BufferFlattener : public StmtExprMutator {
 public:
  BufferFlattener(const std::unordered_map<Buffer, ObjectRef, ObjectHash, ObjectEqual>* buffers_lca,
                  const std::unordered_map<const VarNode*, Buffer>* buffer_map,
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
  }

  Stmt VisitStmt(const Stmt& s) final {
    for (const auto& buffer_lca : (*buffers_lca_)) {
      if (buffer_lca.second == s) {
        buffers_lca_pos_[buffer_lca.first] =
            s.as<LoopNode>() != nullptr ? loop_stack_.size() + 1 : loop_stack_.size();
      }
    }
    return StmtExprMutator::VisitStmt(s);
  }

  PrimExpr VisitExpr(const PrimExpr& e) final {
    for (const auto& buffer_lca : (*buffers_lca_)) {
      if (buffer_lca.second == e) {
        buffers_lca_pos_[buffer_lca.first] = loop_stack_.size();
      }
    }
    return StmtExprMutator::VisitExpr(e);
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    std::vector<arith::IntSet> empty_region(op->extents.size(), arith::IntSet::nothing());
    CHECK(buffer_map_->count(op->buffer_var.get()));
    const Buffer& buffer = buffer_map_->at(op->buffer_var.get());
    buffers_region_[buffer] = empty_region;
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<AllocateNode>();
    CHECK(op != nullptr);
    PrimExpr extents = 1;
    for (const auto& extent : buffers_region_[buffer]) {
      extents *= extent.max() - extent.min() + 1;
    }
    auto o = make_object<AllocateNode>(*op);
    o->extents = {extents};
    return Stmt(o);
  }

  Stmt VisitStmt_(const LoopNode* op) final {
    loop_stack_.push_back(GetRef<Loop>(op));
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    loop_stack_.pop_back();

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
    auto begins = VisitBuffer(op);
    return op->buffer.vstore(begins, VisitExpr(op->value));
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto begins = VisitBuffer(op);
    return op->buffer.vload(begins, op->dtype);
  }

 private:
  const std::unordered_map<Buffer, ObjectRef, ObjectHash, ObjectEqual>* buffers_lca_;
  const std::unordered_map<const VarNode*, Buffer>* buffer_map_;
  std::unordered_map<Buffer, std::vector<arith::IntSet>, ObjectHash, ObjectEqual> buffers_region_;
  std::unordered_map<Buffer, size_t, ObjectHash, ObjectEqual> buffers_lca_pos_;
  std::vector<Loop> loop_stack_;

  template <typename T>
  std::vector<PrimExpr> VisitBuffer(T op) {
    CHECK(buffers_region_.count(op->buffer));
    std::vector<arith::IntSet> region = GatherRegion(op);
    const std::vector<arith::IntSet>& buffer_region = buffers_region_[op->buffer];
    CHECK_EQ(buffer_region.size(), region.size());
    std::vector<PrimExpr> begins;
    for (size_t i = 0; i < region.size(); ++i) {
      begins.push_back(op->indices[i] - region[i].min());
      region[i] = arith::Union({buffer_region[i], region[i]});
    }
    buffers_region_[op->buffer] = region;
    return begins;
  }

  template <typename T>
  std::vector<arith::IntSet> GatherRegion(T op) {
    std::unordered_map<const VarNode*, arith::IntSet> dom_map;
    CHECK(buffers_lca_pos_.count(op->buffer));
    size_t pos = buffers_lca_pos_[op->buffer];
    for (size_t i = 0; i < loop_stack_.size(); ++i) {
      const Loop& loop = loop_stack_[i];
      const VarNode* var = loop->loop_var.get();
      if (i < pos) {
        dom_map[var] = arith::IntSet::single_point(loop->loop_var);
      } else {
        dom_map[var] = arith::IntSet::range(Range::make_by_min_extent(loop->min, loop->extent));
      }
    }
    std::vector<arith::IntSet> region;
    for (const auto& e : op->indices) {
      region.push_back(arith::EvalSet(e, dom_map));
    }
    return region;
  }
};

Function BufferFlatten(Function func) {
  BlockFlattener block_flattener;
  Stmt stmt = block_flattener(func->body);
  LCADetector lca_detector(func->buffer_map);
  lca_detector(stmt);
  BufferFlattener
      flattener(&lca_detector.buffers_lca_, &block_flattener.buffer_map_, func->buffer_map);
  auto new_func = make_object<FunctionNode>(*func.operator->());
  new_func->body = flattener(stmt);
  return Function(new_func);
}

}  // namespace tir
}  // namespace tvm
