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
 * \file te_lower.cc
 */
#include <tvm/te/ir.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/te/transform.h>

namespace tvm {
namespace te {

// Lower Te expression and statement to current tvm
class TeLowerMutator : public StmtExprMutator {
 public:
  explicit TeLowerMutator(Map<Buffer, Tensor> tensor_map) {
    for (const auto& pair : tensor_map) {
      op_map_[pair.first] = pair.second->op;
    }
  }

  // delete block annotation and inline block vars
  Stmt VisitStmt_(const BlockNode* op) final {
    for (size_t i = 0; i < op->iter_vars.size(); ++i) {
      const auto& iter = op->iter_vars[i];
      const auto& v = op->values[i];
      block_var_[iter->var.get()] = v;
    }
    for (const auto& allocate : op->allocations) {
      const auto& buffer = allocate->buffer;
      Operation op_ = PlaceholderOpNode::make(buffer->name,
                                              buffer->shape,
                                              buffer->dtype);
      op_map_[buffer] = op_;
    }
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BlockNode>();
    CHECK(op != nullptr);
    for (size_t i = 0; i < op->iter_vars.size(); ++i) {
      const auto& iter = op->iter_vars[i];
      const auto& v = op->values[i];
      block_var_.erase(iter->var.get());
    }
    Stmt last_stmt;
    if (is_one(op->predicate)) {
      last_stmt = op->body;
    } else {
      last_stmt = IfThenElse::make(op->predicate, op->body);
    }
    for (const auto& allocate : op->allocations) {
      // TODO(siyuan): enhance realize
      const auto& buffer = allocate->buffer;
      Region region;
      for (const auto& extent : buffer->shape) {
        region.push_back(Range::make_by_min_extent(0, extent));
      }
      Stmt realize = Realize::make(op_map_.at(allocate->buffer), 0,
                                   buffer->dtype, region, const_true(), last_stmt);
      last_stmt = AttrStmt::make(op_map_.at(allocate->buffer),
                                 attr::realize_scope,
                                 allocate->scope,
                                 realize);
    }
    return last_stmt;
  }

  // transform Loop to ir::For
  Stmt VisitStmt_(const LoopNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<LoopNode>();
    CHECK(op != nullptr);
    return For::make(op->loop_var, op->min, op->extent,
                     ForType::Serial, DeviceAPI::None, op->body);
  }

  // transform BufferStore to ir::Provide
  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();
    CHECK(op != nullptr);
    Operation operation = op_map_.at(op->buffer);
    return Provide::make(operation, 0, op->value, op->indices);
  }

  // replace black var with expr
  Expr VisitExpr_(const Variable* op) final {
    auto it = block_var_.find(op);
    if (it != block_var_.end()) {
      return it->second;
    } else {
      return GetRef<Expr>(op);
    }
  }

  // transform BufferLoad to ir::Call
  Expr VisitExpr_(const BufferLoadNode* op) final {
    Expr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();
    Operation operation = op_map_.at(op->buffer);
    return Call::make(op->dtype, op->buffer->name, op->indices,
                      Call::CallType::Halide, operation, 0);
  }

 private:
  // maps the buffer to the corresponding operation
  std::unordered_map<Buffer, Operation, ObjectHash, ObjectEqual> op_map_;
  // maps the block variable to the binded expression
  std::unordered_map<const Variable*, Expr> block_var_;
};

Function TeLower(Function func, Map<Buffer, Tensor> tensor_map) {
  Stmt stmt = TeLowerMutator(tensor_map)(func->body);
  return Function(func->params, func->buffer_map, func->name, stmt);
}

}  // namespace te
}  // namespace tvm
