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
 *  Copyright (c) by Contributors
 * \file te_lower.cc
 */
#include <tvm/ir_pass.h>
#include <tvm/te/ir.h>
#include <tvm/ir_mutator.h>

namespace tvm {
namespace ir {

class TeLowerMutator : public IRMutator {
 public:
  explicit TeLowerMutator(Array<Tensor> tensors) {
    this->tensors = std::move(tensors);
  }

  Stmt Mutate_(const te::FunctionNode* op, const Stmt& s) final {
    CHECK_EQ(op->match_buffer.size(), tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
      op_map.Set(op->match_buffer[i], tensors[i]->op);
    }
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<te::FunctionNode>();
    return op->body;
  }

  Stmt Mutate_(const te::BlockNode* op, const Stmt& s) final {
    for (size_t i = 0; i < op->iter_vars.size(); ++i) {
      const auto& iter = op->iter_vars[i];
      const auto& v = op->values[i];
      block_var[iter->var.get()] = v;
    }
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<te::BlockNode>();
    for (size_t i = 0; i < op->iter_vars.size(); ++i) {
      const auto& iter = op->iter_vars[i];
      const auto& v = op->values[i];
      block_var.erase(iter->var.get());
    }
    return op->body;
  }

  Stmt Mutate_(const te::LoopNode* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<te::LoopNode>();

    return For::make(op->loop_var, op->min, op->extent,
                     ForType::Serial, DeviceAPI::None, op->body);
  }

  Stmt Mutate_(const te::BufferStoreNode* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<te::BufferStoreNode>();
    Operation operation = op_map.at(op->buffer);
    return Provide::make(operation, 0, op->value, op->indices);
  }

  Stmt Mutate_(const ir::Block* op, const Stmt& s) final {
    if (const te::BufferAllocateNode* allocate = op->first.as<te::BufferAllocateNode>()) {
      const auto& buffer = allocate->buffer;
      Operation op = PlaceholderOpNode::make(buffer->name,
                                             buffer->shape,
                                             buffer->dtype);
      op_map.Set(buffer, op);
    }
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<ir::Block>();
    if (const te::BufferAllocateNode* allocate = op->first.as<te::BufferAllocateNode>()) {
      const auto& buffer = allocate->buffer;
      Region region;
      for (const auto& extent : buffer->shape) {
        region.push_back(Range::make_by_min_extent(0, extent));
      }
      Stmt realize = Realize::make(op_map.at(allocate->buffer), 0,
                           buffer->dtype, region, const_true(), op->rest);
      return AttrStmt::make(op_map.at(allocate->buffer),
                            attr::realize_scope,
                            allocate->scope,
                            realize);
    } else {
      return stmt;
    }

  }

  Expr Mutate_(const Variable* op, const Expr& e) final {
    auto it = block_var.find(op);
    if (it != block_var.end()) {
      return it->second;
    } else {
      return e;
    }
  }

  Expr Mutate_(const te::BufferLoadNode* op, const Expr& e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<te::BufferLoadNode>();
    Operation operation = op_map.at(op->buffer);
    return Call::make(op->type, op->buffer->name, op->indices,
                      Call::CallType::Halide, operation, 0);
  }

 private:
  Map<Buffer, Operation> op_map;
  std::unordered_map<const Variable*, Expr> block_var;
  Array<Tensor> tensors;
};

Stmt TeLower(Stmt stmt, Array<Tensor> tensors) {
  return TeLowerMutator(tensors).Mutate(stmt);
}

}  // namespace ir
}  // namespace tvm
