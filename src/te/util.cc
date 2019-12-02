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
 * \file util.cc
 */

#include "util.h"
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>

namespace tvm {
namespace te {

inline Array<Expr> MutateArray(Array<Expr> arr, IRMutator* m) {
  std::vector<Expr> new_arr(arr.size());
  bool changed = false;
  for (size_t i = 0; i < arr.size(); ++i) {
    Expr old_elem = arr[i];
    Expr new_elem = m->Mutate(old_elem);
    if (!new_elem.same_as(old_elem)) changed = true;
    new_arr[i] = new_elem;
  }
  if (!changed) {
    return arr;
  } else {
    return Array<Expr>(new_arr);
  }
}

Stmt ScheduleMutator::Mutate_(const BlockNode* op, const Stmt& s) {
  te::Block block = GetRef<te::Block>(op);
  block.Mutable()->values = MutateArray(op->values, this);
  block.Mutable()->predicate = this->Mutate(op->predicate);
  block.Mutable()->body = this->Mutate(op->body);
}

Stmt ScheduleMutator::Mutate_(const LoopNode* op, const Stmt& s) {
  te::Loop loop = GetRef<te::Loop>(op);
  loop.Mutable()->min = Mutate(loop->min);
  loop.Mutable()->extent = Mutate(loop->extent);
  loop.Mutable()->body = Mutate(loop->body);
  return std::move(loop);
}

Array<Var> GatherVars(const NodeRef& expr_or_stmt) {
  Array<Var> ret;
  ir::PostOrderVisit(expr_or_stmt, [&ret](const NodeRef& node) {
    if (node.as<Variable>() && !Downcast<Var>(node).type().is_handle()) {
      ret.push_back(Downcast<Var>(node));
    }
  });
  return ret;
}

void TensorAccessGather::Visit_(const class BufferLoadNode* op) {
  if (target_tensor_.defined()) {
    if (target_tensor_ == op->buffer) {
      std::vector<Expr> indices;
      for (const auto& x : op->indices) {
        indices.push_back(x);
      }
      access_one.push_back(indices);
    }
  } else {
    std::pair<Buffer, std::vector<Expr>> acc;
    acc.first = op->buffer;
    for (const auto& x : op->indices) {
      acc.second.push_back(x);
    }
    access_all.push_back(acc);
    if (!access_grouped.count(acc.first)) {
      tensor_order.push_back(acc.first);
    }
    access_grouped[acc.first].push_back(acc.second);
  }
}

Array<TensorRegion> CreateInputRegions(const Stmt& stmt) {
  Array<TensorRegion> inputs;

  TensorAccessGather gather;
  gather.Visit(stmt);

  for (const auto& t : gather.tensor_order) {  // for all tensors
    Array<Range> ranges;
    const auto& access_info = gather.access_grouped[t];

    for (size_t i = 0; i < t->shape.size(); ++i) {  // for all dimensions
      Array<arith::IntSet> sets;
      for (const auto& x : access_info) {   // for multiple accesses
        sets.push_back(arith::IntSet::single_point(x[i]));
      }
      arith::IntSet unioned = arith::Union(sets);
      ranges.push_back(Range::make_by_min_extent(unioned.min(),
                                                 unioned.max() - unioned.min() + 1));
    }

    inputs.push_back(TensorRegion(t, ranges));
  }
  return inputs;
}

}  // namespace te
}  // namespace tvm
