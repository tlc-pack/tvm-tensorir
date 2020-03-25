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

#include <tvm/tir/ir.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/schedule.h>
#include <tvm/tir/ir_pass.h>
#include <unordered_map>
#include <vector>
#include <utility>
#include "schedule_common.h"


namespace tvm {
namespace tir {

/*! \note Nested SeqStmt is not allowed in schedule. */
Array<Stmt> GetChildren(const Stmt& stmt, bool keep_realize) {
  Stmt body;
  if (const auto* block = stmt.as<BlockNode>()) {
    body = block->body;
  } else if (const auto* loop = stmt.as<LoopNode>()) {
    body = loop->body;
  } else {
    return Array<Stmt>();
  }
  if (const auto* seq = body.as<SeqStmtNode>()) {
    Array<Stmt> ret;
    for (const Stmt& child : seq->seq)
      if (child->IsInstance<BlockRealizeNode>() && !keep_realize) {
        ret.push_back(child.as<BlockRealizeNode>()->block);
      } else {
        ret.push_back(child);
      }
    return ret;
  } else {
    return Array<Stmt>{body};
  }
}

class IRSubstitueInScope : public StmtExprMutator {
 public:
  explicit IRSubstitueInScope(
      std::function<PrimExpr(const VarNode*)> fmap)
      : fmap_(std::move(fmap)) {}

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = fmap_(op);
    if (it.defined()) {
      return it;
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    auto fmutate = [this](const PrimExpr& e) { return this->VisitExpr(e); };
    Array<PrimExpr> v = op->binding_values;
    v.MutateByApply(fmutate);
    PrimExpr pred = this->VisitExpr(op->predicate);
    if (v.same_as(op->binding_values) && pred.same_as(op->predicate)) {
      return GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->binding_values = std::move(v);
      n->predicate = std::move(pred);
      return Stmt(n);
    }
  }

 private:
  const std::function<PrimExpr(const VarNode*)> fmap_;
};

Stmt SubstituteInScope(const Stmt& stmt,
                       const std::function<PrimExpr(const VarNode*)>& value_func) {
  return IRSubstitueInScope(value_func)(stmt);
}

// Only Block and Loop are allowed here.
template <typename T>
Stmt GetStmtFromSeq(const T* op,
                    const Stmt& target,
                    const std::function<bool(const Stmt&, const Stmt&)>& f_equal,
                    int64_t seq_index) {
  if (const auto* seq = op->body.template as<SeqStmtNode>()) {
    if (seq_index >= 0) {
      // fast path
      CHECK(f_equal((*seq)[seq_index], target));
      return (*seq)[seq_index];
    } else {
      // apply slow path when seq_index == -1
      for (const auto& s : seq->seq) {
        if (f_equal(s, target)) return (*seq)[seq_index];
      }
      LOG(FATAL) << "Can not find target stmt";
    }
  } else {
    CHECK(f_equal(op->body, target));
    return op->body;
  }
  return NullValue<Stmt>();
}

BlockRealize GetBlockRealize(const StmtSRef& block_sref) {
  Stmt s = GetRef<Stmt>(block_sref->node);
  CHECK(GetRef<Stmt>(block_sref->node).as<BlockNode>());
  const auto* parent = block_sref->parent;
  Stmt parent_stmt = GetRef<Stmt>(parent->node);

  auto f_equal = [](const Stmt& s, const Stmt& target) {
    CHECK(target.as<BlockNode>());
    const auto* block_realize = s.as<BlockRealizeNode>();
    if (block_realize != nullptr) {
      return block_realize->block.same_as(target);
    } else {
      return false;
    }
  };

  if (const auto* block = parent_stmt.as<BlockNode>()) {
    return Downcast<BlockRealize>(GetStmtFromSeq(block, s, f_equal, block_sref->seq_index));
  } else if (const auto* loop = parent_stmt.as<LoopNode>()) {
    return Downcast<BlockRealize>(GetStmtFromSeq(loop, s, f_equal, block_sref->seq_index));
  } else {
    LOG(FATAL) << "Unknown SRef Type";
  }
  return NullValue<BlockRealize>();
}

StmtSRef LowestCommonAncestor(const std::vector<StmtSRef>& nodes, const StmtSRef& root) {
  // alg: count the visit times for each node from the bottom to the root
  CHECK_GE(nodes.size(), 2);
  std::unordered_map<StmtSRef, size_t, ObjectHash, ObjectEqual> visit_cnt;

  auto f_visit = [&visit_cnt](const StmtSRef& node) {
    auto it = visit_cnt.find(node);
    if (it == visit_cnt.end()) {
      visit_cnt[node] = 1;
    } else {
      it->second++;
    }
  };

  for (auto node : nodes) {
    while (!node.same_as(root)) {
      f_visit(node);
      if (visit_cnt[node] == nodes.size()) {
        return node;
      }
      node = GetRef<StmtSRef>(node->parent);
    }
  }

  return root;
}

void RelaxRegion(const StmtSRef& block_sref, const StmtSRef& root,
                 std::vector<TensorRegion>* reads,
                 std::vector<TensorRegion>* writes) {
  const auto* block = DowncastPtr<BlockNode>(block_sref->node);
  const auto* block_realize = GetBlockRealize(block_sref).operator->();
  CHECK(block != nullptr);

  // Update block_var map
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    vmap[block->iter_vars[i]->var.get()] = block_realize->binding_values[i];
  }

  // Gather iteration domain
  std::unordered_map<const VarNode*, arith::IntSet> dom_map;
  auto sref = GetRef<StmtSRef>(block_sref->parent);
  while (sref.defined()) {
    const auto* loop = DowncastPtr<LoopNode>(sref->node);
    // The root may not be a loop
    if (loop == nullptr) break;
    Range range = Range::make_by_min_extent(loop->min, loop->extent);
    dom_map[loop->loop_var.get()] = arith::IntSet::range(range);
    sref = GetRef<StmtSRef>(sref->parent);
    if (sref.same_as(root)) break;
  }

  auto relax = [&vmap, &dom_map](const TensorRegion& tensor_region) {
    auto n = make_object<TensorRegionNode>();
    Array<Range> region;
    n->buffer = tensor_region->buffer;
    for (auto range : tensor_region->region) {
      range = Range::make_by_min_extent(Substitute(range->min, vmap),
                                        Substitute(range->extent, vmap));
      auto int_set = arith::EvalSet(range, dom_map);
      region.push_back(Range::make_by_min_extent(int_set.min(), int_set.max() - int_set.min() + 1));
    }
    n->region = std::move(region);
    return TensorRegion(n);
  };

  if (reads != nullptr) {
    for (const auto& tensor_region : block->reads) {
      reads->push_back(relax(tensor_region));
    }
  }
  if (writes != nullptr) {
    for (const auto& tensor_region : block->writes) {
      writes->push_back(relax(tensor_region));
    }
  }
}

}  // namespace tir
}  // namespace tvm
