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

#include "./schedule_common.h"

#include <tvm/tir/analysis.h>
#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tir {

bool IsLoopVarParallelizable(const Var& loop_var, const Stmt& block_realize,
                             const ScheduleNode* schedule) {
  const BlockRealizeNode* realize = block_realize.as<BlockRealizeNode>();
  CHECK(realize != nullptr)
      << "InternalError: in IsLoopVarParallelizable, expect BlockRealize, but get type: "
      << block_realize->GetTypeKey();
  const BlockNode* block = realize->block.get();
  // Cond 1. Binding is validated
  if (!schedule->stmt2ref.at(block)->binding_valid) {
    return false;
  }
  CHECK_EQ(realize->binding_values.size(), block->iter_vars.size())
      << "InternalError: BlockRealize is inconsistent with its Block";
  int n = realize->binding_values.size();
  // Cond 2. For each iter var that is not data parallel, the binding does not involve loop_var
  for (int i = 0; i < n; ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    const PrimExpr& binding = realize->binding_values[i];
    if (iter_var->iter_type != kDataPar && ExprContainsVar(binding, loop_var)) {
      return false;
    }
  }
  return true;
}

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
    if (body->IsInstance<BlockRealizeNode>() && !keep_realize) {
      return Array<Stmt>{body.as<BlockRealizeNode>()->block};
    } else {
      return Array<Stmt>{body};
    }
  }
}

class IRSubstitueInScope : public StmtExprMutator {
 public:
  explicit IRSubstitueInScope(std::function<PrimExpr(const VarNode*)> fmap)
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

Stmt SubstituteInScope(const Stmt& stmt,
                       const std::unordered_map<const VarNode*, const VarNode*>& var_map) {
  auto vmap = [&](const VarNode* v) -> PrimExpr {
    const auto& it = var_map.find(v);
    if (it != var_map.end()) {
      return GetRef<Var>(it->second);
    } else {
      return NullValue<PrimExpr>();
    }
  };
  return IRSubstitueInScope(vmap)(stmt);
}

PrimExpr SubstituteInScope(const PrimExpr& expr,
                           const std::unordered_map<const VarNode*, const VarNode*>& var_map) {
  auto vmap = [&](const VarNode* v) -> PrimExpr {
    const auto& it = var_map.find(v);
    if (it != var_map.end()) {
      return GetRef<Var>(it->second);
    } else {
      return NullValue<PrimExpr>();
    }
  };
  return IRSubstitueInScope(vmap)(expr);
}

TensorRegion SubstituteTensorRegion(
    const TensorRegion& tensor_region,
    const std::unordered_map<const VarNode*, const VarNode*>& var_map) {
  auto new_tensor_region = make_object<TensorRegionNode>(*tensor_region.operator->());
  new_tensor_region->region = Array<Range>(make_object<ArrayNode>());
  for (const auto& range : tensor_region->region) {
    new_tensor_region->region.push_back(Range::FromMinExtent(
        SubstituteInScope(range->min, var_map), SubstituteInScope(range->extent, var_map)));
  }
  return TensorRegion(new_tensor_region);
}

// Only Block and Loop are allowed here.
template <typename T>
Stmt GetStmtFromSeq(const T* op, const Stmt& target,
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
  Stmt s = GetRef<Stmt>(block_sref->stmt);
  CHECK(GetRef<Stmt>(block_sref->stmt).as<BlockNode>());
  const auto* parent = block_sref->parent;
  Stmt parent_stmt = GetRef<Stmt>(parent->stmt);

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

std::function<TensorRegion(const TensorRegion)> RelaxGenerator(
    const StmtSRef& block_sref, const StmtSRef& root,
    std::unordered_map<const VarNode*, PrimExpr>* vmap,
    std::unordered_map<const VarNode*, arith::IntSet>* dom_map) {
  const auto* block = block_sref->GetStmt<BlockNode>();
  const auto* block_realize = GetBlockRealize(block_sref).operator->();
  CHECK(block != nullptr);

  // Update block_var map
  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    (*vmap)[block->iter_vars[i]->var.get()] = block_realize->binding_values[i];
  }

  // Gather iteration domain
  auto sref = GetRef<StmtSRef>(block_sref->parent);
  while (sref.defined() && !sref.same_as(root)) {
    const auto* loop = sref->GetStmt<LoopNode>();
    // The root may not be a loop
    if (loop == nullptr) break;
    Range range = Range::FromMinExtent(loop->min, loop->extent);
    (*dom_map)[loop->loop_var.get()] = arith::IntSet::FromRange(range);
    sref = GetRef<StmtSRef>(sref->parent);
  }

  return [vmap, dom_map](const TensorRegion& tensor_region) {
    auto n = make_object<TensorRegionNode>();
    Array<Range> region;
    n->buffer = tensor_region->buffer;
    for (auto range : tensor_region->region) {
      range = Range::FromMinExtent(Substitute(range->min, *vmap), Substitute(range->extent, *vmap));
      auto int_set = arith::EvalSet(range, *dom_map);
      region.push_back(Range::FromMinExtent(int_set.min(), int_set.max() - int_set.min() + 1));
    }
    n->region = std::move(region);
    return TensorRegion(n);
  };
}

void RelaxRegion(const StmtSRef& block_sref, const StmtSRef& root, std::vector<TensorRegion>* reads,
                 std::vector<TensorRegion>* writes) {
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  std::unordered_map<const VarNode*, arith::IntSet> dom_map;
  auto relax = RelaxGenerator(block_sref, root, &vmap, &dom_map);
  const auto* block = block_sref->GetStmt<BlockNode>();
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

TensorRegion RelaxRegion(const StmtSRef& block_sref, const StmtSRef& root,
                         const TensorRegion& region) {
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  std::unordered_map<const VarNode*, arith::IntSet> dom_map;
  auto relax = RelaxGenerator(block_sref, root, &vmap, &dom_map);
  return relax(region);
}

/*!
 * \brief remove the AST leaf and its parent subtree which has only one leaf
 * \param sref The sref of Block/Loop to be removed
 * \param root The AST root
 * \return The orginal stmt and the removed stmt of the subtree rooted by the parent node
 */
std::pair<Stmt, Stmt> RemoveLeaf(StmtSRef sref, const StmtSRef& root) {
  CHECK(sref != root);

  // go upwards until find a father with more than two children
  Stmt last = GetRef<Stmt>(sref->stmt);
  sref = GetRef<StmtSRef>(sref->parent);
  Stmt stmt = GetRef<Stmt>(sref->stmt);
  while (!sref.same_as(root) || stmt.as<BlockNode>() == nullptr) {
    const auto* loop = stmt.as<LoopNode>();
    CHECK(loop != nullptr);
    const auto* seq = loop->body.as<SeqStmtNode>();
    if (seq != nullptr && seq->size() > 1) break;

    sref = GetRef<StmtSRef>(sref->parent);
    last = stmt;
    stmt = GetRef<Stmt>(sref->stmt);
  }

  auto get_body = [&last](const SeqStmtNode* seq) {
    CHECK_GT(seq->size(), 1);
    std::vector<Stmt> stmts;
    for (const auto& s : seq->seq) {
      const auto* ptr = s.as<BlockRealizeNode>();
      if (ptr != nullptr) {
        if (!ptr->block.same_as(last)) stmts.push_back(s);
      } else {
        if (!s.same_as(last)) stmts.push_back(s);
      }
    }
    return SeqStmt::Flatten(stmts);
  };

  if (const auto* block = stmt.as<BlockNode>()) {
    const auto* seq = block->body.as<SeqStmtNode>();
    CHECK(seq != nullptr);
    auto node = make_object<BlockNode>(*block);
    node->body = get_body(seq);
    return std::make_pair(stmt, Stmt(node));
  } else if (const auto* loop = stmt.as<LoopNode>()) {
    const auto* seq = loop->body.as<SeqStmtNode>();
    CHECK(seq != nullptr);
    auto node = make_object<LoopNode>(*loop);
    node->body = get_body(seq);
    return std::make_pair(stmt, Stmt(node));
  } else {
    LOG(FATAL) << "unknown stmt";
    return std::make_pair(Stmt(), Stmt());
  }
}

/*! \brief Wrapper function for VarRelatedDetector */
bool ExprContainsVar(const PrimExpr& expr, const Var& var) {
  bool result = false;
  const auto* expected = var.get();
  PostOrderVisit(expr, [&result, expected](const ObjectRef& obj) {
    if (obj.as<VarNode>() == expected) {
      result = true;
    }
  });
  return result;
}

void PatternMatcher::VisitExpr_(const VarNode* op) {
  auto it = filled_map_.find(op);
  if (it == filled_map_.end()) {
    filled_map_[op] = expr_to_match_;
  } else {
    ExprDeepEqual equal;
    if (it->second.same_as(expr_to_match_) || equal(it->second, expr_to_match_)) return;
    match_success_ = false;
  }
}

void PatternMatcher::VisitExpr_(const LoadNode* op) {
  const auto* ptr = expr_to_match_.as<LoadNode>();
  if (ptr == nullptr) {
    match_success_ = false;
  } else {
    if (!op->buffer_var.same_as(ptr->buffer_var)) {
      match_success_ = false;
    } else {
      PrimExpr tmp = expr_to_match_;
      expr_to_match_ = ptr->predicate;
      VisitExpr(op->predicate);
      expr_to_match_ = ptr->index;
      VisitExpr(op->index);
      std::swap(expr_to_match_, tmp);
    }
  }
}

void PatternMatcher::VisitExpr_(const LetNode* op) {
  const auto* ptr = expr_to_match_.as<LetNode>();
  if (ptr == nullptr) {
    match_success_ = false;
  } else {
    PrimExpr tmp = expr_to_match_;
    expr_to_match_ = ptr->var;
    VisitExpr(op->var);
    expr_to_match_ = ptr->value;
    VisitExpr(op->value);
    expr_to_match_ = ptr->body;
    VisitExpr(op->body);
    std::swap(expr_to_match_, tmp);
  }
}

#define TVM_DECLARE_PATTERN_MATCHER_BIN_OP(OpName)    \
  void PatternMatcher::VisitExpr_(const OpName* op) { \
    const auto* ptr = expr_to_match_.as<OpName>();    \
    if (ptr == nullptr) {                             \
      match_success_ = false;                         \
    } else {                                          \
      PrimExpr current = expr_to_match_;              \
      expr_to_match_ = ptr->a;                        \
      VisitExpr(op->a);                               \
      expr_to_match_ = ptr->b;                        \
      VisitExpr(op->b);                               \
      std::swap(expr_to_match_, current);             \
    }                                                 \
  }

TVM_DECLARE_PATTERN_MATCHER_BIN_OP(AddNode);
TVM_DECLARE_PATTERN_MATCHER_BIN_OP(SubNode);
TVM_DECLARE_PATTERN_MATCHER_BIN_OP(MulNode);
TVM_DECLARE_PATTERN_MATCHER_BIN_OP(DivNode);
TVM_DECLARE_PATTERN_MATCHER_BIN_OP(ModNode);
TVM_DECLARE_PATTERN_MATCHER_BIN_OP(EQNode);
TVM_DECLARE_PATTERN_MATCHER_BIN_OP(NENode);
TVM_DECLARE_PATTERN_MATCHER_BIN_OP(LTNode);
TVM_DECLARE_PATTERN_MATCHER_BIN_OP(LENode);
TVM_DECLARE_PATTERN_MATCHER_BIN_OP(GTNode);
TVM_DECLARE_PATTERN_MATCHER_BIN_OP(GENode);
TVM_DECLARE_PATTERN_MATCHER_BIN_OP(AndNode);
TVM_DECLARE_PATTERN_MATCHER_BIN_OP(OrNode);
TVM_DECLARE_PATTERN_MATCHER_BIN_OP(FloorDivNode);
TVM_DECLARE_PATTERN_MATCHER_BIN_OP(FloorModNode);
TVM_DECLARE_PATTERN_MATCHER_BIN_OP(MinNode);
TVM_DECLARE_PATTERN_MATCHER_BIN_OP(MaxNode);

void PatternMatcher::VisitExpr_(const CallNode* op) {
  const auto* ptr = expr_to_match_.as<CallNode>();
  if (ptr == nullptr) {
    match_success_ = false;
  } else {
    if (!op->op.same_as(ptr->op)) {
      match_success_ = false;
    } else {
      PrimExpr tmp = expr_to_match_;
      for (size_t i = 0; i < op->args.size(); ++i) {
        expr_to_match_ = ptr->args[i];
        VisitExpr(op->args[i]);
      }
      std::swap(expr_to_match_, tmp);
    }
  }
}

void PatternMatcher::VisitExpr_(const CastNode* op) {
  const auto* ptr = expr_to_match_.as<CastNode>();
  if (ptr == nullptr) {
    match_success_ = false;
  } else {
    if (!runtime::TypeEqual(op->dtype, ptr->dtype)) {
      match_success_ = false;
    } else {
      PrimExpr tmp = expr_to_match_;
      expr_to_match_ = ptr->value;
      VisitExpr(op->value);
      std::swap(expr_to_match_, tmp);
    }
  }
}

void PatternMatcher::VisitExpr_(const NotNode* op) {
  const auto* ptr = expr_to_match_.as<NotNode>();
  if (ptr == nullptr) {
    match_success_ = false;
  } else {
    PrimExpr tmp = expr_to_match_;
    expr_to_match_ = ptr->a;
    VisitExpr(op->a);
    std::swap(expr_to_match_, tmp);
  }
}

void PatternMatcher::VisitExpr_(const SelectNode* op) {
  const auto* ptr = expr_to_match_.as<SelectNode>();
  if (ptr == nullptr) {
    match_success_ = false;
  } else {
    PrimExpr tmp = expr_to_match_;
    expr_to_match_ = ptr->condition;
    VisitExpr(op->condition);
    expr_to_match_ = ptr->true_value;
    VisitExpr(op->true_value);
    expr_to_match_ = ptr->false_value;
    VisitExpr(op->false_value);
    std::swap(expr_to_match_, tmp);
  }
}

void PatternMatcher::VisitExpr_(const RampNode* op) {
  const auto* ptr = expr_to_match_.as<RampNode>();
  if (ptr == nullptr) {
    match_success_ = false;
  } else {
    if (op->lanes != ptr->lanes) {
      match_success_ = false;
    } else {
      PrimExpr tmp = expr_to_match_;
      expr_to_match_ = ptr->base;
      VisitExpr(op->base);
      expr_to_match_ = ptr->stride;
      VisitExpr(op->stride);
      std::swap(expr_to_match_, tmp);
    }
  }
}

void PatternMatcher::VisitExpr_(const BroadcastNode* op) {
  const auto* ptr = expr_to_match_.as<RampNode>();  // TODO(@junrushao1994): i dont understand
  if (ptr == nullptr) {
    match_success_ = false;
  } else {
    if (op->lanes != ptr->lanes) {
      match_success_ = false;
    } else {
      PrimExpr tmp = expr_to_match_;
      expr_to_match_ = ptr->base;
      VisitExpr(op->value);
      std::swap(expr_to_match_, tmp);
    }
  }
}

void PatternMatcher::VisitExpr_(const ShuffleNode* op) {
  const auto* ptr = expr_to_match_.as<ShuffleNode>();
  if (ptr == nullptr) {
    match_success_ = false;
  } else {
    if (op->vectors.size() != ptr->vectors.size() || op->indices.size() != ptr->indices.size()) {
      match_success_ = false;
    } else {
      PrimExpr tmp = expr_to_match_;
      for (size_t i = 0; i < op->indices.size(); ++i) {
        expr_to_match_ = ptr->indices[i];
        VisitExpr(op->indices[i]);
      }
      for (size_t i = 0; i < op->vectors.size(); ++i) {
        expr_to_match_ = ptr->vectors[i];
        VisitExpr(op->vectors[i]);
      }
      std::swap(expr_to_match_, tmp);
    }
  }
}

void PatternMatcher::VisitExpr_(const IntImmNode* op) {
  const auto* ptr = expr_to_match_.as<IntImmNode>();
  match_success_ = ptr != nullptr && op->value == ptr->value;
}

void PatternMatcher::VisitExpr_(const FloatImmNode* op) {
  const auto* ptr = expr_to_match_.as<FloatImmNode>();
  match_success_ = ptr != nullptr && op->value == ptr->value;
}

void PatternMatcher::VisitExpr_(const StringImmNode* op) {
  const auto* ptr = expr_to_match_.as<StringImmNode>();
  match_success_ = ptr != nullptr && op->value == ptr->value;
}

void PatternMatcher::VisitExpr_(const BufferLoadNode* op) {
  const auto* ptr = expr_to_match_.as<BufferLoadNode>();
  if (ptr == nullptr) {
    match_success_ = false;
  } else {
    if (!op->buffer.same_as(ptr->buffer) || op->indices.size() != ptr->indices.size()) {
      match_success_ = false;
    } else {
      PrimExpr tmp = expr_to_match_;
      for (size_t i = 0; i < op->indices.size(); ++i) {
        expr_to_match_ = ptr->indices[i];
        VisitExpr(op->indices[i]);
      }
      std::swap(expr_to_match_, tmp);
    }
  }
}

}  // namespace tir
}  // namespace tvm
