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
#include "../../../arith/pattern_match.h"
#include "../../ir/functor_common.h"
#include "../utils.h"

namespace tvm {
namespace tir {

Block GetRootBlock(StmtSRef sref) {
  const StmtSRefNode* p_sref = sref.get();
  while (p_sref->parent != nullptr) {
    p_sref = p_sref->parent;
  }
  const BlockNode* root_block = TVM_SREF_TO_BLOCK(root_block, GetRef<StmtSRef>(p_sref));
  return GetRef<Block>(root_block);
}

bool TensorizeComparator::VisitStmt(const Stmt& n, const Stmt& other) {
  if (n.same_as(other)) return true;
  if (n->type_index() != other->type_index()) return false;
  bool equal = StmtComparator::VisitStmt(n, other);
  if (!equal && assert_mode_)
    LOG(FATAL) << "Stmts are not matching between:\n" << n << "\nand\n" << other;
  return equal;
}

bool TensorizeComparator::VisitStmt_(const ForNode* op, const Stmt& other) {
  const auto* rhs = other.as<ForNode>();
  if (!DefEqual(op->loop_var, rhs->loop_var)) return false;
  if (!VisitExpr(op->min, rhs->min)) return false;
  if (!VisitExpr(op->extent, rhs->extent)) return false;
  if (!VisitStmt(op->body, rhs->body)) return false;
  if (op->kind != rhs->kind) return false;
  if (op->thread_binding.defined() ^ rhs->thread_binding.defined()) return false;
  if (op->thread_binding.defined() &&
      !VisitExpr(op->thread_binding.value(), rhs->thread_binding.value()))
    return false;
  return CompareAnnotationMap(op->annotations, rhs->annotations);
}

bool TensorizeComparator::VisitStmt_(const SeqStmtNode* op, const Stmt& other) {
  const auto* rhs = other.as<SeqStmtNode>();
  return CompareArray(op->seq, rhs->seq, &TensorizeComparator::VisitStmt);
}

bool TensorizeComparator::VisitStmt_(const BufferStoreNode* op, const Stmt& other) {
  const auto* rhs = other.as<BufferStoreNode>();
  return CompareBufferAccess(op, rhs) && VisitExpr(op->value, rhs->value);
}

bool TensorizeComparator::VisitStmt_(const BlockRealizeNode* op, const Stmt& other) {
  const auto* rhs = other.as<BlockRealizeNode>();
  // Skip Compare binding values if the block is scope block (the outermost one).
  if (!is_scope_block) {
    size_t offset = op->iter_values.size() - rhs->iter_values.size();
    if (rhs->iter_values.size() > op->iter_values.size()) return false;
    if (is_inner_block) {
      // weak pattern matching for the inner block (the son of the scope block)
      // where the pattern is v + iter <=> expr + iter
      for (size_t i = 0; i < rhs->iter_values.size(); ++i) {
        PrimExpr lhs_expr, rhs_expr;
        Optional<Var> lhs_iter, rhs_iter;
        auto detect = [](const PrimExpr& binding) -> std::pair<PrimExpr, Optional<Var>> {
          arith::PVar<PrimExpr> expr;
          arith::PVar<Var> iter;
          if (iter.Match(binding)) {
            return std::make_pair(0, iter.Eval());
          } else if ((expr + iter).Match(binding)) {
            return std::make_pair(expr.Eval(), iter.Eval());
          } else if ((iter + expr).Match(binding)) {
            return std::make_pair(expr.Eval(), iter.Eval());
          } else {
            return std::make_pair(expr.Eval(), NullOpt);
          }
        };
        std::tie(lhs_expr, lhs_iter) = detect(op->iter_values[i + offset]);
        std::tie(rhs_expr, rhs_iter) = detect(rhs->iter_values[i]);
        CHECK((lhs_iter && rhs_iter) || (!lhs_iter && !rhs_iter)) << "Incompatible binding";
        if (lhs_iter) VisitExpr(lhs_iter.value(), rhs_iter.value());
        if (is_zero(rhs_expr)) {
          CHECK(is_zero(lhs_expr)) << "Incompatible binding";
        } else {
          const auto* bv = rhs_expr.as<VarNode>();
          if (!bv) {
            VisitExpr(lhs_expr, rhs_expr);
          } else {
            auto it = equal_map_.find(GetRef<Var>(bv));
            if (it == equal_map_.end()) {
              equal_map_[GetRef<Var>(bv)] = lhs_expr;
            } else {
              CHECK(it->second->IsInstance<PrimExprNode>());
              VisitExpr(lhs_expr, Downcast<PrimExpr>(it->second));
            }
          }
        }
      }
    } else {
      for (size_t i = 0; i < rhs->iter_values.size(); ++i) {
        if (!VisitExpr(op->iter_values[i + offset], rhs->iter_values[i])) return false;
      }
      const Block& block = op->block;
      for (size_t i = 0; i < offset; ++i) {
        Var block_var = Downcast<Var>(op->iter_values[i]);
        auto it = equal_map_.find(block_var);
        equal_map_[block->iter_vars[i]->var] = (it == equal_map_.end() ? block_var : it->second);
      }
    }
  }

  return VisitExpr(op->predicate, rhs->predicate) && VisitStmt(op->block, rhs->block);
}

bool TensorizeComparator::VisitStmt_(const BlockNode* op, const Stmt& other) {
  const auto* rhs = other.as<BlockNode>();
  // Check block equal
  // All iter var and buffer region should matches including the order

  // Check iterVar
  // need to use DefEqual to remap vars
  // Note:
  //    We only compare the inner most several axis
  if (op->iter_vars.size() < rhs->iter_vars.size()) return false;

  size_t offset = op->iter_vars.size() - rhs->iter_vars.size();
  for (size_t i = 0; i < rhs->iter_vars.size(); ++i) {
    auto lhs_var = op->iter_vars[i + offset], rhs_var = rhs->iter_vars[i];
    // Skip iter dom
    if (!DefEqual(lhs_var->var, rhs_var->var)) return false;
    if (lhs_var->iter_type != rhs_var->iter_type) return false;
  }

  for (size_t i = 0; i < offset; ++i) {
    if (is_scope_block) {
      extra_block_vars_.push_back(op->iter_vars[i]);
    }
  }

  if (!is_scope_block) {
    if (!CompareArray(op->writes, rhs->writes, &TensorizeComparator::CompareBufferRegion)) {
      return false;
    }
    if (!CompareArray(op->reads, rhs->reads, &TensorizeComparator::CompareBufferRegion)) {
      return false;
    }
    if (!CompareAnnotationMap(op->annotations, rhs->annotations)) {
      return false;
    }
    if (!CompareArray(op->alloc_buffers, rhs->alloc_buffers, &TensorizeComparator::CompareBuffer)) {
      return false;
    }
  }
  if (!is_scope_block) is_inner_block = false;
  is_scope_block = false;
  return VisitStmt(op->body, rhs->body);
}

// Exprs
#define TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(OpName)                            \
  bool TensorizeComparator::VisitExpr_(const OpName* op, const PrimExpr& other) { \
    const auto* rhs = other.as<OpName>();                                         \
    return VisitExpr(op->a, rhs->a) && VisitExpr(op->b, rhs->b);                  \
  }

TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(AddNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(SubNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(MulNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(DivNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(ModNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(EQNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(NENode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(LTNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(LENode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(GTNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(GENode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(AndNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(OrNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(MinNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(MaxNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(FloorDivNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(FloorModNode);

bool TensorizeComparator::VisitExpr_(const IntImmNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<IntImmNode>();
  return CompareType(op->dtype, rhs->dtype) && op->value == rhs->value;
}

bool TensorizeComparator::VisitExpr_(const FloatImmNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<FloatImmNode>();
  return CompareType(op->dtype, rhs->dtype) && op->value == rhs->value;
}

bool TensorizeComparator::VisitExpr_(const CastNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<CastNode>();
  return CompareType(op->dtype, rhs->dtype) && VisitExpr(op->value, rhs->value);
}

bool TensorizeComparator::VisitExpr_(const VarNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<VarNode>();
  auto lhs = GetRef<Var>(op);
  if (lhs.same_as(other)) return true;
  if (!CompareType(op->dtype, rhs->dtype)) return false;
  auto it = equal_map_.find(lhs);
  return it != equal_map_.end() && it->second.same_as(other);
}

bool TensorizeComparator::VisitExpr_(const BufferLoadNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<BufferLoadNode>();
  return CompareBufferAccess(op, rhs);
}

bool TensorizeComparator::DefEqual(const ObjectRef& lhs, const ObjectRef& rhs) {
  if (lhs.same_as(rhs)) return true;
  if (lhs->type_index() != rhs->type_index()) return false;
  auto it = equal_map_.find(lhs);
  // If there is already a mapping
  if (it != equal_map_.end()) return it->second.same_as(rhs);
  equal_map_[lhs] = rhs;
  return true;
}

bool TensorizeComparator::CompareAnnotation(const std::pair<String, ObjectRef>& lhs,
                                            const std::pair<String, ObjectRef>& rhs) {
  if (lhs.first != rhs.first) return false;
  if (!lhs.second.same_as(rhs.second)) return false;
  return VisitExpr(Downcast<PrimExpr>(lhs.second), Downcast<PrimExpr>(rhs.second));
}

bool TensorizeComparator::CompareAnnotationMap(const Map<String, ObjectRef>& lhs,
                                               const Map<String, ObjectRef>& rhs) {
  if (lhs.same_as(rhs)) return true;
  if (lhs.size() != rhs.size()) return false;

  auto sort_map =
      [](const Map<String, ObjectRef>& map) -> std::vector<std::pair<String, ObjectRef>> {
    std::vector<std::pair<String, ObjectRef>> ret;
    ret.reserve(map.size());
    for (const auto& pair : map) {
      ret.emplace_back(pair);
    }
    sort(ret.begin(), ret.end());
    return ret;
  };

  auto lhs_array = sort_map(lhs), rhs_array = sort_map(rhs);

  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!CompareAnnotation(lhs_array[i], rhs_array[i])) return false;
  }
  return true;
}

bool TensorizeComparator::CompareBuffer(const Buffer& lhs, const Buffer& rhs) {
  if (lhs.same_as(rhs)) return true;
  // Remap both buffer itself and buffer data
  // Skip buffer shape
  bool equal = DefEqual(lhs, rhs) && DefEqual(lhs->data, rhs->data) &&
               CompareType(lhs->dtype, rhs->dtype) && lhs.scope() == rhs.scope();
  if (equal) {
    rhs_buffer_map_[rhs] = lhs;
  } else if (assert_mode_) {
    LOG(FATAL) << "Buffers are not matching between:" << lhs << " and " << rhs;
  }
  return equal;
}

bool TensorizeComparator::CompareBufferRegion(const BufferRegion& lhs, const BufferRegion& rhs) {
  // Only for block region declaration
  if (!CompareBuffer(lhs->buffer, rhs->buffer)) return false;
  // Number of indices in desc_block must be smaller than it in AST
  if (rhs->region.size() > lhs->region.size()) return false;

  std::vector<Range> lhs_region;
  for (const auto& range : lhs->region) {
    lhs_region.push_back(Range::FromMinExtent(range->min, range->extent));
  }
  // special judge size 1 buffer
  if (rhs->region.size() == 1 && is_zero(rhs->region[0]->min) && is_one(rhs->region[0]->extent)) {
    lhs_region.push_back(Range::FromMinExtent(0, 1));
  }
  size_t offset = lhs_region.size() - rhs->region.size();
  // initialize buffer indices
  bool need_update = false;
  if (!buffer_indices_.count(lhs->buffer)) {
    need_update = true;
    buffer_indices_[lhs->buffer] = std::vector<PrimExpr>();
  } else {
    if (offset != buffer_indices_[lhs->buffer].size()) return false;
  }
  std::vector<PrimExpr>& indices = buffer_indices_[lhs->buffer];
  for (size_t i = 0; i < offset; ++i) {
    const Range& range = lhs_region[i];
    // High-dim region must be element-wise
    if (!is_one(range->extent)) return false;
    if (need_update) {
      indices.push_back(range->min);
    } else {
      // The order matters since we only map inner block_var to outside block_var
      if (!VisitExpr(range->min, indices[i])) return false;
    }
  }
  for (size_t i = 0; i < rhs->region.size(); ++i) {
    if (!CompareRange(lhs_region[i + offset], rhs->region[i])) return false;
  }
  return true;
}

// Only for BufferStoreNode and BufferLoadNode
template <typename T>
bool TensorizeComparator::CompareBufferAccess(const T* lhs, const T* rhs) {
  if (!CompareBuffer(lhs->buffer, rhs->buffer)) return false;

  if (rhs->indices.size() > lhs->indices.size()) return false;
  // special judge size 1 buffer
  if (rhs->indices.size() == 1 && is_zero(rhs->indices[0])) return true;
  // otherwise
  size_t offset = lhs->indices.size() - rhs->indices.size();
  for (size_t i = 0; i < rhs->indices.size(); ++i) {
    if (!VisitExpr(lhs->indices[i + offset], rhs->indices[i])) return false;
  }
  return true;
}

template <typename T, typename F>
bool TensorizeComparator::CompareArray(const Array<T>& lhs, const Array<T>& rhs, F cmp) {
  if (lhs.same_as(rhs)) return true;
  if (lhs.size() != rhs.size()) return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!(this->*cmp)(lhs[i], rhs[i])) return false;
  }
  return true;
}

bool TensorizeComparator::CompareRange(const Range& lhs, const Range& rhs) {
  return VisitExpr(lhs->min, rhs->min) && VisitExpr(lhs->extent, rhs->extent);
}

bool TensorizeComparator::CompareType(const DataType& lhs, const DataType& rhs) {
  if (lhs == rhs) return true;
  return lhs.code() == rhs.code() && lhs.bits() == rhs.bits() && lhs.lanes() == rhs.lanes();
}

// Deep comparison to check if two IR graph are equivalent
bool TensorizeComparator::VisitExpr(const PrimExpr& n, const PrimExpr& other) {
  bool equal = (n->type_index() == other->type_index()) && ExprComparator::VisitExpr(n, other);
  if (!equal && assert_mode_)
    LOG(FATAL) << "Exprs are not matching between:" << n << " and " << other;
  return equal;
}

Array<Array<arith::IterMark>> TrivialSubspaceDivision(const Array<IterVar>& iter_vars,
                                                      const Array<PrimExpr>& bindings,
                                                      const std::vector<Var>& outer_loops,
                                                      const std::vector<Var>& inner_loops,
                                                      const PrimExpr& predicate) {
  if (!is_one(predicate)) return {};
  std::vector<Array<arith::IterMark>> res;
  std::unordered_set<const VarNode*> outer_loop_vars;
  std::unordered_set<const VarNode*> inner_loop_vars;
  for (const Var& var : outer_loops) {
    outer_loop_vars.insert(var.get());
  }
  for (const Var& var : inner_loops) {
    inner_loop_vars.insert(var.get());
  }
  for (size_t i = 0; i < bindings.size(); ++i) {
    bool outer = UsesVar(
        bindings[i], [&outer_loop_vars](const VarNode* var) { return outer_loop_vars.count(var); });
    bool inner = UsesVar(
        bindings[i], [&inner_loop_vars](const VarNode* var) { return inner_loop_vars.count(var); });
    bool is_var = bindings[i]->IsInstance<VarNode>();
    if (outer && !inner) {
      arith::IterMark outer{nullptr};
      if (is_var) {
        outer = arith::IterMark(
            arith::IterSplitExpr(arith::IterMark(bindings[i], iter_vars[i]->dom->extent)),
            iter_vars[i]->dom->extent);
      } else {
        outer = arith::IterMark(arith::IterSumExpr({}, bindings[i]), iter_vars[i]->dom->extent);
      }
      arith::IterMark inner(arith::IterSumExpr({}, 0), 1);
      res.push_back(Array<arith::IterMark>({outer, inner}));
    } else if (inner && !outer) {
      arith::IterMark inner{nullptr};
      if (is_var) {
        inner = arith::IterMark(
            arith::IterSplitExpr(arith::IterMark(bindings[i], iter_vars[i]->dom->extent)),
            iter_vars[i]->dom->extent);
      } else {
        inner = arith::IterMark(arith::IterSumExpr({}, bindings[i]), iter_vars[i]->dom->extent);
      }
      arith::IterMark outer(arith::IterSumExpr({}, 0), 1);
      res.push_back(Array<arith::IterMark>({outer, inner}));
    } else if (!outer && !inner) {
      arith::IterMark outer(arith::IterSumExpr({}, 0), 1);
      arith::IterMark inner(arith::IterSumExpr({}, 0), 1);
      res.push_back(Array<arith::IterMark>({outer, inner}));
    } else {
      return {};
    }
  }
  res.push_back({arith::IterMark(arith::IterSumExpr({}, 0), Bool(true)),
                 arith::IterMark(arith::IterSumExpr({}, 0), Bool(true))});
  return res;
}

StmtSRef Blockize(ScheduleState self, const StmtSRef& loop_sref) {
  /*!
   * Check:
   *   - The sub AST is one-line with only one block
   *
   * Mutate:
   *   - extra block var from the only block
   *   - Update block binding
   */
  const auto* loop = loop_sref->StmtAs<ForNode>();
  CHECK(loop) << "TypeError: Only support blockize a loop for now, but get type: "
              << loop_sref->stmt->GetTypeKey();
  // check there exists no SeqStmt under loop
  CHECK(CheckOneLine(GetRef<Stmt>(loop))) << "ValueError: Only one line subtree can be blockize";
  // get the inner Block, BlockRealize and StmtSRef
  Array<StmtSRef> child_blocks = GetChildBlocks(self, loop_sref, false);
  CHECK_EQ(child_blocks.size(), 1) << "ValueError: Only one line subtree can be blockize";
  StmtSRef block_sref = child_blocks[0];
  BlockRealize block_realize = GetBlockRealize(self, block_sref);
  Block block = block_realize->block;
  // collect loops inside/outside loop_sref
  std::vector<const ForNode*> outer_loops, inner_loops;
  std::vector<Var> outer_iters, inner_iters;
  std::unordered_map<Var, Range, ObjectPtrHash, ObjectPtrEqual> iters;
  bool inner = true;
  for (StmtSRef current_sref = block_sref;;) {
    current_sref = GetRef<StmtSRef>(current_sref->parent);
    if (!current_sref.defined()) break;
    const auto* current_loop = current_sref->StmtAs<ForNode>();
    if (!current_loop) break;
    if (inner) {
      inner_loops.push_back(current_loop);
      inner_iters.push_back(current_loop->loop_var);
    } else {
      outer_loops.push_back(current_loop);
      outer_iters.push_back(current_loop->loop_var);
    }
    iters[current_loop->loop_var] = Range::FromMinExtent(current_loop->min, current_loop->extent);
    if (current_sref == loop_sref) inner = false;
  }
  arith::Analyzer analyzer;
  Array<Array<arith::IterMark>> division = arith::SubspaceDivide(
      block_realize->iter_values, iters, inner_iters, block_realize->predicate, false, &analyzer);
  if (division.empty()) {
    // It is possible to blockize if we can not do perfect subspace division if we can divide
    // the block var bindings into two categories
    // 1. The binding covers no inner loop var
    // 2. The binding covers only inner loop vars
    division = TrivialSubspaceDivision(block->iter_vars, block_realize->iter_values, outer_iters,
                                       inner_iters, block_realize->predicate);
  }
  CHECK(!division.empty()) << "ValueError: The bindings of the block below can not be blockized";
  // Generate a new inner block
  Array<IterVar> inner_block_vars, outer_block_vars;
  Array<PrimExpr> inner_bindings, outer_bindings;
  std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> block_var_no;
  std::unordered_map<Var, Range, ObjectPtrHash, ObjectPtrEqual> bv_iters;
  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    const arith::IterMapExprNode* outer_binding =
        division[i][0]->source.as<arith::IterMapExprNode>();
    const arith::IterMapExprNode* inner_binding =
        division[i][1]->source.as<arith::IterMapExprNode>();
    ICHECK(outer_binding);
    ICHECK(inner_binding);
    if (is_one(division[i][1]->extent)) {  // IsOuter
      // extract this iter var to outer block directly
      outer_bindings.push_back(
          arith::NormalizeIterMapToExpr(GetRef<arith::IterMapExpr>(outer_binding)));
      outer_block_vars.push_back(iter_var);
      bv_iters[iter_var->var] = Range::FromMinExtent(iter_var->var, 1);
    } else {
      const IterVar outer_var(Range::FromMinExtent(0, division[i][0]->extent),
                              iter_var->var.copy_with_suffix("o"), iter_var->iter_type);
      outer_bindings.push_back(
          arith::NormalizeIterMapToExpr(GetRef<arith::IterMapExpr>(outer_binding)));
      outer_block_vars.push_back(outer_var);
      // generate a new iter var for outer block
      PrimExpr base = is_one(division[i][0]->extent) ? 0 : outer_var * division[i][1]->extent;
      if (const auto* op = division[i][1]->source.as<arith::IterSumExprNode>()) {
        base = base + op->base;
        inner_bindings.push_back(base +
                                 arith::NormalizeIterMapToExpr(arith::IterSumExpr(op->args, 0)));
      } else {
        inner_bindings.push_back(
            base + arith::NormalizeIterMapToExpr(GetRef<arith::IterMapExpr>(inner_binding)));
      }
      inner_block_vars.push_back(iter_var);
      bv_iters[iter_var->var] = Range::FromMinExtent(base, division[i][1]->extent);
    }
    block_var_no[iter_var->var] = i;
  }
  Block inner_block = block;
  inner_block.CopyOnWrite()->iter_vars = inner_block_vars;
  inner_block.CopyOnWrite()->init = NullOpt;
  BlockRealize inner_br = block_realize;
  inner_br.CopyOnWrite()->iter_values = inner_bindings;
  inner_br.CopyOnWrite()->predicate = division.back()[1]->extent;
  inner_br.CopyOnWrite()->block = inner_block;
  // Regenerate inner_loops
  Stmt body = inner_br;
  for (const auto& inner_loop : inner_loops) {
    auto loop_node = make_object<ForNode>(*inner_loop);
    loop_node->body = body;
    body = For(loop_node);
  }
  // Regenerate init for outer block
  Optional<Stmt> new_init = NullOpt;
  if (block->init.defined()) {
    std::vector<For> init_loops;
    std::vector<size_t> init_block_vars;
    std::vector<IterVar> init_block_vars_copy;
    std::vector<PrimExpr> init_bindings;
    std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> binding_replace_map;
    std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> bv_replace_map;
    std::unordered_map<const IterVarNode*, int> new_block_vars2old_index;
    for (size_t i = 0; i < inner_block_vars.size(); ++i) {
      if (inner_block_vars[i]->iter_type == IterVarType::kDataPar &&
          UsesVar(block->init.value(),
                  [v = inner_block_vars[i]->var](const VarNode* var) { return var == v.get(); })) {
        // copy init block vars and ignore reduce block vars
        init_block_vars.push_back(i);
        IterVar init_block_var = inner_block_vars[i];
        init_block_var.CopyOnWrite()->var = inner_block_vars[i]->var.copy_with_suffix("_init");
        init_block_vars_copy.push_back(init_block_var);
        bv_replace_map[inner_block_vars[i]->var] = init_block_var->var;
        new_block_vars2old_index[init_block_var.get()] = i;
      }
    }
    for (const ForNode* inner_loop : inner_loops) {
      for (size_t i = 0; i < init_block_vars.size(); ++i) {
        if (UsesVar(inner_bindings[new_block_vars2old_index[init_block_vars_copy[i].get()]],
                    [v = inner_loop->loop_var](const VarNode* var) { return var == v.get(); })) {
          // copy loops related to init block vars
          For init_loop = GetRef<For>(inner_loop);
          init_loop.CopyOnWrite()->loop_var = inner_loop->loop_var.copy_with_suffix("");
          // replace loop vars with copied loop vars
          binding_replace_map[inner_loop->loop_var] = init_loop->loop_var;
          init_loops.push_back(init_loop);
          break;
        }
      }
    }
    for (size_t i = 0; i < init_block_vars.size(); ++i) {
      init_bindings.push_back(Substitute(inner_bindings[init_block_vars[i]], binding_replace_map));
    }
    new_init = Substitute(Block(/*iter_vars=*/init_block_vars_copy,        //
                                /*reads=*/{},                              //
                                /*writes=*/block->writes,                  //
                                /*name_hint=*/block->name_hint + "_init",  //
                                /*body=*/block->init.value(),              //
                                /*init=*/NullOpt),
                          bv_replace_map);
    new_init =
        BlockRealize(init_bindings, division.back()[1]->extent, Downcast<Block>(new_init.value()));
    for (const auto& init_loop : init_loops) {
      For new_init_loop = init_loop;
      new_init_loop.CopyOnWrite()->body = new_init.value();
      new_init = new_init_loop;
    }
  }
  // Calculate outer block's IO region
  auto rewrite_range = [&](const Range& range) -> Range {
    const Array<arith::IterSumExpr>& res =
        arith::DetectIterMap({range->min}, bv_iters, true, false, &analyzer);
    ICHECK_EQ(res.size(), 1);
    const arith::IterSumExpr& normalized_expr = res[0];
    PrimExpr extent = 1;
    if (normalized_expr->args.size() == 1) {
      CHECK(analyzer.CanProve(normalized_expr->args[0]->scale - range->extent == 0));
      extent = normalized_expr->args[0]->extent;
    }
    return Range::FromMinExtent(normalized_expr->base, extent * range->extent);
  };
  std::vector<BufferRegion> reads, writes;
  auto rewrite_region = [&](std::vector<BufferRegion>* regions, Array<BufferRegion> old_regions) {
    for (auto buffer_region : old_regions) {
      std::vector<Range> region;
      for (const auto& range : buffer_region->region) {
        region.push_back(rewrite_range(range));
      }
      (*regions).emplace_back(buffer_region->buffer, region);
    }
  };
  rewrite_region(&reads, block->reads);
  rewrite_region(&writes, block->writes);
  // Generate a new outer block
  auto outer_block = Block(/*iter_vars=*/outer_block_vars,                 //
                           /*reads=*/reads,                                //
                           /*writes=*/writes,                              //
                           /*name_hint=*/"blockized_" + block->name_hint,  //
                           /*body=*/std::move(body),                       //
                           /*init=*/new_init);
  auto outer_realize = BlockRealize(outer_bindings, division.back()[0]->extent, outer_block);

  self->Replace(loop_sref, outer_realize, {{block, inner_block}});
  {
    StmtSRef block_sref = self->stmt2ref.at(inner_block.get());
    UpdateAffineFlag(self, block_sref);
  }
  {
    StmtSRef block_sref = self->stmt2ref.at(outer_block.get());
    StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false,
                                       /*require_compact_dataflow*/false);
    UpdateScope(self, scope_sref);
    UpdateAffineFlag(self, scope_sref);
  }
  {
    StmtSRef block_sref = self->stmt2ref.at(outer_block.get());
    UpdateScope(self, block_sref);
    UpdateAffineFlag(self, block_sref);
  }

  // Check loop binding

  {
    struct BindingValidator : public StmtVisitor {
      void VisitStmt_(const BlockRealizeNode* realize) final {
        StmtSRef& sref = self->stmt2ref.at(realize->block.get());
        UpdateAffineFlag(self, sref);
        VisitStmt(realize->block->body);
      }
      ScheduleState self;
    };
    BindingValidator validator;
    validator.self = self;
    const PrimFuncNode* func = GetRootPrimFunc(self->mod, GetRootBlock(loop_sref).get(), nullptr);
    validator(func->body);
  }
  return self->stmt2ref.at(outer_block.get());
}

// Stmts

void BufferRemap(const TensorIntrin& intrinsic,
                 std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>* buffer_map) {
  ICHECK_EQ(intrinsic->description->params.size(), intrinsic->implementation->params.size());
  for (size_t i = 0; i < intrinsic->description->params.size(); ++i) {
    const auto& lhs_var = intrinsic->description->params[i];
    const auto& lhs_buffer = intrinsic->description->buffer_map[lhs_var];
    const auto& rhs_var = intrinsic->implementation->params[i];
    const auto& rhs_buffer = intrinsic->implementation->buffer_map[rhs_var];
    (*buffer_map)[rhs_buffer] = lhs_buffer;
  }
}

// Replace buffer with its data, element_offset
class BufferReplacer : public StmtExprMutator {
 public:
  explicit BufferReplacer(
      const std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>& buffer_map,
      const std::unordered_map<const VarNode*, const PrimExprNode*>& var_map,
      std::vector<IterVar>&& extra_block_vars,
      const std::unordered_map<Buffer, std::vector<PrimExpr>, ObjectPtrHash, ObjectPtrEqual>&
          buffer_indices)
      : buffer_map_(buffer_map),
        var_map_(var_map),
        extra_block_vars_(std::move(extra_block_vars)),
        buffer_indices_(buffer_indices) {}

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto s = StmtExprMutator::VisitStmt_(op);
    op = s.as<BufferStoreNode>();
    CHECK(op);
    auto it = buffer_map_.find(op->buffer);
    if (it != buffer_map_.end()) {
      auto n = CopyOnWrite(op);
      n->buffer = it->second;
      auto it2 = buffer_indices_.find(n->buffer);
      CHECK(it2 != buffer_indices_.end());
      n->indices.insert(n->indices.begin(), it2->second.begin(), it2->second.end());
      return Stmt(n);
    } else {
      return GetRef<Stmt>(op);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto s = StmtExprMutator::VisitExpr_(op);
    op = s.as<BufferLoadNode>();
    CHECK(op);
    auto it = buffer_map_.find(op->buffer);
    if (it != buffer_map_.end()) {
      auto n = make_object<BufferLoadNode>(*op);
      n->buffer = it->second;
      auto it2 = buffer_indices_.find(n->buffer);
      CHECK(it2 != buffer_indices_.end());
      n->indices.insert(n->indices.begin(), it2->second.begin(), it2->second.end());
      return PrimExpr(n);
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = var_map_.find(op);
    if (it != var_map_.end()) {
      return GetRef<PrimExpr>(it->second);
    } else {
      auto it2 = block_var_map_.find(op);
      if (it2 != block_var_map_.find(op)) {
        return GetRef<PrimExpr>(it2->second);
      } else {
        return GetRef<PrimExpr>(op);
      }
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    std::vector<IterVar> extra_block_var;
    std::unordered_map<const VarNode*, const PrimExprNode*> block_var_map;
    for (const auto& iter_var : extra_block_vars_) {
      auto n = runtime::make_object<IterVarNode>(*(iter_var.get()));
      IterVar block_var(n);
      extra_block_var.push_back(block_var);
      block_var_map[iter_var->var.get()] = block_var->var.get();
    }
    std::swap(block_var_map, block_var_map_);
    auto s = StmtExprMutator::VisitStmt_(op);
    op = s.as<BlockNode>();
    CHECK(op);

    auto iter_vars = op->iter_vars;
    iter_vars.insert(iter_vars.begin(), extra_block_var.begin(), extra_block_var.end());
    auto reads = UpdateBufferViaMap(op->reads);
    auto writes = UpdateBufferViaMap(op->writes);

    std::swap(block_var_map, block_var_map_);

    if (reads.same_as(op->reads) && writes.same_as(op->writes) &&
        iter_vars.same_as(op->iter_vars)) {
      return GetRef<Block>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      n->iter_vars = std::move(iter_vars);
      return Block(n);
    }
  }

 private:
  const std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>& buffer_map_;
  const std::unordered_map<const VarNode*, const PrimExprNode*>& var_map_;
  std::unordered_map<const VarNode*, const PrimExprNode*> block_var_map_;
  const std::vector<IterVar>& extra_block_vars_;
  const std::unordered_map<Buffer, std::vector<PrimExpr>, ObjectPtrHash, ObjectPtrEqual>&
      buffer_indices_;

  Array<BufferRegion> UpdateBufferViaMap(const Array<BufferRegion>& buffer_regions) {
    auto f_mutate = [this](const BufferRegion& buffer_region) {
      auto it = buffer_map_.find(buffer_region->buffer);
      if (it != buffer_map_.end()) {
        auto n = make_object<BufferRegionNode>(*buffer_region.get());
        n->buffer = it->second;
        auto it2 = buffer_indices_.find(n->buffer);
        if (it2 != buffer_indices_.end()) {
          Region region;
          for (const auto& min : it2->second) {
            region.push_back(Range::FromMinExtent(VisitExpr(min), 1));
          }
          n->region.insert(n->region.begin(), region.begin(), region.end());
        }
        while (n->region.size() > n->buffer->shape.size()) {
          const Range& range = n->region.back();
          ICHECK(is_one(range->extent) && is_zero(range->min));
          n->region.pop_back();
        }
        return BufferRegion(n);
      } else {
        return buffer_region;
      }
    };
    return MutateArray(buffer_regions, f_mutate);
  }
};

void Tensorize(ScheduleState self, const StmtSRef& loop_sref, const TensorIntrin& intrinsic) {
  /*!
   * Check:
   *   - Check buffer binding, including type, alignment, shape and etc.
   *   - Check the sub AST is equal to the description function.
   *
   * Mutate:
   *   - Blockize the sub AST (please refer blockize for details)
   *   - Bind buffers
   *   - Mutate implement function with buffer binding
   *   - Replace the sub tree with the mutated function.
   */
  const auto* loop = loop_sref->StmtAs<ForNode>();
  CHECK(loop) << "Only support tensorize a loop for now";

  const auto* desc_block_realize =
      Downcast<BlockRealize>(intrinsic->description->body)->block->body.as<BlockRealizeNode>();
  const Block& desc_block = desc_block_realize->block;
  const auto* impl_block_realize =
      Downcast<BlockRealize>(intrinsic->implementation->body)->block->body.as<BlockRealizeNode>();
  Block impl_block = impl_block_realize->block;

  const StmtSRef& block_sref = Blockize(self, loop_sref);
  const BlockRealize& block_realize = GetBlockRealize(self, block_sref);

  TensorizeComparator comparator;
  bool equal = comparator.VisitStmt(block_realize, GetRef<Stmt>(desc_block_realize));
  CHECK(equal) << "The AST subtree does not match intrinsic description";
  // Map from intrinsic func buffer to description func buffer
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> intrin_buffer_map;
  BufferRemap(intrinsic, &intrin_buffer_map);
  // Map form intrinsic func buffer to current AST buffer
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map;
  for (const auto& pair : intrin_buffer_map) {
    auto it = comparator.rhs_buffer_map_.find(pair.second);
    CHECK(it != comparator.rhs_buffer_map_.end());
    buffer_map[pair.first] = it->second;
  }
  // Build Var map, which is the map from intrin buffer data to AST buffer data
  std::unordered_map<const VarNode*, const PrimExprNode*> var_map;
  auto update_var_map = [&var_map](const PrimExpr& lhs, const PrimExpr& rhs) {
    if (const auto* var = lhs.as<VarNode>()) {
      var_map[var] = rhs.get();
    }
  };
  for (const auto& pair : buffer_map) {
    update_var_map(pair.first->data, pair.second->data);
  }
  std::unordered_map<Buffer, Array<Range>, ObjectPtrHash, ObjectPtrEqual> buffer_region_map;
  for (const auto& read : impl_block->reads) {
    buffer_region_map.emplace(read->buffer, read->region);
  }
  for (const auto& write : impl_block->writes) {
    buffer_region_map.emplace(write->buffer, write->region);
  }

  Array<MatchBufferRegion> match_buffer_regions;
  for (size_t i = 0; i < intrinsic->implementation->params.size(); ++i) {
    const auto& param = intrinsic->implementation->params[i];
    const auto& buffer = intrinsic->implementation->buffer_map.at(param);
    const auto& source = buffer_map.at(buffer);
    Region region = buffer_region_map.at(buffer);
    auto extra_indices = comparator.buffer_indices_.at(source);
    std::vector<Range> extra_buffer_ranges;
    std::transform(extra_indices.begin(), extra_indices.end(),
                   std::back_inserter(extra_buffer_ranges),
                   [](const PrimExpr& index) { return Range::FromMinExtent(index, 1); });
    region.insert(region.begin(), extra_buffer_ranges.begin(), extra_buffer_ranges.end());
    match_buffer_regions.push_back(MatchBufferRegion(buffer, BufferRegion(source, region)));
  }

  impl_block.CopyOnWrite()->match_buffers = match_buffer_regions;
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> bv_map;
  for (size_t i = 0; i < desc_block->iter_vars.size(); ++i) {
    auto it = comparator.equal_map_.find(desc_block->iter_vars[i]->var);
    if (it != comparator.equal_map_.end()) {
      bv_map[impl_block->iter_vars[i]->var] = Downcast<PrimExpr>(it->second);
    } else {
      bv_map[impl_block->iter_vars[i]->var] = 0;
    }
  }
  Stmt new_body = SubstituteInScope(impl_block,
                                    [&](const VarNode* var) -> PrimExpr {
                                      auto it = bv_map.find(GetRef<Var>(var));
                                      if (it == bv_map.end())
                                        return GetRef<Var>(var);
                                      else
                                        return it->second;
                                    });
  // Replace
  ObjectPtr<BlockNode> new_block_ptr = make_object<BlockNode>(*block_realize->block.get());
  new_block_ptr->body = Downcast<Block>(new_body)->body;
  ICHECK(new_block_ptr->match_buffers.empty());
  new_block_ptr->match_buffers = Downcast<Block>(new_body)->match_buffers;
  Block new_block(new_block_ptr);
  self->Replace(self->stmt2ref.at(block_realize->block.get()), new_block,
                {{block_realize->block, new_block}});
  {
    struct BindingValidator : public StmtVisitor {
      void VisitStmt_(const BlockRealizeNode* realize) final {
        StmtSRef& sref = self->stmt2ref.at(realize->block.get());
        UpdateAffineFlag(self, sref);
        VisitStmt(realize->block->body);
      }
      ScheduleState self;
    };
    BindingValidator validator;
    StmtSRef block_sref = self->stmt2ref.at(new_block.get());
    const PrimFuncNode* func = GetRootPrimFunc(self->mod, GetRootBlock(block_sref).get(), nullptr);
    validator.self = self;
    validator(func->body);
  }
}

struct BlockizeTraits : public UnpackedInstTraits<BlockizeTraits> {
  static constexpr const char* kName = "Blockize";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv) {
    return sch->Blockize(loop_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv) {
    PythonAPICall py("blockize");
    py.Input("loop", loop_rv);
    py.SingleOutput(outputs);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

struct TensorizeTraits : public UnpackedInstTraits<TensorizeTraits> {
  static constexpr const char* kName = "Tensorize";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv, String intrin_name) {
    return sch->Tensorize(loop_rv, intrin_name);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv, String intrin_name) {
    PythonAPICall py("tensorize");
    py.Input("loop", loop_rv);
    py.Input("intrin", intrin_name);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(BlockizeTraits);
TVM_REGISTER_INST_KIND_TRAITS(TensorizeTraits);

}  // namespace tir
}  // namespace tvm
