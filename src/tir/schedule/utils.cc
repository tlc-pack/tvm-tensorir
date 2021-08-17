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

#include "./utils.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tir {

/*! \note Nested SeqStmt is not allowed in schedule. */
Array<Stmt> GetChildren(const Stmt& stmt, bool keep_realize) {
  Stmt body;
  if (const auto* block = stmt.as<BlockNode>()) {
    body = block->body;
  } else if (const auto* loop = stmt.as<ForNode>()) {
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

class IRSubstituteInScope : public StmtExprMutator {
 public:
  explicit IRSubstituteInScope(std::function<PrimExpr(const VarNode*)> fmap)
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
    arith::Analyzer analyzer;
    auto fmutate = [&](const PrimExpr& e) { return this->VisitExpr(e); };
    Array<PrimExpr> v = op->iter_values;
    v.MutateByApply(fmutate);
    PrimExpr pred = this->VisitExpr(op->predicate);
    if (v.same_as(op->iter_values) && pred.same_as(op->predicate)) {
      return GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->iter_values = std::move(v);
      n->predicate = std::move(analyzer.Simplify(pred));
      return Stmt(n);
    }
  }

 private:
  const std::function<PrimExpr(const VarNode*)> fmap_;
};

Stmt SubstituteInScope(const Stmt& stmt,
                       const std::function<PrimExpr(const VarNode*)>& value_func) {
  return IRSubstituteInScope(value_func)(stmt);
}

Stmt SubstituteInScope(const Stmt& stmt,
                       const std::unordered_map<const VarNode*, PrimExpr>& var_map) {
  auto vmap = [&](const VarNode* v) -> PrimExpr {
    const auto& it = var_map.find(v);
    if (it != var_map.end()) {
      return it->second;
    } else {
      return NullValue<PrimExpr>();
    }
  };
  return IRSubstituteInScope(vmap)(stmt);
}

PrimExpr SubstituteInScope(const PrimExpr& expr,
                           const std::unordered_map<const VarNode*, PrimExpr>& var_map) {
  auto vmap = [&](const VarNode* v) -> PrimExpr {
    const auto& it = var_map.find(v);
    if (it != var_map.end()) {
      return it->second;
    } else {
      return NullValue<PrimExpr>();
    }
  };
  return IRSubstituteInScope(vmap)(expr);
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
  return IRSubstituteInScope(vmap)(stmt);
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
  return IRSubstituteInScope(vmap)(expr);
}

BufferRegion SubstituteBufferRegion(
    const BufferRegion& buffer_region,
    const std::unordered_map<const VarNode*, const VarNode*>& var_map) {
  auto new_buffer_region = make_object<BufferRegionNode>(*buffer_region.operator->());
  new_buffer_region->region = Array<Range>(make_object<ArrayNode>());
  for (const auto& range : buffer_region->region) {
    new_buffer_region->region.push_back(Range::FromMinExtent(
        SubstituteInScope(range->min, var_map), SubstituteInScope(range->extent, var_map)));
  }
  return BufferRegion(new_buffer_region);
}

BufferRegion SubstituteBufferRegion(const BufferRegion& buffer_region,
                                    const std::unordered_map<const VarNode*, PrimExpr>& var_map) {
  auto new_buffer_region = make_object<BufferRegionNode>(*buffer_region.operator->());
  new_buffer_region->region = Array<Range>(make_object<ArrayNode>());
  for (const auto& range : buffer_region->region) {
    new_buffer_region->region.push_back(Range::FromMinExtent(
        SubstituteInScope(range->min, var_map), SubstituteInScope(range->extent, var_map)));
  }
  return BufferRegion(new_buffer_region);
}

BlockRealize GetBlockRealize(const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  // We cannot support getting the BlockRealize of the root block, since the parent sref of the root
  // block sref is `nullptr`.
  CHECK(block_sref->parent != nullptr)
      << "ValueError: Get the BlockRealize of the root block is not supported";
  // Step 1. Get the stmt corresponding to the parent sref of the input block.
  Stmt parent = GetRef<Stmt>(block_sref->parent->stmt);
  // Step 2. Get the body of the parent stmt, given that the parent is either a Block or a For.
  Stmt child;
  if (const auto* block_parent = parent.as<BlockNode>()) {
    child = block_parent->body;
  } else if (const auto* loop_parent = parent.as<ForNode>()) {
    child = loop_parent->body;
  } else {
    ICHECK(false) << "TypeError: A StmtSRef can only points to a Block or a For";
  }
  // Step 3.
  //  - If the child stmt is a BlockRealize, it indicates that the input block sref is the only
  //    child of the parent sref. In this case this BlockRealize is exactly what we want.
  //  - If the child stmt is a SeqStmt, it indicates that the input block sref is one child among
  //    the parent block's children. In this case we can find the wanted BlockRealize according to
  //    the `seq_index` of the input block sref, which must be non-negative.
  //  - None of the other cases could happen - the child stmt cannot be any other kind of Stmt.
  if (child->IsInstance<BlockRealizeNode>()) {
    BlockRealize block_realize = Downcast<BlockRealize>(child);
    ICHECK_EQ(block_realize->block.get(), block)
        << "ValueError: The BlockRealize is expected to have the input block as its body block";
    return block_realize;
  } else if (child->IsInstance<SeqStmtNode>()) {
    SeqStmt seq_stmt = Downcast<SeqStmt>(child);
    ICHECK_GE(block_sref->seq_index, 0) << "ValueError: `seq_index` is out of range";
    ICHECK_LT(block_sref->seq_index, static_cast<int64_t>(seq_stmt->seq.size()))
        << "ValueError: `seq_index` is out of range";
    const auto* block_realize = seq_stmt->seq[block_sref->seq_index].as<BlockRealizeNode>();
    ICHECK(block_realize)
        << "TypeError: The stmt indicated by `seq_index` is expected to be a BlockRealize";
    ICHECK_EQ(block_realize->block.get(), block)
        << "ValueError: The BlockRealize is expected to have the input block as its body block";
    return GetRef<BlockRealize>(block_realize);
  } else {
    LOG(FATAL) << "TypeError: The child stmt can only be a BlockRealize or a SeqStmt here";
    throw;
  }
}

StmtSRef LowestCommonAncestor(const std::vector<StmtSRef>& nodes, const StmtSRef& root) {
  // alg: count the visit times for each node from the bottom to the root
  ICHECK_GE(nodes.size(), 2);
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

bool CheckOneLine(const Stmt& s) {
  bool legal = true, meet_block = false;
  PostOrderVisit(s, [&legal, &meet_block](const ObjectRef& obj) {
    if (obj->IsInstance<SeqStmtNode>() && !meet_block) {
      legal = false;
    } else if (obj->IsInstance<BlockRealizeNode>()) {
      meet_block = true;
    }
  });
  return legal;
}

std::function<BufferRegion(const BufferRegion)> RelaxGenerator(
    const StmtSRef& block_sref, const StmtSRef& root,
    std::unordered_map<const VarNode*, PrimExpr>* vmap,
    std::unordered_map<const VarNode*, arith::IntSet>* dom_map) {
  const auto* block = block_sref->StmtAs<BlockNode>();
  const auto* block_realize = GetBlockRealize(block_sref).operator->();
  ICHECK(block != nullptr);

  // Update block_var map
  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    (*vmap)[block->iter_vars[i]->var.get()] = block_realize->iter_values[i];
  }

  // Gather iteration domain
  auto sref = GetRef<StmtSRef>(block_sref->parent);
  while (sref.defined() && !sref.same_as(root)) {
    const auto* loop = sref->StmtAs<ForNode>();
    // The root may not be a loop
    if (loop == nullptr) break;
    Range range = Range::FromMinExtent(loop->min, loop->extent);
    (*dom_map)[loop->loop_var.get()] = arith::IntSet::FromRange(range);
    sref = GetRef<StmtSRef>(sref->parent);
  }

  return [vmap, dom_map](const BufferRegion& buffer_region) {
    arith::Analyzer analyzer;
    auto n = make_object<BufferRegionNode>();
    Array<Range> region;
    n->buffer = buffer_region->buffer;
    for (auto range : buffer_region->region) {
      range = Range::FromMinExtent(Substitute(range->min, *vmap), Substitute(range->extent, *vmap));
      auto int_set = arith::EvalSet(range, *dom_map);
      region.push_back(Range::FromMinExtent(analyzer.Simplify(int_set.min()),
                                            analyzer.Simplify(int_set.max() - int_set.min() + 1)));
    }
    n->region = std::move(region);
    return BufferRegion(n);
  };
}

void RelaxRegion(const StmtSRef& block_sref, const StmtSRef& root, std::vector<BufferRegion>* reads,
                 std::vector<BufferRegion>* writes,
                 const std::unordered_map<const VarNode*, Range>& relax_vars) {
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  std::unordered_map<const VarNode*, arith::IntSet> dom_map;
  auto relax = RelaxGenerator(block_sref, root, &vmap, &dom_map);
  for (const auto& pair : relax_vars) {
    dom_map[pair.first] = arith::IntSet::FromRange(pair.second);
  }
  const auto* block = block_sref->StmtAs<BlockNode>();
  if (reads != nullptr) {
    for (const auto& buffer_region : block->reads) {
      reads->push_back(relax(buffer_region));
    }
  }
  if (writes != nullptr) {
    for (const auto& buffer_region : block->writes) {
      writes->push_back(relax(buffer_region));
    }
  }
}

BufferRegion RelaxRegion(const StmtSRef& block_sref, const StmtSRef& root,
                         const BufferRegion& region) {
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  std::unordered_map<const VarNode*, arith::IntSet> dom_map;
  auto relax = RelaxGenerator(block_sref, root, &vmap, &dom_map);
  return relax(region);
}

/*!
 * \brief remove the AST leaf and its parent subtree which has only one leaf
 * \param sref The sref of Block/Loop to be removed
 * \param root The AST root
 * \return The original stmt and the removed stmt of the subtree rooted by the parent node
 */
std::pair<Stmt, Stmt> RemoveLeaf(StmtSRef sref, const StmtSRef& root) {
  ICHECK(sref != root);

  // go upwards until find a father with more than two children
  Stmt last = GetRef<Stmt>(sref->stmt);
  sref = GetRef<StmtSRef>(sref->parent);
  Stmt stmt = GetRef<Stmt>(sref->stmt);
  while (!sref.same_as(root) && stmt.as<BlockNode>() == nullptr) {
    const auto* loop = stmt.as<ForNode>();
    ICHECK(loop != nullptr);
    const auto* seq = loop->body.as<SeqStmtNode>();
    if (seq != nullptr && seq->size() > 1) break;

    sref = GetRef<StmtSRef>(sref->parent);
    last = stmt;
    stmt = GetRef<Stmt>(sref->stmt);
  }

  auto get_body = [&last](const SeqStmtNode* seq) {
    ICHECK_GT(seq->size(), 1);
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
    ICHECK(seq != nullptr);
    auto node = make_object<BlockNode>(*block);
    node->body = get_body(seq);
    return std::make_pair(stmt, Stmt(node));
  } else if (const auto* loop = stmt.as<ForNode>()) {
    const auto* seq = loop->body.as<SeqStmtNode>();
    ICHECK(seq != nullptr);
    auto node = make_object<ForNode>(*loop);
    node->body = get_body(seq);
    return std::make_pair(stmt, Stmt(node));
  } else {
    LOG(FATAL) << "unknown stmt";
    return std::make_pair(Stmt(), Stmt());
  }
}

Stmt StmtReplacer::VisitStmt(const Stmt& stmt) {
  auto it = replace_map.find(stmt.get());
  if (it == replace_map.end()) {
    return StmtMutator::VisitStmt(stmt);
  } else {
    return StmtMutator::VisitStmt(GetRef<Stmt>(it->second));
  }
}

void UpdateScope(ScheduleState self, const StmtSRef& block_sref) {
  BlockScope scope(tir::GetChildBlocks(self, block_sref));
  // The caller is responsible for correcting the flags
  bool affine_binding = false;
  bool region_cover = false;
  // TODO: stage_pipeline
  self->block_info[block_sref] = BlockInfo(std::move(scope), affine_binding, region_cover);
}

void UpdateAffineFlag(ScheduleState self, const StmtSRef& block_sref) {
  if (block_sref->parent == nullptr) {
    ICHECK(self->block_info.count(block_sref));
    self->block_info[block_sref].affine_binding = true;
    return;
  }
  BlockRealize realize = GetBlockRealize(block_sref);
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  Map<Var, Range> loop_var_ranges;
  for (StmtSRefNode* loop_sref = block_sref->parent; loop_sref != nullptr;
       loop_sref = loop_sref->parent) {
    if (const auto* loop = loop_sref->StmtAs<ForNode>()) {
      loop_var_ranges.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    } else {
      break;
    }
  }
  ICHECK(self->block_info.count(block_sref));
  self->block_info[block_sref].affine_binding = ValidateBlockBinding(realize, loop_var_ranges);
}

Array<BufferRegion> BlockReadWriteCollector::reads() {
  std::vector<BufferRegion> res;
  for (size_t i = 0; i < read_regions_.size(); ++i) {
    std::vector<Range> region;
    for (const auto& range : read_regions_[i])
      region.push_back(range.CoverRange(Range::FromMinExtent(0, 0)));
    res.emplace_back(read_buffers_[i], region);
  }
  return res;
}

Array<BufferRegion> BlockReadWriteCollector::writes() {
  std::vector<BufferRegion> res;
  for (size_t i = 0; i < write_regions_.size(); ++i) {
    std::vector<Range> region;
    for (const auto& range : write_regions_[i])
      region.push_back(range.CoverRange(Range::FromMinExtent(0, 0)));
    res.emplace_back(writes_buffers_[i], region);
  }
  return res;
}

void BlockReadWriteCollector::VisitStmt_(const ForNode* op) {
  Range range = Range::FromMinExtent(op->min, op->extent);
  dom_map_[op->loop_var.get()] = arith::IntSet::FromRange(range);
  StmtVisitor::VisitStmt_(op);
  dom_map_.erase(op->loop_var.get());
}

void BlockReadWriteCollector::Update(std::vector<Buffer>* buffers,
                                     std::vector<std::vector<arith::IntSet>>* regions,
                                     const Buffer& buffer,
                                     const std::vector<arith::IntSet>& region) {
  if (inner_buffers_.find(buffer.get()) != inner_buffers_.end()) return;
  bool find = false;
  for (size_t i = 0; i < regions->size(); ++i)
    if ((*buffers)[i].same_as(buffer)) {
      find = true;
      ICHECK_EQ((*regions)[i].size(), region.size()) << "Inconsistent buffer dimension";
      for (size_t j = 0; j < region.size(); ++j) {
        (*regions)[i][j] = arith::Union({(*regions)[i][j], region[j]});
      }
    }
  if (!find) {
    buffers->push_back(buffer);
    regions->push_back(region);
  }
}

void BlockReadWriteCollector::VisitExpr_(const BufferLoadNode* op) {
  std::vector<arith::IntSet> relaxed_region;
  for (size_t j = 0; j < op->indices.size(); ++j) {
    relaxed_region.push_back(arith::EvalSet(op->indices[j], dom_map_));
  }
  Update(&read_buffers_, &read_regions_, op->buffer, relaxed_region);
  ExprVisitor::VisitExpr_(op);
}

void BlockReadWriteCollector::VisitStmt_(const BufferStoreNode* op) {
  std::vector<arith::IntSet> relaxed_region;
  for (size_t j = 0; j < op->indices.size(); ++j) {
    relaxed_region.push_back(arith::EvalSet(op->indices[j], dom_map_));
  }
  Update(&writes_buffers_, &write_regions_, op->buffer, relaxed_region);
  StmtVisitor::VisitStmt_(op);
}

void BlockReadWriteCollector::VisitStmt_(const BlockRealizeNode* op) {
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  for (size_t i = 0; i < op->block->iter_vars.size(); ++i) {
    vmap[op->block->iter_vars[i]->var.get()] = op->iter_values[i];
  }
  for (const auto& read : op->block->reads) {
    std::vector<arith::IntSet> relaxed_region;
    for (const auto& range : read->region) {
      relaxed_region.push_back(
          arith::EvalSet(arith::IntSet::FromRange(Range::FromMinExtent(
                             Substitute(range->min, vmap), Substitute(range->extent, vmap))),
                         dom_map_));
    }
    Update(&read_buffers_, &read_regions_, read->buffer, relaxed_region);
  }
  for (const auto& write : op->block->writes) {
    std::vector<arith::IntSet> relaxed_region;
    for (const auto& range : write->region) {
      relaxed_region.push_back(
          arith::EvalSet(arith::IntSet::FromRange(Range::FromMinExtent(
                             Substitute(range->min, vmap), Substitute(range->extent, vmap))),
                         dom_map_));
    }
    Update(&writes_buffers_, &write_regions_, write->buffer, relaxed_region);
  }
}

}  // namespace tir
}  // namespace tvm
