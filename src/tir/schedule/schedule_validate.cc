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

#include <tvm/arith/analyzer.h>

#include "../../arith/pattern_match.h"
#include "./schedule_common.h"

namespace tvm {
namespace tir {

bool IsFusePattern(const PrimExpr& n, Var* outer, Var* inner, PrimExpr* inner_extent) {
  arith::Analyzer analyzer;
  arith::PVar<Var> outer_p;
  arith::PVar<Var> inner_p;
  arith::PVar<PrimExpr> inner_extent_p;
  // Check if n is in form of `(outer * inner_extent + inner)`
  if ((outer_p * inner_extent_p + inner_p).Match(n)) {
    // Extract the variables from the pattern
    *outer = outer_p.Eval();
    *inner = inner_p.Eval();
    *inner_extent = analyzer.Simplify(inner_extent_p.Eval());
    return true;
  }
  return false;
}

bool IsFloorDivPattern(const PrimExpr& n, Var* lhs, PrimExpr* rhs) {
  arith::Analyzer analyzer;
  arith::PVar<Var> lhs_p;
  arith::PVar<PrimExpr> rhs_p;
  // Check if n is in form of `floordiv(lhs, rhs)`
  if (floordiv(lhs_p, rhs_p).Match(n)) {
    // Extract the variables from the pattern
    *lhs = lhs_p.Eval();
    *rhs = analyzer.Simplify(rhs_p.Eval());
    return true;
  }
  return false;
}

bool IsFloorModPattern(const PrimExpr& n, Var* lhs, PrimExpr* rhs) {
  arith::Analyzer analyzer;
  arith::PVar<Var> lhs_p;
  arith::PVar<PrimExpr> rhs_p;
  // Check if n is in form of `floormod(lhs, rhs)`
  if (floormod(lhs_p, rhs_p).Match(n)) {
    // Extract the variables from the pattern
    *lhs = lhs_p.Eval();
    *rhs = analyzer.Simplify(rhs_p.Eval());
    return true;
  }
  return false;
}

std::vector<std::pair<PrimExpr, PrimExpr>> SplitPredicate(PrimExpr pred) {
  // Split the predicate into `(a < b) && (c < d) && ...`
  std::vector<std::pair<PrimExpr, PrimExpr>> result;
  arith::PVar<PrimExpr> lhs, rhs, rest;
  for (;;) {
    if (((lhs < rhs) && rest).Match(pred)) {
      result.emplace_back(lhs.Eval(), rhs.Eval());
      pred = rest.Eval();
    } else if ((lhs < rhs).Match(pred)) {
      result.emplace_back(lhs.Eval(), rhs.Eval());
      break;
    } else {
      // TODO(@junrushao1994): do nothing?
      break;
    }
  }
  return result;
}

bool IsAllUniqueVars(const std::vector<PrimExpr>& list) {
  // Check if all elements in the given list are `Var` and unique
  std::unordered_set<const PrimExprNode*> exists;
  for (const PrimExpr& item : list) {
    if (!item->IsInstance<VarNode>()) {
      return false;
    }
    const PrimExprNode* p = item.get();
    if (exists.count(p)) {
      return false;
    }
    exists.insert(p);
  }
  return true;
}

class FuseSplitDetecter : public ExprVisitor {
 public:
  bool SetFuseFunctor(const PrimExpr& n) {
    Var outer, inner;
    PrimExpr inner_extent;
    // Detect if the pattern exists
    if (!IsFusePattern(n, &outer, &inner, &inner_extent)) {
      return false;
    }
    // Retrieve the extent of corresponding variables
    auto outer_extent_itr = loop_var_extents->find(outer.get());
    auto inner_extent_itr = loop_var_extents->find(inner.get());
    // Rule out the possibility of
    // 1) Not loop-vars
    // 2) Extents do not match
    if (outer_extent_itr == loop_var_extents->end() ||
        inner_extent_itr == loop_var_extents->end() ||
        !ExprDeepEqual()(inner_extent, inner_extent_itr->second)) {
      return false;
    }
    // New variable for fusion result and its extent
    Var fused = Var(outer->name_hint + "_" + inner->name_hint + "_fuse");
    PrimExpr fused_extent = outer_extent_itr->second * inner_extent_itr->second;
    // Set the functor for replacing all occurrence
    this->replace = [outer, inner, inner_extent, fused](PrimExpr n) -> Optional<PrimExpr> {
      Var e_outer, e_inner;
      PrimExpr e_inner_extent;
      if (!IsFusePattern(n, &e_outer, &e_inner, &e_inner_extent)) {
        return NullOpt;
      }
      return (e_outer.same_as(outer) && e_inner.same_as(inner) &&
              ExprDeepEqual()(e_inner_extent, inner_extent))
                 ? Optional<PrimExpr>(fused)
                 : NullOpt;
    };
    // Set the post-processing functor to manipulate loop_var_extents
    this->postproc = [this, outer, inner, fused, fused_extent]() -> void {
      arith::Analyzer analyzer;
      // TODO(@junrushao1994): just erase it without any checking?
      this->loop_var_extents->erase(outer.get());
      this->loop_var_extents->erase(inner.get());
      this->loop_var_extents->emplace(fused.get(), analyzer.Simplify(fused_extent));
    };
    return true;
  }

  bool SetSplitFunctor(const PrimExpr& n) {
    Var lhs;
    PrimExpr rhs;
    // Detect if the pattern exists
    if (!IsFloorDivPattern(n, &lhs, &rhs) && !IsFloorModPattern(n, &lhs, &rhs)) {
      return false;
    }
    // Retrieve the extent of corresponding variables
    auto lhs_itr = loop_var_extents->find(lhs.get());
    // Rule out the possibility that lhs is not loop_var
    if (lhs_itr == loop_var_extents->end()) {
      return false;
    }
    // New variables as result of split
    Var outer = Var(lhs->name_hint + "_o");
    Var inner = Var(lhs->name_hint + "_i");
    PrimExpr outer_extent = floordiv(lhs_itr->second, rhs);
    PrimExpr inner_extent = rhs;
    // Set the functor for replacing all occurrence
    this->replace = [outer, inner, lhs, rhs](PrimExpr n) -> Optional<PrimExpr> {
      Var e_lhs;
      PrimExpr e_rhs;
      if (IsFloorDivPattern(n, &e_lhs, &e_rhs)) {
        return (e_lhs.same_as(lhs) && ExprDeepEqual()(e_rhs, rhs)) ? Optional<PrimExpr>(outer)
                                                                   : NullOpt;
      }
      if (IsFloorModPattern(n, &e_lhs, &e_rhs)) {
        return (e_lhs.same_as(lhs) && ExprDeepEqual()(e_rhs, rhs)) ? Optional<PrimExpr>(inner)
                                                                   : NullOpt;
      }
      return NullOpt;
    };
    // Set the post-processing functor to manipulate loop_var_extents
    this->postproc = [this, lhs, outer, inner, outer_extent, inner_extent]() -> void {
      arith::Analyzer analyzer;
      // TODO(@junrushao1994): just erase it without any checking?
      this->loop_var_extents->erase(lhs.get());
      this->loop_var_extents->emplace(outer.get(), analyzer.Simplify(outer_extent));
      this->loop_var_extents->emplace(inner.get(), inner_extent);
    };
    return true;
  }

  explicit FuseSplitDetecter(std::unordered_map<const VarNode*, PrimExpr>* loop_var_extents)
      : loop_var_extents(loop_var_extents), replace(nullptr), postproc(nullptr) {}

  void VisitExpr(const PrimExpr& n) override {
    if (replace != nullptr) {
      return;
    }
    if (SetFuseFunctor(n)) {
      return;
    }
    if (SetSplitFunctor(n)) {
      return;
    }
    ExprVisitor::VisitExpr(n);
  }

  std::unordered_map<const VarNode*, PrimExpr>* loop_var_extents;
  std::function<Optional<PrimExpr>(PrimExpr)> replace;
  std::function<void()> postproc;
};

class FuseSplitNormalizer : public ExprMutator {
 public:
  explicit FuseSplitNormalizer(const FuseSplitDetecter& detector)
      : detector(detector), replaced(false) {}

  ~FuseSplitNormalizer() {
    if (replaced) {
      detector.postproc();
    }
  }

  PrimExpr VisitExpr(const PrimExpr& n) override {
    PrimExpr expr = ExprMutator::VisitExpr(n);
    Optional<PrimExpr> mutated = detector.replace(expr);
    if (!mutated.defined()) {
      return expr;
    }
    this->replaced = true;
    return mutated.value();
  }

  const FuseSplitDetecter& detector;
  bool replaced;
};

class LoopValidator : public StmtVisitor {
 public:
  explicit LoopValidator(std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref)
      : stmt2ref(stmt2ref) {}

  void VisitStmt_(const LoopNode* loop) final {
    // Update `loop_var_extents` with the current loop
    const VarNode* loop_var = loop->loop_var.get();
    CHECK(!loop_var_extents.count(loop_var))
        << "ValueError: duplicate loop variable \"" << loop_var->name_hint << "\"";
    // TODO(@junrushao1994): loop->min is always 0?
    loop_var_extents.emplace(loop_var, analyzer.Simplify(loop->extent));
    StmtVisitor::VisitStmt_(loop);
    loop_var_extents.erase(loop_var);
  }

  void VisitStmt_(const BlockRealizeNode* realize) final {
    // Check StmtSRef's binding validity on all blocks
    stmt2ref->at(realize->block.get())->binding_valid = ValidateBlockBinding(realize);
    StmtVisitor::VisitStmt_(realize);
  }

 private:
  bool ValidateBlockBinding(const BlockRealizeNode* realize) {
    // validate the bindings to loop variables
    std::vector<PrimExpr> bindings{realize->binding_values.begin(), realize->binding_values.end()};
    std::unordered_map<const VarNode*, PrimExpr> loop_vars{loop_var_extents};
    std::vector<std::pair<PrimExpr, PrimExpr>> predicates = SplitPredicate(realize->predicate);

    for (;;) {
      // Detect fuse/split pattern
      FuseSplitDetecter detector(&loop_vars);
      for (const auto& binding : bindings) {
        detector(binding);
        if (detector.replace) {
          break;
        }
      }
      // If there is not fuse/split pattern left, break
      if (!detector.replace) {
        break;
      }
      // Substitute pattern
      FuseSplitNormalizer normalizer(detector);
      // Update all bindings, remove split/fuse pattern, and replace with loop variable
      for (auto& binding : bindings) {
        binding = normalizer(binding);
      }
      // Update lhs of all predicates
      std::vector<std::pair<PrimExpr, PrimExpr>> new_predicates;
      for (const auto& kv : predicates) {
        PrimExpr lhs = normalizer(kv.first);
        PrimExpr rhs = kv.second;
        // once they can be reduced to `Var`, they can be considered as a loop variable
        if (const auto* var = lhs.as<VarNode>()) {
          loop_vars[var] = rhs;
        } else {
          new_predicates.emplace_back(lhs, rhs);
        }
      }
      predicates.swap(new_predicates);
    }
    return predicates.empty() && IsAllUniqueVars(bindings);
  }

  std::unordered_map<const VarNode*, PrimExpr> loop_var_extents;
  std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref;
  arith::Analyzer analyzer;
};

void ScheduleNode::ValidateLoops() {
  LoopValidator validator(&stmt2ref);
  validator(func->body);
}

bool ScheduleNode::ValidateRegionCover(const StmtSRef& consumer_block_sref) const {
  if (consumer_block_sref->parent == nullptr) {
    return true;
  }
  const auto* consumer_block = consumer_block_sref->GetStmt<BlockNode>();
  const StmtSRef& parent_block_sref = GetParentBlockSRef(consumer_block_sref);
  // Gather all the producers
  struct Producer {
    /*! \brief The block that writes the buffer */
    StmtSRef block_sref;
    /*! \brief The region the buffer is written */
    TensorRegion region;
    /*! \brief Constructor */
    Producer(const StmtSRef& block_sref, const TensorRegion& region)
        : block_sref(block_sref), region(region) {}
  };
  // Maps a buffer var to its producers
  std::unordered_map<const VarNode*, std::vector<Producer>> buffer_producers;
  // Collect all producers to a buffer by enumerating all RAW predecessors of the consumer
  for (const DepEdge& edge : scopes.at(parent_block_sref).GetPredecessors(consumer_block_sref)) {
    if (edge->type != DepType::kRAW) {
      continue;
    }
    // i.e. the RAW predecessor is producer
    const StmtSRef& producer_block_sref = edge->dst;
    for (const TensorRegion& output_region : producer_block_sref->GetStmt<BlockNode>()->writes) {
      const VarNode* buffer_var = output_region->buffer->data.get();
      buffer_producers[buffer_var].emplace_back(producer_block_sref, output_region);
    }
  }
  // Check the region cover property for each buffer that the consumer reads
  for (const TensorRegion& consumer_region : consumer_block->reads) {
    const VarNode* buffer_var = consumer_region->buffer->data.get();
    if (!buffer_producers.count(buffer_var)) {
      continue;
    }
    // Producers of the current buffer
    const std::vector<Producer>& producers = buffer_producers.at(buffer_var);
    // Figure out LCA of consumer and all producers
    StmtSRef lca = [&producers, &consumer_block_sref, &parent_block_sref]() {
      // inputs include consumer and all producers
      std::vector<StmtSRef> inputs = {consumer_block_sref};
      for (const Producer& producer : producers) {
        inputs.push_back(producer.block_sref);
      }
      return LowestCommonAncestor(inputs, parent_block_sref);
    }();
    arith::Analyzer analyzer;
    // Relax the read region with the loops under LCA
    TensorRegion read = RelaxRegion(consumer_block_sref, lca, consumer_region);
    int ndim = read->region.size();
    for (const Producer& producer : producers) {
      // Relax the write region with the loops under LCA
      TensorRegion write = RelaxRegion(producer.block_sref, lca, producer.region);
      CHECK_EQ(read->region.size(), write->region.size())
          << "InternalError: Inconsistent rank of the same buffer between reads and writes";
      // Check if the write domain covers the read domain
      for (int i = 0; i < ndim; ++i) {
        PrimExpr read_min = read->region[i]->min;
        PrimExpr read_max = read_min + read->region[i]->extent;
        PrimExpr write_min = write->region[i]->min;
        PrimExpr write_max = write_min + write->region[i]->extent;
        if (!analyzer.CanProve(write_min <= read_min) ||
            !analyzer.CanProve(read_max <= write_max)) {
          LOG(WARNING) << "InternalError: Cannot prove the region cover property on dimension " << i
                       << ". The read range is [" << read_min << ", " << read_max
                       << "), and the write range is [" << write_min << ", " << write_max
                       << "). The producer is :\n " << read << "\nThe consumer is:\n"
                       << write;
          return false;
        }
      }
    }
  }
  return true;
}

class SRefValidator : public StmtVisitor {
 public:
  explicit SRefValidator(const ScheduleNode* sch) : sch_(sch) {}

  void VisitStmt_(const BlockNode* op) override {
    CheckParent(op);
    auto sref = sch_->stmt2ref.at(op);
    CHECK(sch_->scopes.count(sref)) << "Cannot find scope information of the block:\n"
                                    << GetRef<Stmt>(op);
  }

  void VisitStmt_(const LoopNode* op) override { CheckParent(op); }

 private:
  const ScheduleNode* sch_;
  const StmtSRefNode* parent_{nullptr};

  void CheckParent(const StmtNode* op) {
    auto it = sch_->stmt2ref.find(op);
    Stmt s = GetRef<Stmt>(op);
    CHECK(it != sch_->stmt2ref.end()) << "Cannot find Stmt in stmt2ref map:\n" << s;
    StmtSRef sref = it->second;
    CHECK(sref->parent == parent_) << "The parent of the node is mismatch:\n" << s;
    parent_ = sref.get();
  }
};

// TODO(@spectrometerHBH): The bugfix to validator triggers the following issues in unittests:
// [FAILED] tests/python/tir/test_schedule_replace.py::test_replace_copy
// [FAILED] tests/python/tir/test_schedule_replace.py::test_replace_partial_copy0
// [FAILED] tests/python/tir/test_schedule_replace.py::test_replace_partial_copy1
// [FAILED] tests/python/tir/test_schedule_replace.py::test_replace_root_copy1
#if (false)
class SRefValidator : public StmtVisitor {
 public:
  explicit SRefValidator(const ScheduleNode* sch) : sch(sch), ancestors({nullptr}) {}

  void VisitStmt_(const BlockNode* block) override {
    CHECK(sch->stmt2ref.count(block))
        << "InternalError: A BlockNode should appear in sref map, but it didn't\n"
        << GetRef<Stmt>(block);
    const StmtSRef& sref = sch->stmt2ref.at(block);
    CHECK(sch->scopes.count(sref))
        << "InternalError: Cannot find scope information of the BlockNode:\n"
        << GetRef<Stmt>(block);
    CHECK(sref->parent == ancestors.back())
        << "InternalError: Parent information mismatch for BlockNode:\n"
        << GetRef<Stmt>(block) << "\nIts parent is supposed to be:\n"
        << GetRef<Stmt>(ancestors.back()->stmt) << "\nHowever, its parent is incorrect and is:\n"
        << (sref->parent ? Optional<Stmt>(GetRef<Stmt>(sref->parent->stmt))
                         : Optional<Stmt>(NullOpt));
    ancestors.push_back(sref.get());
    StmtVisitor::VisitStmt_(block);
    ancestors.pop_back();
  }

  void VisitStmt_(const LoopNode* loop) override {
    CHECK(sch->stmt2ref.count(loop))
        << "InternalError: A LoopNode should appear in sref map, but it didn't\n"
        << GetRef<Stmt>(loop);
    const StmtSRef& sref = sch->stmt2ref.at(loop);
    Optional<Stmt> stmt = NullOpt;
    CHECK(sref->parent == ancestors.back())
        << "InternalError: Parent information mismatch for LoopNode:\n"
        << GetRef<Stmt>(loop) << "\nIts parent is supposed to be:\n"
        << GetRef<Stmt>(ancestors.back()->stmt) << "\nHowever, its parent is incorrect and is:\n"
        << (sref->parent ? Optional<Stmt>(GetRef<Stmt>(sref->parent->stmt))
                         : Optional<Stmt>(NullOpt));
    ancestors.push_back(sref.get());
    StmtVisitor::VisitStmt_(loop);
    ancestors.pop_back();
  }

  const ScheduleNode* sch;
  std::vector<const StmtSRefNode*> ancestors;
};
#endif

bool ScheduleNode::ValidateSRef() const {
  // TODO(@junrushao1994): idk if we need to return
  SRefValidator(this)(func->body);
  return true;
}

TVM_REGISTER_GLOBAL("tir.schedule.ValidateSRef")
    .set_body_typed<bool(Schedule)>([](Schedule schedule) { return schedule->ValidateSRef(); });

}  // namespace tir
}  // namespace tvm
