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
#include "../../arith/pattern_match.h"
#include "./schedule_common.h"

namespace tvm {
namespace tir {
/*!
 * \brief Checks if an expression is of pattern `(outer * inner_extent + inner)`, where `outer` and
 * `inner` are variables and `inner_extent` is an expression
 * \param n The expression to be checked
 * \param outer The `outer` in the pattern if it matches
 * \param inner The `inner` in the pattern if it matches
 * \param inner_extent The `inner_extent` in the pattern if it matches
 * \return A boolean indicating if `n` matches this pattern
 */
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
/*!
 * \brief Checks if an expression is of pattern `floordiv(lhs, rhs)`, where `lhs` is a variable
 * and `rhs` is an expression
 * \param n The expression to be checked
 * \param lhs The `lhs` in the pattern if it matches
 * \param rhs The `rhs` in the pattern if it matches
 * \return A boolean indicating if `n` matches this pattern
 */
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
/*!
 * \brief Checks if an expression is of pattern `floordiv(lhs, rhs)`, where `lhs` is a variable
 * and `rhs` is an expression
 * \param n The expression to be checked
 * \param lhs The `lhs` in the pattern if it matches
 * \param rhs The `rhs` in the pattern if it matches
 * \return A boolean indicating if `n` matches this pattern
 */
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
/*!
 * \brief Split the predicate into `(a < b) && (c < d) && ...`
 * \param pred The predicate to be splitted
 * \return A list of pairs, each element of which are lhs and rhs of the '<' sign
 */
std::vector<std::pair<PrimExpr, PrimExpr>> SplitPredicate(PrimExpr pred) {
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
/*!
 * \brief Check if all elements of the list are of type Var and are unique
 * \param list The list to be checked
 * \return A boolean indicating the check result
 */
bool IsAllUniqueVars(const std::vector<PrimExpr>& list) {
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

/*!
 * \brief A helper class that detects if there is any split/fuse patterns in an Expr
 * If so, it provides two functions, replace and postproc, for replacing this pattern
 * and removing them
 */
class FuseSplitDetecter : public ExprVisitor {
 public:
  /*! \brief Constructor */
  explicit FuseSplitDetecter(std::unordered_map<const VarNode*, PrimExpr>* loop_var_extents)
      : loop_var_extents(loop_var_extents) {}

  /*! \brief Check if the PrimExpr is in fuse pattern. If so, set replace and postproc for it */
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
  /*! \brief Check if the PrimExpr is in split pattern. If so, set replace and postproc for it */
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
  // Detect this pattern for all sub-expressions
  void VisitExpr(const PrimExpr& n) override {
    // If the functors have been set, exist
    if (replace != nullptr) {
      return;
    }
    // Detect if there is fuse pattern
    if (SetFuseFunctor(n)) {
      return;
    }
    // Detect if there is split pattern
    if (SetSplitFunctor(n)) {
      return;
    }
    // If not, then detect recursively
    ExprVisitor::VisitExpr(n);
  }

 public:
  /*! \brief The replace functor to be used by FuseSplitNormalizer */
  std::function<Optional<PrimExpr>(PrimExpr)> replace = nullptr;
  /*! \brief The postproc functor to be used by FuseSplitNormalizer */
  std::function<void()> postproc = nullptr;

 private:
  /*! \brief Extents to be manipulated by the functors */
  std::unordered_map<const VarNode*, PrimExpr>* loop_var_extents;
};

/*! \brief A class helps to replace patterns once they are detected */
class FuseSplitNormalizer : public ExprMutator {
 public:
  /*! \brief Constructor */
  explicit FuseSplitNormalizer(const FuseSplitDetecter& detector) : detector_(detector) {}
  /*! \brief Destructor. Invoke postproc only if replacement happens at least once. */
  ~FuseSplitNormalizer() {
    if (replaced_) {
      detector_.postproc();
    }
  }
  // Do replacement recursively for all sub-expressions
  PrimExpr VisitExpr(const PrimExpr& n) override {
    PrimExpr expr = ExprMutator::VisitExpr(n);
    Optional<PrimExpr> mutated = detector_.replace(expr);
    if (!mutated.defined()) {
      return expr;
    }
    this->replaced_ = true;
    return mutated.value();
  }

 private:
  /*! \brief The detector that has detected some pattern */
  const FuseSplitDetecter& detector_;
  /*! \brief Indicating if replacement happens at least once */
  bool replaced_ = false;
};

/*! \brief A helper class to validate loops and store them into StmtSRefNode::binding_valid */
class LoopValidator : public StmtVisitor {
 public:
  /*! \brief Constructor */
  explicit LoopValidator(std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref)
      : stmt2ref_(stmt2ref) {}
  // Collect the extent for each loop variable
  void VisitStmt_(const LoopNode* loop) final {
    // Update `loop_var_extents` with the current loop
    const VarNode* loop_var = loop->loop_var.get();
    CHECK(!loop_var_extents_.count(loop_var))
        << "ValueError: duplicate loop variable \"" << loop_var->name_hint << "\"";
    // TODO(@junrushao1994): loop->min is always 0?
    loop_var_extents_.emplace(loop_var, analyzer_.Simplify(loop->extent));
    StmtVisitor::VisitStmt_(loop);
    loop_var_extents_.erase(loop_var);
  }
  // Validate loop binding for each block
  void VisitStmt_(const BlockRealizeNode* realize) final {
    // Check StmtSRef's binding validity on all blocks
    stmt2ref_->at(realize->block.get())->binding_valid = ValidateBlockBinding(realize);
    StmtVisitor::VisitStmt_(realize);
  }
  /*! \brief Validate the binding of a given block */
  bool ValidateBlockBinding(const BlockRealizeNode* realize) {
    // validate the bindings to loop variables
    std::vector<PrimExpr> bindings{realize->binding_values.begin(), realize->binding_values.end()};
    std::unordered_map<const VarNode*, PrimExpr> loop_vars{loop_var_extents_};
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

 private:
  /*! \brief Extents for loop variables */
  std::unordered_map<const VarNode*, PrimExpr> loop_var_extents_;
  /*! \brief ScheduleNode::stmt2ref whose StmtSRef::binding_valid needs updating */
  std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref_;
  /*! \brief An analyzer used to simplify expressions */
  arith::Analyzer analyzer_;
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
                       << "\nThe producer is:\n  " << write << ", write range: [" << write_min
                       << ", " << write_max << ")"
                       << "\nThe consumer is:\n  " << read << ", read range: [" << read_min << ", "
                       << read_max << ")";
          return false;
        }
      }
    }
  }
  return true;
}

class GPUValidator : public StmtVisitor {
 public:
  void VisitStmt_(const LoopNode* loop) final {
    std::string thread_tag;
    for (const auto& annotation : loop->annotations) {
      if (annotation->attr_key == attr::loop_type) {
        thread_tag = Downcast<StringImm>(annotation->value)->value;
      }
    }

    bool new_kernel = false;
    if ((IsBlockIdx(thread_tag) || IsThreadIdx(thread_tag)) && thread_tag != "vthread") {
      // Check thread binding extents are same in one single kernel

      // If there is no binding, we can regard it as a new kernel
      new_kernel = thread_extents_.empty();

      auto it = thread_extents_.find(thread_tag);
      if (it != thread_extents_.end()) {
        CHECK(ExprDeepEqual()(loop->extent, it->second))
            << "All loops with the same thread binding must have the same extent, but get "
            << loop->extent << " vs " << it->second;
      } else {
        thread_extents_[thread_tag] = loop->extent;
      }
    }

    // Check execution scope and
    if ((current_scope_ == "gpu_thread") && (IsBlockIdx(thread_tag) || IsThreadIdx(thread_tag))) {
      // If the current scope is gpu_thread, any inside threadIdx or blockIdx is illegal.
      LOG(FATAL) << "threadIdx or blockIdx can not be binded under the exec_scope gpu_thread";
    } else if (current_scope_ == "gpu_warp" &&
               ((IsBlockIdx(thread_tag) || IsThreadIdx(thread_tag)) &&
                (thread_tag != "threadIdx.x" || !ExprDeepEqual()(loop->extent, 32)))) {
      LOG(FATAL) << "threadIdx or blockIdx can not be binded under the exec_scope "
                    "gpu_thread except threadIdx.x with extents 32";
    } else if (current_scope_ == "gpu_block" && IsBlockIdx(thread_tag)) {
      // If the current scope is gpu_block, any inside blockIdx is illegal.
      LOG(FATAL) << "blockIdx can not be binded under the exec_scope gpu_block";
    }

    StmtVisitor::VisitStmt_(loop);

    if (new_kernel) {
      if (check_thread_x_) {
        auto it = thread_extents_.find("threadIdx.x");
        CHECK(it != thread_extents_.end())
            << "can not find threadIdx.x but find warp level execution scope";
        CHECK(ExprDeepEqual()(it->second, 32))
            << "threadIdx.x extent is expected to be 32 with warp level block but get "
            << it->second;
      }
      check_thread_x_ = false;
      thread_extents_.clear();
    }
  }

  void VisitStmt_(const BlockRealizeNode* realize) final {
    std::string exec_scope = realize->exec_scope;
    std::string current_scope;
    std::swap(current_scope, current_scope_);

    if (exec_scope != "") {
      if (exec_scope == "gpu_block") {
        CHECK(current_scope == "gpu_block" || current_scope == "gpu_global");
      } else if (exec_scope == "gpu_warp") {
        CHECK(current_scope == "gpu_warp" || current_scope == "gpu_block" ||
              current_scope == "gpu_global");
      } else if (exec_scope == "gpu_warp") {
        CHECK(exec_scope == "gpu_thread" || current_scope == "gpu_warp" ||
              current_scope == "gpu_block" || current_scope == "gpu_global");
      }
    }

    current_scope_ = exec_scope;
    StmtVisitor::VisitStmt_(realize);
    std::swap(current_scope, current_scope_);
  }

  /*! \brief The final result */

 private:
  inline bool IsThreadIdx(std::string thread_tag) {
    return thread_tag.substr(0, 9) == "threadIdx" || thread_tag.substr(0, 7) == "vthread";
  }

  inline bool IsBlockIdx(std::string thread_tag) { return thread_tag.substr(0, 9) == "BlockIdx"; }

  /*! \brief The current execution scope (gpu_global, gpu_block, gpu_warp or gpu_thread) */
  std::string current_scope_ = "gpu_global";
  /*! \brief The extents of each threadIdx or blockIdx */
  std::unordered_map<std::string, PrimExpr> thread_extents_;
  /*! \brief Whether need to check threadIdx.x extents = 32 */
  bool check_thread_x_ = false;
};

void ScheduleNode::ValidateHierarchy(const PrimFunc& f) {
  GPUValidator gpu_validator;
  gpu_validator(f->body);
}

/*! \brief A helper class to validate correctness of StmtSRef */
class SRefValidator : public StmtVisitor {
 public:
  /*! \brief Constructor */
  explicit SRefValidator(const ScheduleNode* sch) : sch(sch), ancestors({nullptr}) {}
  // Valida each block
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
    ancestors.push_back(sref.operator->());
    StmtVisitor::VisitStmt_(block);
    ancestors.pop_back();
  }
  // Validate each loop
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
    ancestors.push_back(sref.operator->());
    StmtVisitor::VisitStmt_(loop);
    ancestors.pop_back();
  }
  /*! \brief The schedule it belongs to */
  const ScheduleNode* sch;
  /*! \brief Parent information during the visit */
  std::vector<const StmtSRefNode*> ancestors;
};

bool ScheduleNode::ValidateSRef() const {
  SRefValidator(this)(func->body);
  return true;
}

TVM_REGISTER_GLOBAL("tir.schedule.ValidateSRef")
    .set_body_typed<bool(Schedule)>([](Schedule schedule) { return schedule->ValidateSRef(); });

}  // namespace tir
}  // namespace tvm
