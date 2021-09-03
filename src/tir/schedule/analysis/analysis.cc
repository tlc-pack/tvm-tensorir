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
#include "../utils.h"

namespace tvm {
namespace tir {

/******** IR Module ********/

Block GetRootBlock(StmtSRef sref) {
  const StmtSRefNode* p_sref = sref.get();
  while (p_sref->parent != nullptr) {
    p_sref = p_sref->parent;
  }
  const BlockNode* root_block = TVM_SREF_TO_BLOCK(root_block, GetRef<StmtSRef>(p_sref));
  return GetRef<Block>(root_block);
}

const PrimFuncNode* GetRootPrimFunc(const IRModule& mod, const StmtNode* root_block,
                                    GlobalVar* result_g_var) {
  for (const auto& kv : mod->functions) {
    const GlobalVar& g_var = kv.first;
    const BaseFunc& base_func = kv.second;
    if (const auto* func = base_func.as<PrimFuncNode>()) {
      if (const auto* realize = func->body.as<BlockRealizeNode>()) {
        if (realize->block.get() == root_block) {
          if (result_g_var != nullptr) {
            *result_g_var = g_var;
          }
          return func;
        }
      }
    }
  }
  LOG(FATAL) << "IndexError: Could not get the corresponding function in the schedule state of the "
                "statement:\n"
             << GetRef<Stmt>(root_block);
  throw;
}

/******** Scope ********/

/*!
 * \brief Gets the sref to the scope root block, exclusive
 * \param sref The block or loop sref to be retrieved
 * \return The sref to the scope root block. NullOpt if `sref` is the root block of the IR
 */
Optional<StmtSRef> GetScopeRoot(const StmtSRef& sref) {
  for (const StmtSRefNode* p = sref->parent; p != nullptr; p = p->parent) {
    if (p->stmt->IsInstance<BlockNode>()) {
      return GetRef<StmtSRef>(p);
    }
  }
  return NullOpt;
}

StmtSRef GetScopeRoot(const ScheduleState& self, const StmtSRef& sref,
                      bool require_stage_pipeline) {
  class RootBlockError : public ScheduleError {
   public:
    explicit RootBlockError(IRModule mod) : mod_(mod) {}
    IRModule mod() const final { return mod_; }
    String FastErrorString() const final {
      return "ScheduleError: The primitive does not operate on the root block";
    }
    String DetailRenderTemplate() const final {
      return "The primitive does not operate on the root block";
    }
    Array<ObjectRef> LocationsOfInterest() const final { return {}; }
    IRModule mod_;
  };

  class NotStagePipelineError : public ScheduleError {
   public:
    explicit NotStagePipelineError(IRModule mod, Block block) : mod_(mod), block_(block) {}
    IRModule mod() const final { return mod_; }
    String FastErrorString() const final {
      return "ScheduleError: The scope root is not a stage pipeline";
    }
    String DetailRenderTemplate() const final {
      return R"(The scope {0} is not a stage pipeline.
Definition of a scope that is a stage pipeline:
- The region cover property holds for every of its child blocks
- No write-after-read dependency or opaque dependency,
- only read-after-write and write-after-write are allowed
- All the statements in the scope are schedulable statements, i.e. Block and For
)";
    }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    Block block_;
  };

  StmtSRef scope_root_sref{nullptr};
  if (Optional<StmtSRef> opt_scope_root_sref = GetScopeRoot(sref)) {
    scope_root_sref = opt_scope_root_sref.value();
  } else {
    throw RootBlockError(self->mod);
  }
  bool stage_pipeline = self->GetBlockInfo(scope_root_sref).scope->stage_pipeline;
  if (require_stage_pipeline && stage_pipeline == false) {
    const BlockNode* block = TVM_SREF_TO_BLOCK(block, scope_root_sref);
    throw NotStagePipelineError(self->mod, GetRef<Block>(block));
  }
  return scope_root_sref;
}

/*!
 * \brief Check the dominant property of a block:
 * the block is the only writer of its output, dominating the reader of its output buffers
 * \param self The schedule state
 * \param block_sref The block whose dominant property is to be checked
 * \return A boolean indicating if the block is a dominant block
 */
bool IsDominantBlock(const BlockScope& self, const StmtSRef& block_sref) {
  // Check whether the input block is the only writer of its outputs
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  const std::unordered_map<Buffer, Array<StmtSRef>, ObjectPtrHash, ObjectPtrEqual>& buffer_writers =
      self->buffer_writers;
  for (const BufferRegion& write_region : block->writes) {
    ICHECK(buffer_writers.count(write_region->buffer))
        << "InternalError: buffer \"" << write_region->buffer->name
        << "\" does not exist in the current scope, when querying block:\n"
        << GetRef<Block>(block);
    if (buffer_writers.at(write_region->buffer).size() != 1) {
      return false;
    }
  }
  return true;
}

/*!
 * \brief A helper function that checks whether a given block is a complete block under the scope,
 * or return the condition it violates if it is not a complete block
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \param scope_root_sref The sref to the root block of the scope that `block_sref` is in
 * \return 0 if the block is a complete block, or a positive integer indicating which condition is
 * first violated
 */
int CheckCompleteBlockErrorCode(const ScheduleState& self, const StmtSRef& block_sref,
                                const StmtSRef& scope_root_sref) {
  BlockScope scope = self->GetBlockScope(scope_root_sref);
  // Cond 1. All block vars are data parallel
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  for (const IterVar& iter_var : block->iter_vars) {
    if (iter_var->iter_type != kDataPar) {
      return 1;
    }
  }
  // Cond 2. Dominant: the block is the only writer of its output,
  // dominating the reader of its output buffers
  if (!IsDominantBlock(scope, block_sref)) {
    return 2;
  }
  // Cond 3. No overlap between the buffers the block reads and writes
  std::unordered_set<const BufferNode*> written_buffers;
  written_buffers.reserve(block->writes.size());
  for (const BufferRegion& write : block->writes) {
    written_buffers.insert(write->buffer.get());
  }
  for (const BufferRegion& read : block->reads) {
    if (written_buffers.count(read->buffer.get())) {
      return 3;
    }
  }
  return 0;
}

bool IsCompleteBlock(const ScheduleState& self, const StmtSRef& block_sref,
                     const StmtSRef& scope_root_sref) {
  return CheckCompleteBlockErrorCode(self, block_sref, scope_root_sref) == 0;
}

void CheckCompleteBlock(const ScheduleState& self, const StmtSRef& block_sref,
                        const StmtSRef& scope_root_sref) {
  class IncompleteBlockError : public ScheduleError {
   public:
    explicit IncompleteBlockError(IRModule mod, Block block, int violated_cond)
        : mod_(std::move(mod)), block_(std::move(block)), violated_cond_(violated_cond) {}
    String FastErrorString() const final { return "ScheduleError: Incomplete block"; }
    String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "The block {0} is not a complete block - it violates condition #" << violated_cond_
         << ".\n"
         << R"(Definition of a complete block:
1) All block vars are data parallel
2) Dominant: the block is the only writer of its output, dominating the reader of its output buffers
            3) No overlap between the buffers the block reads and writes)";
      return os.str();
    }
    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    Block block_;
    int violated_cond_;
  };

  int error_code = CheckCompleteBlockErrorCode(self, block_sref, scope_root_sref);
  if (error_code != 0) {
    const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
    throw IncompleteBlockError(self->mod, GetRef<Block>(block), error_code);
  }
}

/*!
 * \brief A helper function that checks whether a given block is a reduction block under the scope,
 * or return the condition it violates if it is not a reduction block
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \param scope_root_sref The sref to the root block of the scope that `block_sref` is in
 * \return 0 if the block is a reduction block, or a positive integer indicating which condition is
 * first violated
 */
int CheckReductionBlockErrorCode(const ScheduleState& self, const StmtSRef& block_sref,
                                 const StmtSRef& scope_root_sref) {
  BlockScope scope = self->GetBlockScope(scope_root_sref);
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  // Cond 1. The block has the `init` statement.
  if (!block->init.defined()) {
    return 1;
  }
  // Cond 2. All the block bindings are quasi-affine expressions.
  if (!self->IsAffineBlockBinding(block_sref)) {
    return 2;
  }
  // Cond 3. All block vars are either data parallel block vars or reduction block vars. Meanwhile,
  // we collect all the reduction block vars.
  std::unordered_set<const VarNode*> reduction_block_vars;
  reduction_block_vars.reserve(block->iter_vars.size());
  for (const IterVar& iter_var : block->iter_vars) {
    if (iter_var->iter_type != kDataPar && iter_var->iter_type != kCommReduce) {
      return 3;
    } else if (iter_var->iter_type == kCommReduce) {
      reduction_block_vars.insert(iter_var->var.get());
    }
  }
  // Cond 4. Dominant: the block is the only writer of its output, dominating the reader of its
  // output buffers.
  if (!IsDominantBlock(scope, block_sref)) {
    return 4;
  }
  // Cond 5. The reduction block vars are not used to index the output buffers.
  std::unordered_set<const BufferNode*> buffer_written;
  buffer_written.reserve(block->writes.size());
  for (const BufferRegion& write_region : block->writes) {
    buffer_written.insert(write_region->buffer.get());
  }
  bool affected = false;
  PreOrderVisit(block->body, [&](const ObjectRef& obj) {
    if (affected) {
      return false;
    }
    if (const auto* store = obj.as<BufferStoreNode>()) {
      ICHECK(buffer_written.count(store->buffer.get()))
          << "ValueError: The buffer \"" << store->buffer
          << "\" is written in the block but is not in the block's signature";
      for (const PrimExpr& index : store->indices) {
        if (UsesVar(index, [&reduction_block_vars](const VarNode* var) {
              return reduction_block_vars.count(var);
            })) {
          affected = true;
          return false;
        }
      }
      return false;
    }
    return true;
  });
  return !affected ? 0 : 5;
}

bool IsReductionBlock(const ScheduleState& self, const StmtSRef& block_sref,
                      const StmtSRef& scope_root_sref) {
  return CheckReductionBlockErrorCode(self, block_sref, scope_root_sref) == 0;
}

void CheckReductionBlock(const ScheduleState& self, const StmtSRef& block_sref,
                         const StmtSRef& scope_root_sref) {
  class NotReductionBlockError : public ScheduleError {
   public:
    explicit NotReductionBlockError(IRModule mod, Block block, int violated_cond)
        : mod_(std::move(mod)), block_(std::move(block)), violated_cond_(violated_cond) {}
    String FastErrorString() const final { return "ScheduleError: Not a reduction block"; }
    String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "The block {0} is not a reduction block - it violates condition #" << violated_cond_
         << ".\n"
         << R"(Definition of a reduction block:
1) The block has the `init` statement
2) All the block bindings are quasi-affine expressions
3) All block vars are either data parallel block vars or reduction block vars
4) Dominant: the block is the only writer of its output, dominating the reader of its output buffers
            5) The reduction block vars are not used to index the output buffers)";
      return os.str();
    }
    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    Block block_;
    int violated_cond_;
  };

  int error_code = CheckReductionBlockErrorCode(self, block_sref, scope_root_sref);
  if (error_code != 0) {
    const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
    throw NotReductionBlockError(self->mod, GetRef<Block>(block), error_code);
  }
}

void CheckSRefSubtreeCompactDataFlow(const ScheduleState& self, const StmtSRef& subtree_root_sref) {
  class NotCompactDataFlowError : public ScheduleError {
   public:
    explicit NotCompactDataFlowError(IRModule mod, Stmt subtree_root, Block violate_block)
        : mod_(std::move(mod)),
          subtree_root_(std::move(subtree_root)),
          violate_block_(std::move(violate_block)) {
      ICHECK(subtree_root_->IsInstance<BlockNode>() || subtree_root_->IsInstance<ForNode>());
    }
    String FastErrorString() const final {
      return "ScheduleError: The queried subtree root in SRef tree does not have compact data "
             "flow, because some of its child block on SRef tree is neither a complete block nor a "
             "reduction block";
    }
    String DetailRenderTemplate() const final {
      return "The queried subtree root {0} in SRef tree does not have compact data flow, because "
             "its child block {1} on SRef tree is neither a complete block nor a reduction block";
    }
    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {subtree_root_, violate_block_}; }

    IRModule mod_;
    Stmt subtree_root_;
    Block violate_block_;
  };

  // Turn off `require_stage_pipeline` temporarily.
  StmtSRef scope_root = GetScopeRoot(self, subtree_root_sref, /*require_stage_pipeline=*/false);
  Array<StmtSRef> child_blocks = GetChildBlockSRefOnSRefTree(self, scope_root);
  for (const StmtSRef& block : child_blocks) {
    if (!IsCompleteBlock(self, block, scope_root) && !IsReductionBlock(self, block, scope_root)) {
      const BlockNode* violate_block = TVM_SREF_TO_BLOCK(violate_block, block);
      throw NotCompactDataFlowError(self->mod, GetRef<Stmt>(subtree_root_sref->stmt),
                                    GetRef<Block>(violate_block));
    }
  }
}

/******** Binding ********/

bool IsAffineBinding(const BlockRealize& realize, const Map<Var, Range>& loop_var_ranges,
                     arith::Analyzer* analyzer) {
  if (loop_var_ranges.empty()) {
    return true;
  }
  Array<arith::IterSumExpr> results = arith::DetectIterMap(
      /*indices=*/realize->iter_values,
      /*input_iters=*/loop_var_ranges,
      /*predicate=*/realize->predicate,
      /*require_bijective=*/false,
      /*analyzer=*/analyzer);
  if (results.empty()) {
    return false;
  }
  for (const arith::IterSumExpr& sum_expr : results) {
    const Array<arith::IterSplitExpr>& args = sum_expr->args;
    if (!args.empty() && !is_one(args[0]->scale)) {
      return false;
    }
  }
  return true;
}

Map<Var, Range> LoopDomainOfSRefTreePath(const StmtSRef& low_inclusive,
                                         const Optional<StmtSRef>& high_exclusive,
                                         const runtime::StorageScope& extra_relax_scope) {
  Map<Var, Range> result;
  const StmtSRefNode* p = low_inclusive.get();
  const StmtSRefNode* limit = static_cast<const StmtSRefNode*>(high_exclusive.get());
  for (; p != limit; p = p->parent) {
    const ForNode* loop = p->StmtAs<ForNode>();
    if (loop == nullptr) {
      break;
    }
    result.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
  }
  if (extra_relax_scope.rank != runtime::StorageRank::kGlobal) {
    for (; p; p = p->parent) {
      if (const ForNode* loop = p->StmtAs<ForNode>()) {
        if (loop->kind == ForKind::kThreadBinding) {
          const String& thread_tag = loop->thread_binding.value()->thread_tag;
          if (CanRelaxStorageUndereThread(extra_relax_scope,
                                          runtime::ThreadScope::Create(thread_tag))) {
            result.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
          }
        }
      }
    }
  }
  return result;
}

void CheckAffineBinding(const ScheduleState& self, Block block) {
  class NotAffineBindingError : public ScheduleError {
   public:
    explicit NotAffineBindingError(IRModule mod, Block block)
        : mod_(std::move(mod)), block_(std::move(block)) {}
    String FastErrorString() const final {
      return "ScheduleError: The block is required to have an affine binding";
    }
    String DetailRenderTemplate() const final {
      return "The block {0} is required to have an affine binding";
    }
    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    Block block_;
  };

  if (!self->IsAffineBlockBinding(self->stmt2ref.at(block.get()))) {
    throw NotAffineBindingError(self->mod, std::move(block));
  }
}

Map<Var, PrimExpr> GetBindings(const BlockRealize& realize) {
  const BlockNode* block = realize->block.get();
  const Array<IterVar>& all_lhs = block->iter_vars;
  const Array<PrimExpr>& all_rhs = realize->iter_values;
  ICHECK_EQ(all_lhs.size(), all_rhs.size());
  Map<Var, PrimExpr> result;
  for (int i = 0, n = all_lhs.size(); i < n; ++i) {
    const IterVar& lhs = all_lhs[i];
    const PrimExpr& rhs = all_rhs[i];
    result.Set(lhs->var, rhs);
  }
  return result;
}

bool GetVarsTouchedByBlockIters(const BlockRealize& block_realize,
                                std::unordered_set<const VarNode*>* data_par_vars,
                                std::unordered_set<const VarNode*>* reduce_vars) {
  Block block = block_realize->block;
  ICHECK(block_realize->block.same_as(block))
      << "ValueError: The input `block_realize` is required to be the exact BlockRealize of the "
         "input block";

  bool has_block_vars_of_other_types = false;
  ICHECK_EQ(block->iter_vars.size(), block_realize->iter_values.size());
  int n = static_cast<int>(block->iter_vars.size());
  for (int i = 0; i < n; ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    const PrimExpr& iter_value = block_realize->iter_values[i];
    std::unordered_set<const VarNode*>* set = nullptr;
    if (iter_var->iter_type == IterVarType::kDataPar) {
      set = data_par_vars;
    } else if (iter_var->iter_type == IterVarType::kCommReduce) {
      set = reduce_vars;
    } else {
      has_block_vars_of_other_types = true;
    }

    Array<Var> vars_in_binding = UndefinedVars(iter_value);
    for (const Var& var : vars_in_binding) {
      set->insert(var.get());
    }
  }

  return has_block_vars_of_other_types;
}

/******** Block-loop relation ********/

Array<StmtSRef> GetChildBlockSRefOnSRefTree(const ScheduleState& self,
                                            const StmtSRef& parent_sref) {
  Array<BlockRealize> child_block_realize = GetChildBlockRealizeOnSRefTree(parent_sref);
  Array<StmtSRef> child_block_srefs;
  child_block_srefs.reserve(child_block_realize.size());

  for (BlockRealize realize : child_block_realize) {
    child_block_srefs.push_back(self->stmt2ref.at(realize->block.get()));
  }
  return child_block_srefs;
}

Array<BlockRealize> GetChildBlockRealizeOnSRefTree(const StmtSRef& parent_sref) {
  struct Collector : public StmtVisitor {
    static Array<BlockRealize> Collect(const Stmt& stmt) {
      Collector collector;
      collector(stmt);
      return std::move(collector.result_);
    }

    void VisitStmt_(const BlockRealizeNode* block_realize) final {
      result_.push_back(GetRef<BlockRealize>(block_realize));
    }

    Array<BlockRealize> result_;
  };

  if (parent_sref->stmt->IsInstance<ForNode>()) {
    const auto* loop = static_cast<const ForNode*>(parent_sref->stmt);
    return Collector::Collect(loop->body);
  } else if (parent_sref->stmt->IsInstance<BlockNode>()) {
    const auto* block = static_cast<const BlockNode*>(parent_sref->stmt);
    return Collector::Collect(block->body);
  }
  ICHECK(false) << "Unreachable";
  throw;
}

BlockRealize CheckGetSingleChildBlockRealizeOnSRefTree(const ScheduleState& self,
                                                       const StmtSRef& parent_sref) {
  class NonSingleChildBlockError : public ScheduleError {
   public:
    explicit NonSingleChildBlockError(IRModule mod, const StmtSRef& sref)
        : mod_(std::move(mod)), stmt_(GetRef<Stmt>(sref->stmt)) {
      sref_type_ = stmt_.as<BlockNode>() != nullptr ? "block" : "loop";
    }

    String FastErrorString() const final {
      std::ostringstream os;
      os << "ScheduleError: The " << sref_type_ << " is required to have only one child block";
      return os.str();
    }

    String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "The " << sref_type_ << " {0} is required to have only one child block";
      return os.str();
    }

    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {stmt_}; }

    IRModule mod_;
    Stmt stmt_;
    String sref_type_;
  };

  Array<BlockRealize> child_block_realize = GetChildBlockRealizeOnSRefTree(parent_sref);
  if (child_block_realize.size() != 1) {
    throw NonSingleChildBlockError(self->mod, parent_sref);
  }
  return child_block_realize[0];
}

BlockRealize GetBlockRealize(const ScheduleState& self, const StmtSRef& block_sref) {
  struct BlockRealizeFinder : public StmtVisitor {
    explicit BlockRealizeFinder(const BlockNode* target_block)
        : target_block(target_block), result(nullptr) {}

    void VisitStmt(const Stmt& stmt) final {
      if (result != nullptr) {
        return;
      }
      StmtVisitor::VisitStmt(stmt);
    }

    void VisitStmt_(const BlockRealizeNode* block_realize) final {
      if (block_realize->block.get() == target_block) {
        result = block_realize;
      }
      // No need to visit recursively, since the deeper BlockRealizes must not be the result.
    }

    const BlockNode* target_block;
    const BlockRealizeNode* result;
  };

  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  if (block_sref->parent == nullptr) {
    const PrimFuncNode* func = GetRootPrimFunc(self->mod, block, nullptr);
    return Downcast<BlockRealize>(func->body);
  } else {
    BlockRealizeFinder finder(block);
    finder(GetRef<Stmt>(block_sref->parent->stmt));
    ICHECK(finder.result != nullptr)
        << "InternalError: Cannot find the BlockRealize of block " << GetRef<Block>(block);
    return GetRef<BlockRealize>(finder.result);
  }
}

/******** Pattern Matcher ********/

/*!
 * \brief PrimExpr pattern matcher.
 *
 * It is different from the pattern matcher in arith/pattern_match.h, which is dedicated
 * for compile-time constant patterns. This pattern matcher can work on dynamic user-specific
 * patterns.
 *
 * The code below shows how to use the pattern matcher.
 *
 * \code
 *
 * Var x("x"), y("y");
 * // use PrimExpr to declare patterns, x, y are holes that can be filled with
 * PatternMatcher pattern_matcher(x + y);
 * // expr = C[i, j] + A[i, k] * B[k, j], which is the expr we want to match
 * pattern_matcher.Match(expr);
 *
 * if (pattern_matcher.Success()) {
 *   pattern_matcher.Eval(x) // C[i, j]
 *   pattern_matcher.Eval(y) // A[i, k] * B[k, j]
 * }
 *
 * \endcode
 */
class PatternMatcher : public ExprVisitor {
 public:
  explicit PatternMatcher(PrimExpr pattern) : pattern_(std::move(pattern)) {}

  void VisitExpr_(const VarNode* op) final {
    auto it = filled_map_.find(op);
    if (it == filled_map_.end()) {
      filled_map_[op] = expr_to_match_;
    } else {
      ExprDeepEqual equal;
      if (it->second.same_as(expr_to_match_) || equal(it->second, expr_to_match_)) return;
      match_success_ = false;
    }
  }

  void VisitExpr_(const LoadNode* op) final {
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

  void VisitExpr_(const LetNode* op) final {
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

  void VisitExpr_(const CallNode* op) final {
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

#define TVM_DECLARE_PATTERN_MATCHER_BIN_OP(OpName) \
  void VisitExpr_(const OpName* op) {              \
    const auto* ptr = expr_to_match_.as<OpName>(); \
    if (ptr == nullptr) {                          \
      match_success_ = false;                      \
    } else {                                       \
      PrimExpr current = expr_to_match_;           \
      expr_to_match_ = ptr->a;                     \
      VisitExpr(op->a);                            \
      expr_to_match_ = ptr->b;                     \
      VisitExpr(op->b);                            \
      std::swap(expr_to_match_, current);          \
    }                                              \
  }

  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(AddNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(SubNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(MulNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(DivNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(ModNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(FloorDivNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(FloorModNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(MinNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(MaxNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(EQNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(NENode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(LTNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(LENode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(GTNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(GENode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(AndNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(OrNode);

  void VisitExpr_(const CastNode* op) final {
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

  void VisitExpr_(const NotNode* op) final {
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

  void VisitExpr_(const SelectNode* op) final {
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

  void VisitExpr_(const RampNode* op) final {
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

  void VisitExpr_(const BroadcastNode* op) final {
    const auto* ptr = expr_to_match_.as<BroadcastNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (op->lanes != ptr->lanes) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        expr_to_match_ = ptr->value;
        VisitExpr(op->value);
        std::swap(expr_to_match_, tmp);
      }
    }
  }

  void VisitExpr_(const ShuffleNode* op) final {
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

  void VisitExpr_(const IntImmNode* op) final {
    const auto* ptr = expr_to_match_.as<IntImmNode>();
    match_success_ = ptr != nullptr && op->value == ptr->value;
  }

  void VisitExpr_(const FloatImmNode* op) final {
    const auto* ptr = expr_to_match_.as<FloatImmNode>();
    match_success_ = ptr != nullptr && op->value == ptr->value;
  }

  void VisitExpr_(const StringImmNode* op) final {
    const auto* ptr = expr_to_match_.as<StringImmNode>();
    match_success_ = ptr != nullptr && op->value == ptr->value;
  }

  void VisitExpr_(const BufferLoadNode* op) final {
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

  void Match(const PrimExpr& expr_to_match) {
    this->match_success_ = true;
    this->filled_map_.clear();
    this->expr_to_match_ = expr_to_match;
    this->operator()(pattern_);
  }

  PrimExpr Eval(const Var& var) {
    auto it = filled_map_.find(var.operator->());
    ICHECK(it != filled_map_.end()) << "Unknown pattern variable";
    ICHECK(match_success_) << "Match failed";
    return it->second;
  }

  bool Success() const { return match_success_; }

 private:
  bool match_success_{true};
  PrimExpr pattern_, expr_to_match_;
  std::unordered_map<const VarNode*, PrimExpr> filled_map_;
};

/******** Commutative Reducer ********/

bool MatchReducer(const CommReducer& reducer, const PrimExpr& identity, const PrimExpr& combiner,
                  const BufferLoad& load, PrimExpr* lhs, PrimExpr* rhs) {
  if (!ExprDeepEqual()(reducer->identity_element[0], identity)) {
    return false;
  }
  PatternMatcher pattern_matcher(reducer->result[0]);
  pattern_matcher.Match(combiner);
  if (pattern_matcher.Success()) {
    PrimExpr lhs_tmp = pattern_matcher.Eval(reducer->lhs[0]);
    PrimExpr rhs_tmp = pattern_matcher.Eval(reducer->rhs[0]);
    if (ExprDeepEqual()(load, lhs_tmp)) {
      *lhs = std::move(lhs_tmp);
      *rhs = std::move(rhs_tmp);
    }
    return true;
  }
  return false;
}

bool FromIdentityCombiner(const PrimExpr& identity, const BufferStore& combiner,
                          CommReducer* result_reducer, PrimExpr* lhs, PrimExpr* rhs) {
  BufferLoad load(combiner->buffer, combiner->indices);
  // Check reduction patterns.
  for (const TypedPackedFunc<CommReducer(DataType)>& reducer_getter : GetReducerGetters()) {
    CommReducer reducer = reducer_getter(identity.dtype());
    if (MatchReducer(reducer, identity, combiner->value, load, lhs, rhs)) {
      *result_reducer = std::move(reducer);
      return true;
    }
  }
  return false;
}

/******** Misc: not upstream-ed ********/

// TODO(Siyuan,Junru)
// std::unordered_set<const VarNode*> Vars(const ObjectRef& stmt_or_expr) {
//   std::unordered_set<const VarNode*> result;
//   auto f_visit = [&result](const ObjectRef& obj) -> void {
//     if (const auto* var = obj.as<VarNode>()) {
//       result.insert(var);
//     }
//   };
//   PostOrderVisit(stmt_or_expr, f_visit);
//   return result;
// }

bool ValidateBlockBinding(const BlockRealize& realize, const Map<Var, Range>& loop_var_ranges) {
  if (loop_var_ranges.empty()) {
    return true;
  }
  arith::Analyzer analyzer;
  Array<arith::IterSumExpr> results = arith::DetectIterMap(
      /*indices=*/realize->iter_values,
      /*input_iters=*/loop_var_ranges,
      /*predicate=*/realize->predicate,
      /*require_bijective=*/true, /*analyzer=*/&analyzer);
  if (results.empty()) {
    return false;
  }
  for (const arith::IterSumExpr& sum_expr : results) {
    const Array<arith::IterSplitExpr>& args = sum_expr->args;
    if (args.empty()) {
      continue;
    }
    if (!is_one(args[0]->scale)) {
      return false;
    }
  }
  return true;
}

bool RegionCoveredConsumer(const ScheduleState& self, const StmtSRef& consumer_block_sref,
                           const StmtSRef& scope_root) {
  if (consumer_block_sref->parent == nullptr) {
    return true;
  }
  const BlockNode* consumer_block = TVM_SREF_TO_BLOCK(consumer_block, consumer_block_sref);
  BlockScope scope = self->GetBlockScope(scope_root);
  // Step 1. Gather all the producers
  struct Producer {
    /*! \brief The block that writes the buffer */
    StmtSRef block_sref;
    /*! \brief The region the buffer is written */
    BufferRegion region;
    /*! \brief Constructor */
    explicit Producer(StmtSRef block_sref, BufferRegion region)
        : block_sref(std::move(block_sref)), region(std::move(region)) {}
  };
  // Maps a buffer var to its producers
  std::unordered_map<const VarNode*, std::vector<Producer>> buffer_producers;
  // Collect all producers to a buffer by enumerating all RAW predecessors of the consumer
  for (const Dependency& edge : scope->GetDepsByDst(consumer_block_sref)) {
    // i.e. the RAW predecessor is producer
    if (edge->kind == DepKind::kRAW) {
      const StmtSRef& producer_block_sref = edge->src;
      const BlockNode* producer_block = TVM_SREF_TO_BLOCK(producer_block, producer_block_sref);
      for (const BufferRegion& output_region : producer_block->writes) {
        const VarNode* buffer_var = output_region->buffer->data.get();
        buffer_producers[buffer_var].emplace_back(producer_block_sref, output_region);
      }
    }
  }
  // Step 2. For each buffer that the consumer reads, check the region cover property
  arith::Analyzer analyzer;
  for (const BufferRegion& consumer_region : consumer_block->reads) {
    // Step 2.1. Find the producers of the buffer
    const VarNode* buffer_var = consumer_region->buffer->data.get();
    auto it = buffer_producers.find(buffer_var);
    if (it == buffer_producers.end()) {
      continue;
    }
    const std::vector<Producer>& producers = it->second;
    // Step 2.2. Figure out LCA of consumer and all producers
    StmtSRef lca{nullptr};
    {
      std::vector<StmtSRef> inputs;
      inputs.reserve(producers.size() + 1);
      inputs.emplace_back(consumer_block_sref);
      for (const Producer& producer : producers) {
        inputs.emplace_back(producer.block_sref);
      }
      lca = LowestCommonAncestor(inputs, scope_root);
    }
    // Step 2.3. Relax the read region with the loops under LCA
    BufferRegion read = RelaxRegion(consumer_block_sref, lca, consumer_region);
    int ndim = read->region.size();
    for (const Producer& producer : producers) {
      // Relax the write region with the loops under LCA
      BufferRegion write = RelaxRegion(producer.block_sref, lca, producer.region);
      ICHECK_EQ(read->region.size(), write->region.size())
          << "ValueError: Inconsistent rank of the same buffer between reads and writes";
      // Check if the write domain covers the read domain
      for (int i = 0; i < ndim; ++i) {
        PrimExpr read_min = read->region[i]->min;
        PrimExpr read_max = read_min + read->region[i]->extent;
        PrimExpr write_min = write->region[i]->min;
        PrimExpr write_max = write_min + write->region[i]->extent;
        PrimExpr cond = (write_min <= read_min) && (read_max <= write_max);
        if (!analyzer.CanProve(cond)) {
          return false;
        }
      }
    }
  }
  return true;
}

class SRefTreeVerifier : public StmtVisitor {
 public:
  static void Verify(const ScheduleStateNode* self) { SRefTreeVerifier(self).Verify(); }

 private:
  /*! \brief Constructor */
  explicit SRefTreeVerifier(const ScheduleStateNode* self) : self_(self) {}

  void Verify() {
    VisitPrimFuncs(self_->mod, [this](const PrimFuncNode* func) { this->VisitStmt(func->body); });
    ICHECK_EQ(n_sref_visited_, static_cast<int>(self_->stmt2ref.size()));
    for (const auto& kv : self_->block_info) {
      const StmtSRef& sref = kv.first;
      ICHECK(sref->stmt != nullptr)
          << "InternalError: An expired sref is found in the block_scope mapping";
      ICHECK(self_->stmt2ref.count(sref->stmt))
          << "InternalError: The sref points to a statement that does not exist in stmt2ref";
      const StmtSRef& sref2 = self_->stmt2ref.at(sref->stmt);
      ICHECK(sref.same_as(sref2))
          << "InternalError: The sref points to a statement whose corresponding sref in stmt2ref "
             "is not the same object as itself";
    }
    ICHECK_EQ(n_block_sref_visited_, static_cast<int>(self_->block_info.size()));
  }

  void VisitStmt_(const BlockNode* block) override {
    if (init_block_depth_) {
      ICHECK(!self_->stmt2ref.count(block)) << "InternalError: A block inside init block has its "
                                               "corresponding sref, which is not allowed";
      StmtVisitor::VisitStmt_(block);
      return;
    }
    ICHECK(self_->stmt2ref.count(block))
        << "InternalError: A BlockNode should appear in sref map, but it didn't\n"
        << GetRef<Stmt>(block);
    ++n_sref_visited_;
    ++n_block_sref_visited_;
    const StmtSRef& sref = self_->stmt2ref.at(block);
    ICHECK(self_->block_info.count(sref))
        << "InternalError: Cannot find scope information of the BlockNode:\n"
        << GetRef<Stmt>(block);
    ICHECK(sref->parent == ancestors_.back())
        << "InternalError: Parent information mismatch for BlockNode:\n"
        << GetRef<Stmt>(block) << "\nIts parent is supposed to be:\n"
        << GetRef<Stmt>(ancestors_.back()->stmt) << "\nHowever, its parent is incorrect and is:\n"
        << (sref->parent ? Optional<Stmt>(GetRef<Stmt>(sref->parent->stmt))
                         : Optional<Stmt>(NullOpt));
    ancestors_.push_back(sref.operator->());
    if (block->init.defined()) {
      ++init_block_depth_;
      VisitStmt(block->init.value());
      --init_block_depth_;
    }
    VisitStmt(block->body);
    ancestors_.pop_back();
  }

  void VisitStmt_(const ForNode* loop) override {
    if (init_block_depth_) {
      ICHECK(!self_->stmt2ref.count(loop)) << "InternalError: A loop inside init block has its "
                                              "corresponding sref, which is not allowed";
      StmtVisitor::VisitStmt_(loop);
      return;
    }
    ICHECK(self_->stmt2ref.count(loop))
        << "InternalError: A ForNode should appear in sref map, but it didn't\n"
        << GetRef<Stmt>(loop);
    ++n_sref_visited_;
    const StmtSRef& sref = self_->stmt2ref.at(loop);
    Optional<Stmt> stmt = NullOpt;
    ICHECK(sref->parent == ancestors_.back())
        << "InternalError: Parent information mismatch for ForNode:\n"
        << GetRef<Stmt>(loop) << "\nIts parent is supposed to be:\n"
        << GetRef<Stmt>(ancestors_.back()->stmt) << "\nHowever, its parent is incorrect and is:\n"
        << (sref->parent ? Optional<Stmt>(GetRef<Stmt>(sref->parent->stmt))
                         : Optional<Stmt>(NullOpt));
    ancestors_.push_back(sref.operator->());
    StmtVisitor::VisitStmt_(loop);
    ancestors_.pop_back();
  }

  void VisitStmt_(const SeqStmtNode* seq_stmt) override {
    // Verify seq_index
    if (init_block_depth_) {
      StmtVisitor::VisitStmt_(seq_stmt);
      return;
    }
    int n = static_cast<int>(seq_stmt->seq.size());
    for (int i = 0; i < n; ++i) {
      const Stmt& child = seq_stmt->seq[i];
      StmtSRef sref{nullptr};
      if (const auto* realize = child.as<BlockRealizeNode>()) {
        const auto* block = realize->block.get();
        ICHECK(self_->stmt2ref.count(block));
        sref = self_->stmt2ref.at(block);
      } else if (child->IsInstance<ForNode>()) {
        ICHECK(self_->stmt2ref.count(child.get()));
        sref = self_->stmt2ref.at(child.get());
      } else {
        continue;
      }
      ICHECK_EQ(sref->seq_index, i) << "InternalError: A StmtSRef has incorrect seq_index";
    }
    StmtVisitor::VisitStmt_(seq_stmt);
  }

  /*! \brief The schedule it belongs to */
  const ScheduleStateNode* self_;
  /*! \brief Parent information during the visit */
  std::vector<const StmtSRefNode*> ancestors_ = {nullptr};
  /*! \brief If the visitor is currently in the init block */
  int init_block_depth_ = 0;
  /*! \brief Number of srefs that are visited */
  int n_sref_visited_ = 0;
  /*! \brief Number of block srefs that are visited */
  int n_block_sref_visited_ = 0;
};

class BlockInfoVerifier : public StmtVisitor {
 public:
  static void Verify(const ScheduleStateNode* self) { BlockInfoVerifier(self).Verify(); }

 private:
  /*! \brief Constructor */
  explicit BlockInfoVerifier(const ScheduleStateNode* self) : self_(self) {}

  void Verify() {
    VisitPrimFuncs(self_->mod, [this](const PrimFuncNode* func) { this->VisitStmt(func->body); });
  }

  void CheckAffineBinding(const BlockRealizeNode* realize) const {
    const auto* block = realize->block.get();
    StmtSRef block_sref = self_->stmt2ref.at(block);
    Map<Var, Range> loop_var_ranges;
    for (StmtSRefNode* loop_sref = block_sref->parent; loop_sref != nullptr;
         loop_sref = loop_sref->parent) {
      if (const auto* loop = loop_sref->StmtAs<ForNode>()) {
        loop_var_ranges.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
      } else {
        break;
      }
    }
    bool affine_binding = ValidateBlockBinding(GetRef<BlockRealize>(realize), loop_var_ranges);
    ICHECK_EQ(affine_binding, self_->IsAffineBlockBinding(block_sref))
        << "Block: " << realize->block->name_hint << "\n"
        << Repr(self_->mod) << "\n"
        << loop_var_ranges;
  }

  void VisitStmt_(const BlockRealizeNode* realize) override {
    this->VisitStmt(realize->block->body);
    CheckAffineBinding(realize);
  }
  /*! \brief The schedule it belongs to */
  const ScheduleStateNode* self_;
};

void VerifyBlockInfo(const ScheduleState& self) { BlockInfoVerifier::Verify(self.get()); }

bool IsCompactDataFlow(const ScheduleState& self, const StmtSRef& scope_root,
                       const Array<StmtSRef>& child_blocks) {
  for (const StmtSRef& block : child_blocks) {
    if (!IsCompleteBlock(self, block, scope_root) && !IsReductionBlock(self, block, scope_root)) {
      return false;
    }
  }
  return true;
}

/*!
 * \brief Check if each reduction instance is valid. Particularly, check:
 * 1) Each iteration variable is either data parallel or reduction
 * 2) Indices used to access the output buffer are not related to or affected by reduction iteration
 * variables.
 * \param iter_vars Iteration variables of the reduction
 * \param output_buffer_indices Indices used to access the output buffer
 * \return A boolean indicating if the reduction instance is valid
 */
bool CheckReductionInstance(const Array<IterVar>& iter_vars,
                            const Array<PrimExpr>& output_buffer_indices) {
  std::unordered_set<const VarNode*> reduction_block_vars;
  reduction_block_vars.reserve(iter_vars.size());
  // Check 1. Each iter_var can only be data parallel or reduction
  for (const IterVar& iter_var : iter_vars) {
    IterVarType kind = iter_var->iter_type;
    if (kind != kDataPar && kind != kCommReduce) {
      return false;
    }
    if (kind == kCommReduce) {
      reduction_block_vars.insert(iter_var->var.get());
    }
  }
  // Check 2. Each reduction iter_var should not be used to index output buffer
  for (const PrimExpr& idx : output_buffer_indices) {
    if (UsesVar(idx, [&](const VarNode* node) { return reduction_block_vars.count(node); })) {
      return false;
    }
  }
  return true;
}

bool CanMergeReduction(const ScheduleState& self, const StmtSRef& init_block_sref,
                       const StmtSRef& update_block_sref, const StmtSRef& scope_root) {
  BlockScope scope = self->GetBlockScope(scope_root);
  const BlockNode* init = TVM_SREF_TO_BLOCK(init, init_block_sref);
  const BlockNode* update = TVM_SREF_TO_BLOCK(update, update_block_sref);
  // Cond 1. Check the binding of update block is valid
  if (!self->IsAffineBlockBinding(update_block_sref)) {
    return false;
  }
  // Cond 2. Check init_block and update_block are the only two producers for their output buffer
  for (const BufferRegion& write_region : update->writes) {
    const Array<StmtSRef>& writers = scope->buffer_writers.at(write_region->buffer);
    if (writers.size() != 2) {
      return false;
    }
    if (!writers[0].same_as(init_block_sref) && !writers[0].same_as(update_block_sref)) {
      return false;
    }
    if (!writers[1].same_as(init_block_sref) && !writers[1].same_as(update_block_sref)) {
      return false;
    }
  }
  // Cond 3. init and update share the same buffer
  const auto* init_body = TVM_TYPE_AS(init_body, init->body, BufferStoreNode);
  const auto* update_body = TVM_TYPE_AS(update_body, update->body, BufferStoreNode);
  if (!init_body->buffer.same_as(update_body->buffer)) {
    return false;
  }
  // Access must be the same dimensional
  ICHECK_EQ(init_body->indices.size(), update_body->indices.size())
      << "InternalError: indexing to the same buffer with different dimensions";
  // Cond 4. All block vars of update_block are either data parallel or reduction,
  // and reduction vars of update_block should not affect indexing the output buffer
  return CheckReductionInstance(update->iter_vars, update_body->indices);
}

bool HasSingleChild(const StmtSRef& loop_or_block_sref) {
  const StmtNode* body = nullptr;
  if (const auto* loop = loop_or_block_sref->StmtAs<ForNode>()) {
    body = loop->body.get();
  } else if (const auto* block = loop_or_block_sref->StmtAs<BlockNode>()) {
    body = block->body.get();
  } else {
    LOG(FATAL) << "TypeError: Unable to recognize the type of `loop_or_block_sref`: "
               << loop_or_block_sref->stmt->GetTypeKey();
  }
  if (body->IsInstance<SeqStmtNode>()) {
    const auto* seq_stmt = static_cast<const SeqStmtNode*>(body);
    return seq_stmt->seq.size() == 1;
  }
  return true;
}

IterVarType GetLoopIterType(const ScheduleState& self, const StmtSRef& loop_sref) {
  int n_spatial = 0;
  int n_reduce = 0;
  int n_other = 0;
  const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
  const Var& loop_var = loop->loop_var;
  auto f_visit = [&loop_var, &n_spatial, &n_reduce, &n_other](const ObjectRef& obj) -> bool {
    if (const auto* realize = obj.as<BlockRealizeNode>()) {
      const BlockNode* block = realize->block.get();
      // Number of block vars and their bindings
      ICHECK_EQ(realize->iter_values.size(), block->iter_vars.size());
      int n = realize->iter_values.size();
      for (int i = 0; i < n; ++i) {
        const IterVar& iter_var = block->iter_vars[i];
        const PrimExpr& binding = realize->iter_values[i];
        // Categorize the current block var
        int* ref = nullptr;
        if (iter_var->iter_type == IterVarType::kDataPar) {
          ref = &n_spatial;
        } else if (iter_var->iter_type == IterVarType::kCommReduce) {
          ref = &n_reduce;
        } else {
          ref = &n_other;
        }
        // Visit the binding to see if `loop_var` appears
        PostOrderVisit(binding, [&ref, &loop_var](const ObjectRef& obj) -> void {
          if (obj.same_as(loop_var)) {
            (*ref) += 1;
          }
        });
      }
      return false;
    }
    return true;
  };
  PreOrderVisit(loop->body, f_visit);
  if (n_other) {
    return IterVarType::kOpaque;
  } else if (n_spatial && n_reduce) {
    return IterVarType::kOpaque;
  } else if (n_reduce) {
    return IterVarType::kCommReduce;
  } else if (loop->kind == ForKind::kUnrolled) {
    return IterVarType::kUnrolled;
  } else if (loop->kind == ForKind::kVectorized) {
    return IterVarType::kVectorized;
  } else if (loop->kind == ForKind::kParallel) {
    return IterVarType::kParallelized;
  }
  return IterVarType::kDataPar;
}

Array<StmtSRef> CollectComputeLocation(const ScheduleState& self, const StmtSRef& block_sref) {
  Array<StmtSRef> loop_srefs = GetLoops(block_sref);
  Array<StmtSRef> result;
  result.reserve(loop_srefs.size());
  bool visited_reduce = false;
  for (const StmtSRef& loop_sref : loop_srefs) {
    const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
    IterVarType iter_type = GetLoopIterType(self, loop_sref);
    if (iter_type == IterVarType::kDataPar) {
      if (visited_reduce) {
        break;
      }
    } else {
      visited_reduce = true;
    }
    result.push_back(loop_sref);
  }
  return result;
}

StmtSRef GetSRefTreeRoot(const StmtSRef& sref) {
  const StmtSRefNode* p = sref.get();
  for (; p->parent != nullptr; p = p->parent) {
  }
  return GetRef<StmtSRef>(p);
}

const PrimFuncNode* GetRootPrimFunc(const ScheduleState& self, const StmtSRef& sref) {
  const StmtSRefNode* p = sref.get();
  for (; p->parent != nullptr; p = p->parent) {
  }
  for (const auto& kv : self->mod->functions) {
    const BaseFunc& base_func = kv.second;
    if (const auto* func = base_func.as<PrimFuncNode>()) {
      if (const auto* realize = func->body.as<BlockRealizeNode>()) {
        if (realize->block.get() == p->stmt) {
          return func;
        }
      }
    }
  }
  LOG(FATAL) << "IndexError: Could not get the correpsonding function in the schedule state of the "
                "statement:\n"
             << GetRef<Stmt>(sref->stmt);
  throw;
}

}  // namespace tir
}  // namespace tvm
