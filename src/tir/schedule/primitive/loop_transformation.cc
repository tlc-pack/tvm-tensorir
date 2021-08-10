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

/*! \brief Append a new predicate to the each child of type BlockRealize (not recursively) */
class BlockPredicateAppender : public StmtMutator {
 public:
  /*!
   * \brief Constructor
   * \param to_append The predicate to be appended to BlockRealizeNode
   */
  explicit BlockPredicateAppender(const PrimExpr& to_append) : to_append_(to_append) {}

 private:
  // For each direct child of type BlockRealizeNode, append the predicate
  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    // We do not recursively do this
    ObjectPtr<BlockRealizeNode> n = CopyOnWrite(realize);
    n->predicate = n->predicate && to_append_;
    return BlockRealize(n);
  }

  /*! \brief The predicate to be appended */
  const PrimExpr& to_append_;
};

/*! \brief Substitute vars and collect the reuse mapping of opaque blocks */
class SubstituteVarAndCollectOpaqueBlock : public StmtExprMutator {
 public:
  explicit SubstituteVarAndCollectOpaqueBlock(std::function<Optional<PrimExpr>(const Var&)> vmap,
                                              Map<Block, Block>* opaque_blocks)
      : vmap_(vmap), opaque_blocks_(opaque_blocks) {}

 private:
  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    if (Optional<PrimExpr> ret = vmap_(var)) {
      return ret.value();
    } else {
      return std::move(var);
    }
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    BlockRealize realize = Downcast<BlockRealize>(StmtMutator::VisitStmt_(op));
    if (realize->block->iter_vars.empty()) {
      opaque_blocks_->Set(op->block, realize->block);
    }
    return std::move(realize);
  }

  /*! \brief The substitute function */
  std::function<Optional<PrimExpr>(const Var&)> vmap_;
  /*! \brief The reuse mapping of opaque blocks */
  Map<Block, Block>* opaque_blocks_;
};

/*! \brief Simplify the binding of block realize and update the opaque block reuse mapping */
class IterMapSimplifyBlockBinding : public StmtExprMutator {
 public:
  explicit IterMapSimplifyBlockBinding(MapNode* opaque_blocks, Map<Var, Range> loop_var2extent)
      : opaque_blocks_(opaque_blocks), loop_var2extent_(loop_var2extent) {}

  static For SimplifyBindings(Stmt stmt, const Array<StmtSRef>& loop_srefs,
                              MapNode* opaque_blocks) {
    Map<Var, Range> loop_var2extent;
    for (const StmtSRef& sref : loop_srefs) {
      const ForNode* loop = TVM_SREF_TO_FOR(loop, sref);
      loop_var2extent.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    }
    return Downcast<For>(
        IterMapSimplifyBlockBinding(opaque_blocks, std::move(loop_var2extent))(std::move(stmt)));
  }

 private:
  Stmt VisitStmt_(const ForNode* op) final {
    loop_var2extent_.Set(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    Stmt res = StmtMutator::VisitStmt_(op);
    loop_var2extent_.erase(op->loop_var);
    return res;
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    // skip opaque block and update mapping
    if (op->iter_values.empty()) {
      Block block = op->block;
      BlockRealize realize = Downcast<BlockRealize>(StmtMutator::VisitStmt_(op));
      for (const std::pair<ObjectRef, ObjectRef>& entry : *opaque_blocks_) {
        if (entry.second.same_as(block)) {
          opaque_blocks_->at(entry.first) = realize->block;
          break;
        }
      }
      return std::move(realize);
    }
    Array<PrimExpr> v = arith::IterMapSimplify(/*indices=*/op->iter_values,
                                               /*input_iters=*/loop_var2extent_,
                                               /*input_pred=*/op->predicate,
                                               /*require_bijective=*/false);
    if (v.same_as(op->iter_values)) {
      return GetRef<Stmt>(op);
    } else {
      ObjectPtr<BlockRealizeNode> n = CopyOnWrite(op);
      n->iter_values = std::move(v);
      return Stmt(n);
    }
  }

  /*! \brief The reuse mapping */
  MapNode* opaque_blocks_;
  /*! \brief The range of loops */
  Map<Var, Range> loop_var2extent_;
};

std::vector<const StmtSRefNode*> GetLoopsPostOrder(const ScheduleState self,
                                                   const StmtSRef& root_sref) {
  std::vector<const StmtSRefNode*> loops;
  // Gather all the loops under parent_block
  PreOrderVisit(root_sref->StmtAs<BlockNode>()->body, [&loops, self](const ObjectRef& node) {
    // Stops at a new BlockNode
    if (node->IsInstance<BlockNode>()) {
      return false;
    }
    // Collects every LoopNode
    if (const auto* loop = node.as<ForNode>()) {
      loops.push_back(self->stmt2ref.at(loop).operator->());
    }
    return true;
  });
  // Reverse to get bottom-up order
  std::reverse(loops.begin(), loops.end());
  return loops;
}

class HasAnnotationOrThreadBindingError : public ScheduleError {
 public:
  explicit HasAnnotationOrThreadBindingError(IRModule mod, For loop)
      : mod_(mod), loop_(std::move(loop)) {}

  String FastErrorString() const final {
    return "ScheduleError: The primitive can't be applied because the loop has annotation or "
           "thread binding";
  }

  String DetailRenderTemplate() const final {
    return "The primitive can't be applied because the loop {0} has annotation or thread binding";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
};

class OuterNotInnerParent : public ScheduleError {
 public:
  explicit OuterNotInnerParent(IRModule mod, For outer, For inner)
      : mod_(mod), outer_(std::move(outer)), inner_(std::move(inner)) {}

  String FastErrorString() const final {
    return "ScheduleError: The outer loop is not the parent of the inner loop";
  }

  String DetailRenderTemplate() const final {
    return "The loops can't be fused because the outer loop {0} is not the parent of the inner "
           "loop {1}";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {outer_, inner_}; }

  IRModule mod_;
  For outer_;
  For inner_;
};

class NotOnlyChildError : public ScheduleError {
 public:
  explicit NotOnlyChildError(IRModule mod, For outer, For inner)
      : mod_(mod), outer_(std::move(outer)), inner_(std::move(inner)) {}

  String FastErrorString() const final {
    return "ScheduleError: The inner loop is not the only child of outer loop";
  }

  String DetailRenderTemplate() const final {
    return "The loops can't be fused because the inner loop {1} is not the only child of outer "
           "loop {0}.";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {outer_, inner_}; }

  IRModule mod_;
  For outer_;
  For inner_;
};

class LoopNotStartWithZeroError : public ScheduleError {
 public:
  explicit LoopNotStartWithZeroError(IRModule mod, For loop) : mod_(mod), loop_(std::move(loop)) {}

  String FastErrorString() const final {
    return "ScheduleError: The primitive only supports loop starting with 0";
  }

  String DetailRenderTemplate() const final {
    return "The loop {0} does not start with 0, which is not supported";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
};

class NotSingleInferFactorError : public ScheduleError {
 public:
  explicit NotSingleInferFactorError(IRModule mod) : mod_(mod) {}

  String FastErrorString() const final {
    return "ScheduleError: only one factor can be specified as -1 or none";
  }

  String DetailRenderTemplate() const final {
    return "Only one factor can be specified as -1 or none";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

  IRModule mod_;
};

class WrongFactorProductError : public ScheduleError {
 public:
  explicit WrongFactorProductError(IRModule mod, For loop) : mod_(mod), loop_(std::move(loop)) {}

  String FastErrorString() const final {
    return "ScheduleError: The product of factors is not larger than or equal to the extent of "
           "loop";
  }

  String DetailRenderTemplate() const final {
    return "The product of factors is not larger than or equal to the extent of loop {0}";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
};

Array<StmtSRef> Split(ScheduleState self, const StmtSRef& loop_sref,
                      const Array<PrimExpr>& factors) {
  // Invariance
  // - The total repeat number has not changed for each direct child block with updating predicate.
  // - The execution order has not changed. (The block executes with the same args and the same
  // order with before.
  // Step 1. Check correctness
  const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
  if (!loop->annotations.empty() || loop->thread_binding.defined()) {
    throw HasAnnotationOrThreadBindingError(self->mod, GetRef<For>(loop));
  }
  // Currently, loops not starting with 0 are not supported
  arith::Analyzer analyzer;
  if (!analyzer.CanProve(loop->min == 0)) {
    throw LoopNotStartWithZeroError(self->mod, GetRef<For>(loop));
  }
  // Step 2. Replace all occurrences of the original loop var with new variables
  int n = factors.size();
  PrimExpr substitute_value = 0;
  std::vector<Var> new_loop_vars;
  new_loop_vars.reserve(n);
  for (int i = 0; i < n; i++) {
    const PrimExpr& factor = factors[i];
    Var var = loop->loop_var.copy_with_suffix("_" + std::to_string(i));
    substitute_value = substitute_value * factor + var;
    analyzer.Bind(var, Range::FromMinExtent(0, factor));
    new_loop_vars.emplace_back(std::move(var));
  }
  Map<Block, Block> opaque_block_reuse;
  Stmt new_stmt = loop->body;
  new_stmt = SubstituteVarAndCollectOpaqueBlock(
      [&](const Var& v) -> Optional<PrimExpr> {
        if (v.same_as(loop->loop_var)) {
          return substitute_value;
        } else {
          return NullOpt;
        }
      },
      &opaque_block_reuse)(std::move(new_stmt));
  // Step 3. Update predicate to guard the loop
  PrimExpr predicate = substitute_value < loop->extent;
  if (!analyzer.CanProve(predicate)) {
    new_stmt = BlockPredicateAppender(/*predicate=*/predicate)(std::move(new_stmt));
  }
  // Step 4. Generate nested loops to replace the original loop and simplify the binding
  for (int i = n - 1; i >= 0; i--) {
    new_stmt = For(new_loop_vars[i], 0, factors[i], ForKind::kSerial, new_stmt);
  }
  new_stmt = IterMapSimplifyBlockBinding::SimplifyBindings(std::move(new_stmt), GetLoops(loop_sref),
                                                           opaque_block_reuse.CopyOnWrite());
  self->Replace(loop_sref, new_stmt, opaque_block_reuse);
  Array<StmtSRef> result_srefs;
  result_srefs.reserve(n);
  for (int i = 0; i < n; i++) {
    result_srefs.push_back(self->stmt2ref.at(new_stmt.get()));
    const ForNode* outer_loop = TVM_TYPE_AS(outer_loop, new_stmt, ForNode);
    new_stmt = outer_loop->body;
  }
  return result_srefs;
}

StmtSRef Fuse(ScheduleState self, const Array<StmtSRef>& loop_srefs) {
  // Invariance
  // - The total repeat number has not changed for each direct child block.
  // - The execution order has not changed. (The block executes with the same
  //   args and the same order with before.)
  std::vector<const ForNode*> loops;
  loops.reserve(loop_srefs.size());
  StmtSRef outer_loop_sref{nullptr};
  const ForNode* outer_loop = nullptr;
  arith::Analyzer analyzer;
  // Step 1. check correctness
  for (const StmtSRef& sref : loop_srefs) {
    const ForNode* loop = TVM_SREF_TO_FOR(loop, sref);
    if (!loop->annotations.empty() || loop->thread_binding.defined()) {
      throw HasAnnotationOrThreadBindingError(self->mod, GetRef<For>(loop));
    }
    if (outer_loop_sref.defined()) {
      if (sref->parent != outer_loop_sref.get()) {
        throw OuterNotInnerParent(self->mod, GetRef<For>(outer_loop), GetRef<For>(loop));
      }
      if (!outer_loop->body.same_as(GetRef<For>(loop))) {
        throw NotOnlyChildError(self->mod, GetRef<For>(outer_loop), GetRef<For>(loop));
      }
    }
    outer_loop_sref = sref;
    outer_loop = loop;
    if (!analyzer.CanProve(loop->min == 0)) {
      throw LoopNotStartWithZeroError(self->mod, GetRef<For>(loop));
    }
    loops.push_back(loop);
  }
  // Step 2. Create fused loop var and replace the original loop vars
  std::string suffix;
  int n = loops.size();
  for (int i = 1; i < n; i++) {
    suffix += "_" + loops[i]->loop_var->name_hint;
  }
  suffix += "_fused";
  Var fused_var = loops[0]->loop_var.copy_with_suffix(suffix);
  Array<PrimExpr> substitute_value;
  substitute_value.resize(loops.size());
  PrimExpr tot = fused_var;
  for (int i = static_cast<int>(loops.size()) - 1; i >= 0; i--) {
    substitute_value.Set(i, floormod(tot, loops[i]->extent));
    tot = floordiv(tot, loops[i]->extent);
  }
  Stmt new_stmt = loops.back()->body;
  Map<Block, Block> opaque_block_reuse;
  auto f_substitute = [&](const Var& v) -> Optional<PrimExpr> {
    for (int i = 0; i < n; i++) {
      if (v.same_as(loops[i]->loop_var)) {
        return substitute_value[i];
      }
    }
    return NullOpt;
  };
  new_stmt =
      SubstituteVarAndCollectOpaqueBlock(f_substitute, &opaque_block_reuse)(std::move(new_stmt));
  // Step 3. Generate a loop to replace the original loops
  PrimExpr fused_extent = 1;
  for (int i = 0; i < n; i++) {
    fused_extent *= loops[i]->extent;
  }
  fused_extent = analyzer.Simplify(fused_extent);
  new_stmt = For(fused_var, 0, fused_extent, ForKind::kSerial, new_stmt);
  new_stmt = IterMapSimplifyBlockBinding::SimplifyBindings(
      std::move(new_stmt), GetLoops(loop_srefs[0]), opaque_block_reuse.CopyOnWrite());
  self->Replace(loop_srefs[0], new_stmt, opaque_block_reuse);
  return self->stmt2ref.at(new_stmt.get());
}

void Reorder(ScheduleState self, const Array<StmtSRef>& order) {
  /*
   * Check:
   * - check loops are in the same line and are single-branch
   * - the block below has all its block_var to be data_par or reduce.
   * Mutate:
   * - reorder the loops
   */
  CHECK(!order.empty()) << "ValueError: 'reorder' expects 'order' to be an non-empty list";
  // Check 1. type checks and uniqueness check
  std::unordered_set<const StmtSRefNode*> loops;
  for (const StmtSRef& loop_sref : order) {
    // type check
    const auto* loop = loop_sref->StmtAs<ForNode>();
    CHECK(loop) << "TypeError: 'reorder' expects an array of loops, but get type: "
    << loop_sref->stmt->GetTypeKey();
    // uniqueness check
    const StmtSRefNode* loop_sref_ptr = loop_sref.operator->();
    CHECK_EQ(loops.count(loop_sref_ptr), 0U)
    << "ValueError: 'reorder' expects an array of unique array, but get duplicate: "
    << GetRef<Stmt>(loop_sref->stmt);
    loops.insert(loop_sref_ptr);
  }
  // Check 2. Loops are in the same line
  // The algorithm now is to scan the inverse DFS order of the whole loop tree in the scope.
  // For some Loop x, it is potentially in the reorder range if
  //   - x is in the reorder list
  //   - exactly 1 son y of x is potentially in the reorder range
  //     (If there are more, then the loops are not in the same line).
  //     Put (x, y) in the map.
  // If x is potentially in the reorder range, check x is single branch
  // After the inverse DFS, we can know how to catch the loop line by the map.
  // Top and bottom denotes the range of loops need reordering
  const StmtSRefNode* top = nullptr;
  const StmtSRefNode* bottom = nullptr;
  // Maps a parent to its child
  std::unordered_map<const StmtSRefNode*, const StmtSRefNode*> successor;
  // Gather all the loops under parent_block
  int n_loops_not_found = order.size();
  for (const StmtSRefNode* loop : GetLoopsPostOrder(self, GetScopeRoot(order[0]).value())) {
    bool is_in_reorder_list = loops.count(loop);
    bool has_inner_loop = successor.count(loop);
    if (is_in_reorder_list || has_inner_loop) {
      const StmtSRefNode* parent = loop->parent;
      // If the successor of `parent` exists, then it is not the current loop
      CHECK(!successor.count(parent))
      << "ValueError: 'reorder' expects the loops be in the same line";
      successor[parent] = loop;
      // `bottom` is the first loop encountered
      if (bottom == nullptr) {
        bottom = loop;
      }
      // `top` is the last loop encountered
      if (is_in_reorder_list) {
        top = loop;
        --n_loops_not_found;
      }
    }
  }
  // Check 3. Loops are in the same block scope
  CHECK_EQ(n_loops_not_found, 0)
  << "ValueError: 'reorder' expects loops to be under the same block scope";
  // Check 4. Loops are single-branch
  const BlockNode* block = nullptr;
  for (const StmtSRefNode* loop = top; !(block = loop->StmtAs<BlockNode>());) {
    Array<Stmt> children = GetChildren(GetRef<Stmt>(loop->stmt));
    CHECK_EQ(children.size(), 1) << "ValueError: 'reorder' expects the loops to be single-branch";
    loop = self->stmt2ref.at(children[0].get()).operator->();
  }
  // Check 5. the block below has all its block_var to be data_par or reduce
  for (const IterVar& iter_var : block->iter_vars) {
    IterVarType kind = iter_var->iter_type;
    // TODO(@junrushao1994): remove kThreadIndex
    CHECK(kind == kDataPar || kind == kCommReduce || kind == kThreadIndex)
    << "ValueError: 'reorder' expects block var to be data parallel or reduction";
  }
  std::function<Stmt(const StmtSRefNode*, int index)> f_reorder =
      [&bottom, &loops, &successor, &order, &f_reorder](const StmtSRefNode* loop,
          int index) -> Stmt {
    // The order list maybe incomplete, so we may copy the old_loop rather than order
    const ForNode* copy =
        loops.count(loop) ? order[index++]->StmtAs<ForNode>() : loop->StmtAs<ForNode>();
    ObjectPtr<ForNode> n = make_object<ForNode>(*copy);
    if (loop == bottom) {
      // bottom loop
      n->body = loop->StmtAs<ForNode>()->body;
    } else {
      // reorder recursively
      n->body = f_reorder(successor.at(loop), index);
    }
    return Stmt(n);
  };
  self->Replace(GetRef<StmtSRef>(top), f_reorder(top, 0), {});
}

struct FuseTraits : public UnpackedInstTraits<FuseTraits> {
  static constexpr const char* kName = "Fuse";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  template <size_t delta>
  static TVM_ALWAYS_INLINE void _SetInputs(const runtime::TVMArgsSetter& setter,
                                           const Array<ObjectRef>& inputs) {
    setter(delta, inputs);
  }

  static LoopRV UnpackedApplyToSchedule(Schedule sch, Array<LoopRV> loop_rvs) {
    return sch->Fuse(loop_rvs);
  }

  static String UnpackedAsPython(Array<String> outputs, Array<String> loop_rvs) {
    PythonAPICall py("fuse");
    for (const String& loop_rv : loop_rvs) {
      py.Input("", loop_rv);
    }
    py.SingleOutput(outputs);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

struct SplitTraits : public UnpackedInstTraits<SplitTraits> {
  static constexpr const char* kName = "Split";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  template <size_t delta>
  static TVM_ALWAYS_INLINE void _SetInputs(const runtime::TVMArgsSetter& setter,
                                           const Array<ObjectRef>& inputs) {
    thread_local ObjectRef loop_rv{nullptr};
    thread_local Array<ObjectRef> factors{nullptr};
    loop_rv = inputs[0];
    factors = Array<ObjectRef>{inputs.begin() + 1, inputs.end()};
    setter(delta, loop_rv);
    setter(delta + 1, factors);
  }

  static Array<LoopRV> UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv,
                                               Array<Optional<ExprRV>> factors) {
    return sch->Split(loop_rv, factors);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv, Array<ObjectRef> factors) {
    PythonAPICall py("split");
    py.Input("loop", loop_rv);
    py.Input("factors", factors);
    py.OutputList(outputs);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

struct ReorderTraits : public UnpackedInstTraits<ReorderTraits> {
  static constexpr const char* kName = "Reorder";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  template <size_t delta>
  static TVM_ALWAYS_INLINE void _SetInputs(const runtime::TVMArgsSetter& setter,
                                           const Array<ObjectRef>& inputs) {
    setter(delta, inputs);
  }

  static void UnpackedApplyToSchedule(Schedule sch, Array<LoopRV> loop_rvs) {
    return sch->Reorder(loop_rvs);
  }

  static String UnpackedAsPython(Array<String> outputs, Array<String> loop_rvs) {
    PythonAPICall py("reorder");
    for (const String& loop_rv : loop_rvs) {
      py.Input("", loop_rv);
    }
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(FuseTraits);
TVM_REGISTER_INST_KIND_TRAITS(SplitTraits);
TVM_REGISTER_INST_KIND_TRAITS(ReorderTraits);

}  // namespace tir
}  // namespace tvm
