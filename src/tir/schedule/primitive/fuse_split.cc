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

/*! \brief Append a new predicate to the each children of type BlockRealize */
class PredicateUpdater : public StmtMutator {
 public:
  /*!
   * \brief Constructor
   * \param predicate The predicate to be apppend to BlockRealizeNode
   */
  explicit PredicateUpdater(const PrimExpr& predicate) : predicate_(predicate) {}
  // For each direct child of type BlockRealizeNode, append the predicate
  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    // We do not recursively do this
    ObjectPtr<BlockRealizeNode> n = make_object<BlockRealizeNode>(*realize);
    n->predicate = n->predicate && predicate_;
    return BlockRealize(n);
  }

 private:
  /*! \brief The predicate to be added */
  const PrimExpr& predicate_;
};

class BlockRealizeRewriter : public StmtExprMutator {
 public:
  explicit BlockRealizeRewriter(
      const std::unordered_map<Var, Range, ObjectPtrHash, ObjectPtrEqual>& loop_map) {
    loop_map_.insert(loop_map.begin(), loop_map.end());
  }

  Stmt VisitStmt_(const ForNode* op) final {
    loop_map_[op->loop_var] = Range::FromMinExtent(op->min, op->extent);
    Stmt res = StmtMutator::VisitStmt_(op);
    loop_map_.erase(op->loop_var);
    return res;
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    auto v = arith::IterMapSimplify(op->iter_values, loop_map_, op->predicate, false);
    if (v.same_as(op->iter_values)) {
      return GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->iter_values = std::move(v);
      return Stmt(n);
    }
  }

 private:
  std::unordered_map<Var, Range, ObjectPtrHash, ObjectPtrEqual> loop_map_;
};

Stmt SimplifyBindings(const Stmt& stmt, const Array<StmtSRef>& loops) {
  std::unordered_map<Var, Range, ObjectPtrHash, ObjectPtrEqual> loop_map;
  for (const auto& sref : loops) {
    const auto* loop = sref->StmtAs<ForNode>();
    loop_map[loop->loop_var] = Range::FromMinExtent(loop->min, loop->extent);
  }
  BlockRealizeRewriter rewriter(loop_map);
  return rewriter(stmt);
}

class SplitNotLoopError : public ScheduleError {
 public:
  explicit SplitNotLoopError(IRModule mod, String type) : mod_(mod), type_(type) {}

  String FastErrorString() const final { return "ScheduleError: 'split' only operates on a loop"; }

  String DetailRenderTemplate() const final {
    return "'split' only operates on a loop, but the StmtSref passed in points to"
           "type: {0} ";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {type_}; }

  IRModule mod_;
  String type_;
};

class SplitHasAnnotationError : public ScheduleError {
 public:
  explicit SplitHasAnnotationError(IRModule mod, For loop) : mod_(mod), loop_(loop) {}

  String FastErrorString() const final {
    return "ScheduleError: The loop can't be split because it has annotation";
  }

  String DetailRenderTemplate() const final {
    return "The loop {0} can't be split because it has annotation ";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
};

class FuseNotLoopError : public ScheduleError {
 public:
  explicit FuseNotLoopError(IRModule mod, String type, bool inner)
      : mod_(mod), type_(type), inner_(inner) {}

  String FastErrorString() const final { return "ScheduleError: 'fuse' only operates on loops"; }

  String DetailRenderTemplate() const final {
    if (inner_) {
      return "'fuse' only operates on loops, but the inner StmtSref passed in "
             "points to type: {0} ";
    } else {
      return "'fuse' only operates on loops, but the outer StmtSref passed in "
             "points to type: {0} ";
    }
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {type_}; }

  IRModule mod_;
  String type_;
  bool inner_;
};

class FuseHasAnnotationError : public ScheduleError {
 public:
  explicit FuseHasAnnotationError(IRModule mod, For loop, bool inner)
      : mod_(mod), loop_(loop), inner_(inner) {}

  String FastErrorString() const final {
    return "ScheduleError: The loops can't be fused because one of the loops has annotation";
  }

  String DetailRenderTemplate() const final {
    if (inner_) {
      return "The inner loop {0} can't be fused because it has annotation";
    } else {
      return "The outer loop {0} can't be fused because it has annotation";
    }
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
  bool inner_;
};

class OuterNotInnerParent : public ScheduleError {
 public:
  explicit OuterNotInnerParent(IRModule mod, For outer, For inner)
      : mod_(mod), outer_(outer), inner_(inner) {}

  String FastErrorString() const final {
    return "ScheduleError: the outer loop is not the parent of the inner loop";
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
      : mod_(mod), outer_(outer), inner_(inner) {}

  String FastErrorString() const final {
    return "ScheduleError: the inner loop is not the only child of outer loop";
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
  explicit LoopNotStartWithZeroError(IRModule mod, For loop) : mod_(mod), loop_(loop) {}

  String FastErrorString() const final {
    return "ScheduleError: the primitive only supports loop starting with 0";
  }

  String DetailRenderTemplate() const final {
    return "The loop {0} does not start with 0, which is not supported";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
};

Array<StmtSRef> Split(ScheduleState self, const StmtSRef& loop_sref, const PrimExpr& nparts,
                      const PrimExpr& factor) {
  // Invariance
  // - The total repeat number has not changed for each direct child block with updating predicate.
  // - The execution order has not changed. (The block executes with the same args and the same
  // order with before.
  const auto* loop = loop_sref->StmtAs<ForNode>();
  if (loop == nullptr) {
    throw SplitNotLoopError(self->mod, loop_sref->stmt->GetTypeKey());
  }
  if (!loop->annotations.empty()) {
    throw SplitHasAnnotationError(self->mod, GetRef<For>(loop));
  }
  // Currently, loops starting with 0 is not supported
  arith::Analyzer analyzer;
  if (!analyzer.CanProve(loop->min == 0)) {
    throw LoopNotStartWithZeroError(self->mod, GetRef<For>(loop));
  }
  // Step 1. Replace all occurrence of the original loop var with new variables
  Var outer_var = loop->loop_var.copy_with_suffix("_outer");
  Var inner_var = loop->loop_var.copy_with_suffix("_inner");
  Stmt new_loop_body = Substitute(loop->body, [&](const Var& v) -> PrimExpr {
    if (v.same_as(loop->loop_var)) {
      return outer_var * factor + inner_var;
    } else {
      return PrimExpr{nullptr};
    }
  });
  // Step 2. Update predicate to guard the loop
  PrimExpr outer_min = 0;
  PrimExpr outer_extent = nparts;
  PrimExpr inner_min = 0;
  PrimExpr inner_extent = factor;
  analyzer.Bind(outer_var, Range::FromMinExtent(outer_min, outer_extent));
  analyzer.Bind(inner_var, Range::FromMinExtent(inner_min, inner_extent));
  PrimExpr predicate = outer_var * factor + inner_var < loop->extent;
  if (!analyzer.CanProve(predicate)) {
    new_loop_body = PredicateUpdater(predicate)(new_loop_body);
  }
  // Step 3. Generate two nested loops to replace the original loop and simplify the binding
  // created by replacement in Step 1
  For inner_loop(inner_var, inner_min, inner_extent, loop->kind, new_loop_body);
  For outer_loop(outer_var, outer_min, outer_extent, loop->kind, inner_loop);
  outer_loop = Downcast<For>(SimplifyBindings(outer_loop, GetLoops(loop_sref)));
  self->Replace(loop_sref, outer_loop, {});
  return {self->stmt2ref.at(outer_loop.get()), self->stmt2ref.at(outer_loop->body.get())};
}

StmtSRef Fuse(ScheduleState self, const StmtSRef& outer_sref, const StmtSRef& inner_sref) {
  // Equivalence
  // - The total repeat number has not changed for each direct child block.
  // - The execution order has not changed. (The block executes with the same
  //   args and the same order with before.)

  // Can only fuse neighbor loop without any extra branches.
  // Future enhancement: this condition can be eliminated by lifting all siblings of inner
  // as the children of the father of outer
  const auto* outer = outer_sref->StmtAs<ForNode>();
  const auto* inner = inner_sref->StmtAs<ForNode>();
  if (outer == nullptr) {
    throw FuseNotLoopError(self->mod, outer_sref->stmt->GetTypeKey(), false);
  }
  if (inner == nullptr) {
    throw FuseNotLoopError(self->mod, inner_sref->stmt->GetTypeKey(), true);
  }
  if (!outer->annotations.empty()) {
    throw FuseHasAnnotationError(self->mod, GetRef<For>(outer), false);
  }
  if (!inner->annotations.empty()) {
    throw FuseHasAnnotationError(self->mod, GetRef<For>(inner), true);
  }
  // Step 1. Check `inner` is the only children of `outer` and they are in the same scope
  if (inner_sref->parent != outer_sref.get()) {
    throw OuterNotInnerParent(self->mod, GetRef<For>(outer), GetRef<For>(inner));
  }
  Array<Stmt> outer_children = GetChildren(GetRef<Stmt>(outer));
  if (outer_children.size() != 1 || outer_children[0].get() != inner) {
    throw NotOnlyChildError(self->mod, GetRef<For>(outer), GetRef<For>(inner));
  }
  // Step 2. Create fused loop var and replace the loop var used in inner and outer loop
  arith::Analyzer analyzer;
  if (!analyzer.CanProve(inner->min == 0)) {
    throw LoopNotStartWithZeroError(self->mod, GetRef<For>(inner));
  }
  if (!analyzer.CanProve(outer->min == 0)) {
    throw LoopNotStartWithZeroError(self->mod, GetRef<For>(outer));
  }

  Var fused_var = outer->loop_var.copy_with_suffix("_" + inner->loop_var->name_hint + "_fused");
  Stmt new_loop_body = Substitute(inner->body, [&](const Var& v) -> PrimExpr {
    if (v.same_as(outer->loop_var)) {
      return floordiv(fused_var, inner->extent) + outer->min;
    } else if (v.same_as(inner->loop_var)) {
      return floormod(fused_var, inner->extent) + inner->min;
    } else {
      return PrimExpr{nullptr};
    }
  });
  // Step 3. Generate a loop to replace the original two nested loops
  PrimExpr fused_min = 0;
  PrimExpr fused_extent = analyzer.Simplify(outer->extent * inner->extent);
  For fused_loop = For(fused_var, fused_min, fused_extent, outer->kind, new_loop_body);
  fused_loop = Downcast<For>(SimplifyBindings(fused_loop, GetLoops(outer_sref)));
  self->Replace(outer_sref, fused_loop, {});
  return self->stmt2ref.at(fused_loop.get());
}

}  // namespace tir
}  // namespace tvm
