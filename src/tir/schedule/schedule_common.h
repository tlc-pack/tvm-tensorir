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
 * \file tir/schedule/schedule_common.h
 * \brief Common utils for implementing schedule primitives
 */
#ifndef TVM_TIR_SCHEDULE_SCHEDULE_COMMON_H_
#define TVM_TIR_SCHEDULE_SCHEDULE_COMMON_H_

#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt_functor.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <algorithm>

namespace tvm {
namespace tir {

/*! Gather all direct blocks in ast subtree. */
class ChildBlockGatherer : public StmtExprVisitor {
 public:
  ChildBlockGatherer(const ScheduleNode* sch,
                     std::unordered_set<StmtSRef, ObjectHash, ObjectEqual>* child_blocks)
      : sch_(sch), child_blocks_(child_blocks) {}

  void VisitStmt_(const BlockNode* op) final {
    const auto* node = static_cast<const StmtNode*>(op);
    child_blocks_->insert(sch_->stmt2ref.at(node));
  }

 private:
  const ScheduleNode* sch_;
  std::unordered_set<StmtSRef, ObjectHash, ObjectEqual>* child_blocks_;
};

/*!
 * \brief Get the direct child Schedulable Stmt (Block and Loop)
 * \param stmt the parent stmt.
 * \param keep_realize if true, get block_realize for blocks
 * \return the list of child stmts
 */
Array<Stmt> GetChildren(const Stmt& stmt, bool keep_realize = false);

/*!
 * \brief Substitute the var in current block scope specified in key->var to be value.
 * \param stmt The source stmt to be substituted
 * \param value_func The function of new values mapping.
 * \return The converted stmt.
 */
Stmt SubstituteInScope(const Stmt& stmt,
                       const std::function<PrimExpr(const VarNode*)>& value_func);

/*!
 * \brief Substitute the var in current block scope specified in var map
 * \param stmt The source stmt to be substituted
 * \param var_map The mapping of var
 * \return The converted stmt
 */
Stmt SubstituteInScope(const Stmt& stmt,
                       const std::unordered_map<const VarNode*, const VarNode*>& var_map);

/*!
 * \brief Substitute the var in TensorRegion
 * \param tensor_region The source TensorRegion to be substituted
 * \param var_map the mapping of var
 * \return The converted tensor region
 */
TensorRegion SubstituteTensorRegion(const TensorRegion& tensor_region,
                                    const std::unordered_map<const VarNode*,
                                                             const VarNode*>& var_map);
/*!
 * \brief Get BlockRealize with by Block
 * \param block The queried block
 * \return BlockRealize.
 */
BlockRealize GetBlockRealize(const StmtSRef& block_sref);

/*!
 * \brief Get lowest common ancestor of all nodes
 * \param nodes The queried nodes
 * \param root The root of the tree / subtree
 * \return The LCA StmtSRef
 */
StmtSRef LowestCommonAncestor(const std::vector<StmtSRef>& nodes, const StmtSRef& root);

/*!
 * \brief Relax the TensorRegion with the loops under root
 * \param block_sref The block sref
 * \param root The root node
 * \param reads The vector to store the reads result
 * \param writes The vector to store the writes result
 * \note reads and writes can be nullptr. In that case, we will ignore relax reads or writes region.
 * \example
 *   Before relax
 *   \code
 *     for i = 0 to 10
 *       Block(reads=A[i: i+1]
 *   After relax, the relaxed region would be A[0: 10]
 */
void RelaxRegion(const StmtSRef& block_sref, const StmtSRef& root,
                 std::vector<TensorRegion>* reads,
                 std::vector<TensorRegion>* writes);

/*!
 * \brief Relax the TensorRegion with the loops under root
 * \param block_sref The block sref
 * \param root The root node
 * \param region The region to be relaxed
 * \return The relaxed region
 */
TensorRegion RelaxRegion(const StmtSRef& block_sref,
                         const StmtSRef& root,
                         const TensorRegion& region);

/*!
 * \brief remove the AST leaf and its parent subtree which has only one leaf
 * \param sref The sref of Block/Loop to be removed
 * \param root The AST root
 * \return The orginal stmt and the removed stmt of the subtree rooted by the parent node
 */
std::pair<Stmt, Stmt> RemoveLeaf(StmtSRef sref, const StmtSRef& root);

/*!
 * \brief Whether expr is related with var
 * \param var the expected var
 * \param expr the expected expr
 * \return Whether expr is related with var
 */
bool RelatedWithVar(const Var& var, const PrimExpr& expr);

class SRefValidator : public StmtVisitor {
 public:
  explicit SRefValidator(const ScheduleNode* sch) : sch_(sch) {}

  void VisitStmt_(const BlockNode* op) final {
    CheckParent(op);
    auto sref = sch_->stmt2ref.at(op);
    CHECK(sch_->scopes_.count(sref))
      << "Cannot find scope information of the block:\n" << GetRef<Stmt>(op);
  }

  void VisitStmt_(const LoopNode* op) final {
    CheckParent(op);
  }

 private:
  const ScheduleNode* sch_;
  const StmtSRefNode* parent_{nullptr};

  template <typename T>
  void CheckParent(const T* op) {
    auto it = sch_->stmt2ref.find(op);
    Stmt s = GetRef<Stmt>(op);
    CHECK(it != sch_->stmt2ref.end()) << "Cannot find Stmt in stmt2ref map:\n" << s;
    StmtSRef sref = it->second;
    CHECK(sref->parent == parent_) << "The parent of the node is mismatch:\n" << s;
    parent_ = sref.get();
  }
};

/*!
 * \brief PrimExpr pattern matcher.
 *
 * It is different from the pattern matcher in arith/pattern_match.h, which is dedicated
 * for compile-time constant patterns. This pattern matcher can work on dynamic user-specic
 * patterns.
 *
 * The code below shows how to use the pattern matcher.
 *
 * \code
 *
 * Var x("x"), y("y");
 * // use PrimExpr to declare patterns, x, y are holes that can be filled with
 * PatternMatcher pattern_matcher(x + y);
 * // expr = C[i,j] + A[i,k]*B[k,j], which is the expr we want to match
 * pattern_matcher.Match(expr);
 *
 * if (pattern_matcher.Success()) {
 *   pattern_matcher.Eval(x) // C[i,j]
 *   pattern_matcher.Eval(y) // A[i,k]*B[k,j]
 * }
 *
 * \endcode
 */
class PatternMatcher : public ExprVisitor {
 public:
  explicit PatternMatcher(const PrimExpr& pattern) : pattern_(pattern) {}

  void VisitExpr_(const VarNode* op) final;
  void VisitExpr_(const LoadNode* op) final;
  void VisitExpr_(const LetNode* op) final;
  void VisitExpr_(const CallNode* op) final;
  void VisitExpr_(const AddNode* op) final;
  void VisitExpr_(const SubNode* op) final;
  void VisitExpr_(const MulNode* op) final;
  void VisitExpr_(const DivNode* op) final;
  void VisitExpr_(const ModNode* op) final;
  void VisitExpr_(const FloorDivNode* op) final;
  void VisitExpr_(const FloorModNode* op) final;
  void VisitExpr_(const MinNode* op) final;
  void VisitExpr_(const MaxNode* op) final;
  void VisitExpr_(const EQNode* op) final;
  void VisitExpr_(const NENode* op) final;
  void VisitExpr_(const LTNode* op) final;
  void VisitExpr_(const LENode* op) final;
  void VisitExpr_(const GTNode* op) final;
  void VisitExpr_(const GENode* op) final;
  void VisitExpr_(const AndNode* op) final;
  void VisitExpr_(const OrNode* op) final;
  void VisitExpr_(const CastNode* op) final;
  void VisitExpr_(const NotNode* op) final;
  void VisitExpr_(const SelectNode* op) final;
  void VisitExpr_(const RampNode* op) final;
  void VisitExpr_(const BroadcastNode* op) final;
  void VisitExpr_(const ShuffleNode* op) final;
  void VisitExpr_(const IntImmNode* op) final;
  void VisitExpr_(const FloatImmNode* op) final;
  void VisitExpr_(const StringImmNode* op) final;
  void VisitExpr_(const BufferLoadNode* op) final;

  void Match(const PrimExpr& expr_to_match) {
    this->match_success_ = true;
    this->filled_map_.clear();
    this->expr_to_match_ = expr_to_match;
    this->operator()(pattern_);
  }

  PrimExpr Eval(const Var& var) {
    auto it = filled_map_.find(var.operator->());
    CHECK(it != filled_map_.end()) << "Unknown pattern variable";
    CHECK(match_success_) << "Match failed";
    return it->second;
  }

  bool Success() const {
    return match_success_;
  }

 private:
  bool match_success_{true};
  PrimExpr pattern_, expr_to_match_;
  std::unordered_map<const VarNode*, PrimExpr> filled_map_;
};

/*! \brief namespace for default reducer patterns */
namespace default_reducer {

class DefaultReducer {
 public:
  explicit DefaultReducer(const std::function<PrimExpr(Var, Var)>& combiner,
                          std::function<PrimExpr(DataType)> identity)
      : lhs_("x"), rhs_("y"), identity_(std::move(identity)) {
    result_  = combiner(lhs_, rhs_);
  }

  CommReducer GetReducer(DataType dtype) const {
    return CommReducerNode::make({lhs_}, {rhs_}, {result_}, {identity_(dtype)});
  }

 private:
  Var lhs_, rhs_;
  PrimExpr result_;
  const std::function<PrimExpr(DataType)> identity_;
};

static DefaultReducer default_reducers[4] = {
    DefaultReducer([](const Var& x, const Var& y) { return x + y; },
                   [](DataType dtype) { return make_const(dtype, 0); }),
    DefaultReducer([](const Var& x, const Var& y) { return x * y; },
                   [](DataType dtype) { return make_const(dtype, 1); }),
    DefaultReducer([](const Var& x, const Var& y) { return min(x, y); }, max_value),
    DefaultReducer([](const Var& x, const Var& y) { return max(x, y); }, min_value)
};

}  // namespace default_reducer

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_SCHEDULE_SCHEDULE_COMMON_H_
