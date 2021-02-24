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

#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tir {

#define TVM_SREF_TO_BLOCK(Result, SRef)                                                       \
  SRef->GetStmt<::tvm::tir::BlockNode>();                                                     \
  ICHECK(Result) << "TypeError: Expects SRef `" << #SRef << "` points to `Block`, but gets: " \
                 << (SRef->stmt ? SRef->stmt->GetTypeKey() : "None");

#define TVM_SREF_TO_LOOP(Result, SRef)                                                       \
  SRef->GetStmt<::tvm::tir::LoopNode>();                                                     \
  ICHECK(Result) << "TypeError: Expects SRef `" << #SRef << "` points to `Loop`, but gets: " \
                 << (SRef->stmt ? SRef->stmt->GetTypeKey() : "None");

#define TVM_TYPE_AS(Result, From, Type)                                                      \
  From.as<Type>();                                                                           \
  ICHECK(Result) << "TypeError: Expects `" << #From << "` to have type `" << Type::_type_key \
                 << "`, but gets: " << (From.defined() ? From->GetTypeKey() : "None")

inline String ReprFunc(PrimFunc func) {
  const auto* f = runtime::Registry::Get("script.AsTVMScript");
  CHECK(f) << "IndexError: global function \"script.AsTVMScript\" not found";
  String s = (*f)(func, true);
  return s;
}

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
Stmt SubstituteInScope(const Stmt& stmt, const std::function<PrimExpr(const VarNode*)>& value_func);

/*!
 * \brief Substitute the var in current block scope specified in var map
 * \param stmt The source stmt to be substituted
 * \param var_map The mapping of var
 * \return The converted stmt
 */
Stmt SubstituteInScope(const Stmt& stmt,
                       const std::unordered_map<const VarNode*, const VarNode*>& var_map);

/*!
 * \brief Substitute the var in current block scope specified in var map
 * \param stmt The source stmt to be substituted
 * \param var_map The mapping of var
 * \return The converted stmt
 */
Stmt SubstituteInScope(const Stmt& stmt,
                       const std::unordered_map<const VarNode*, PrimExpr>& var_map);

/*!
 * \brief Substitute the var in BufferRegion
 * \param buffer_region The source BufferRegion to be substituted
 * \param var_map the mapping of var
 * \return The converted tensor region
 */
BufferRegion SubstituteBufferRegion(
    const BufferRegion& buffer_region,
    const std::unordered_map<const VarNode*, const VarNode*>& var_map);

/*!
 * \brief Substitute the var in BufferRegion
 * \param buffer_region The source BufferRegion to be substituted
 * \param var_map the mapping of var
 * \return The converted tensor region
 */
BufferRegion SubstituteBufferRegion(const BufferRegion& buffer_region,
                                    const std::unordered_map<const VarNode*, PrimExpr>& var_map);

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
 * \brief Relax the BufferRegion with the loops under root
 * \param block_sref The block sref
 * \param root The root node
 * \param reads The vector to store the reads result
 * \param writes The vector to store the writes result
 * \param relax_vars The additional vars should be relaxed
 * \note reads and writes can be nullptr. In that case, we will ignore relax reads or writes region.
 * For example
 *   Before relax
 *   \code
 *     for i = 0 to 10
 *       Block(reads=A[i: i+1])
 *   After relax, the relaxed region would be A[0: 10]
 */
void RelaxRegion(const StmtSRef& block_sref, const StmtSRef& root, std::vector<BufferRegion>* reads,
                 std::vector<BufferRegion>* writes,
                 const std::unordered_map<const VarNode*, Range>& relax_vars =
                     std::unordered_map<const VarNode*, Range>());

/*!
 * \brief Relax the BufferRegion with the loops under root
 * \param block_sref The block sref
 * \param root The root node
 * \param region The region to be relaxed
 * \return The relaxed region
 */
BufferRegion RelaxRegion(const StmtSRef& block_sref, const StmtSRef& root,
                         const BufferRegion& region);

/*!
 * \brief remove the AST leaf and its parent subtree which has only one leaf
 * \param sref The sref of Block/Loop to be removed
 * \param root The AST root
 * \return The orginal stmt and the removed stmt of the subtree rooted by the parent node
 */
std::pair<Stmt, Stmt> RemoveLeaf(StmtSRef sref, const StmtSRef& root);

/*!
 * \brief Inspect whether the stmt/expr contains any var of vars
 * \param obj the expected stmt/expr
 * \param vars the expected expr with vars
 * \return Whether any var appears in stmt/expr
 */
bool StmtExprContainsVar(const ObjectRef& obj, const PrimExpr& vars);

/*!
 * \brief Inspect whether the stmt/expr contains any var of vars
 * \param obj the expected stmt/expr
 * \param vars the vars to be inspected
 * \return Whether the stmt/expr contains any var of vars
 */
bool StmtExprContainsVar(const ObjectRef& obj, const std::vector<Var>& vars);

/*!
 * \brief Inspect whether the stmt/expr contains any var of vars
 * \param obj the expected stmt/expr
 * \param vars the vars to be inspected
 * \return Whether the stmt/expr contains any var of vars
 */
bool StmtExprContainsVar(const ObjectRef& obj, const std::unordered_set<const VarNode*>& vars);

/*!
 * \brief Update the scope (dependency) information of a given block statement
 * \param stmt The block statement to be updated
 * \param stmt2ref The ScheduleNode::stmt2ref from ScheduleNode
 * \param scopes The ScheduleNode::stmt2ref from ScheduleNode that is to be updated
 */
void UpdateScope(const StmtNode* stmt,
                 const std::unordered_map<const StmtNode*, StmtSRef>& stmt2ref,
                 std::unordered_map<StmtSRef, BlockScope, ObjectPtrHash, ObjectPtrEqual>* scopes);

class StmtReplacer : public StmtMutator {
 public:
  explicit StmtReplacer(const std::unordered_map<const StmtNode*, const StmtNode*>& replace_map)
      : replace_map(replace_map) {}

  Stmt VisitStmt(const Stmt& stmt) override;

  const std::unordered_map<const StmtNode*, const StmtNode*>& replace_map;
};

bool CheckOneLine(const Stmt& s);

void CollectVars(std::unordered_set<const VarNode*>& res, const PrimExpr& expr);

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

/* \brief Auto calculate the block read write region */
class BlockReadWriteCollector : public StmtExprVisitor {
 public:
  explicit BlockReadWriteCollector(const Array<Buffer>& allocations) {
    for (const auto& allocate : allocations) inner_buffers_.insert(allocate.get());
  }

  Array<BufferRegion> reads();
  Array<BufferRegion> writes();

 private:
  std::unordered_map<const VarNode*, arith::IntSet> dom_map_;
  std::vector<Buffer> read_buffers_, writes_buffers_;
  std::vector<std::vector<tvm::arith::IntSet>> read_regions_, write_regions_;
  std::unordered_set<const BufferNode*> inner_buffers_;

  void VisitStmt_(const ForNode* op) override;
  void Update(std::vector<Buffer>* buffers, std::vector<std::vector<arith::IntSet>>* regions,
              const Buffer& buffer, const std::vector<arith::IntSet>& region);
  void VisitExpr_(const BufferLoadNode* op) override;
  void VisitStmt_(const BufferStoreNode* op) override;
  void VisitStmt_(const BlockRealizeNode* op) override;
};

/* \brief Deep comparison to check if two IR graph are equivalent */
using ExprComparator = ExprFunctor<bool(const PrimExpr& n, const PrimExpr& other)>;
using StmtComparator = StmtFunctor<bool(const Stmt& n, const Stmt& other)>;

class TensorizeComparator : public ExprComparator, public StmtComparator {
 public:
  explicit TensorizeComparator(bool assert_mode = true) : assert_mode_(assert_mode) {}

  // Map from rhs buffer to lhs buffer
  std::unordered_map<Buffer, Buffer, ObjectHash, ObjectEqual> rhs_buffer_map_;
  // Buffer indices mapping
  std::unordered_map<Buffer, std::vector<PrimExpr>, ObjectPtrHash, ObjectPtrEqual> buffer_indices_;
  std::vector<IterVar> extra_block_vars_;
  // variable remap if any
  std::unordered_map<ObjectRef, ObjectRef, ObjectPtrHash, ObjectPtrEqual> equal_map_;

  bool VisitExpr(const PrimExpr& n, const PrimExpr& other) override;
  bool VisitStmt(const Stmt& n, const Stmt& other) override;

  bool VisitStmt_(const ForNode* op, const Stmt& other) override;
  bool VisitStmt_(const SeqStmtNode* op, const Stmt& other) override;
  bool VisitStmt_(const BufferStoreNode* op, const Stmt& other) override;
  bool VisitStmt_(const BlockRealizeNode* op, const Stmt& other) override;
  bool VisitStmt_(const BlockNode* op, const Stmt& other) override;

  bool VisitExpr_(const AddNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const SubNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const MulNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const DivNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const ModNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const EQNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const NENode* op, const PrimExpr& other) override;
  bool VisitExpr_(const LTNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const LENode* op, const PrimExpr& other) override;
  bool VisitExpr_(const GTNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const GENode* op, const PrimExpr& other) override;
  bool VisitExpr_(const AndNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const OrNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const MinNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const MaxNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const FloorDivNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const FloorModNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const IntImmNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const FloatImmNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const CastNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const VarNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const BufferLoadNode* op, const PrimExpr& other) override;

  bool DefEqual(const ObjectRef& lhs, const ObjectRef& rhs);
  virtual bool CompareBuffer(const Buffer& lhs, const Buffer& rhs);
  bool CompareBufferRegion(const BufferRegion& lhs, const BufferRegion& rhs);
  bool CompareAnnotation(const std::pair<String, ObjectRef>& lhs,
                         const std::pair<String, ObjectRef>& rhs);
  bool CompareAnnotationMap(const Map<String, ObjectRef>& lhs, const Map<String, ObjectRef>& rhs);
  template <typename T>
  bool CompareBufferAccess(const T* lhs, const T* rhs);
  template <typename T, typename F>
  bool CompareArray(const Array<T>& lhs, const Array<T>& rhs, F cmp);
  bool CompareRange(const Range& lhs, const Range& rhs);
  bool CompareType(const DataType& lhs, const DataType& rhs);

 protected:
  bool assert_mode_;
  bool is_scope_block = true, is_inner_block = true;
};

/*! \brief namespace for default reducer patterns */
namespace default_reducer {

class DefaultReducer {
 public:
  explicit DefaultReducer(const std::function<PrimExpr(Var, Var)>& combiner,
                          std::function<PrimExpr(DataType, Span)> identity)
      : lhs_("x"), rhs_("y"), identity_(std::move(identity)) {
    result_ = combiner(lhs_, rhs_);
  }

  CommReducer GetReducer(DataType dtype, Span span) const {
    return CommReducer({lhs_}, {rhs_}, {result_}, {identity_(dtype, span)});
  }

 private:
  Var lhs_, rhs_;
  PrimExpr result_;
  const std::function<PrimExpr(DataType, Span)> identity_;
};

static DefaultReducer default_reducers[4] = {
    DefaultReducer([](const Var& x, const Var& y) { return x + y; },
                   [](DataType dtype, Span span) { return make_const(dtype, 0, span); }),
    DefaultReducer([](const Var& x, const Var& y) { return x * y; },
                   [](DataType dtype, Span span) { return make_const(dtype, 1, span); }),
    DefaultReducer([](const Var& x, const Var& y) { return min(x, y); }, max_value),
    DefaultReducer([](const Var& x, const Var& y) { return max(x, y); }, min_value)};

}  // namespace default_reducer

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_SCHEDULE_SCHEDULE_COMMON_H_