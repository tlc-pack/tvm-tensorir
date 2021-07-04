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
#ifndef TVM_TIR_SCHEDULE_UTILS_H_
#define TVM_TIR_SCHEDULE_UTILS_H_

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/schedule/state.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>
#include <utility>

#include "../../printer/text_printer.h"
#include "../../runtime/thread_storage_scope.h"
#include "../../support/array.h"
#include "./analysis.h"
#include "./error.h"
#include "./primitive.h"

#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/schedule/state.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tir {

/*!
 * \brief A helper macro to convert an sref to the statement it points to,
 * then check if the downcasting succeeded.
 * \param Result The result variable, used for checking
 * \param SRef The SRef to be casted
 * \param Type The type to be casted to, can be Block or For
 */
#define TVM_SREF_AS_OR_ERR(Result, SRef, Type) \
  SRef->StmtAs<Type>();                        \
  ICHECK(Result)

/*!
 * \brief A helper macro to convert an sref to the block it points to,
 * throwing an internal error if downcasting fails
 * \param Result The result variable, used for checking
 * \param SRef The SRef to be casted
 */
#define TVM_SREF_TO_BLOCK(Result, SRef)                   \
  TVM_SREF_AS_OR_ERR(Result, SRef, ::tvm::tir::BlockNode) \
      << "TypeError: Expects StmtSRef `" << #SRef         \
      << "` points to `Block`, but gets: " << (SRef->stmt ? SRef->stmt->GetTypeKey() : "None")

/*!
 * \brief A helper macro to convert an sref to the for-loop it points to,
 * throwing an internal error if downcasting fails
 * \param Result The name of the result variable, used for checking
 * \param SRef The SRef to be casted
 */
#define TVM_SREF_TO_FOR(Result, SRef)                   \
  TVM_SREF_AS_OR_ERR(Result, SRef, ::tvm::tir::ForNode) \
      << "TypeError: Expects StmtSRef `" << #SRef       \
      << "` points to `Loop`, but gets: " << (SRef->stmt ? SRef->stmt->GetTypeKey() : "None")

/*!
 * \brief Downcast a TVM ObjectRef to its corresponding container using `ObjectRef::as<Type>`,
 * then check if the downcasting succeeded.
 * \param Result The result variable, used for checking
 * \param From The ObjectRef to be downcasted
 * \param Type The type to be downcasted to
 */
#define TVM_TYPE_AS_OR_ERR(Result, From, Type) \
  From.as<Type>();                             \
  ICHECK(Result)

/*!
 * \brief Downcast a TVM ObjectRef to its corresponding container using `ObjectRef::as<Type>`,
 * throwing an internal error if downcast fails.
 * \param Result The result variable, used for checking
 * \param From The ObjectRef to be downcasted
 * \param Type The type to be downcasted to
 */
#define TVM_TYPE_AS(Result, From, Type)                                           \
  TVM_TYPE_AS_OR_ERR(Result, From, Type)                                          \
      << "TypeError: Expects `" << #From << "` to have type `" << Type::_type_key \
      << "`, but gets: " << (From.defined() ? From->GetTypeKey() : "None")

/******** Storage scope ********/

/*!
 * \brief Determine if iterators of a storage scope should be relaxed
 * under a specific thread scope
 * \param storage_scope The storage scope that the iterators are on
 * \param thread_scope The thread scope to be relaxed
 * \return A boolean indicating the result
 */
inline bool CanRelaxStorageUndereThread(const runtime::StorageScope& storage_scope,
                                        const runtime::ThreadScope& thread_scope) {
  if (storage_scope.rank == runtime::StorageRank::kWarp) {
    // for warp memory, we only relax threadIdx.x
    return thread_scope.rank == 1 && thread_scope.dim_index == 0;
  }
  return static_cast<int>(storage_scope.rank) <= static_cast<int>(thread_scope.rank);
}

/******** SeqStmt ********/

/*!
 * \brief Remove a specific Stmt from a SeqStmt. If a SeqStmt contains a BlockRealize,
 * whose block is the Stmt to be removed, then remove that BlockRealize too.
 * \param seq The SeqStmt to be removed from
 * \param to_remove The Stmt to be removed
 * \return The removal result
 */
inline Stmt RemoveFromSeqStmt(const SeqStmt& seq, const Stmt& to_remove) {
  ICHECK_GT(seq->size(), 1);
  Array<Stmt> new_stmts;
  new_stmts.reserve(seq->size());
  for (const Stmt& stmt : seq->seq) {
    if (to_remove.same_as(stmt)) {
      continue;
    }
    if (const auto* realize = stmt.as<BlockRealizeNode>()) {
      if (to_remove.same_as(realize->block)) {
        continue;
      }
    }
    new_stmts.push_back(stmt);
  }
  return SeqStmt::Flatten(new_stmts);
}

/******** Integer set ********/

/*!
 * \brief Converts the Ranges to IntSets
 * \param var_dom The ranges of variables
 * \return The integer sets of the variables
 */
inline Map<Var, arith::IntSet> AsIntSet(const Map<Var, Range>& var_dom) {
  std::unordered_map<Var, arith::IntSet, ObjectPtrHash, ObjectPtrEqual> result;
  result.reserve(var_dom.size());
  for (auto kv : var_dom) {
    Var& var = kv.first;
    Range& range = kv.second;
    result.emplace(std::move(var), arith::IntSet::FromRange(std::move(range)));
  }
  return {result.begin(), result.end()};
}

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_UTILS_H_
#define TVM_SREF_TO_E(Result, SRef, Type) \
  SRef->GetStmt<Type>();                  \
  ICHECK(Result)

#define TVM_SREF_TO_BLOCK(Result, SRef)              \
  TVM_SREF_TO_E(Result, SRef, ::tvm::tir::BlockNode) \
      << "TypeError: Expects StmtSRef `" << #SRef    \
      << "` points to `Block`, but gets: " << (SRef->stmt ? SRef->stmt->GetTypeKey() : "None");

#define TVM_SREF_TO_FOR(Result, SRef)              \
  TVM_SREF_TO_E(Result, SRef, ::tvm::tir::ForNode) \
      << "TypeError: Expects StmtSRef `" << #SRef  \
      << "` points to `Loop`, but gets: " << (SRef->stmt ? SRef->stmt->GetTypeKey() : "None");

#define TVM_TYPE_AS_E(Result, From, Type) \
  From.as<Type>();                        \
  ICHECK(Result)

#define TVM_TYPE_AS(Result, From, Type)                                           \
  TVM_TYPE_AS_E(Result, From, Type)                                               \
      << "TypeError: Expects `" << #From << "` to have type `" << Type::_type_key \
      << "`, but gets: " << (From.defined() ? From->GetTypeKey() : "None")

inline String ReprFunc(PrimFunc func) {
  const auto* f = runtime::Registry::Get("script.AsTVMScript");
  CHECK(f) << "IndexError: global function \"script.AsTVMScript\" not found";
  String s = (*f)(func, true);
  return s;
}

/*!
 * \brief Convert a tvm::runtime::Array to std::vector
 * \tparam TSrc The type of elements in the source Array
 * \tparam TDst The type of elements in the result vector
 * \return The result vector
 */
template <class TSrc, class TDst>
std::vector<TDst> AsVector(const Array<TSrc>& vec);

/*!
 * \brief Convert an std::vector to tvm::runtime::Array
 * \tparam TSrc The type of elements in the source vector
 * \tparam TDst The type of elements in the result Array
 * \return The result Array
 */
template <class TSrc, class TDst>
Array<TDst> AsArray(const std::vector<TSrc>& vec);

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

inline int64_t GetLoopIntExtent(const ForNode* loop) {
  const auto* int_extent = loop->extent.as<IntImmNode>();
  return int_extent ? int_extent->value : -1;
}

inline int64_t GetLoopIntExtent(const StmtSRef& loop_sref) {
  const auto* loop = TVM_SREF_TO_FOR(loop, loop_sref);
  return GetLoopIntExtent(loop);
}

/**************** AsArray<TSrc, TDst> ****************/

namespace details {
template <class TSrc, class TDst>
struct AsArrayImpl {};

template <class TSrc>
struct AsArrayImpl<TSrc, TSrc> {
  inline Array<TSrc> operator()(const std::vector<TSrc>& vec) const {
    return Array<TSrc>(vec.begin(), vec.end());
  }
};

template <class TDstObjectRef>
struct AsArrayImpl<int, TDstObjectRef> {
  inline Array<TDstObjectRef> operator()(const std::vector<int>& vec) const {
    Array<TDstObjectRef> result;
    result.reserve(vec.size());
    for (int x : vec) {
      result.push_back(Integer(x));
    }
    return result;
  }
};

template <class TDstObjectRef>
struct AsArrayImpl<int64_t, TDstObjectRef> {
  inline Array<TDstObjectRef> operator()(const std::vector<int64_t>& vec) const {
    Array<TDstObjectRef> result;
    result.reserve(vec.size());
    for (int64_t x : vec) {
      result.push_back(Integer(x));
    }
    return result;
  }
};

template <class TDstObjectRef>
struct AsArrayImpl<double, TDstObjectRef> {
  inline Array<TDstObjectRef> operator()(const std::vector<double>& vec) const {
    Array<TDstObjectRef> result;
    result.reserve(vec.size());
    for (double x : vec) {
      result.push_back(FloatImm(tvm::DataType::Float(64), x));
    }
    return result;
  }
};
}  // namespace details

template <class TSrc, class TDst>
inline Array<TDst> AsArray(const std::vector<TSrc>& vec) {
  return details::AsArrayImpl<TSrc, TDst>()(vec);
}

/**************** AsVector<TSrc, TDst> ****************/

namespace details {

template <class TSrc, class TDst>
struct AsVectorImpl {};

template <class TSrc>
struct AsVectorImpl<TSrc, TSrc> {
  inline std::vector<TSrc> operator()(const Array<TSrc>& vec) const {
    return std::vector<TSrc>(vec.begin(), vec.end());
  }
};

template <class TSrcObjectRef>
struct AsVectorImpl<TSrcObjectRef, int> {
  inline std::vector<int> operator()(const Array<TSrcObjectRef>& vec) const {
    std::vector<int> results;
    for (const TSrcObjectRef& x : vec) {
      const auto* n = x.template as<IntImmNode>();
      ICHECK(n) << "TypeError: Expects IntImm, but gets: " << x->GetTypeKey();
      results.push_back(n->value);
    }
    return results;
  }
};

template <class TSrcObjectRef>
struct AsVectorImpl<TSrcObjectRef, int64_t> {
  inline std::vector<int64_t> operator()(const Array<TSrcObjectRef>& vec) const {
    std::vector<int64_t> results;
    for (const TSrcObjectRef& x : vec) {
      const auto* n = x.template as<IntImmNode>();
      ICHECK(n) << "TypeError: Expects IntImm, but gets: " << x->GetTypeKey();
      results.push_back(n->value);
    }
    return results;
  }
};

template <class TSrcObjectRef>
struct AsVectorImpl<TSrcObjectRef, double> {
  inline std::vector<double> operator()(const Array<TSrcObjectRef>& array) const {
    std::vector<double> results;
    for (const TSrcObjectRef& x : array) {
      const auto* n = x.template as<FloatImmNode>();
      ICHECK(n) << "TypeError: Expects FloatImm, but gets: " << x->GetTypeKey();
      results.push_back(n->value);
    }
    return results;
  }
};
}  // namespace details

template <class TSrc, class TDst>
inline std::vector<TDst> AsVector(const Array<TSrc>& vec) {
  return details::AsVectorImpl<TSrc, TDst>()(vec);
}

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_SCHEDULE_SCHEDULE_COMMON_H_
