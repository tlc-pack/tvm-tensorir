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

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_SCHEDULE_SCHEDULE_COMMON_H_
