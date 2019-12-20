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
 * \brief Stmt mutator that implements COW semantics
 * \file cow_stmt_mutator.h
 */
#ifndef TVM_TE_COW_STMT_MUTATOR_H_
#define TVM_TE_COW_STMT_MUTATOR_H_

#include <tvm/ir_functor_ext.h>

namespace tvm {
namespace te {

class COWStmtMutator :
    protected StmtFunctor<Stmt(const Stmt&)> {
 public:
  /*!
   * \brief Run a COW mutation on stmt.
   * \param stmt The input statement as a rvalue reference.
   *
   * \note COW will only happen iff stmt is the only reference
   *       to the statement. The rvalue reference forces
   *       the user to move the stmt and avoid keeping an additional copy.
   * \return The result of the mutation.
   */
  Stmt operator()(Stmt&& stmt) {
    return VisitStmt(stmt);
  }


 protected:
  /*!
   * \brief Mutate expression.
   *
   * \note Do not mutate expresison for now.
   * \return the mutated expression.
   */
  Expr Mutate(const Expr& expr) {
    return expr;
  }
  /*!
   * \brief Mutate stmt.
   *  All the subclasses must call this function to create new children.
   *
   * \note This function maintains the condition about whether
   *       we can perform CopyOnWrite optimization.
   * \return The transformed stmt.
   */
  Stmt Mutate(const Stmt& stmt)  {
    if (allow_copy_on_write_ && !stmt.unique()) {
      allow_copy_on_write_ = false;
      Stmt ret = VisitStmt(stmt);
      allow_copy_on_write_ = true;
      return ret;
    } else {
      return VisitStmt(stmt);
    }
  }
  /*!
   * \brief Perform copy on write on node.
   *
   *  If CopyOnWrite is allowed, directly return
   *  a strong reference to the node container.
   *  Otherwise, return a copy of the node.
   *
   * \return The result node pointer.
   */
  template<typename TNode>
  NodePtr<TNode> CopyOnWrite(const TNode* node) {
    if (allow_copy_on_write_) {
      // return the old node.
      auto n = runtime::GetObjectPtr<TNode>(const_cast<TNode*>(node));
      return n;
    } else {
      // Make a new copy of the node.
      return runtime::make_object<TNode>(*node);
    }
  }

  Stmt VisitStmt_(const IfThenElse* op) override;
  Stmt VisitStmt_(const Evaluate* op) override;
  Stmt VisitStmt_(const LoopNode* op) override;
  Stmt VisitStmt_(const BlockNode* op) override;
  Stmt VisitStmt_(const SeqStmtNode* op) override;

 protected:
  bool allow_copy_on_write_{true};
};
}  // namespace te
}  // namespace tvm
#endif  // TVM_TE_COW_STMT_MUTATOR_H_
