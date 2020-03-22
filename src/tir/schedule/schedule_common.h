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

#include <vector>

namespace tvm {
namespace tir {

/*!
 * \brief Get the direct child Schedulable Stmt (Block and Loop)
 * \param stmt the parent stmt.
 * \param keep_realize if true, get block_realize for blocks
 * \return the list of child stmts
 */
Array<Stmt> GetChildren(const Stmt& stmt, bool keep_realize = false);

/*!
 * \brief Substitute the var in current block scope specified in key->var to be value.
 * \param expr The source expression to be substituted
 * \param value_func The function of new values mapping.
 * \return The converted expression.
 */
Stmt SubstituteInScope(const Stmt& stmt,
                       const std::function<PrimExpr(const VarNode*)>& value_func);

/*!
 * \brief Get BlockRealize with by Block
 * \param block The queried block
 * \return BlockRealize.
 */
BlockRealize GetBlockRealize(const StmtSRef& block_sref);

/*! \brief Get lowest common ancestor of all nodes */
StmtSRef LowestCommonAncestor(const std::vector<StmtSRef>& nodes, const StmtSRef& root);

/*!
 * \brief Relax the TensorRegion with the loops under root
 * \param block_sref The block sref
 * \param root The root node
 * \param reads The vector to store the reads result
 * \param writes The vector to store the writes result
 * \note reads and writes can be nullptr. In that case, we will ignore relax reads or writes region.
 */
void RelaxRegion(const StmtSRef& block_sref, const StmtSRef& root,
                 std::vector<TensorRegion>* reads,
                 std::vector<TensorRegion>* writes);

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_SCHEDULE_SCHEDULE_COMMON_H_
