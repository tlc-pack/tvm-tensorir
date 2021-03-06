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
#ifndef TVM_TIR_SCHEDULE_ANALYSIS_H_
#define TVM_TIR_SCHEDULE_ANALYSIS_H_

#include <tvm/tir/schedule/schedule.h>

#include <unordered_set>

namespace tvm {
namespace tir {

/******** ContainsVar ********/
/*!
 * \brief Checks if an Expr or Stmt contains a list of specific Vars
 * \param stmt_or_expr The Stmt or Expr
 * \return A boolean indicating if any var in the list is found in stmt/expr
 */
bool ContainsVar(const ObjectRef& stmt_or_expr, const Array<Var>& var);
/*!
 * \brief Checks if an Expr or Stmt contains a specific Var
 * \param stmt_or_expr The Stmt or Expr
 * \return A boolean indicating if the var is found in stmt/expr
 */
bool ContainsVar(const ObjectRef& stmt_or_expr, const Var& var);
/*!
 * \brief Checks if an Expr or Stmt contains a list of specific Vars
 * \param stmt_or_expr The Stmt or Expr
 * \return A boolean indicating if any var in the list is found in stmt/expr
 */
bool ContainsVar(const ObjectRef& stmt_or_expr, const std::unordered_set<const VarNode*>& var);

/******** Verification ********/
/*!
 * \brief Verify the correctness of the sref tree
 * \param self The schedule state containing the sref to be verified
 * \note An exception will be thrown out if the sref tree is not valid
 */
void VerifySRefTree(const ScheduleState& self);
/*!
 * \brief Check the region cover for the single consumer block
 */
void VerifyRegionCover(const ScheduleState& self, const StmtSRef& consumer_block_sref);

/******** Binding ********/

bool ValidateBlockBinding(const BlockRealize& realize, const Map<Var, Range>& loop_var_ranges);

IterVarType GetLoopIterType(const ScheduleState& self, const StmtSRef& loop_sref);

/******** Scope ********/

/*!
 * \brief Get the sref to the scope root block, exclusive
 * \param sref The block or loop sref to be retrieved
 * \return The sref to the scope root block
 */
StmtSRef GetScopeRoot(const StmtSRef& sref);

/******** Block-loop relation ********/
/*!
 * \brief Get block from its tag
 * \param tag The query tag
 * \return the block schedulable reference list
 */
Array<StmtSRef> GetBlocks(const ScheduleState& self, const String& name);

/*!
 * \brief Get loops of the block
 * \param block The query block
 * \return the loop sref list
 */
Array<StmtSRef> GetAxes(const ScheduleState& self, const StmtSRef& block_sref);

/*!
 * \brief Get the child blocks of a specific parent block/loop
 * \param parent_sref The StmtSRef that points to the parent block/loop
 * \param inclusive If true and parent_sref is a block, return a single-element list containing
 * parent_sref
 * \return A list of child blocks
 */
Array<StmtSRef> GetChildBlocks(const ScheduleState& self, const StmtSRef& parent_sref,
                               bool inclusive = false);

/*!
 * \brief Get the producer of a specific block
 * \return The producers
 */
Array<StmtSRef> GetProducers(const ScheduleState& self, const StmtSRef& block_sref);

/*!
 * \brief Get the consumers of a specific block
 * \return The consumers
 */
Array<StmtSRef> GetConsumers(const ScheduleState& self, const StmtSRef& block_sref);

StmtSRef GetSRefTreeRoot(const StmtSRef& sref);

/******** Misc ********/

bool HasSingleChild(const StmtSRef& loop_or_block_sref);

Array<StmtSRef> CollectComputeLocation(const ScheduleState& self, const StmtSRef& block_sref);

/*!
 * \brief Get the pointer to the PrimFunc that the statement pointed by sref belongs to
 * \param self The state of scheduling
 * \param sref The sref to the statement in the query
 * \return A pointer to the PrimFunc the statement belongs to
 */
const PrimFuncNode* GetBelongFunc(const ScheduleState& self, const StmtSRef& sref);

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_ANALYSIS_H_
