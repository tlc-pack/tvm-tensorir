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
#include <tvm/arith/analyzer.h>
#include <tvm/ir/error.h>
#include <tvm/tir/op.h>
#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#ifndef SRC_AUTO_SCHEDULER_AUTO_SCHEDULER_UTILS_H_ /* TODO(@junrushao1994): name convention */
#define SRC_AUTO_SCHEDULER_AUTO_SCHEDULER_UTILS_H_

namespace tvm {
namespace auto_scheduler {

/*!
 * \brief Whether the expr contains var
 * \param expr the expected expr
 * \param var the expected var
 * \return A boolean indicating if var appears in expr
 */
bool ExprContainsVar(const PrimExpr& expr, const tir::Var& var);

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // SRC_AUTO_SCHEDULER_AUTO_SCHEDULER_UTILS_H_
