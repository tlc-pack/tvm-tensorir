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
 * \file tvm/include/te/transform.h
 * \brief Additional transform util for TE
 */
#ifndef TVM_TE_TRANSFORM_H_
#define TVM_TE_TRANSFORM_H_

#include <tvm/expr.h>
#include <tvm/buffer.h>
#include <tvm/tensor.h>
#include <tvm/te/ir.h>

namespace tvm {
namespace te {
/*!
 * \brief Lower TE IR to current TVM IR. It is a temporary pass and will be
 * removed after rewriting all IR passes.
 *
 * \param func The TeFunc to be lowerd
 * \param tensor_map Tensors to bind to the argument during lowering.
 * \return Transformed stmt.
 */
Function TeLower(Function func, Map<Buffer, Tensor> tensor_map);

Stmt Substitute(Stmt stmt,
                const std::unordered_map<const Variable*, Expr>& value_map);

Expr Substitute(Expr expr,
                const std::unordered_map<const Variable*, Expr>& value_map);

Stmt Substitute(Stmt stmt, const Map<Var, Expr>& value_map);

Expr Substitute(Expr expr, const Map<Var, Expr>& value_map);

}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_TRANSFORM_H_
