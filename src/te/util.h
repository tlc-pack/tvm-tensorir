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

#ifndef TVM_TE_UTIL_H_
#define TVM_TE_UTIL_H_

#include <tvm/te/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <utility>
#include <vector>

namespace tvm {
namespace te {

class TensorAccessGather : public IRVisitor {
 public:
  TensorAccessGather() = default;
  explicit TensorAccessGather(Buffer target_tensor) :
      target_tensor_(target_tensor) {}

  void Visit_(const BufferLoadNode* op) final;

  // grouped accesses by target tensor
  std::unordered_map<Buffer, std::vector<std::vector<Expr>>, NodeHash, NodeEqual> access_grouped;
  std::vector<std::pair<Buffer, std::vector<Expr>>> access_all;  // all accesses
  std::vector<std::vector<Expr>> access_one;                      // accesses to the target buffer

  std::vector<Buffer> tensor_order;  // a list to keep the original order of tensors

 private:
  Buffer target_tensor_;
};

Array<Var> GatherVars(const NodeRef& expr_or_stmt);

Array<TensorRegion> CreateInputRegions(const Stmt& stmt);

}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_UTIL_H_
