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

#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/node/reflection.h>
#include <tvm/node/structural_equal.h>
#include <tvm/runtime/registry.h>
#include "schedule_common.h"


namespace tvm {
namespace tir {

class TensorizeSEqualHandler : public SEqualReducer::Handler {
 public:
  // use direct recursion.
  bool SEqualReduce(const ObjectRef& lhs, const ObjectRef& rhs, bool map_free_vars) final {
    if (lhs.same_as(rhs)) return true;
    if (!lhs.defined() && rhs.defined()) return false;
    if (!rhs.defined() && lhs.defined()) return false;
    if (lhs->type_index() != rhs->type_index()) return false;
    return vtable_->SEqualReduce(lhs.get(), rhs.get(), SEqualReducer(this, map_free_vars));
  }

  ObjectRef MapLhsToRhs(const ObjectRef& lhs) final { return ObjectRef(nullptr); }

  void MarkGraphNode() final {}

 private:
  // reflection vtable
  ReflectionVTable* vtable_ = ReflectionVTable::Global();
};

void ScheduleNode::tensorize(const StmtSRef& sref, const Intrinsic& intrinsic) {
  //TODO(Siyuan): support directly tensorize a block
  const auto* n = DowncastPtr<LoopNode>(sref->stmt);
  CHECK(n) << "Only support tensorize a loop for now";


}

}  // namespace tir
}  // namespace tvm
