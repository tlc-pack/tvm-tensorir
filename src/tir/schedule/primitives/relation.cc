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
#include "../schedule_common.h"
#include "./primitives.h"

namespace tvm {
namespace tir {
namespace schedule {

Array<StmtSRef> GetBlock(const ScheduleState& self, const String& name) {
  Array<StmtSRef> result;
  for (const auto& kv : self->scopes) {
    const StmtSRef& block_sref = kv.first;
    const auto* block = TVM_SREF_TO_BLOCK(block, block_sref);
    if (block->name_hint == name) {
      result.push_back(block_sref);
    }
  }
  return result;
}

Array<StmtSRef> GetAxes(const ScheduleState& self, const StmtSRef& block_sref) {
  std::vector<StmtSRef> result;
  for (StmtSRefNode* parent = block_sref->parent; parent && parent->stmt->IsInstance<ForNode>();
       parent = parent->parent) {
    result.push_back(GetRef<StmtSRef>(parent));
  }
  return {result.rbegin(), result.rend()};
}

Array<StmtSRef> GetChildBlocks(const ScheduleState& self, const StmtSRef& parent_sref) {
  struct Collector : public StmtVisitor {
   private:
    void VisitStmt_(const BlockNode* block) final { result.push_back(self->stmt2ref.at(block)); }

   public:
    explicit Collector(const ScheduleState& self) : self(self) {}

    const ScheduleState& self;
    Array<StmtSRef> result;
  };
  Collector collector(self);
  collector(GetRef<Stmt>(parent_sref->stmt));
  return std::move(collector.result);
}

}  // namespace schedule
}  // namespace tir
}  // namespace tvm
