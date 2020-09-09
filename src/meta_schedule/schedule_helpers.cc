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
#ifndef SRC_META_SCHEDULE_SCHEDULE_H_
#define SRC_META_SCHEDULE_SCHEDULE_H_

#include <tvm/tir/schedule.h>

#include <unordered_map>
#include <utility>

#include "./instruction.h"
#include "./random_variable.h"
#include "./sampler.h"
#include "./schedule.h"

namespace tvm {
namespace meta_schedule {

tir::Block BlockFromStmtSRef(tir::StmtSRef sref) {
  return GetRef<tir::Block>(sref->GetStmt<tir::BlockNode>());
}

tir::Loop LoopFromStmtSRef(tir::StmtSRef sref) {
  return GetRef<tir::Loop>(sref->GetStmt<tir::LoopNode>());
}

TVM_REGISTER_GLOBAL("meta_schedule.BlockFromStmtSRef").set_body_typed(BlockFromStmtSRef);
TVM_REGISTER_GLOBAL("meta_schedule.LoopFromStmtSRef").set_body_typed(LoopFromStmtSRef);

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SCHEDULE_H_
