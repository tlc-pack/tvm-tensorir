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
#ifndef SRC_META_SCHEDULE_ANALYSIS_H_
#define SRC_META_SCHEDULE_ANALYSIS_H_

#include <utility>

#include "./schedule.h"

namespace tvm {
namespace meta_schedule {

TVM_DLL bool IsTrivialBinding(Schedule sch, BlockRV block);

TVM_DLL Array<Integer> GetIterType(Schedule sch, BlockRV block);

TVM_DLL bool IsLeaf(Schedule sch, BlockRV block);

TVM_DLL bool IsBodySingleStmt(Schedule sch, BlockRV block);

TVM_DLL tir::BufferLoad GetBufferStore(Schedule sch, BlockRV block);

TVM_DLL Array<tir::BufferLoad> GetBufferLoad(Schedule sch, BlockRV block);

TVM_DLL int CountOp(Schedule sch, BlockRV block, Op op);

TVM_DLL bool HasBranch(Schedule sch, BlockRV block);

TVM_DLL Optional<Array<tir::Var>> BlockVarsAsStoreAxes(Schedule sch, BlockRV block);

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_ANALYSIS_H_
