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

#include <unordered_map>

#include "./schedule_common.h"

namespace tvm {
namespace tir {

TVM_DLL bool ValidateBlockBinding(const BlockRealize& realize,
                                  const Map<Var, Range>& loop_var_ranges);

TVM_DLL StmtSRef GetScopeSRef(const StmtSRef& sref);

/*! \brief Check the region cover for the single consumer block */
TVM_DLL void VerifyRegionCover(const ScheduleState& self, const StmtSRef& consumer_block_sref);

/*! \brief Verify the correctness of the sref tree */
TVM_DLL void VerifySRefTree(const ScheduleState& self);

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_ANALYSIS_H_
