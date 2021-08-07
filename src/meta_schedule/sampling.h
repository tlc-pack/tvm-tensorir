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
#ifndef TVM_META_SCHEDULE_SAMPLING_H_
#define TVM_META_SCHEDULE_SAMPLING_H_

#include <vector>

#include "./schedule.h"

namespace tvm {
namespace meta_schedule {

/******** Sampling Instructions ********/

TVM_DLL std::vector<int64_t> SamplePerfectTile(tir::ScheduleState self, int64_t* random_state,
                                               const tir::StmtSRef& loop_sref,  //
                                               int n,                           //
                                               int max_innermost_factor,        //
                                               Optional<Array<Integer>>* decision);

TVM_DLL int64_t SampleCategorical(tir::ScheduleState self, int64_t* random_state,  //
                                  const Array<Integer>& candidates,                //
                                  const Array<FloatImm>& probs,                    //
                                  Optional<Integer>* decision);

TVM_DLL tir::StmtSRef SampleComputeLocation(tir::ScheduleState self, int64_t* random_state,  //
                                            const tir::StmtSRef& block_sref,                 //
                                            Optional<Integer>* decision);

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SAMPLING_H_
