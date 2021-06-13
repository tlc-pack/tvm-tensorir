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
#ifndef TVM_META_SCHEDULE_FEATURE_H_
#define TVM_META_SCHEDULE_FEATURE_H_

#include <vector>

#include "./schedule.h"

namespace tvm {
namespace meta_schedule {

TVM_DLL runtime::NDArray PerBlockFeature(const Schedule& sch, int max_num_buffer_access_features);

TVM_DLL Array<runtime::NDArray> PerBlockFeatureBatched(const Array<Schedule>& schs,
                                                       int max_num_buffer_access_features);

TVM_DLL Array<String> PerBlockFeatureNames(int max_num_buffer_access_features);

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_FEATURE_H_
