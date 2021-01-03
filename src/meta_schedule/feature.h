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
#ifndef SRC_META_SCHEDULE_FEATURE_H_
#define SRC_META_SCHEDULE_FEATURE_H_

#include <tvm/tir/function.h>

#include <vector>

namespace tvm {
namespace meta_schedule {

class PrimFuncFeature {
 public:
  runtime::NDArray AsNDArray();

  std::vector<double> feature;
  std::vector<int64_t> shape;
};

TVM_DLL void CalcPerBlockFeature(const tir::PrimFunc& func, int max_num_buffer_access_features,
                                 PrimFuncFeature* result);

TVM_DLL Array<String> PerBlockFeatureNames(const tir::PrimFunc& func,
                                           int max_num_buffer_access_features);

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_FEATURE_H_
