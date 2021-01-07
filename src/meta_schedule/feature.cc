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
#include "./feature.h"  // NOLINT(build/include)

namespace tvm {
namespace meta_schedule {

runtime::NDArray PrimFuncFeature::AsNDArray() {
  static thread_local DLManagedTensor* dl_tensor = new DLManagedTensor();
  dl_tensor->dl_tensor.data = feature.data();
  dl_tensor->dl_tensor.ctx = DLContext{kDLCPU, 0};
  dl_tensor->dl_tensor.ndim = shape.size();
  dl_tensor->dl_tensor.dtype = DLDataType{kDLFloat, 64, 1};
  dl_tensor->dl_tensor.shape = shape.data();
  dl_tensor->dl_tensor.strides = nullptr;
  dl_tensor->dl_tensor.byte_offset = 0;
  dl_tensor->manager_ctx = nullptr;
  dl_tensor->deleter = nullptr;
  return runtime::NDArray::FromDLPack(dl_tensor);
}

}  // namespace meta_schedule
}  // namespace tvm
