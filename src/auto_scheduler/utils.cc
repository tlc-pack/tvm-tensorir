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

/*!
 * \file auto_scheduler/utils.cc
 * \brief Common utilities.
 */

#include "utils.h"

namespace tvm {
namespace auto_scheduler {

NullStream& NullStream::Global() {
  static NullStream stream;
  return stream;
}


std::unordered_map<size_t, size_t>
TopKDispatcher::dispatch(const std::vector<float>& scores,
                         const size_t num_states) {
  const size_t num_instances = scores.size() / num_states;
  float max_acc_score;
  size_t k = 1;

  do {
    max_acc_score = 1e-10;

    for (size_t inst_id = 0; inst_id < num_instances; ++inst_id) {
      
    }
  } while (max_acc_score > 1e-10);
  LOG(INFO) << "k=" << k;
}

}  // namespace auto_scheduler
}  // namespace tvm
