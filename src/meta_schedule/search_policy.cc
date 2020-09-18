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

#include "./search_policy.h"  // NOLINT(build/include)

#include "./measure.h"
#include "./schedule.h"
#include "./search_task.h"

namespace tvm {
namespace meta_schedule {

TVM_REGISTER_OBJECT_TYPE(SearchPolicyNode);

struct Internal {
  static Schedule SearchPolicySearch(SearchPolicy policy, SearchTask task, ProgramMeasurer measurer,
                                     int verbose) {
    // TODO(@junrushao1994): it is not exposed to python because ProgramMeasurer is not exposed yet
    return policy->Search(task, measurer, verbose);
  }
};

TVM_REGISTER_GLOBAL("meta_schedule.SearchPolicySearch")
    .set_body_typed(Internal::SearchPolicySearch);

}  // namespace meta_schedule
}  // namespace tvm
