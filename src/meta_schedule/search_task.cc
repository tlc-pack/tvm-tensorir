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

#include "./search_task.h"  // NOLINT(build/include)

namespace tvm {
namespace meta_schedule {

TVM_REGISTER_NODE_TYPE(SearchTaskNode);

SearchTask::SearchTask(tir::PrimFunc func, Array<ObjectRef> build_args, Target target,
                       Target target_host) {
  ObjectPtr<SearchTaskNode> n = make_object<SearchTaskNode>();
  n->func = std::move(func);
  n->build_args = std::move(build_args);
  n->target = std::move(target);
  n->target_host = std::move(target_host);
  data_ = std::move(n);
}

}  // namespace meta_schedule
}  // namespace tvm
