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

#include <tvm/runtime/registry.h>

namespace tvm {
namespace meta_schedule {

SearchTask::SearchTask(tir::PrimFunc func, String task_name, Target target, Target target_host) {
  ObjectPtr<SearchTaskNode> n = make_object<SearchTaskNode>();
  n->func = std::move(func);
  n->task_name = std::move(task_name);
  n->target = std::move(target);
  n->target_host = std::move(target_host);
  data_ = std::move(n);
}

struct Internal {
  /*!
   * \brief Wrap for SearchTask::SearchTask
   * \param func The function to be optimized
   * \param task_name Name of this search task
   * \param target The target to be built at
   * \param target_host The target host to be built at
   * \return SearchTask, the new object constructed
   * \sa SearchTask::SearchTask
   */
  static SearchTask New(tir::PrimFunc func, String task_name, Target target, Target target_host) {
    return SearchTask(func, task_name, target, target_host);
  }
};

TVM_REGISTER_NODE_TYPE(SearchTaskNode);
TVM_REGISTER_GLOBAL("meta_schedule.SearchTask").set_body_typed(Internal::New);

}  // namespace meta_schedule
}  // namespace tvm
