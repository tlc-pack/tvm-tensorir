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
#ifndef SRC_META_SCHEDULE_SEARCH_TASK_H_
#define SRC_META_SCHEDULE_SEARCH_TASK_H_

#include <tvm/target/target.h>
#include <tvm/tir/function.h>

namespace tvm {
namespace meta_schedule {

/********** SearchTask **********/

class SearchTaskNode : public Object {
 public:
  tir::PrimFunc func;
  String task_name;
  Array<ObjectRef> build_args;
  Target target;
  Target target_host;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("func", &func);
    v->Visit("task_name", &task_name);
    v->Visit("build_args", &build_args);
    v->Visit("target", &target);
    v->Visit("target_host", &target_host);
  }

  static constexpr const char* _type_key = "meta_schedule.SearchTask";
  TVM_DECLARE_FINAL_OBJECT_INFO(SearchTaskNode, Object);
};

class SearchTask : public ObjectRef {
 public:
  explicit SearchTask(tir::PrimFunc func, String task_name, Array<ObjectRef> build_args,
                      Target target, Target target_host);
  TVM_DEFINE_OBJECT_REF_METHODS(SearchTask, ObjectRef, SearchTaskNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SEARCH_TASK_H_
