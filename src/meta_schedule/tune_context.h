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

#include <tvm/ir/module.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/object.h>

#include "./space_generator.h"

#ifndef SRC_META_SCHEDULE_TUNE_CONTEXT_H_
#define SRC_META_SCHEDULE_TUNE_CONTEXT_H_

namespace tvm {
namespace meta_schedule {

class TuneContextNode : public runtime::Object {
 public:
  /*! \brief The function to be optimized */
  Optional<IRModule> workload;
  Optional<SpaceGenerator> space_generator;

  int64_t seed;
  int num_threads;
  int verbose;
  // todo @zxybazh : Add other classes
};

class TuneContext : public runtime::ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TuneContext, ObjectRef, TuneContextNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_TUNE_CONTEXT_H_
