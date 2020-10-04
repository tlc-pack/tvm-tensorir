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
#include "./mutator.h"  // NOLINT(build/include)

namespace tvm {
namespace meta_schedule {

/********** Constructor **********/

MutateTileSize::MutateTileSize(double p) {
  ObjectPtr<MutateTileSizeNode> n = make_object<MutateTileSizeNode>();
  n->p = p;
  data_ = std::move(n);
}

/********** MutateTileSize **********/

Optional<Schedule> MutateTileSizeNode::Apply(const SearchTask& task, const Schedule& sch,
                                             Sampler* sampler) {
  return NullOpt;
}

/********** FFI **********/

TVM_REGISTER_OBJECT_TYPE(MutatorNode);
TVM_REGISTER_NODE_TYPE(MutateTileSizeNode);

}  // namespace meta_schedule
}  // namespace tvm
