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
#include "./builder.h"  // NOLINT(build/include)

namespace tvm {
namespace meta_schedule {

/******** Constructors ********/

BuildInput::BuildInput(IRModule mod, Target target) {
  ObjectPtr<BuildInputNode> n = make_object<BuildInputNode>();
  n->mod = std::move(mod);
  n->target = std::move(target);
  data_ = std::move(n);
}

BuildResult::BuildResult(Optional<String> artifact_path, Optional<String> error_msg) {
  ObjectPtr<BuildResultNode> n = make_object<BuildResultNode>();
  n->artifact_path = std::move(artifact_path);
  n->error_msg = std::move(error_msg);
  data_ = std::move(n);
}

Builder Builder::PyBuilder(BuilderNode::FBuild build_func) {
  ObjectPtr<PyBuilderNode> n = make_object<PyBuilderNode>();
  n->build_func = std::move(build_func);
  return Builder(std::move(n));
}

/******** FFI ********/

TVM_REGISTER_NODE_TYPE(BuildInputNode);
TVM_REGISTER_NODE_TYPE(BuildResultNode);
TVM_REGISTER_OBJECT_TYPE(BuilderNode);
TVM_REGISTER_NODE_TYPE(PyBuilderNode);

TVM_REGISTER_GLOBAL("meta_schedule.BuildInput")
    .set_body_typed([](IRModule mod, Target target) -> BuildInput {
      return BuildInput(mod, target);
    });

TVM_REGISTER_GLOBAL("meta_schedule.BuildResult")
    .set_body_typed([](Optional<String> artifact_path, Optional<String> error_msg) -> BuildResult {
      return BuildResult(artifact_path, error_msg);
    });

TVM_REGISTER_GLOBAL("meta_schedule.PyBuilder").set_body_typed(Builder::PyBuilder);

}  // namespace meta_schedule
}  // namespace tvm
