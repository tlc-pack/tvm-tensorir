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
#ifndef SRC_META_SCHEDULE_BUILDER_H_
#define SRC_META_SCHEDULE_BUILDER_H_

#include <tvm/ir/module.h>
#include <tvm/target/target.h>

namespace tvm {
namespace meta_schedule {

class BuildInputNode : public runtime::Object {
 public:
  IRModule mod;
  Target target;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("mod", &mod);
    v->Visit("target", &target);
  }

  static constexpr const char* _type_key = "meta_schedule.BuildInput";
  TVM_DECLARE_FINAL_OBJECT_INFO(BuildInputNode, runtime::Object);
};

class BuildInput : public runtime::ObjectRef {
 public:
  TVM_DLL explicit BuildInput(IRModule mod, Target target);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BuildInput, runtime::ObjectRef, BuildInputNode);
};

class BuildResultNode : public runtime::Object {
 public:
  Optional<String> artifact_path;
  Optional<String> error_msg;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("artifact_path", &artifact_path);
    v->Visit("error_msg", &error_msg);
  }

  static constexpr const char* _type_key = "meta_schedule.BuildResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(BuildResultNode, runtime::Object);
};

class BuildResult : public runtime::ObjectRef {
 public:
  TVM_DLL explicit BuildResult(Optional<String> artifact_path, Optional<String> error_msg);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BuildResult, runtime::ObjectRef, BuildResultNode);
};

class BuilderNode : public runtime::Object {
 public:
  virtual ~BuilderNode() = default;
  virtual Array<BuildResult> Build(const Array<BuildInput>& build_inputs) = 0;

  using FBuild = runtime::TypedPackedFunc<Array<BuildResult>(const Array<BuildInput>&)>;

  static constexpr const char* _type_key = "meta_schedule.Builder";
  TVM_DECLARE_BASE_OBJECT_INFO(BuilderNode, runtime::Object);
};

class Builder : public runtime::ObjectRef {
 public:
  static Builder PyBuilder(BuilderNode::FBuild f_build);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Builder, runtime::ObjectRef, BuilderNode);
};

class PyBuilderNode : public BuilderNode {
 public:
  FBuild f_build;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `f_build` is not visited
  }

  Array<BuildResult> Build(const Array<BuildInput>& build_inputs) final {
    return f_build(build_inputs);
  }

  static constexpr const char* _type_key = "meta_schedule.PyBuilder";
  TVM_DECLARE_BASE_OBJECT_INFO(PyBuilderNode, BuilderNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_BUILDER_H_