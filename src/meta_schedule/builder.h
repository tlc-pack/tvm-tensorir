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

/*! \brief The builder's input. */
class BuildInputNode : public runtime::Object {
 public:
  /*! \brief The IRModule to be built. */
  IRModule mod;
  /*! \brief The target to be built for. */
  Target target;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("mod", &mod);
    v->Visit("target", &target);
  }

  static constexpr const char* _type_key = "meta_schedule.BuildInput";
  TVM_DECLARE_FINAL_OBJECT_INFO(BuildInputNode, runtime::Object);
};

/*!
 * \brief Managed reference to BuildInputNode
 * \sa BuildInputNode
 */
class BuildInput : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor of BuildInput.
   * \param mod The IRModule to be built.
   * \param target The target to be built for.
   */
  TVM_DLL explicit BuildInput(IRModule mod, Target target);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BuildInput, runtime::ObjectRef, BuildInputNode);
};

/*! \brief The builder's output. */
class BuildResultNode : public runtime::Object {
 public:
  /*! \brief The path to the built artifact. */
  Optional<String> artifact_path;
  /*! \brief The error message if any. */
  Optional<String> error_msg;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("artifact_path", &artifact_path);
    v->Visit("error_msg", &error_msg);
  }

  static constexpr const char* _type_key = "meta_schedule.BuildResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(BuildResultNode, runtime::Object);
};

/*!
 * \brief Managed reference to BuildResultNode
 * \sa BuildResultNode
 */
class BuildResult : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor of BuildResult.
   * \param artifact_path The path to the built artifact.
   * \param error_msg The error message if any.
   */
  TVM_DLL explicit BuildResult(Optional<String> artifact_path, Optional<String> error_msg);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BuildResult, runtime::ObjectRef, BuildResultNode);
};

/*! \brief The abstract builder interface. */
class BuilderNode : public runtime::Object {
 public:
  /*! \brief Default destructor */
  virtual ~BuilderNode() = default;
  /*!
   * \brief Generate the build results from build inputs.
   * \param build_inputs The inputs to be built.
   * \return The build results.
   */
  virtual Array<BuildResult> Build(const Array<BuildInput>& build_inputs) = 0;
  /*!
   * \brief The function type of `Build` method.
   * \param build_inputs The inputs to be built.
   * \return The build results.
   */
  using FBuild = runtime::TypedPackedFunc<Array<BuildResult>(const Array<BuildInput>&)>;

  static constexpr const char* _type_key = "meta_schedule.Builder";
  TVM_DECLARE_BASE_OBJECT_INFO(BuilderNode, runtime::Object);
};

/*!
 * \brief Managed reference to BuilderNode
 * \sa BuilderNode
 */
class Builder : public runtime::ObjectRef {
 public:
  /*!
   * \brief Create a builder with customized build method on the python-side.
   * \param build_func The function pointer to the `Build` function.
   * \return The Builder created.
   */
  static Builder PyBuilder(BuilderNode::FBuild build_func);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Builder, runtime::ObjectRef, BuilderNode);
};

/*! \brief A builder with customized build method on the python-side. */
class PyBuilderNode : public BuilderNode {
 public:
  /*! \brief The packed function to the `Build` function. */
  FBuild build_func;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `build_func` is not visited
  }

  Array<BuildResult> Build(const Array<BuildInput>& build_inputs) final {
    return build_func(build_inputs);
  }

  static constexpr const char* _type_key = "meta_schedule.PyBuilder";
  TVM_DECLARE_FINAL_OBJECT_INFO(PyBuilderNode, BuilderNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_BUILDER_H_
