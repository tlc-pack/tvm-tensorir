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

/*! \brief The class for builder's input type. */
class BuildInputNode : public runtime::Object {
 public:
  /*! \brief The workload to be optimized. */
  IRModule workload;
  /*! \brief The target to be optimized for. */
  Target target;

  /*!
   * \brief Visitor for variables in python.
   * \note required for non-abstract classes.
   */
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("workload", &workload);
    v->Visit("target", &target);
  }

  /*! \brief Class name `PySpaceGenerator` */
  static constexpr const char* _type_key = "meta_schedule.BuildInput";
  TVM_DECLARE_FINAL_OBJECT_INFO(BuildInputNode, runtime::Object);  // Concrete class
};

/*!
 * \brief Managed reference to BuildInputNode
 * \sa BuildInputNode
 */
class BuildInput : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor function of BuildInput class.
   * \param workload The workload to be optimized.
   * \param target The target to be optimized for.
   */
  TVM_DLL explicit BuildInput(IRModule mod, Target target);
  /*! \brief Declare reference relationship. */
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BuildInput, runtime::ObjectRef, BuildInputNode);
};

/*! \brief The class for builder's output type. */
class BuildResultNode : public runtime::Object {
 public:
  /*! \brief The String typed path to the built artifact. */
  Optional<String> artifact_path;
  /*! \brief The error message if any. */
  Optional<String> error_msg;

  /*!
   * \brief Visitor for variables in python.
   * \note required for non-abstract classes.
   */
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("artifact_path", &artifact_path);
    v->Visit("error_msg", &error_msg);
  }

  /*! \brief Class name `BuildResult` */
  static constexpr const char* _type_key = "meta_schedule.BuildResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(BuildResultNode, runtime::Object);  // Concrete class
};

/*!
 * \brief Managed reference to BuildResultNode
 * \sa BuildResultNode
 */
class BuildResult : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor function of BuildResult class.
   * \param artifact_path The String typed path to the built artifact.
   * \param error_msg The error message if any.
   */
  TVM_DLL explicit BuildResult(Optional<String> artifact_path, Optional<String> error_msg);
  /*! \brief Declare reference relationship. */
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BuildResult, runtime::ObjectRef, BuildResultNode);
};

/*! \brief The class of builder. */

class BuilderNode : public runtime::Object {
 public:
  /*! \brief Virtual destructor, required for abstract class. */
  virtual ~BuilderNode() = default;

  /*!
   * \brief Virtual function to generate the build result from build input.
   * \param build_inputs The given array of build inputs from meansure candidates.
   * \return The array of build results.
   */
  virtual Array<BuildResult> Build(const Array<BuildInput>& build_inputs) = 0;

  /*! \brief The function type of `Build` method. */
  using FBuild = runtime::TypedPackedFunc<Array<BuildResult>(const Array<BuildInput>&)>;

  /*! \brief Class name `Builder` */
  static constexpr const char* _type_key = "meta_schedule.Builder";
  TVM_DECLARE_BASE_OBJECT_INFO(BuilderNode, runtime::Object);  // Absract class
};

/*!
 * \brief Managed reference to BuilderNode
 * \sa BuilderNode
 */
class Builder : public runtime::ObjectRef {
 public:
  /*!
   * \brief Member function to create the PyBuilder class.
   * \param build_func The function pointer to the `Build` function.
   * \return The constructed PyBuilder object but in Builder type.
   */
  static Builder PyBuilder(BuilderNode::FBuild build_func);

  /*! \brief Declare reference relationship. */
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Builder, runtime::ObjectRef, BuilderNode);
};

/*! \brief The python side customizable class for builder. */
class PyBuilderNode : public BuilderNode {
 public:
  /*! \brief The function pointer to the `Build` function. */
  FBuild build_func;

  /*!
   * \brief Visitor for variables in python.
   * \note required for non-abstract classes.
   */
  void VisitAttrs(tvm::AttrVisitor* v) {
    // `build_func` is not visited
  }

  /*!
   * \brief Use the given function pointer to override the `Build` function.
   * \param build_inputs The given array of build inputs from meansure candidates.
   * \return The array of build results.
   */
  Array<BuildResult> Build(const Array<BuildInput>& build_inputs) final {
    return build_func(build_inputs);
  }

  /*! \brief Class name `PyBuilder` */
  static constexpr const char* _type_key = "meta_schedule.PyBuilder";
  TVM_DECLARE_FINAL_OBJECT_INFO(PyBuilderNode, BuilderNode);  // Concrete class
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_BUILDER_H_