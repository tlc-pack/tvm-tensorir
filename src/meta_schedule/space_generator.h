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

#ifndef SRC_META_SCHEDULE_SPACE_GENERATOR_H_
#define SRC_META_SCHEDULE_SPACE_GENERATOR_H_

#include <tvm/ir/module.h>
#include <tvm/tir/schedule/trace.h>

namespace tvm {
namespace meta_schedule {

// Forward declaration
class TuneContext;

/*! \brief The class for design space generation. */
class SpaceGeneratorNode : public runtime::Object {
 public:
  /*! \brief The function type of `InitializeWithTuneContext` method. */
  using FInitializeWithTuneContext = runtime::TypedPackedFunc<void(const TuneContext&)>;
  /*! \brief The function type of `Generate` method. Both typedefs are used for customization. */
  using FGenerate = runtime::TypedPackedFunc<Array<tir::Trace>(const IRModule&)>;

  /*! \brief Virtual destructor, required for abstract class. */
  virtual ~SpaceGeneratorNode() = default;

  /*!
   * \brief Virtual function to initialize the design space generator with TuneContext.
   * \param context The TuneContext object for initialization.
   */
  virtual void InitializeWithTuneContext(const TuneContext& context) = 0;

  /*!
   * \brief Virtual function to generate design spaces out of the design space generator.
   * \return The generated design spaces, i.e., traces.
   */
  virtual runtime::Array<tir::Trace> Generate(const IRModule& workload) = 0;

  /*! \brief Class name `SpaceGenerator` */
  static constexpr const char* _type_key = "meta_schedule.SpaceGenerator";
  TVM_DECLARE_BASE_OBJECT_INFO(SpaceGeneratorNode, Object);  // Abstract class
};

/*!
 * \brief Managed reference to SpaceGeneratorNode.
 * \sa SpaceGeneratorNode
 */
class SpaceGenerator : public runtime::ObjectRef {
 public:
  /*! \brief Declare reference relationship. */
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SpaceGenerator, ObjectRef, SpaceGeneratorNode);

  /*!
   * \brief Member function to create the python side customizable PySpaceGenerator class.
   * \param initialize_with_tune_context_func The function pointer to the `Init...` function.
   * \param generate_func The function pointer to the `Generate` function.
   * \return The constructed PySpaceGenerator object but in SpaceGenerator type.
   */
  static SpaceGenerator PySpaceGenerator(
      SpaceGeneratorNode::FInitializeWithTuneContext initialize_with_tune_context_func,
      SpaceGeneratorNode::FGenerate generate_func);

  /*!
   * \brief Member function to create the SpaceGeneratorUnion class.
   * \param space_generators Array of the SpaceGenerator objects to be unioned.
   * \return The constructed SpaceGeneratorUnion object but in SpaceGenerator type.
   */
  static SpaceGenerator SpaceGeneratorUnion(runtime::Array<ObjectRef> space_generators);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SPACE_GENERATOR_H_
