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

class TuneContext;

/*! \brief The abstract class for design space generation. */
class SpaceGeneratorNode : public Object {
 public:
  /*!
   * \brief The function type of `InitializeWithTuneContext` method.
   * \param tune_context The tuning context for initialization.
   */
  using FInitializeWithTuneContext = runtime::TypedPackedFunc<void(const TuneContext&)>;
  /*!
   * \brief The function type of `GenerateDesignSpace` method.
   * \param mod The module used for design space generation.
   * \return The generated design spaces, i.e., traces.
   */
  using FGenerateDesignSpace = runtime::TypedPackedFunc<Array<tir::Trace>(const IRModule&)>;

  /*! \brief Virtual destructor */
  virtual ~SpaceGeneratorNode();

  /*!
   * \brief Initialize the design space generator with tuning context.
   * \param tune_context The tuning context for initialization.
   */
  virtual void InitializeWithTuneContext(const TuneContext& tune_context) = 0;

  /*!
   * \brief Generate design spaces given a module.
   * \param mod The module used for design space generation.
   * \return The generated design spaces, i.e., traces.
   */
  virtual Array<tir::Trace> GenerateDesignSpaces(const IRModule& mod) = 0;

  static constexpr const char* _type_key = "meta_schedule.SpaceGenerator";
  TVM_DECLARE_BASE_OBJECT_INFO(SpaceGeneratorNode, Object);
};

/*!
 * \brief Managed reference to SpaceGeneratorNode.
 * \sa SpaceGeneratorNode
 */
class SpaceGenerator : public ObjectRef {
 public:
  /*!
   * \brief Create a design space generator with customized methods on the python-side.
   * \param initialize_with_tune_context_func The packed function of `InitializeWithTuneContext`.
   * \param generate_design_space_func The packed function of `GenerateDesignSpace`.
   * \return The design space generator created.
   */
  static SpaceGenerator PySpaceGenerator(
      SpaceGeneratorNode::FInitializeWithTuneContext initialize_with_tune_context_func,
      SpaceGeneratorNode::FGenerateDesignSpace generate_design_space_func);

  /*!
   * \brief Create a design space generator that is union of multiple design space generators.
   * \param space_generators An array of design space generators to be unioned.
   * \return The design space generator created.
   */
  static SpaceGenerator SpaceGeneratorUnion(Array<ObjectRef> space_generators);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SpaceGenerator, ObjectRef, SpaceGeneratorNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SPACE_GENERATOR_H_
