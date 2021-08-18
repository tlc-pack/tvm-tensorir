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

#include <tvm/tir/schedule/schedule.h>

namespace tvm {
namespace meta_schedule {

// Forward declaration
class TuneContext;

class SpaceGeneratorNode : public runtime::Object {
 public:
  using FInitializeWithTuneContext = void(const TuneContext&);
  using FGenerate = Array<tir::Schedule>(const IRModule&);

  /*! \brief Virtual destructor */
  virtual ~SpaceGeneratorNode() = default;

  /*! \brief Initialize the design space generator with TuneContext */
  virtual void InitializeWithTuneContext(const TuneContext& context) = 0;

  /*!
   * \brief Generate a schedule out of the design space generator
   * \return The generated schedule
   */
  // TODO(zxybazh): Change to Traces class
  virtual runtime::Array<tir::Schedule> Generate(const IRModule& workload) = 0;

  static constexpr const char* _type_key = "meta_schedule.SpaceGenerator";
  TVM_DECLARE_BASE_OBJECT_INFO(SpaceGeneratorNode, Object);
};

/*!
 * \brief Managed reference to Design Space Generator Node
 * \sa SpaceGeneratorNode
 */
class SpaceGenerator : public runtime::ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SpaceGenerator, ObjectRef, SpaceGeneratorNode);
  static SpaceGenerator PySpaceGenerator(
      runtime::TypedPackedFunc<SpaceGeneratorNode::FInitializeWithTuneContext>
          initialize_with_tune_context_func,
      runtime::TypedPackedFunc<SpaceGeneratorNode::FGenerate> generate_func);
  static SpaceGenerator SpaceGeneratorUnion(runtime::Array<ObjectRef> space_generators);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SPACE_GENERATOR_H_
