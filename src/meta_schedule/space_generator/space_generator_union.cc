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

#include "../space_generator.h"
#include "../tune_context.h"

namespace tvm {
namespace meta_schedule {

/*! \brief The cost model returning random value for all predictions */
class SpaceGeneratorUnionNode : public SpaceGeneratorNode {
 public:
  runtime::Array<SpaceGenerator> space_generators;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  void InitializeWithTuneContext(const TuneContext& context) override {
    for (const SpaceGenerator& space_gen : space_generators) {
      space_gen->InitializeWithTuneContext(context);
    }
  }

  Array<Trace> Generate(const IRModule& workload) override {
    Array<Trace> result;
    for (const SpaceGenerator& space_gen : space_generators) {
      Array<Trace> partial = space_gen->Generate(workload);
      result.insert(result.end(), partial.begin(), partial.end());
    }
    return result;
  }

  static constexpr const char* _type_key = "meta_schedule.SpaceGeneratorUnion";
  TVM_DECLARE_FINAL_OBJECT_INFO(SpaceGeneratorUnionNode, SpaceGeneratorNode);
};

/*!
 * \brief Managed reference to SpaceGeneratorUnionNode.
 * \sa SpaceGeneratorUnionNode
 */
class SpaceGeneratorUnion : public SpaceGenerator {
 public:
  explicit SpaceGeneratorUnion(runtime::Array<ObjectRef> space_generators) {
    ObjectPtr<SpaceGeneratorUnionNode> n = make_object<SpaceGeneratorUnionNode>();
    for (const ObjectRef& space_gen : space_generators)
      n->space_generators.push_back(Downcast<SpaceGenerator>(space_gen));
    data_ = std::move(n);
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(SpaceGeneratorUnion, SpaceGenerator,
                                                    SpaceGeneratorUnionNode);
};

SpaceGenerator SpaceGenerator::SpaceGeneratorUnion(runtime::Array<ObjectRef> space_generators) {
  return meta_schedule::SpaceGeneratorUnion(space_generators);
}

TVM_REGISTER_NODE_TYPE(SpaceGeneratorUnionNode);
TVM_REGISTER_GLOBAL("meta_schedule.SpaceGeneratorUnionNew")
    .set_body_typed(SpaceGenerator::SpaceGeneratorUnion);
TVM_REGISTER_GLOBAL("meta_schedule.SpaceGeneratorUnionInitializeWithTuneContext")
    .set_body_method<SpaceGeneratorUnion>(&SpaceGeneratorUnionNode::InitializeWithTuneContext);
TVM_REGISTER_GLOBAL("meta_schedule.SpaceGeneratorUnionGenerate")
    .set_body_method<SpaceGeneratorUnion>(&SpaceGeneratorUnionNode::Generate);

}  // namespace meta_schedule
}  // namespace tvm
