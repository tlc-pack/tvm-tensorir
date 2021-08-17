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

/**************** PySpaceGenerator ****************/

/*! \brief The cost model returning random value for all predictions */
class PySpaceGeneratorNode : public SpaceGeneratorNode {
 public:
  /*! \brief Pointer to the Init funcion in python */
  runtime::TypedPackedFunc<FInitializeWithTuneContext> initialize_with_tune_context_func;
  /*! \brief Pointer to the Generate funcion in python */
  runtime::TypedPackedFunc<FGenerate> generate_func;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  void InitializeWithTuneContext(const TuneContext& context) override {
    this->initialize_with_tune_context_func(context);
  }

  Array<Trace> Generate(const IRModule& workload) override { return this->generate_func(workload); }

  static constexpr const char* _type_key = "meta_schedule.PySpaceGenerator";
  TVM_DECLARE_FINAL_OBJECT_INFO(PySpaceGeneratorNode, SpaceGeneratorNode);
};

/*!
 * \brief Managed reference to PySpaceGeneratorNode.
 * \sa PySpaceGeneratorNode
 */
class PySpaceGenerator : public SpaceGenerator {
 public:
  explicit PySpaceGenerator(runtime::TypedPackedFunc<SpaceGeneratorNode::FInitializeWithTuneContext>
                                initialize_with_tune_context_func,
                            runtime::TypedPackedFunc<SpaceGeneratorNode::FGenerate> generate_func) {
    ObjectPtr<PySpaceGeneratorNode> n = make_object<PySpaceGeneratorNode>();
    n->initialize_with_tune_context_func = std::move(initialize_with_tune_context_func);
    n->generate_func = std::move(generate_func);
    data_ = std::move(n);
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PySpaceGenerator, SpaceGenerator,
                                                    PySpaceGeneratorNode);
};

SpaceGenerator SpaceGenerator::PySpaceGenerator(
    runtime::TypedPackedFunc<SpaceGeneratorNode::FInitializeWithTuneContext>
        initialize_with_tune_context_func,
    runtime::TypedPackedFunc<SpaceGeneratorNode::FGenerate> generate_func) {
  return meta_schedule::PySpaceGenerator(initialize_with_tune_context_func, generate_func);
}

TVM_REGISTER_NODE_TYPE(PySpaceGeneratorNode);
TVM_REGISTER_GLOBAL("meta_schedule.PySpaceGenerator")
    .set_body_typed(SpaceGenerator::PySpaceGenerator);

}  // namespace meta_schedule
}  // namespace tvm
