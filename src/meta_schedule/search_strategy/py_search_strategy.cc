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

#include <tvm/ir/module.h>
#include <tvm/node/node.h>
#include <tvm/runtime/packed_func.h>

#include "../search_strategy.h"
#include "../tune_context.h"

namespace tvm {
namespace meta_schedule {

/**************** PySearchStrategy ****************/

/*! \brief The cost model returning random value for all predictions */
class PySearchStrategyNode : public SearchStrategyNode {
 public:
  /*! \brief Pointer to the Init funcion in python */
  runtime::TypedPackedFunc<FGenerateMeasureCandidates> generate_measure_candidates_func;
  /*! \brief Pointer to the Generate funcion in python */
  runtime::TypedPackedFunc<FUpdateResults> update_results_func;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  runtime::Array<ObjectRef> GenerateMeasureCandidates(const TuneContext& context) override {
    return this->generate_measure_candidates_func(context);
  }

  void UpdateResults(const Array<ObjectRef>& results) override {
    this->update_results_func(results);
  }

  static constexpr const char* _type_key = "meta_schedule.PySearchStrategy";
  TVM_DECLARE_FINAL_OBJECT_INFO(PySearchStrategyNode, SearchStrategyNode);
};

/*!
 * \brief Managed reference to PySearchStrategyNode.
 * \sa PySearchStrategyNode
 */
class PySearchStrategy : public SearchStrategy {
 public:
  explicit PySearchStrategy(
      runtime::TypedPackedFunc<SearchStrategyNode::FGenerateMeasureCandidates>
          generate_measure_candidates_func,
      runtime::TypedPackedFunc<SearchStrategyNode::FUpdateResults> update_results_func) {
    ObjectPtr<PySearchStrategyNode> n = make_object<PySearchStrategyNode>();
    n->generate_measure_candidates_func = std::move(generate_measure_candidates_func);
    n->update_results_func = std::move(update_results_func);
    data_ = std::move(n);
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PySearchStrategy, SearchStrategy,
                                                    PySearchStrategyNode);
};

SearchStrategy SearchStrategy::PySearchStrategy(
    runtime::TypedPackedFunc<SearchStrategyNode::FGenerateMeasureCandidates>
        generate_measure_candidates_func,
    runtime::TypedPackedFunc<SearchStrategyNode::FUpdateResults> update_results_func) {
  return meta_schedule::PySearchStrategy(generate_measure_candidates_func, update_results_func);
}

TVM_REGISTER_NODE_TYPE(PySearchStrategyNode);
TVM_REGISTER_GLOBAL("meta_schedule.PySearchStrategyNew")
    .set_body_typed(SearchStrategy::PySearchStrategy);

}  // namespace meta_schedule
}  // namespace tvm
