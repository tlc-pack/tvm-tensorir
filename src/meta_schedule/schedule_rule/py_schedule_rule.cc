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

#include "../schedule.h"
#include "../schedule_rule.h"
#include "../tune_context.h"

namespace tvm {
namespace meta_schedule {

/**************** PyScheduleRule ****************/

/*! \brief The cost model returning random value for all predictions */
class PyScheduleRuleNode : public ScheduleRuleNode {
 public:
  /*! \brief Pointer to the Init funcion in python */
  runtime::TypedPackedFunc<FInitializeWithTuneContext> initialize_with_tune_context_func;
  /*! \brief Pointer to the Generate funcion in python */
  runtime::TypedPackedFunc<FApply> apply_func;

  void InitializeWithTuneContext(const TuneContext& context) override {
    this->initialize_with_tune_context_func(context);
  }

  Array<Schedule> Apply(const Schedule& schedule, const BlockRV& block) override {
    return this->apply_func(schedule, block);
  }

  static constexpr const char* _type_key = "meta_schedule.PyScheduleRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(PyScheduleRuleNode, ScheduleRuleNode);
};

/*!
 * \brief Managed reference to PyScheduleRuleNode.
 * \sa PyScheduleRuleNode
 */
class PyScheduleRule : public ScheduleRule {
 public:
  using FInitializeWithTuneContext = PyScheduleRuleNode::FInitializeWithTuneContext;
  using FApply = PyScheduleRuleNode::FApply;

  explicit PyScheduleRule(String name,
                          runtime::TypedPackedFunc<ScheduleRuleNode::FInitializeWithTuneContext>
                              initialize_with_tune_context_func,
                          runtime::TypedPackedFunc<ScheduleRuleNode::FApply> apply_func) {
    ObjectPtr<PyScheduleRuleNode> n = make_object<PyScheduleRuleNode>();
    n->name = name;
    n->initialize_with_tune_context_func = std::move(initialize_with_tune_context_func);
    n->apply_func = std::move(apply_func);
    data_ = std::move(n);
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PyScheduleRule, ScheduleRule,
                                                    PyScheduleRuleNode);
};

ScheduleRule ScheduleRule::PyScheduleRule(
    String name,
    runtime::TypedPackedFunc<ScheduleRuleNode::FInitializeWithTuneContext>
        initialize_with_tune_context_func,
    runtime::TypedPackedFunc<ScheduleRuleNode::FApply> apply_func) {
  return meta_schedule::PyScheduleRule(name, initialize_with_tune_context_func, apply_func);
}

TVM_REGISTER_NODE_TYPE(PyScheduleRuleNode);
TVM_REGISTER_GLOBAL("meta_schedule.PyScheduleRule").set_body_typed(ScheduleRule::PyScheduleRule);

}  // namespace meta_schedule
}  // namespace tvm
