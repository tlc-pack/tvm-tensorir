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

#ifndef TVM_META_SCHEDULE_SCHEDULE_RULE_H_
#define TVM_META_SCHEDULE_SCHEDULE_RULE_H_

#include <tvm/tir/schedule/schedule.h>

namespace tvm {
namespace meta_schedule {

class TuneContext;

/*! \brief Rules to modify a block in a schedule. */
class ScheduleRuleNode : public runtime::Object {
 public:
  /*! \brief Virtual destructor. */
  virtual ~ScheduleRuleNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*! \brief Initialize the schedule rule with TuneContext. */
  virtual void InitializeWithTuneContext(const TuneContext& context) = 0;

  /*!
   * \brief Apply a schedule rule to the specific block in the given schedule.
   * \return The list of schedules generated by applying the schedule rule.
   */
  virtual runtime::Array<tir::Schedule> Apply(const tir::Schedule& schedule,
                                              const tir::BlockRV& block) = 0;

  static constexpr const char* _type_key = "meta_schedule.ScheduleRule";
  TVM_DECLARE_BASE_OBJECT_INFO(ScheduleRuleNode, Object);
};

/*! \brief The schedule rule with customized methods on the python-side. */
class PyScheduleRuleNode : public ScheduleRuleNode {
 public:
  /*!
   * \brief The function type of `InitializeWithTuneContext` method.
   * \param tune_context The tuning context for initialization.
   */
  using FInitializeWithTuneContext = runtime::TypedPackedFunc<void(const TuneContext&)>;
  /*!
   * \brief The function type of `Apply` method.
   * \param schedule The schedule to be modified.
   * \param block The specific block to apply the schedule rule.
   * \return The generated design spaces, i.e., schedules.
   */
  using FApply =
      runtime::TypedPackedFunc<Array<tir::Schedule>(const tir::Schedule&, const tir::BlockRV&)>;

  /*! \brief The packed function to the `InitializeWithTuneContext` funcion. */
  FInitializeWithTuneContext f_initialize_with_tune_context;
  /*! \brief The packed function to the `Apply` funcion. */
  FApply f_apply;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `f_initialize_with_tune_context` is not visited
    // `f_apply` is not visited
  }

  void InitializeWithTuneContext(const TuneContext& context) final {
    this->f_initialize_with_tune_context(context);
  }

  Array<tir::Schedule> Apply(const tir::Schedule& schedule, const tir::BlockRV& block) final {
    return this->f_apply(schedule, block);
  }

  static constexpr const char* _type_key = "meta_schedule.PyScheduleRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(PyScheduleRuleNode, ScheduleRuleNode);
};

/*!
 * \brief Managed reference to ScheduleRuleNode
 * \sa ScheduleRuleNode
 */
class ScheduleRule : public runtime::ObjectRef {
 public:
  /*!
   * \brief Create a schedule rule with customized methods on the python-side.
   * \param f_initialize_with_tune_context The packed function of `InitializeWithTuneContext`.
   * \param f_apply The packed function of `Apply`.
   * \return The schedule rule created.
   */
  TVM_DLL static ScheduleRule PyScheduleRule(
      PyScheduleRuleNode::FInitializeWithTuneContext f_initialize_with_tune_context,
      PyScheduleRuleNode::FApply f_apply);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScheduleRule, ObjectRef, ScheduleRuleNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_SCHEDULE_RULE_H_
