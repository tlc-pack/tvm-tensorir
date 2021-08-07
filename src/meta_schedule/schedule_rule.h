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

#ifndef SRC_META_SCHEDULE_SCHEDULE_RULE_H_
#define SRC_META_SCHEDULE_SCHEDULE_RULE_H_

#include <tvm/runtime/container.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>

#include "./schedule.h"

namespace tvm {
namespace meta_schedule {

class TuneContext;

class ScheduleRuleNode : public runtime::Object {
 public:
  using FInitializeWithTuneContext = void(const TuneContext&);
  using FApply = Array<Schedule>(const Schedule&, const BlockRV&);

  /*! \brief Virtual destructor */
  virtual ~ScheduleRuleNode() = default;

  /*! \brief Name of the rule */
  String name;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("name", &name); }

  /*! \brief Initialize the f space with TuneContext */
  virtual void InitializeWithTuneContext(const TuneContext& context) = 0;

  /*!
   * \brief Apply a schedule rule to the given workload & block specification
   * \return The schedule after applying the schedule rule
   */
  virtual runtime::Array<Schedule> Apply(const Schedule& schedule, const BlockRV& block) = 0;

  static constexpr const char* _type_key = "meta_schedule.ScheduleRule";
  TVM_DECLARE_BASE_OBJECT_INFO(ScheduleRuleNode, Object);
};

/*!
 * \brief Managed reference to Schedule Rule Node
 * \sa ScheduleRuleNode
 */
class ScheduleRule : public runtime::ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScheduleRule, ObjectRef, ScheduleRuleNode);
  static ScheduleRule PyScheduleRule(
      String name,
      runtime::TypedPackedFunc<ScheduleRuleNode::FInitializeWithTuneContext>
          initialize_with_tune_context_func,
      runtime::TypedPackedFunc<ScheduleRuleNode::FApply> apply_func);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SCHEDULE_RULE_H_
