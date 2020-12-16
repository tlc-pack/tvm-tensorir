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
#ifndef SRC_META_SCHEDULE_TRACE_H_
#define SRC_META_SCHEDULE_TRACE_H_

#include <tvm/tir/schedule.h>

#include "./instruction.h"

namespace tvm {
namespace meta_schedule {

class Schedule;

/*! \brief The trace of program execution */
class TraceNode : public runtime::Object {
 public:
  /*! \brief The instructions used */
  Array<Instruction> insts;
  /*! \brief The decisions made in sampling */
  // TODO: change signature to ObjectRef
  Map<Instruction, Array<ObjectRef>> decisions;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("insts", &insts);
    v->Visit("decisions", &decisions);
  }

  /*!
   * \brief Append a deterministic instruction to the back of the trace
   * \param inst The instruction
   */
  void Append(const Instruction& inst);
  /*!
   * \brief Append a stochastic instruction to the back of the trace
   * \param inst The instrruction
   * \param decision The sampling decision made on the instruction
   */
  void Append(const Instruction& inst, const Array<ObjectRef>& decision);
  /*!
   * \brief Apply the trace to the schedule
   * \param sch The schedule to be applied
   */
  void Apply(const Schedule& sch) const;
  /*!
   * \brief Export the trace into JSON format
   */
  ObjectRef Serialize() const;
  /*!
   * \brief Import the trace from the JSON format
   * \param json The serialized trace of scheduling
   * \param sch The schedule to be deserialized on
   */
  static void Deserialize(const ObjectRef& json, const Schedule& sch);

  static constexpr const char* _type_key = "meta_schedule.Trace";
  TVM_DECLARE_FINAL_OBJECT_INFO(TraceNode, Object);
};

/*!
 * \brief Managed reference to TraceNode
 * \sa TraceNode
 */
class Trace : public runtime::ObjectRef {
 public:
  /*! \brief Default constructor */
  Trace();
  /*!
   * \brief Constructor
   * \param insts The instructions used
   * \param decisions The decisions made in sampling
   */
  explicit Trace(Array<Instruction> insts, Map<Instruction, Array<ObjectRef>> decisions);

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Trace, runtime::ObjectRef, TraceNode);
};

Trace DeadCodeElimination(const Trace& trace);

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_TRACE_H_
