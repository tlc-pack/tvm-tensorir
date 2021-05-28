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

#include <tvm/tir/schedule/schedule.h>

#include "./instruction.h"

namespace tvm {
namespace meta_schedule {

class Schedule;
class Trace;

/*! \brief The trace of program execution */
class TraceNode : public runtime::Object {
 public:
  /*! \brief The instructions used */
  Array<Instruction> insts;
  /*! \brief The decisions made in sampling */
  Map<Instruction, ObjectRef> decisions;

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
  void Append(const Instruction& inst, const ObjectRef& decision);
  /*!
   * \brief Remove the last element of the trace and return
   * \return The last element of the trace, which is removed; NullOpt if the trace has been empty
   */
  Optional<Instruction> Pop();
  /*!
   * \brief Apply the trace to the schedule
   * \param sch The schedule to be applied
   */
  void Apply(const Schedule& sch) const;
  /*!
   * \brief Apply the trace to the schedule
   * \param sch The schedule to be applied
   * \param decision_provider Provide decision to each instruction
   */
  void Apply(const Schedule& sch,
             const std::function<Optional<ObjectRef>(const Instruction& inst,
                                                     const Array<Optional<ObjectRef>>& inputs)>&
                 decision_provider) const;
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
  /*!
   * \brief Return to a list of strings that converts the trace to a series of Python API calling
   * \return The list of strings
   */
  Array<String> AsPython() const;
  /*!
   * \brief Returns a new trace with the decision mutated, and optionally remove instructions
   * in post-processing
   * \param inst The instruction
   * \param decision The decision
   * \return The new trace without postprocessing
   */
  Trace WithDecision(const Instruction& inst, const ObjectRef& decision,
                     bool remove_postproc) const;
  /*!
   * \brief Returns a simplified trace by dead-code elimination and optionally removing instructions
   * in post-processing
   * \param remove_postproc Whether to remove the post-processing code
   * \return The simplified trace
   */
  Trace Simplified(bool remove_postproc) const;
  /*!
   * \brief Stringify the trace as applying a sequence of schedule primitives
   * \return A string, the sequence of schedule primitives
   */
  String Stringify() const;

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
  explicit Trace(Array<Instruction> insts, Map<Instruction, ObjectRef> decisions);

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Trace, runtime::ObjectRef, TraceNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_TRACE_H_
