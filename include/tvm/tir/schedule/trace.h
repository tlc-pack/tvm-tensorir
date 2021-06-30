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
#ifndef TVM_TIR_SCHEDULE_TRACE_H_
#define TVM_TIR_SCHEDULE_TRACE_H_

#include <tvm/tir/schedule/inst.h>

namespace tvm {
namespace tir {

class Trace;

class TraceNode : public runtime::Object {
 public:
  /*! \brief The instructions used */
  Array<Inst> insts;
  /*! \brief The decisions made on sampling instructions */
  Map<Inst, ObjectRef> decisions;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("insts", &insts);
    v->Visit("decisions", &decisions);
  }

  static constexpr const char* _type_key = "tir.Trace";
  TVM_DECLARE_FINAL_OBJECT_INFO(TraceNode, runtime::Object);

 public:
  void Append(const Inst& inst);

  void Append(const Inst& inst, const ObjectRef& decision);

  Optional<Inst> Pop();

  void ApplyToSchedule(
      const Schedule& sch, bool remove_postproc,
      std::function<ObjectRef(const Inst& inst, const Array<ObjectRef>& inputs,
                              const Array<ObjectRef>& attrs, const ObjectRef& decision)>
          decision_provider = nullptr) const;

  ObjectRef AsJSON() const;

  Array<String> AsPython() const;

  Trace WithDecision(const Inst& inst, const ObjectRef& decision, bool remove_postproc) const;

  Trace Simplified(bool remove_postproc) const;
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
  explicit Trace(Array<Inst> insts, Map<Inst, ObjectRef> decisions);

  static void ApplyJSONToSchedule(const ObjectRef& json, const Schedule& sch);

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Trace, runtime::ObjectRef, TraceNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_TRACE_H_
