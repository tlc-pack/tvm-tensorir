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
#ifndef SRC_AUTO_SCHEDULER_INSTRUCTION_H_
#define SRC_AUTO_SCHEDULER_INSTRUCTION_H_

#include <tvm/tir/function.h>
#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace auto_scheduler {

struct ScheduleStatus {
  tir::ScheduleNode* schedule;
  tir::StmtSRef cursor;
  Map<tir::Var, PrimExpr> symbol_table;
};

/**************** Define Instruction ****************/

class InstructionNode : public Object {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "auto_scheduler.Instruction";
  TVM_DECLARE_BASE_OBJECT_INFO(InstructionNode, Object);
};

class Instruction : public ObjectRef {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Instruction, ObjectRef, InstructionNode);

 protected:
  /*! \brief Constructor. The node should never be constructed directly. */
  Instruction() = default;
};

/**************** Define DeclIntVar ****************/

class DeclIntVarNode : public InstructionNode {
 public:
  tir::Var var;
  Array<Integer> choices;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("var", &var);
    v->Visit("choices", &choices);
  }

  /*!
   * \brief Apply the instruction to a TIR schedule
   * \param sampled_vars The variables sampled
   * \param status The schedule to be applied
   */
  void ApplyToSchedule(const Map<tir::Var, PrimExpr>& sampled_vars, ScheduleStatus* status) const;

  static constexpr const char* _type_key = "auto_scheduler.DeclIntVar";
  TVM_DECLARE_FINAL_OBJECT_INFO(DeclIntVarNode, InstructionNode);
};

class DeclIntVar : public Instruction {
 public:
  /*!
   * \brief Constructor
   * \param var The variable
   * \param choices The categorial distribution prior that the integer can be sampled from
   */
  DeclIntVar(tir::Var var, Array<Integer> choices);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(DeclIntVar, Instruction, DeclIntVarNode);
};

/**************** Define SplitInnerToOuter ****************/

class SplitInnerToOuterNode : public InstructionNode {
 public:
  int loop_id;
  Array<Optional<tir::Var>> factors;
  Optional<tir::Var> inferred;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("loop_id", &loop_id);
    v->Visit("factors", &factors);
    v->Visit("inferred", &inferred);
  }

  /*!
   * \brief Apply the instruction to a TIR schedule
   * \param status The schedule to be applied
   */
  void ApplyToSchedule(ScheduleStatus* status) const;

  static constexpr const char* _type_key = "auto_scheduler.SplitInnerToOuter";
  TVM_DECLARE_FINAL_OBJECT_INFO(SplitInnerToOuterNode, InstructionNode);
};

class SplitInnerToOuter : public Instruction {
 public:
  SplitInnerToOuter(int loop_id, Array<Optional<tir::Var>> factors, Optional<tir::Var> inferred);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(SplitInnerToOuter, Instruction, SplitInnerToOuterNode);
};

/**************** Define Reorder ****************/

class ReorderNode : public InstructionNode {
 public:
  Array<Integer> after_ids;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("after_ids", &after_ids); }

  /*!
   * \brief Apply the instruction to a TIR schedule
   * \param status The schedule to be applied
   */
  void ApplyToSchedule(ScheduleStatus* status) const;

  static constexpr const char* _type_key = "auto_scheduler.Reorder";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReorderNode, InstructionNode);
};

class Reorder : public Instruction {
 public:
  Reorder(Array<Integer> after_ids);  // NOLINT(*)
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Reorder, Instruction, ReorderNode);
};

/**************** Define ComputeAtOffset ****************/

class ComputeAtOffsetNode : public InstructionNode {
 public:
  int offset;
  int loop_id;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("offset", &offset);
    v->Visit("loop_id", &loop_id);
  }

  /*!
   * \brief Apply the instruction to a TIR schedule
   * \param status The schedule to be applied
   */
  void ApplyToSchedule(ScheduleStatus* status) const;

  static constexpr const char* _type_key = "auto_scheduler.ComputeAtOffset";
  TVM_DECLARE_FINAL_OBJECT_INFO(ComputeAtOffsetNode, InstructionNode);
};

class ComputeAtOffset : public Instruction {
 public:
  ComputeAtOffset(int offset, int loop_id);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ComputeAtOffset, Instruction, ComputeAtOffsetNode);
};

/**************** Define CursorMoveOffset ****************/

class CursorMoveOffsetNode : public InstructionNode {
 public:
  int offset;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("offset", &offset); }

  /*!
   * \brief Apply the instruction to a TIR schedule
   * \param status The schedule to be applied
   */
  void ApplyToSchedule(ScheduleStatus* status) const;

  static constexpr const char* _type_key = "auto_scheduler.CursorMoveOffset";
  TVM_DECLARE_FINAL_OBJECT_INFO(CursorMoveOffsetNode, InstructionNode);
};

class CursorMoveOffset : public Instruction {
 public:
  CursorMoveOffset(int offset);  // NOLINT(*)
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(CursorMoveOffset, Instruction, CursorMoveOffsetNode);
};

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // SRC_AUTO_SCHEDULER_INSTRUCTION_H_
