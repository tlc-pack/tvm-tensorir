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
#ifndef SRC_AUTO_SCHEDULER_META_SCHEDULE_H_
#define SRC_AUTO_SCHEDULER_META_SCHEDULE_H_

#include <vector>

#include "./instruction.h"
#include "./loop_tree.h"

namespace tvm {
namespace auto_scheduler {

class MetaScheduleNode : public Object {
  // TODO(@junrushao1994): `make_object` requires all fields to have default a constructor, but
  // not-nullable fields may not have one

 public:
  /*! \brief Function used to revoke the effect of a specific instruction */
  using RevokeFunc = std::function<void()>;
  /*! \brief The meta IR to be meta scheduled */
  mutable MetaIR meta_ir{nullptr};
  /*! \brief The cursor pointing to the MetaIR node for the current stage */
  mutable MetaIR cursor{nullptr};
  /*! \brief The instructions applied the meta ir */
  mutable Array<Instruction> instructions;
  /*! \brief The functions used to revoke instructions accordingly, used for Backtracking */
  mutable std::vector<RevokeFunc> revokes;
  /*! \brief The variables that have been declared */
  mutable Array<tir::Var> declared_vars;
  /*!
   * \brief Add an instruction `decl_int_var`
   * \param choices The possible candidates of the value
   * \param name The name hint of the variable
   * \return The variable created
   */
  tir::Var DeclIntVarNode(Array<Integer> choices, String name_hint = "");
  /*!
   * \brief Split the loop specified by `loop_id` by the factors specified
   * \param loop_id The index of the loop to be split
   * \param factors The split factors, allows at most one to be NullOpt, which means this part is
   * inferred
   * \param name The name hint of the inferred variable
   * \return The variable created if there is any, or NullOpt if no inference is used
   */
  Optional<tir::Var> SplitInnerToOuter(int loop_id, Array<Optional<tir::Var>> factors,
                                       String name_hint = "");
  /*!
   * \brief Reorder the loops surrounding the current block
   * \param after_ids The indices of the loop after reordering
   */
  void Reorder(Array<Integer> after_ids);
  /*!
   * \brief Compute the current block at its sibling with a specific offset, on its specific loop
   * \param offset Offset of the sibling
   * \param loop_id The index of the loop to be computed_at on the sibling
   */
  void ComputeAtOffset(int offset, int loop_id);
  /*!
   * \brief Move the cursor to its sibling with a specific offset
   * \param offset Offset of the sibling
   * \return The new cursor
   */
  MetaIR CursorMoveOffset(int offset);

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("meta_ir", &meta_ir);
    v->Visit("cursor", &cursor);
    v->Visit("instructions", &instructions);
    v->Visit("declared_vars", &declared_vars);
  }

  static constexpr const char* _type_key = "auto_scheduler.MetaSchedule";
  TVM_DECLARE_FINAL_OBJECT_INFO(MetaScheduleNode, Object);
};

class MetaSchedule : public ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param meta_ir The meta IR to be meta scheduled
   * \param cursor The cursor pointing to the MetaIR node for the current stage
   * \param instructions The instructions applied the meta ir
   * \param revokes The functions used to revoke instructions accordingly, used for Backtracking
   * \param declared_vars The variables that have been declared
   */
  MetaSchedule(MetaIR meta_ir, MetaIR cursor, Array<Instruction> instructions,
               std::vector<MetaScheduleNode::RevokeFunc> revokes, Array<tir::Var> declared_vars);
  /*!
   * \brief Construct from an initial meta ir
   * \param meta_ir The initial meta ir
   */
  explicit MetaSchedule(MetaIR meta_ir);

  // Below are definition for mutable + TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(MetaSchedule, ObjectRef, MetaScheduleNode);
};

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // SRC_AUTO_SCHEDULER_META_SCHEDULE_H_
