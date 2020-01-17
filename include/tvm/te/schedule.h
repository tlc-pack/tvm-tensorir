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

#ifndef TVM_TE_SCHEDULE_H_
#define TVM_TE_SCHEDULE_H_
#include <tvm/te/ir.h>
#include <tvm/te/stmt_sref.h>
#include <tvm/te/scope.h>
#include <string>
#include <unordered_map>

namespace tvm {
namespace te {


class ScheduleNode : public Object {
 public:
  /*! \brief The function to be scheduled */
  Function func;
  /*! \brief The root of schedulable reference tree */
  StmtSRef root;
  /*!
   * \brief The mapping from stmt to its schedulable reference node
   * \note This is a hint to improve mutation efficiency
   * */
  std::unordered_map<const StmtNode*, StmtSRef> stmt2ref;
  /*! \brief The block scopes of each block */
  std::unordered_map<StmtSRef, Scope, ObjectHash, ObjectEqual> scopes_;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("func", &func);
    v->Visit("root", &root);
  }

  static constexpr const char* _type_key = "te.Schedule";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleNode, Object);
};

class Schedule : public ObjectRef {
 public:
  /*!
   * \brief Create a new schedule
   * \param func The function to be scheduled
   * \return The schedule
   */
  static Schedule Create(Function func);

  /*!
   * \brief replace part of AST with new stmt
   * \param ref The schedulable reference of the old stmt
   * \param target The new stmt
   */
  void Replace(StmtSRef ref, Stmt target);

  /*!
   * \brief Get block from its tag
   * \param scope The block scope
   * \param tag The query tag
   * \return the block schedulable reference list
   */
  Array<StmtSRef> GetBlock(const std::string& tag, StmtSRef scope = StmtSRef()) const;

  /*!
   * \brief Get block from its output tensor
   * \param scope The block scope
   * \param buffer The query buffer
   * \return the block schedulable reference list
   */
  Array<StmtSRef> GetBlock(const Buffer& buffer, StmtSRef scope = StmtSRef()) const;

  /*!
   * \brief Get all blocks in the scope
   * \param scope The block scope
   * \return the block schedulable reference list
   */
  Array<StmtSRef> Blocks(StmtSRef scope) const;

  /*!
   * \brief Get loops of the block
   * \param block The query block
   * \return the loop sref list
   */
  Array<StmtSRef> GetLoopsInScope(const StmtSRef& block) const;

  /*!
   * \brief Get the scope of the schedulable reference
   * \param node The queried node
   * \return the block scope reference
   */
  StmtSRef GetScope(StmtSRef node) const;

  /*!
   * \brief fuse two consecutive loops of one computation.
   * \param outer The outer loop
   * \param inner The inner loop
   * \return the fused loop
   * */
  StmtSRef fuse(const StmtSRef& outer, const StmtSRef& inner);

  /*!
   * \brief split a specified loop into two loops by factor.
   * \param node The loop to be split
   * \param factor The split factor
   * \return the loops after splitting
   * */
  Array<StmtSRef> split(const StmtSRef& node, const Expr& factor);

  TVM_DEFINE_OBJECT_REF_METHODS(Schedule, ObjectRef, ScheduleNode);

  ScheduleNode* operator->() {
    return static_cast<ScheduleNode*>(ObjectRef::get_mutable());
  }

 private:
  void UpdateSRef(StmtSRefNode* sref, const Stmt& stmt);

  static Array<Stmt> GetChildren(const Stmt& stmt);
};

}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_SCHEDULE_H_
