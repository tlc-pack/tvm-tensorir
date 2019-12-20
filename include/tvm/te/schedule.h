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

namespace tvm {
namespace te {

/*!
 * \brief The container of stmt schedulable ref.
 */
class StmtSRefNode : public Node {
 public:
  /*! \brief The corresponding stmt node */
  const StmtNode* node;
  /*! \brief The parent sref */
  StmtSRefNode* parent{nullptr};
  /*! \brief The location in an array if parent contains SeqStmt. */
  int64_t seq_index{-1};

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "StmtSRef";
  TVM_DECLARE_NODE_TYPE_INFO(StmtSRefNode, Node);
};

/*!
 * \brief The stmt schedulable ref.
 */
class StmtSRef : public NodeRef {
 public:
  StmtSRef(const StmtNode* node, StmtSRefNode* parent, int64_t seq_index = -1);

  StmtSRefNode* operator->() {
    return static_cast<StmtSRefNode*>(NodeRef::get_mutable());
  }
  TVM_DEFINE_NODE_REF_METHODS(StmtSRef, NodeRef, StmtSRefNode);
};

class ScheduleNode : public Node {
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

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("func", &func);
  }

  static constexpr const char* _type_key = "te.Schedule";
  TVM_DECLARE_NODE_TYPE_INFO(ScheduleNode, Node);
};

class Schedule : public NodeRef {
 public:
  /*!
   * \brief Create a new schedule
   * \param func The function to be scheduled
   * \return The schedule
   * */
  static Schedule Create(const Function& func);

  /*!
   * \brief replace part of AST with new stmt
   * \param ref The schedulable reference of the old stmt
   * \param target The new stmt
   * */
  void Replace(StmtSRef ref, Stmt target);

  TVM_DEFINE_NODE_REF_METHODS(Schedule, NodeRef, ScheduleNode);

  ScheduleNode* operator->() {
    return static_cast<ScheduleNode*>(NodeRef::get_mutable());
  }

 private:
  void UpdateChildren(const Stmt& stmt, StmtSRefNode* father);

  void UpdateSRef(StmtSRefNode* sref, const Stmt& stmt);
};

}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_SCHEDULE_H_
