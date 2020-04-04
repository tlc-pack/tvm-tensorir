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

#ifndef TVM_TIR_STMT_SREF_H_
#define TVM_TIR_STMT_SREF_H_
#include <tvm/tir/stmt.h>

namespace tvm {
namespace tir {

/*!
 * \brief The container of stmt schedulable ref.
 */
class StmtSRefNode : public Object {
 public:
  /*! \brief The corresponding stmt node */
  const StmtNode* node;
  /*! \brief The parent sref */
  StmtSRefNode* parent{nullptr};
  /*! \brief The location in an array if parent contains SeqStmt. */
  int64_t seq_index{-1};
  /*! \breif Whether the */
  bool binding_valid;

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "StmtSRef";
  TVM_DECLARE_FINAL_OBJECT_INFO(StmtSRefNode, Object);
};

/*!
 * \brief The stmt schedulable ref.
 */
class StmtSRef : public ObjectRef {
 public:
  StmtSRef(const StmtNode* node, StmtSRefNode* parent, int64_t seq_index = -1);

  StmtSRefNode* operator->() {
    return get();
  }

  StmtSRefNode* get() {
    return static_cast<StmtSRefNode*>(ObjectRef::get_mutable());
  }

  StmtSRefNode* const get() const {
    return static_cast<StmtSRefNode*>(ObjectRef::get_mutable());
  }

  TVM_DEFINE_OBJECT_REF_METHODS(StmtSRef, ObjectRef, StmtSRefNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_STMT_SREF_H_
