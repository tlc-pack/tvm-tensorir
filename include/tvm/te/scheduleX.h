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
#include <tvm/te/ir.h>

namespace tvm {
namespace te {

// node container
class StmtSRefNode : public Node {
 public:
  // the corresponding stmt node
  const StmtNode* node;
  // parent sref
  StmtSRefNode* parent{nullptr};
  // location in an array if parent contains SeqStmt.
  int64_t seq_index{-1};

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "StmtSRef";
  TVM_DECLARE_NODE_TYPE_INFO(StmtSRefNode, Node);
};

// strong ref
class StmtSRef : public NodeRef {
 public:
  StmtSRef(const StmtNode* node, StmtSRefNode* parent, int64_t seq_index = -1);
  StmtSRefNode* operator->() {
    return static_cast<StmtSRefNode*>(NodeRef::get_mutable());
  }
  TVM_DEFINE_NODE_REF_METHODS(StmtSRef, NodeRef, StmtSRefNode);
};

class ScheduleXNode : public Node {
 public:
  // function
  Function func;
  // root
  StmtSRef root;
  // stmt
  std::unordered_map<const StmtNode*, StmtSRef> stmt2ref;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("func", &func);
  }

  static constexpr const char* _type_key = "te.ScheduleX";
  TVM_DECLARE_NODE_TYPE_INFO(ScheduleXNode, Node);

  void UpdateSRef(StmtSRefNode* sref, Stmt stmt) {
    stmt2ref[stmt.operator->()] = GetRef<StmtSRef>(sref);
    stmt2ref.erase(sref->node);
    sref->node = stmt.operator->();
  }

};

class ScheduleX : public NodeRef {
 public:
  static ScheduleX Create(Function func);

  void Replace(StmtSRef ref, Stmt target);

  void UpdateChildren(const Stmt& stmt, StmtSRefNode* father);

  TVM_DEFINE_NODE_REF_METHODS(ScheduleX, NodeRef, ScheduleXNode);

  ScheduleXNode* operator->() {
    return static_cast<ScheduleXNode*>(NodeRef::get_mutable());
  }
};
}
}