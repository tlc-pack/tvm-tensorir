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

#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

class SRefMapCreator : public StmtVisitor {
 public:
  static std::unordered_map<const StmtNode*, StmtSRef> Create(const PrimFunc& func) {
    SRefMapCreator visitor;
    visitor(func->body);
    std::unordered_map<const StmtNode*, StmtSRef> ret = std::move(visitor.stmt2ref);
    return ret;
  }

  void PushSRef(const StmtNode* stmt) {
    StmtSRefNode* parent = frames.empty() ? nullptr : frames.back().get();
    frames.push_back(StmtSRef(stmt, parent));
  }

  void PopSRef() {
    const StmtSRef& sref = frames.back();
    stmt2ref[sref->stmt] = sref;
    frames.pop_back();
  }

  void VisitStmt_(const SeqStmtNode* seq_stmt) override {
    StmtVisitor::VisitStmt_(seq_stmt);
    int index = 0;
    for (const Stmt& stmt : seq_stmt->seq) {
      const StmtNode* node;
      if (const auto* realize = stmt.as<BlockRealizeNode>()) {
        node = realize->block.get();
      } else {
        node = stmt.get();
      }
      stmt2ref[node]->seq_index = index++;
    }
  }

  void VisitStmt_(const BlockNode* stmt) override {
    PushSRef(stmt);
    StmtVisitor::VisitStmt_(stmt);
    PopSRef();
  }

  void VisitStmt_(const LoopNode* stmt) override {
    PushSRef(stmt);
    StmtVisitor::VisitStmt_(stmt);
    PopSRef();
  }

  std::vector<StmtSRef> frames;
  std::unordered_map<const StmtNode*, StmtSRef> stmt2ref;
};

class ScopeMapCreator : public StmtVisitor {
 public:
  explicit ScopeMapCreator(const std::unordered_map<const StmtNode*, StmtSRef>& stmt2ref)
      : stmt2ref(stmt2ref) {}

  void VisitStmt_(const BlockNode* block) override {
    // Create a new scope
    frames.push_back(Frame());
    // Visit the children to collect info for the block scope
    StmtVisitor::VisitStmt_(block);
    // Save the info collected to ScopeMap
    const auto& sref = stmt2ref.at(block);
    scopes[sref] = std::move(frames.back().scope);
    // Pop stack
    frames.pop_back();
    // Update parent scope if exists
    if (!frames.empty()) {
      auto& top = frames.back();
      top.scope.AddChildBlock(sref, &top.buffer_readers);
    }
  }

  static std::unordered_map<StmtSRef, Scope, ObjectHash, ObjectEqual> Create(
      const std::unordered_map<const StmtNode*, StmtSRef>& stmt2ref,
      const BlockRealizeNode* realize) {
    ScopeMapCreator creator(stmt2ref);
    creator(realize->block);
    std::unordered_map<StmtSRef, Scope, ObjectHash, ObjectEqual> ret = std::move(creator.scopes);
    return ret;
  }

  struct Frame {
    Scope scope;
    std::unordered_map<Buffer, Array<StmtSRef>, ObjectHash, ObjectEqual> buffer_readers;
  };

  const std::unordered_map<const StmtNode*, StmtSRef>& stmt2ref;
  std::unordered_map<StmtSRef, Scope, ObjectHash, ObjectEqual> scopes;
  std::vector<Frame> frames;
};

Schedule ScheduleNode::Create(PrimFunc func) {
  const BlockRealizeNode* realize = func->body.as<BlockRealizeNode>();
  CHECK(realize != nullptr) << "TypeError: body of PrimFunc is expected to be BlockRealize";
  ObjectPtr<ScheduleNode> n = make_object<ScheduleNode>();
  n->stmt2ref = SRefMapCreator::Create(func);
  n->scopes = ScopeMapCreator::Create(n->stmt2ref, realize);
  n->func = std::move(func);
  n->root = n->stmt2ref.at(realize->block.get());
  n->ValidateLoops();
  for (const auto& it : n->scopes) {
    n->ValidateRegionCover(it.first);
  }
  return Schedule(n);
}

TVM_REGISTER_GLOBAL("tir.schedule.CreateSchedule").set_body_typed(ScheduleNode::Create);

}  // namespace tir
}  // namespace tvm
