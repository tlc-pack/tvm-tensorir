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
#include "./schedule_common.h"

namespace tvm {
namespace tir {

/*! \brief A helper class to create ScheduleNode::stmt2ref */
class SRefMapCreator : public StmtVisitor {
 public:
  /*!
   * \brief The entry function of SRefMapCreator
   * \param func The function to be analyzed
   * \return The stmt2ref map
   */
  static std::unordered_map<const StmtNode*, StmtSRef> Create(const PrimFunc& func) {
    SRefMapCreator visitor;
    visitor(func->body);
    std::unordered_map<const StmtNode*, StmtSRef> ret = std::move(visitor.stmt2ref);
    return ret;
  }
  /*!
   * \brief Add a new statement to the stack, which becomes the current scope
   * \param stmt A loop statement or a block statement
   */
  void PushSRef(const StmtNode* stmt) {
    StmtSRefNode* parent = frames.empty() ? nullptr : frames.back().get();
    frames.push_back(StmtSRef(stmt, parent));
  }
  /*! \brief Pop the top of the scope and record it in stmt2ref map */
  void PopSRef() {
    const StmtSRef& sref = frames.back();
    stmt2ref[sref->stmt] = sref;
    frames.pop_back();
  }
  // Create a StmtSRef for BlockNode
  void VisitStmt_(const BlockNode* stmt) override {
    PushSRef(stmt);
    StmtVisitor::VisitStmt_(stmt);
    PopSRef();
  }
  // Create a StmtSRef for LoopNode
  void VisitStmt_(const LoopNode* stmt) override {
    PushSRef(stmt);
    StmtVisitor::VisitStmt_(stmt);
    PopSRef();
  }
  // Set `seq_index` information for SeqStmtNode
  void VisitStmt_(const SeqStmtNode* seq_stmt) override {
    StmtVisitor::VisitStmt_(seq_stmt);
    int index = 0;
    for (const Stmt& stmt : seq_stmt->seq) {
      const StmtNode* node;
      if (const auto* realize = stmt.as<BlockRealizeNode>()) {
        node = realize->block.get();
      } else {
        // TODO(@junrushao1994): seems that we should assert it as LoopNode?
        node = stmt.get();
      }
      stmt2ref.at(node)->seq_index = index++;
    }
  }
  /*! \brief The stack frame used to indicate the current scope */
  std::vector<StmtSRef> frames;
  /*! \brief The result stmt2ref */
  std::unordered_map<const StmtNode*, StmtSRef> stmt2ref;
};

/*! \brief A helper class to create ScheduleNode::scopes */
class ScopeMapCreator : public StmtVisitor {
 public:
  /*!
   * \brief Entry function, use stmt2ref and the TIR PrimFunc's body for analysis
   * \param stmt2ref The ScopeNode::stmt2ref that is just constructed by SRefMapCreator
   * \param realize The body of the TIR PrimFunc
   * \return The ScheduleNode::scopes created
   */
  static std::unordered_map<StmtSRef, Scope, ObjectPtrHash, ObjectPtrEqual> Create(
      const std::unordered_map<const StmtNode*, StmtSRef>& stmt2ref,
      const BlockRealizeNode* realize) {
    ScopeMapCreator creator(stmt2ref);
    creator(realize->block);
    std::unordered_map<StmtSRef, Scope, ObjectPtrHash, ObjectPtrEqual> ret =
        std::move(creator.scopes);
    return ret;
  }
  /*! \brief Constructor. Requires ScheduleNode::stmt2ref to create ScheduleNode::scopes. */
  explicit ScopeMapCreator(const std::unordered_map<const StmtNode*, StmtSRef>& stmt2ref)
      : stmt2ref(stmt2ref) {}
  // For each BlockNode Create its corresponding ScopeNode
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
  /*! \brief A stack frame indicating the information being gathered but not completed */
  struct Frame {
    /*! \brief The scope to be created. */
    Scope scope;
    /*! \brief ScopeNode::buffer_writers exists, but ScopeNode::buffer_readers does not. */
    std::unordered_map<Buffer, Array<StmtSRef>, ObjectPtrHash, ObjectPtrEqual> buffer_readers;
  };
  /*! \brief The ScheduleNode::stmt2ref provided. */
  const std::unordered_map<const StmtNode*, StmtSRef>& stmt2ref;
  /*! \brief Stack frame of the DFS visit. */
  std::vector<Frame> frames;
  /*! \brief The result ScheduleNode::scopes being created. */
  std::unordered_map<StmtSRef, Scope, ObjectPtrHash, ObjectPtrEqual> scopes;
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
