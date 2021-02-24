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

Schedule::Schedule(PrimFunc func, StmtSRef root,
                   std::unordered_map<const StmtNode*, StmtSRef> stmt2ref,
                   std::unordered_map<StmtSRef, BlockScope, ObjectPtrHash, ObjectPtrEqual> scopes) {
  ObjectPtr<ScheduleNode> n = make_object<ScheduleNode>();
  n->func = std::move(func);
  n->root = std::move(root);
  n->stmt2ref = std::move(stmt2ref);
  n->scopes = std::move(scopes);
  data_ = std::move(n);
}

Array<StmtSRef> ScheduleNode::GetBlock(const String& tag) const {
  std::vector<StmtSRef> ret, scope_stack;
  scope_stack.push_back(root);
  while (!scope_stack.empty()) {
    StmtSRef scope = scope_stack.back();
    scope_stack.pop_back();
    ICHECK(GetRef<Stmt>(scope->stmt).as<BlockNode>());
    for (const auto& block : Blocks(scope)) {
      if (GetRef<Stmt>(block->stmt).as<BlockNode>()->name_hint == tag) {
        ret.push_back(block);
      }
      scope_stack.push_back(block);
    }
  }
  return ret;
}

Array<StmtSRef> ScheduleNode::GetBlock(const Buffer& buffer, StmtSRef scope) const {
  if (!scope.defined()) {
    scope = root;
  }
  ICHECK(GetRef<Stmt>(scope->stmt).as<BlockNode>());
  ICHECK_GT(scopes.count(scope), 0);
  const auto& buffer_writers = scopes.at(scope)->buffer_writers;
  if (buffer_writers.count(buffer)) {
    return buffer_writers.at(buffer);
  } else {
    return Array<StmtSRef>();
  }
}

Array<StmtSRef> ScheduleNode::Blocks(StmtSRef scope) const {
  if (!scope.defined()) {
    scope = root;
  }
  ICHECK(scope->stmt->IsInstance<BlockNode>());
  ICHECK_GT(scopes.count(scope), 0);
  const auto& buffer_writers = scopes.at(scope)->buffer_writers;
  std::unordered_set<StmtSRef, ObjectPtrHash, ObjectPtrEqual> collect;
  for (const auto& x : buffer_writers) {
    for (const auto& block : x.second) {
      collect.insert(block);
    }
  }
  Array<StmtSRef> ret;
  for (const auto& block : collect) ret.push_back(block);
  return ret;
}

Array<StmtSRef> ScheduleNode::GetChildBlocks(const StmtSRef& parent_sref) const {
  std::vector<StmtSRef> result;
  PreOrderVisit(GetRef<Stmt>(parent_sref->stmt), [&result, this](const ObjectRef& node) {
    if (const auto* block = node.as<BlockNode>()) {
      result.push_back(stmt2ref.at(block));
      return false;
    }
    return true;
  });
  return result;
}

StmtSRef ScheduleNode::GetParentBlockSRef(const StmtSRef& sref) const {
  for (const StmtSRefNode* ptr = sref->parent; ptr != nullptr; ptr = ptr->parent) {
    if (ptr->stmt->IsInstance<BlockNode>()) {
      return GetRef<StmtSRef>(ptr);
    }
  }
  LOG(FATAL) << "ValueError: Cannot find a father block";
  throw;
}

BlockScope ScheduleNode::GetParentScope(const StmtSRef& sref) const {
  return scopes.at(GetParentBlockSRef(sref));
}

Array<StmtSRef> ScheduleNode::GetAxes(const StmtSRef& block) const {
  if (!block->parent) return Array<StmtSRef>();
  Array<StmtSRef> ret;
  StmtSRef sref = GetRef<StmtSRef>(block->parent);
  while (!GetRef<Stmt>(sref->stmt).as<BlockNode>()) {
    if (GetRef<Stmt>(sref->stmt).as<ForNode>()) {
      ret.push_back(sref);
    }
    sref = GetRef<StmtSRef>(sref->parent);
  }
  return Array<StmtSRef>(ret.rbegin(), ret.rend());
}

void ScheduleNode::register_reducer(const CommReducer& comm_reducer) {
  this->reducers.push_back(comm_reducer);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<StmtSRefNode>([](const ObjectRef& node, ReprPrinter* p) {
      const auto* op = static_cast<const StmtSRefNode*>(node.get());
      if (const auto* loop = GetRef<Stmt>(op->stmt).as<ForNode>()) {
        p->PrintIndent();
        p->stream << "for ";
        p->Print(loop->loop_var);
        p->stream << " = ";
        p->Print(loop->min);
        p->stream << " to ";
        p->Print(loop->extent);
      } else {
        p->Print(Downcast<Block>(GetRef<Stmt>(op->stmt)));
      }
    });

TVM_REGISTER_NODE_TYPE(ScheduleNode);

TVM_REGISTER_GLOBAL("tir.schedule.GetStmtSRef")
    .set_body_typed<StmtSRef(Schedule, Stmt)>([](Schedule schedule, Stmt stmt) {
      return schedule->stmt2ref.at(stmt.operator->());
    });

TVM_REGISTER_GLOBAL("tir.schedule.GetStmt").set_body_typed<Stmt(StmtSRef)>([](StmtSRef sref) {
  return GetRef<Stmt>(sref->stmt);
});

TVM_REGISTER_GLOBAL("tir.schedule.GetBlocksFromTag")
    .set_body_typed<Array<StmtSRef>(Schedule, std::string)>([](Schedule schedule, std::string tag) {
      return schedule->GetBlock(tag);
    });

TVM_REGISTER_GLOBAL("tir.schedule.GetBlocksFromBuffer")
    .set_body_typed<Array<StmtSRef>(Schedule, Buffer, StmtSRef)>([](Schedule schedule,
                                                                    Buffer buffer, StmtSRef scope) {
      return schedule->GetBlock(buffer, scope);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetAxes")
    .set_body_typed<Array<StmtSRef>(Schedule, StmtSRef)>([](Schedule schedule, StmtSRef scope) {
      return schedule->GetAxes(scope);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleRegisterReducer")
    .set_body_typed<void(Schedule, CommReducer)>([](Schedule schedule, CommReducer comm_reducer) {
      schedule->register_reducer(comm_reducer);
    });

// dependency graph
TVM_REGISTER_GLOBAL("tir.schedule.GetSuccessors")
    .set_body_typed<Array<DepEdge>(Schedule, StmtSRef, StmtSRef)>([](Schedule schedule,
                                                                     StmtSRef scope,
                                                                     StmtSRef block) {
      return schedule->scopes[scope]->GetSuccessors(block);
    });

TVM_REGISTER_GLOBAL("tir.schedule.GetPredecessors")
    .set_body_typed<Array<DepEdge>(Schedule, StmtSRef, StmtSRef)>([](Schedule schedule,
                                                                     StmtSRef scope,
                                                                     StmtSRef block) {
      return schedule->scopes[scope]->GetPredecessors(block);
    });

}  // namespace tir
}  // namespace tvm
