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
#include "./instruction.h"  // NOLINT(build/include)

#include "./auto_scheduler_utils.h"

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_NODE_TYPE(InstructionNode);
TVM_REGISTER_NODE_TYPE(DeclIntVarNode);
TVM_REGISTER_NODE_TYPE(SplitInnerToOuterNode);
TVM_REGISTER_NODE_TYPE(ReorderNode);
TVM_REGISTER_NODE_TYPE(ComputeAtOffsetNode);
TVM_REGISTER_NODE_TYPE(CursorMoveOffsetNode);

DeclIntVar::DeclIntVar(tir::Var var, Array<Integer> choices) {
  ObjectPtr<DeclIntVarNode> n = make_object<DeclIntVarNode>();
  n->var = std::move(var);
  n->choices = std::move(choices);
  data_ = std::move(n);
}

SplitInnerToOuter::SplitInnerToOuter(int loop_id, Array<Optional<tir::Var>> factors,
                                     Optional<tir::Var> inferred) {
  CHECK(!factors.empty()) << "ValueError: SplitInnerToOuter requires non-empty factors";
  ObjectPtr<SplitInnerToOuterNode> n = make_object<SplitInnerToOuterNode>();
  n->loop_id = loop_id;
  n->factors = std::move(factors);
  n->inferred = std::move(inferred);
  data_ = std::move(n);
}

Reorder::Reorder(Array<Integer> after_ids) {
  ObjectPtr<ReorderNode> n = make_object<ReorderNode>();
  n->after_ids = std::move(after_ids);
  data_ = std::move(n);
}

ComputeAtOffset::ComputeAtOffset(int offset, int loop_id) {
  ObjectPtr<ComputeAtOffsetNode> n = make_object<ComputeAtOffsetNode>();
  n->offset = offset;
  n->loop_id = loop_id;
  data_ = std::move(n);
}

CursorMoveOffset::CursorMoveOffset(int offset) {
  ObjectPtr<CursorMoveOffsetNode> n = make_object<CursorMoveOffsetNode>();
  n->offset = offset;
  data_ = std::move(n);
}

inline int FindIndex(const Array<tir::StmtSRef>& siblings, const tir::StmtSRef& element) {
  int n = siblings.size();
  for (int i = 0; i < n; ++i) {
    if (element.same_as(siblings[i])) {
      return i;
    }
  }
  LOG(FATAL) << "ValueError: Cannot find cursor in its parent's children.\nCursor is:\n" << element;
  throw;
}

void DeclIntVarNode::ApplyToSchedule(const Map<tir::Var, PrimExpr>& sampled_vars,
                                     ScheduleStatus* status) const {
  CHECK(sampled_vars.count(var)) << "ValueError: Variable with name hint \"" << var->name_hint
                                 << "\" is not defined";
  PrimExpr value = sampled_vars.at(var);
  // Type check for sampled_vars[var]
  const auto* integer = value.as<IntImmNode>();
  CHECK(integer != nullptr) << "TypeError: A sample of the integer variable \"" << var
                            << "\" is expected to be an integer, but gets type \""
                            << value->GetTypeKey() << "\" of value: " << value;
  // Check if it is in-distribution
  bool found = false;
  for (const Integer& candidate : choices) {
    if (candidate->value == integer->value) {
      found = true;
    }
  }
  CHECK(found) << "TypeError: The integer variable \"" << var
               << "\" is expected to be sampled within " << choices
               << ", but gets out-of-distribution value: " << integer->value;
  // Set the symbol table
  status->symbol_table.Set(var, value);
}

void SplitInnerToOuterNode::ApplyToSchedule(ScheduleStatus* status) const {
  Array<tir::StmtSRef> axes = status->schedule->GetLoopsInScope(status->cursor);
  CHECK(0 <= loop_id && loop_id < static_cast<int>(axes.size()))
      << "IndexError: indexing axes out of bound, index = " << loop_id
      << ", but number of axes are " << axes.size();
  tir::StmtSRef target = axes[loop_id];
  int n_factors = static_cast<int>(factors.size());
  for (int i = n_factors - 1; i >= 0 && factors[i].defined(); --i) {
    tir::Var var = factors[i].value();
    CHECK(status->symbol_table.count(var))
        << "ValueError: The " << i << "-th factor of SplitInnerToOuter, " << var->name_hint
        << ", is not found in the symbol table";
    const auto* loop = target->GetStmt<tir::LoopNode>();
    CHECK(loop != nullptr) << "TypeError: Expects LoopNode, but gets: " << target->GetTypeKey();
    PrimExpr factor = status->symbol_table.at(var);
    PrimExpr nparts = floordiv(loop->extent + factor - 1, factor);
    Array<tir::StmtSRef> split_result = status->schedule->split(target, nparts, factor);
    CHECK_EQ(split_result.size(), 2);
    target = split_result[0];
  }
  if (inferred.defined()) {
    for (int i = 0; i < n_factors && factors[i].defined(); ++i) {
      tir::Var var = factors[i].value();
      CHECK(status->symbol_table.count(var))
          << "ValueError: The " << i << "-th factor of SplitInnerToOuter, " << var->name_hint
          << ", is not found in the symbol table";
      const auto* loop = target->GetStmt<tir::LoopNode>();
      CHECK(loop != nullptr) << "TypeError: Expects LoopNode, but gets: " << target->GetTypeKey();
      PrimExpr nparts = status->symbol_table.at(var);
      PrimExpr factor = floordiv(loop->extent + nparts - 1, nparts);
      Array<tir::StmtSRef> split_result = status->schedule->split(target, nparts, factor);
      CHECK_EQ(split_result.size(), 2);
      target = split_result[1];
    }
    // Set the symbol table
    const auto* loop = target->GetStmt<tir::LoopNode>();
    CHECK(loop != nullptr) << "TypeError: Expects LoopNode, but gets: " << target->GetTypeKey();
    status->symbol_table.Set(inferred.value(), loop->extent);
  }
}

void ReorderNode::ApplyToSchedule(ScheduleStatus* status) const {
  Array<tir::StmtSRef> before_axes = status->schedule->GetLoopsInScope(status->cursor);
  CHECK_EQ(before_axes.size(), after_ids.size())
      << "ValueError: Unequal number of axes, gets before_axes = " << before_axes
      << ", but after_ids = " << after_ids;
  int n = before_axes.size();
  std::vector<int> from(n, -1);
  for (int i = 0; i < n; ++i) {
    int j = after_ids[i];
    from[j] = i;
  }
  std::vector<tir::StmtSRef> after_axes;
  for (int j : from) {
    after_axes.push_back(before_axes[j]);
  }
  status->schedule->reorder(after_axes);
}

void ComputeAtOffsetNode::ApplyToSchedule(ScheduleStatus* status) const {
  tir::StmtSRef parent = status->schedule->GetParentBlockSRef(status->cursor);
  Array<tir::StmtSRef> siblings = status->schedule->GetChildBlocks(parent);
  int index = FindIndex(siblings, status->cursor);
  int target_index = index + offset;
  CHECK(0 <= target_index && target_index < static_cast<int>(siblings.size()))
      << "IndexError: offset out-of-bound, index = " << index << ", offset = " << offset
      << ", number of siblings in the subtree = " << siblings.size();
  tir::StmtSRef target = siblings[target_index];
  Array<tir::StmtSRef> axes = status->schedule->GetLoopsInScope(target);
  CHECK(0 <= loop_id && loop_id < static_cast<int>(axes.size()))
      << "IndexError: indexing axes out of bound, index = " << loop_id
      << ", but number of axes are " << axes.size();
  status->schedule->compute_at(status->cursor, axes[loop_id]);
  status->cursor = siblings[target_index - 1];
}

void CursorMoveOffsetNode::ApplyToSchedule(ScheduleStatus* status) const {
  tir::StmtSRef parent = status->schedule->GetParentBlockSRef(status->cursor);
  Array<tir::StmtSRef> siblings = status->schedule->GetChildBlocks(parent);
  int index = FindIndex(siblings, status->cursor);
  int target_index = index + offset;
  CHECK(0 <= target_index && target_index < static_cast<int>(siblings.size()))
      << "IndexError: offset out-of-bound, index = " << index << ", offset = " << offset
      << ", number of siblings in the subtree = " << siblings.size();
  status->cursor = siblings[target_index];
}

TVM_REGISTER_GLOBAL("auto_scheduler.DeclIntVar")
    .set_body_typed([](tir::Var var, Array<Integer> choices) { return DeclIntVar(var, choices); });

TVM_REGISTER_GLOBAL("auto_scheduler.SplitInnerToOuter")
    .set_body_typed([](int loop_id, Array<Optional<tir::Var>> factors, tir::Var inferred) {
      return SplitInnerToOuter(loop_id, factors, inferred);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.Reorder").set_body_typed([](Array<Integer> after_ids) {
  return Reorder(after_ids);
});

TVM_REGISTER_GLOBAL("auto_scheduler.ComputeAtOffset").set_body_typed([](int offset, int loop_id) {
  return ComputeAtOffset(offset, loop_id);
});

TVM_REGISTER_GLOBAL("auto_scheduler.CursorMoveOffset").set_body_typed([](int offset) {
  return CursorMoveOffset(offset);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DeclIntVarNode>([](const ObjectRef& obj, ReprPrinter* p) {
      const auto* n = obj.as<DeclIntVarNode>();
      CHECK(n);
      p->stream << n->var->name_hint << " = "
                << "DeclIntVarNode(choices=[";
      bool is_first = true;
      for (const Integer& choice : n->choices) {
        if (is_first) {
          is_first = false;
          p->stream << choice;
        } else {
          p->stream << ", " << choice;
        }
      }
      p->stream << "])";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SplitInnerToOuterNode>([](const ObjectRef& obj, ReprPrinter* p) {
      const auto* n = obj.as<SplitInnerToOuterNode>();
      CHECK(n);
      if (n->inferred.defined()) {
        p->stream << n->inferred.value()->name_hint << " = ";
      }
      p->stream << "SplitInnerToOuter(loop_id=" << n->loop_id << ", factors=[";
      bool is_first = true;
      for (const Optional<tir::Var> factor : n->factors) {
        if (is_first) {
          is_first = false;
        } else {
          p->stream << ", ";
        }
        if (factor.defined()) {
          p->stream << factor.value();
        } else {
          p->stream << "None";
        }
      }
      p->stream << "])";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ReorderNode>([](const ObjectRef& obj, ReprPrinter* p) {
      const auto* n = obj.as<ReorderNode>();
      CHECK(n);
      p->stream << "Reorder(after_ids=[";
      bool is_first = true;
      for (const Integer& after_id : n->after_ids) {
        if (is_first) {
          is_first = false;
          p->stream << after_id;
        } else {
          p->stream << ", " << after_id;
        }
      }
      p->stream << "])";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ComputeAtOffsetNode>([](const ObjectRef& obj, ReprPrinter* p) {
      const auto* n = obj.as<ComputeAtOffsetNode>();
      CHECK(n);
      p->stream << "ComputeAt(offset=" << n->offset << ", loop_id=" << n->loop_id << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<CursorMoveOffsetNode>([](const ObjectRef& obj, ReprPrinter* p) {
      const auto* n = obj.as<CursorMoveOffsetNode>();
      CHECK(n);
      p->stream << "CursorMove(offset=" << n->offset << ")";
    });

}  // namespace auto_scheduler
}  // namespace tvm
