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
