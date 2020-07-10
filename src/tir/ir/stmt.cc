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

/*!
 * \file tvm/tir/stmt.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt.h>

#include "../schedule/schedule_common.h"

namespace tvm {
namespace tir {

// LetStmt
LetStmt::LetStmt(Var var, PrimExpr value, Stmt body) {
  CHECK(value.defined());
  CHECK(body.defined());
  CHECK_EQ(value.dtype(), var.dtype());

  ObjectPtr<LetStmtNode> node = make_object<LetStmtNode>();
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.LetStmt").set_body_typed([](Var var, PrimExpr value, Stmt body) {
  return LetStmt(var, value, body);
});

TVM_REGISTER_NODE_TYPE(LetStmtNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LetStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const LetStmtNode*>(node.get());
      p->PrintIndent();
      p->stream << "let " << op->var << " = ";
      p->Print(op->value);
      p->stream << '\n';
      p->Print(op->body);
    });

// AttrStmt
AttrStmt::AttrStmt(ObjectRef node, String attr_key, PrimExpr value, Stmt body) {
  auto n = make_object<AttrStmtNode>();
  n->node = node;
  n->attr_key = std::move(attr_key);
  n->value = std::move(value);
  n->body = std::move(body);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("tir.AttrStmt")
    .set_body_typed([](ObjectRef node, String attr_key, PrimExpr value, Stmt body) {
      return AttrStmt(node, attr_key, value, body);
    });

TVM_REGISTER_NODE_TYPE(AttrStmtNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AttrStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AttrStmtNode*>(node.get());
      p->PrintIndent();
      p->stream << "// attr [";
      p->Print(op->node);
      p->stream << "] " << op->attr_key << " = ";
      p->Print(op->value);
      p->stream << '\n';
      p->Print(op->body);
    });

// AssertStmt
AssertStmt::AssertStmt(PrimExpr condition, PrimExpr message, Stmt body) {
  CHECK(condition.defined());
  CHECK(message.dtype() == DataType::Int(32) || message.as<StringImmNode>())
      << "TypeError: AssertStmt message must be an int or string:" << message << "\n";

  ObjectPtr<AssertStmtNode> node = make_object<AssertStmtNode>();
  node->condition = std::move(condition);
  node->message = std::move(message);
  node->body = std::move(body);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(AssertStmtNode);

TVM_REGISTER_GLOBAL("tir.AssertStmt")
    .set_body_typed([](PrimExpr condition, ObjectRef message, Stmt body) {
      if (const auto* str = message.as<StringObj>()) {
        auto msg = StringImm(str->data);
        return AssertStmt(condition, msg, body);
      } else {
        return AssertStmt(condition, Downcast<PrimExpr>(message), body);
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AssertStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AssertStmtNode*>(node.get());
      p->PrintIndent();
      p->stream << "assert(";
      p->Print(op->condition);
      p->stream << ", ";
      p->Print(op->message);
      p->stream << ")\n";
      p->Print(op->body);
    });

// For
For::For(Var loop_var, PrimExpr min, PrimExpr extent, ForType for_type, DeviceAPI device_api,
         Stmt body) {
  CHECK(min.defined());
  CHECK(extent.defined());
  CHECK(min.dtype().is_scalar());
  CHECK(extent.dtype().is_scalar());
  CHECK(loop_var.dtype().is_scalar());
  CHECK(body.defined());

  ObjectPtr<ForNode> node = make_object<ForNode>();
  node->loop_var = std::move(loop_var);
  node->min = std::move(min);
  node->extent = std::move(extent);
  node->for_type = for_type;
  node->device_api = device_api;
  node->body = std::move(body);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.For").set_body_typed([](Var loop_var, PrimExpr min, PrimExpr extent,
                                                 int for_type, int device_api, Stmt body) {
  return For(loop_var, min, extent, static_cast<ForType>(for_type),
             static_cast<DeviceAPI>(device_api), body);
});

TVM_REGISTER_NODE_TYPE(ForNode);

std::ostream& operator<<(std::ostream& out, ForType type) {  // NOLINT(*)
  switch (type) {
    case ForType::Serial:
      out << "for";
      break;
    case ForType::Parallel:
      out << "parallel";
      break;
    case ForType::Unrolled:
      out << "unrolled";
      break;
    case ForType::Vectorized:
      out << "vectorized";
      break;
  }
  return out;
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ForNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ForNode*>(node.get());
      p->PrintIndent();
      p->stream << op->for_type << " (" << op->loop_var << ", ";
      p->Print(op->min);
      p->stream << ", ";
      p->Print(op->extent);
      p->stream << ") {\n";

      p->indent += 2;
      p->Print(op->body);
      p->indent -= 2;

      p->PrintIndent();
      p->stream << "}\n";
    });

// Store
Store::Store(Var buffer_var, PrimExpr value, PrimExpr index, PrimExpr predicate) {
  CHECK(value.defined());
  CHECK(index.defined());
  CHECK(predicate.defined());
  CHECK_EQ(value.dtype().lanes(), index.dtype().lanes());
  CHECK_EQ(value.dtype().lanes(), predicate.dtype().lanes());

  ObjectPtr<StoreNode> node = make_object<StoreNode>();
  node->buffer_var = std::move(buffer_var);
  node->value = std::move(value);
  node->index = std::move(index);
  node->predicate = std::move(predicate);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.Store").set_body([](TVMArgs args, TVMRetValue* ret) {
  PrimExpr value = args[1];
  if (args.size() == 3) {
    *ret = Store(args[0], value, args[2], const_true(value.dtype().lanes()));
  } else {
    *ret = Store(args[0], value, args[2], args[3]);
  }
});

TVM_REGISTER_NODE_TYPE(StoreNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<StoreNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const StoreNode*>(node.get());
      p->PrintIndent();
      p->stream << op->buffer_var << "[";
      p->Print(op->index);
      p->stream << "] = ";
      p->Print(op->value);
      if (!is_one(op->predicate)) {
        p->stream << " if ";
        p->Print(op->predicate);
      }
      p->stream << '\n';
    });

// ProducerStore
ProducerStore::ProducerStore(DataProducer producer, PrimExpr value, Array<PrimExpr> indices) {
  ObjectPtr<ProducerStoreNode> node = make_object<ProducerStoreNode>();
  node->producer = std::move(producer);
  node->value = std::move(value);
  node->indices = std::move(indices);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.ProducerStore")
    .set_body_typed([](DataProducer producer, PrimExpr value, Array<PrimExpr> indices) {
      return ProducerStore(producer, value, indices);
    });

TVM_REGISTER_NODE_TYPE(ProducerStoreNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ProducerStoreNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ProducerStoreNode*>(node.get());
      p->PrintIndent();
      p->stream << op->producer->GetNameHint() << "[";
      for (size_t i = 0; i < op->indices.size(); ++i) {
        p->Print(op->indices[i]);
        if (i < op->indices.size() - 1) p->stream << ", ";
      }
      p->stream << "]";
      p->stream << " =";
      p->Print(op->value);
      p->stream << '\n';
    });

// Allocate
Allocate::Allocate(Var buffer_var, DataType dtype, Array<PrimExpr> extents, PrimExpr condition,
                   Stmt body) {
  for (size_t i = 0; i < extents.size(); ++i) {
    CHECK(extents[i].defined());
    CHECK(extents[i].dtype().is_scalar());
  }
  CHECK(body.defined());
  CHECK(condition.defined());
  CHECK(condition.dtype().is_bool());

  ObjectPtr<AllocateNode> node = make_object<AllocateNode>();
  node->buffer_var = std::move(buffer_var);
  node->dtype = dtype;
  node->extents = std::move(extents);
  node->condition = std::move(condition);
  node->body = std::move(body);
  data_ = std::move(node);
}

int32_t AllocateNode::constant_allocation_size(const Array<PrimExpr>& extents) {
  int64_t result = 1;
  for (size_t i = 0; i < extents.size(); ++i) {
    if (const IntImmNode* int_size = extents[i].as<IntImmNode>()) {
      result *= int_size->value;
      if (result > std::numeric_limits<int32_t>::max()) {
        return 0;
      }
    } else {
      return 0;
    }
  }
  return static_cast<int32_t>(result);
}

TVM_REGISTER_GLOBAL("tir.Allocate")
    .set_body_typed([](Var buffer_var, DataType type, Array<PrimExpr> extents, PrimExpr condition,
                       Stmt body) { return Allocate(buffer_var, type, extents, condition, body); });

TVM_REGISTER_NODE_TYPE(AllocateNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AllocateNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AllocateNode*>(node.get());
      p->PrintIndent();
      p->stream << "allocate " << op->buffer_var << "[" << op->dtype;
      for (size_t i = 0; i < op->extents.size(); ++i) {
        p->stream << " * ";
        p->Print(op->extents[i]);
      }
      p->stream << "]";
      if (!is_one(op->condition)) {
        p->stream << " if ";
        p->Print(op->condition);
      }
      p->stream << "\n";
      p->Print(op->body);
    });

// ProducerRealize
ProducerRealize::ProducerRealize(DataProducer producer, Region bounds, PrimExpr condition,
                                 Stmt body) {
  for (size_t i = 0; i < bounds.size(); ++i) {
    CHECK(bounds[i]->min.defined());
    CHECK(bounds[i]->extent.defined());
    CHECK(bounds[i]->min.dtype().is_scalar());
    CHECK(bounds[i]->extent.dtype().is_scalar());
  }
  CHECK(body.defined());
  CHECK(condition.defined());
  CHECK(condition.dtype().is_bool());

  ObjectPtr<ProducerRealizeNode> node = make_object<ProducerRealizeNode>();
  node->producer = std::move(producer);
  node->bounds = std::move(bounds);
  node->condition = std::move(condition);
  node->body = std::move(body);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.ProducerRealize")
    .set_body_typed([](DataProducer producer, Region bounds, PrimExpr condition, Stmt body) {
      return ProducerRealize(producer, bounds, condition, body);
    });

TVM_REGISTER_NODE_TYPE(ProducerRealizeNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ProducerRealizeNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ProducerRealizeNode*>(node.get());
      p->PrintIndent();
      p->stream << "producer_realize " << op->producer->GetNameHint() << "(";
      for (size_t i = 0; i < op->bounds.size(); ++i) {
        p->stream << "[";
        p->Print(op->bounds[i]->min);
        p->stream << ", ";
        p->Print(op->bounds[i]->extent);
        p->stream << "]";
        if (i < op->bounds.size() - 1) p->stream << ", ";
      }
      p->stream << ")";
      if (!is_one(op->condition)) {
        p->stream << " if ";
        p->Print(op->condition);
      }
      p->stream << " {\n";

      p->indent += 2;
      p->Print(op->body);
      p->indent -= 2;

      p->PrintIndent();
      p->stream << "}\n";
    });

// Prefetch
Prefetch::Prefetch(Buffer buffer, Array<Range> bounds) {
  data_ = make_object<PrefetchNode>(buffer, bounds);
}

TVM_REGISTER_GLOBAL("tir.Prefetch").set_body_typed([](Buffer buffer, Array<Range> bounds) {
  return Prefetch(buffer, bounds);
});

TVM_REGISTER_NODE_TYPE(PrefetchNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrefetchNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrefetchNode*>(node.get());
      p->PrintIndent();
      p->stream << "prefetch " << op->buffer << "(";
      for (size_t i = 0; i < op->bounds.size(); ++i) {
        p->stream << "[";
        p->Print(op->bounds[i]->min);
        p->stream << ", ";
        p->Print(op->bounds[i]->extent);
        p->stream << "]";
        if (i < op->bounds.size() - 1) p->stream << ", ";
      }
      p->stream << ")";
    });

// SeqStmt
SeqStmt::SeqStmt(Array<Stmt> seq) {
  auto node = make_object<SeqStmtNode>();
  node->seq = std::move(seq);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.SeqStmt").set_body_typed([](Array<Stmt> seq) {
  return SeqStmt(std::move(seq));
});

TVM_REGISTER_NODE_TYPE(SeqStmtNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SeqStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SeqStmtNode*>(node.get());
      for (Stmt stmt : op->seq) {
        p->Print(stmt);
      }
    });

// IfThenElse
IfThenElse::IfThenElse(PrimExpr condition, Stmt then_case, Stmt else_case) {
  CHECK(condition.defined());
  CHECK(then_case.defined());
  // else_case may be null.
  ObjectPtr<IfThenElseNode> node = make_object<IfThenElseNode>();
  node->condition = std::move(condition);
  node->then_case = std::move(then_case);
  node->else_case = std::move(else_case);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(IfThenElseNode);

TVM_REGISTER_GLOBAL("tir.IfThenElse")
    .set_body_typed([](PrimExpr condition, Stmt then_case, Stmt else_case) {
      return IfThenElse(condition, then_case, else_case);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IfThenElseNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IfThenElseNode*>(node.get());
      p->PrintIndent();
      while (true) {
        p->stream << "if (" << op->condition << ") {\n";
        p->indent += 2;
        p->Print(op->then_case);
        p->indent -= 2;

        if (!op->else_case.defined()) {
          break;
        }

        if (const IfThenElseNode* nested_if = op->else_case.as<IfThenElseNode>()) {
          p->PrintIndent();
          p->stream << "} else ";
          op = nested_if;
        } else {
          p->PrintIndent();
          p->stream << "} else {\n";
          p->indent += 2;
          p->Print(op->else_case);
          p->indent -= 2;
          break;
        }
      }
      p->PrintIndent();
      p->stream << "}\n";
    });

// Evaluate
Evaluate::Evaluate(PrimExpr value) {
  CHECK(value.defined());

  ObjectPtr<EvaluateNode> node = make_object<EvaluateNode>();
  node->value = std::move(value);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.Evaluate").set_body_typed([](PrimExpr value) { return Evaluate(value); });

TVM_REGISTER_NODE_TYPE(EvaluateNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<EvaluateNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const EvaluateNode*>(node.get());
      p->PrintIndent();
      p->Print(op->value);
      p->stream << "\n";
    });

// BufferStore
BufferStore::BufferStore(Buffer buffer, PrimExpr value, Array<PrimExpr> indices) {
  ObjectPtr<BufferStoreNode> node = make_object<BufferStoreNode>();
  node->buffer = std::move(buffer);
  node->value = std::move(value);
  node->indices = std::move(indices);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.BufferStore")
    .set_body_typed([](Buffer buffer, PrimExpr value, Array<PrimExpr> indices) {
      return BufferStore(buffer, value, indices);
    });

TVM_REGISTER_NODE_TYPE(BufferStoreNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BufferStoreNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const BufferStoreNode*>(node.get());
      p->PrintIndent();
      p->stream << op->buffer->name << "[";
      for (size_t i = 0; i < op->indices.size(); ++i) {
        p->Print(op->indices[i]);
        if (i < op->indices.size() - 1) p->stream << ", ";
      }
      p->stream << "]";
      p->stream << " = ";
      p->Print(op->value);
      p->stream << '\n';
    });

// BufferRealize
BufferRealize::BufferRealize(Buffer buffer, Array<Range> bounds, PrimExpr condition, Stmt body) {
  data_ = make_object<BufferRealizeNode>(buffer, bounds, condition, body);
}

TVM_REGISTER_GLOBAL("tir.BufferRealize")
    .set_body_typed([](Buffer buffer, Array<Range> bounds, PrimExpr condition, Stmt body) {
      return BufferRealize(buffer, bounds, condition, body);
    });

TVM_REGISTER_NODE_TYPE(BufferRealizeNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BufferRealizeNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const BufferRealizeNode*>(node.get());
      p->PrintIndent();
      p->stream << "buffer_realize " << op->buffer->name << "(";
      for (size_t i = 0; i < op->bounds.size(); ++i) {
        p->stream << "[";
        p->Print(op->bounds[i]->min);
        p->stream << ", ";
        p->Print(op->bounds[i]->extent);
        p->stream << "]";
        if (i < op->bounds.size() - 1) p->stream << ", ";
      }
      p->stream << ")";
      if (!is_one(op->condition)) {
        p->stream << " if ";
        p->Print(op->condition);
      }
      p->stream << " {\n";

      p->indent += 2;
      p->Print(op->body);
      p->indent -= 2;

      p->PrintIndent();
      p->stream << "}\n";
    });

// Annotation
Annotation::Annotation(std::string attr_key, PrimExpr value) {
  ObjectPtr<AnnotationNode> node = make_object<AnnotationNode>();
  node->attr_key = std::move(attr_key);
  node->value = std::move(value);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.Annotation")
    .set_body_typed<Annotation(std::string, PrimExpr)>([](std::string attr_key, PrimExpr value) {
      return Annotation(attr_key, value);
    });

TVM_REGISTER_NODE_TYPE(AnnotationNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AnnotationNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AnnotationNode*>(node.get());
      p->stream << op->attr_key << ": ";
      p->Print(op->value);
    });

// Loop
Loop::Loop(Var loop_var, PrimExpr min, PrimExpr extent, Array<Annotation> annotations, Stmt body) {
  ObjectPtr<LoopNode> node = make_object<LoopNode>();
  node->loop_var = std::move(loop_var);
  node->min = std::move(min);
  node->extent = std::move(extent);
  node->annotations = std::move(annotations);
  node->body = std::move(body);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.Loop")
    .set_body_typed<Loop(Var, PrimExpr, PrimExpr, Array<Annotation>, Stmt)>(
        [](Var loop_var, PrimExpr min, PrimExpr extent, Array<Annotation> annotations, Stmt body) {
      return Loop(loop_var, min, extent, annotations, body);
    });

TVM_REGISTER_NODE_TYPE(LoopNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LoopNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const LoopNode*>(node.get());

      // print loop and annotations
      p->PrintIndent();
      p->stream << "for (" << op->loop_var << ", ";
      p->Print(op->min);
      p->stream << ", ";
      p->Print(op->extent);
      if (!op->annotations.empty()) {
        p->stream << ", attr: ";
        p->Print(op->annotations);
        p->stream << ")";
      }
      p->stream << ") {\n";

      // print body
      p->indent += 2;
      p->Print(op->body);
      p->indent -= 2;
      p->PrintIndent();
      p->stream << "}\n";
    });

// TensorRegion
TensorRegion::TensorRegion(Buffer buffer, Array<Range> region) {
  ObjectPtr<TensorRegionNode> node = make_object<TensorRegionNode>();
  node->buffer = std::move(buffer);
  node->region = std::move(region);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.TensorRegion")
    .set_body_typed<TensorRegion(Buffer, Array<Range>)>([](Buffer buffer, Array<Range> region) {
      return TensorRegion(buffer, region);
    });

TVM_REGISTER_NODE_TYPE(TensorRegionNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<TensorRegionNode>([](const ObjectRef& node, ReprPrinter* p) {
  auto* op = static_cast<const TensorRegionNode*>(node.get());
  p->Print(op->buffer->data);
  p->stream << "[";
  for (size_t i = 0; i < op->region.size(); ++i) {
    const auto& range = op->region[i];
    p->Print(range->min);
    p->stream << ":";
    p->Print(range->min + range->extent);
    if (i != op->region.size() - 1) {
      p->stream << ", ";
    }
  }
  p->stream << "]";
});

// BufferAllocate
BufferAllocate::BufferAllocate(Buffer buffer, std::string scope) {
  ObjectPtr<BufferAllocateNode> node = make_object<BufferAllocateNode>();
  node->buffer = std::move(buffer);
  node->scope = std::move(scope);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.BufferAllocate")
    .set_body_typed<BufferAllocate(Buffer, std::string)>([](Buffer buffer, std::string scope) {
      return BufferAllocate(buffer, scope);
    });

TVM_REGISTER_NODE_TYPE(BufferAllocateNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BufferAllocateNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const BufferAllocateNode*>(node.get());
      p->PrintIndent();
      p->stream << "BufferAllocate(";
      p->stream << op->buffer->name;
      p->Print(op->buffer->shape);
      p->stream << ", " << op->buffer->dtype;
      p->stream << ", \"" << op->scope << "\")\n";
    });

// Block
Block::Block(Array<IterVar> iter_vars,
             Array<TensorRegion> reads,
             Array<TensorRegion> writes,
             Stmt body,
             Array<BufferAllocate> allocations,
             Array<Annotation> annotations,
             std::string tag) {
  ObjectPtr<BlockNode> node = make_object<BlockNode>();
  node->iter_vars = std::move(iter_vars);
  node->reads = std::move(reads);
  node->writes = std::move(writes);
  node->body = std::move(body);
  node->allocations = std::move(allocations);
  node->annotations = std::move(annotations);
  node->tag = std::move(tag);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.Block")
.set_body_typed<Block(Array<IterVar>,
                      Array<TensorRegion>,
                      Array<TensorRegion>,
                      Stmt,
                      Array<BufferAllocate>,
                      Array<Annotation>,
                      std::string)>(
    [](Array<IterVar> iter_vars,
       Array<TensorRegion> reads,
       Array<TensorRegion> writes,
       Stmt body,
       Array<BufferAllocate> allocates,
       Array<Annotation> annotations,
       std::string tag) {
      return Block(iter_vars, reads, writes,
                   body, allocates, annotations, tag);
    });

TVM_REGISTER_NODE_TYPE(BlockNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<BlockNode>([](const ObjectRef& node, ReprPrinter* p) {
  auto* op = static_cast<const BlockNode*>(node.get());

  // print block name and block vars
  p->PrintIndent();
  p->stream << "block " << op->tag << "(";
  for (size_t i = 0; i < op->iter_vars.size(); ++i) {
    const auto& iter_var = op->iter_vars[i];
    if (iter_var->iter_type != kDataPar) {
      std::string str;
      switch (iter_var->iter_type) {
        case kCommReduce:str = "reduce";
          break;
        case kOrdered:str = "ordered";
          break;
        case kOpaque:str = "opaque";
          break;
        default:str = "unknown";
          break;
      }
      p->stream << str << " ";
    }
    p->Print(iter_var->var);
    p->stream << "[";
    p->Print(iter_var->dom->min);
    p->stream << ":";
    p->Print(iter_var->dom->min + iter_var->dom->extent);
    p->stream << "]=None";
    if (i != op->iter_vars.size() - 1) {
      p->stream << ", ";
    }
  }
  p->stream << ")";

  // print tensor region and annotations
  p->stream << " W: ";
  p->Print(op->writes);
  p->stream << " R: ";
  p->Print(op->reads);
  if (!op->annotations.empty()) {
    p->stream << " attr: ";
    p->Print(op->annotations);
  }

  // print body
  p->stream << " {\n";
  p->indent += 2;
  for (const auto& allocate : op->allocations) {
    p->Print(allocate);
  }
  p->Print(op->body);
  p->indent -= 2;
  p->PrintIndent();
  p->stream << "}\n";
});

// BlockRealize
BlockRealize::BlockRealize(Array<PrimExpr> values,
                           PrimExpr predicate,
                           Block block) {
  CHECK_EQ(block->iter_vars.size(), values.size());
  ObjectPtr<BlockRealizeNode> node = make_object<BlockRealizeNode>();
  node->binding_values = std::move(values);
  node->predicate = std::move(predicate);
  node->block = std::move(block);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.BlockRealize")
.set_body_typed<BlockRealize(Array<PrimExpr>, PrimExpr, Block)>(
    [](Array<PrimExpr> values, PrimExpr predicate, Block block) {
      if (!predicate.dtype().is_bool()) {
        // To support python ir_builder
        CHECK(is_one(predicate));
        predicate = IntImm(DataType::Bool(), 1);
      }
      return BlockRealize(values, predicate, block);
    });

TVM_REGISTER_NODE_TYPE(BlockRealizeNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<BlockRealizeNode>([](const ObjectRef& node, ReprPrinter* p) {
  const auto* op_reailze = static_cast<const BlockRealizeNode*>(node.get());
  const auto* op = static_cast<const BlockNode*>(op_reailze->block.get());

  // print block name and block vars
  p->PrintIndent();
  p->stream << "block " << op->tag << "(";
  for (size_t i = 0; i < op->iter_vars.size(); ++i) {
    const auto& iter_var = op->iter_vars[i];
    if (iter_var->iter_type != kDataPar) {
      std::string str;
      switch (iter_var->iter_type) {
        case kCommReduce:str = "reduce";
          break;
        case kOrdered:str = "ordered";
          break;
        case kOpaque:str = "opaque";
          break;
        default:str = "unknown";
          break;
      }
      p->stream << str << " ";
    }
    p->Print(iter_var->var);
    p->stream << "[";
    p->Print(iter_var->dom->min);
    p->stream << ":";
    p->Print(iter_var->dom->min + iter_var->dom->extent);
    p->stream << "]=";
    p->Print(op_reailze->binding_values[i]);
    if (i != op->iter_vars.size() - 1) {
      p->stream << ", ";
    }
  }
  p->stream << ")";

  // print tensor region and annotations
  p->stream << " W: ";
  p->Print(op->writes);
  p->stream << " R: ";
  p->Print(op->reads);
  if (!is_one(op_reailze->predicate)) {
    p->stream << " pred: ";
    p->Print(op_reailze->predicate);
  }
  if (!op->annotations.empty()) {
    p->stream << " attr: ";
    p->Print(op->annotations);
  }

  // print body
  p->stream << " {\n";
  p->indent += 2;
  for (const auto& allocate : op->allocations) {
    p->Print(allocate);
  }
  p->Print(op->body);
  p->indent -= 2;
  p->PrintIndent();
  p->stream << "}\n";
});

// ReduceStep
ReduceStep::ReduceStep(CommReducer comm_reducer, PrimExpr lhs, PrimExpr rhs) {
  ObjectPtr<ReduceStepNode> node = make_object<ReduceStepNode>();
  node->comm_reducer = std::move(comm_reducer);
  node->lhs = std::move(lhs);
  node->rhs = std::move(rhs);
  data_ = std::move(node);
}

PrimExpr ReduceStepNode::ApplyCombiner() const {
  return ApplyCombiner(this->lhs, this->rhs);
}

PrimExpr ReduceStepNode::ApplyCombiner(const PrimExpr& lhs, const PrimExpr& rhs) const {
  CHECK_EQ(comm_reducer->lhs.size(), 1);
  CHECK_EQ(comm_reducer->rhs.size(), 1);
  CHECK_EQ(comm_reducer->result.size(), 1);
  auto vmap = [&](const Var& v) -> Optional<PrimExpr> {
    if (v.same_as(comm_reducer->lhs[0])) {
      return lhs;
    } else if (v.same_as(comm_reducer->rhs[0])) {
      return rhs;
    } else {
      return v;
    }
  };
  return Substitute(comm_reducer->result[0], vmap);
}

std::tuple<bool, PrimExpr, PrimExpr> ReducerMatched(const CommReducer& reducer,
                                                    const PrimExpr& init, const PrimExpr update) {
  ExprDeepEqual equal;
  if (!equal(reducer->identity_element[0], init))
    return std::make_tuple(false, NullValue<PrimExpr>(), NullValue<PrimExpr>());
  PatternMatcher pattern_matcher(reducer->result[0]);
  pattern_matcher.Match(update);
  return std::make_tuple(pattern_matcher.Success(),
                         pattern_matcher.Eval(reducer->lhs[0]),
                         pattern_matcher.Eval(reducer->rhs[0]));
}

Stmt ReduceStep::FromInitUpdate(const Array<CommReducer>& patterns,
                                const PrimExpr& init, const BufferStore& update) {
  ExprDeepEqual equal;
  const auto& lhs = BufferLoad(update->buffer, update->indices);
  // Check user defined patterns
  for (const auto& reducer : patterns) {
    const auto& res = ReducerMatched(reducer, init, update->value);
    if (std::get<0>(res) && equal(lhs, std::get<1>(res))) {
      return ReduceStep(reducer, std::get<1>(res), std::get<2>(res));
    }
  }
  // Check default patterns
  for (const auto& reducer : default_reducer::default_reducers) {
    const auto& res = ReducerMatched(reducer.GetReducer(init.dtype()), init, update->value);
    if (std::get<0>(res) && equal(lhs, std::get<1>(res))) {
      return ReduceStep(reducer.GetReducer(init.dtype()), std::get<1>(res), std::get<2>(res));
    }
  }
  LOG(FATAL) << "No reducer pattern matched for " << init << " " << update;
  return NullValue<ReduceStep>();
}

TVM_REGISTER_GLOBAL("tir.ReduceStep")
.set_body_typed<ReduceStep(CommReducer, PrimExpr, PrimExpr)>(
    [](CommReducer comm_reducer, PrimExpr lhs, PrimExpr rhs) {
      return ReduceStep(comm_reducer, lhs, rhs);
    });

TVM_REGISTER_NODE_TYPE(ReduceStepNode);

PrimExpr TypeAnnotation(DataType dtype) {
  static auto op = Op::Get("tir.type_annotation");
  return tir::Call(dtype, op, {});
}

TVM_REGISTER_OP("tir.type_annotation")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

}  // namespace tir
}  // namespace tvm
