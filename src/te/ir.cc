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
 * \file tvm/te/ir.cc
 * \brief Additional high level nodes in the TensorIR
 */

#include <tvm/te/ir.h>
#include <tvm/api_registry.h>
#include <tvm/arithmetic.h>

namespace tvm {
namespace te {
using namespace ir;

SeqStmt::SeqStmt(Array<Stmt> seq) {
  NodePtr<SeqStmtNode> node = make_node<SeqStmtNode>();
  node->seq = std::move(seq);
  data_ = std::move(node);
}

BufferLoad::BufferLoad(DataType type, Buffer buffer, Array<Expr> indices) {
  NodePtr<BufferLoadNode> node = make_node<BufferLoadNode>();
  node->type = type;
  node->buffer = std::move(buffer);
  node->indices = std::move(indices);
  data_ = std::move(node);
}

BufferStore::BufferStore(Buffer buffer, Expr value, Array<Expr> indices) {
  NodePtr<BufferStoreNode> node = make_node<BufferStoreNode>();
  node->buffer = std::move(buffer);
  node->value = std::move(value);
  node->indices = std::move(indices);
  data_ = std::move(node);
}

Block::Block(Array<IterVar> iter_vars,
             Array<Expr> values,
             Array<TensorRegion> reads,
             Array<TensorRegion> writes,
             Stmt body,
             Expr predicate,
             Array<BufferAllocate> allocations,
             Array<Annotation> annotations,
             std::string tag) {
  NodePtr<BlockNode> node = make_node<BlockNode>();
  CHECK_EQ(iter_vars.size(), values.size());
  node->iter_vars = std::move(iter_vars);
  node->values = std::move(values);
  node->reads = std::move(reads);
  node->writes = std::move(writes);
  node->body = std::move(body);
  node->predicate = std::move(predicate);
  node->allocations = std::move(allocations);
  node->annotations = std::move(annotations);
  node->tag = std::move(tag);
  data_ = std::move(node);
}

TensorRegion::TensorRegion(Buffer buffer, Array<Range> region) {
  NodePtr<TensorRegionNode> node = make_node<TensorRegionNode>();
  node->buffer = std::move(buffer);
  node->region = std::move(region);
  data_ = std::move(node);
}

Annotation::Annotation(std::string attr_key, Expr value) {
  NodePtr<AnnotationNode> node = make_node<AnnotationNode>();
  node->attr_key = std::move(attr_key);
  node->value = std::move(value);
  data_ = std::move(node);
}

Loop::Loop(Var loop_var,
           Expr min,
           Expr extent,
           Array<Annotation> annotations,
           Stmt body) {
  NodePtr<LoopNode> node = make_node<LoopNode>();
  node->loop_var = std::move(loop_var);
  node->min = std::move(min);
  node->extent = std::move(extent);
  node->annotations = std::move(annotations);
  node->body = std::move(body);
  data_ = std::move(node);
}

BufferAllocate::BufferAllocate(Buffer buffer, std::string scope) {
  NodePtr<BufferAllocateNode> node = make_node<BufferAllocateNode>();
  node->buffer = std::move(buffer);
  node->scope = std::move(scope);
  data_ = std::move(node);
}

Function::Function(Array<Var> params,
                   Map<Var, Buffer> buffer_map,
                   std::string name,
                   Stmt body) {
  NodePtr<FunctionNode> node = make_node<FunctionNode>();
  CHECK_EQ(params.size(), buffer_map.size());
  node->params = std::move(params);
  node->buffer_map = std::move(buffer_map);
  node->name = std::move(name);
  node->body = std::move(body);
  data_ = std::move(node);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<SeqStmtNode>([](const ObjectRef& node, IRPrinter* p) {
  auto* op = static_cast<const SeqStmtNode*>(node.get());
  for (size_t i = 0; i < op->size(); ++i) {
    p->Print((*op)[i]);
  }
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<BufferLoadNode>([](const ObjectRef& node, IRPrinter* p) {
  auto* op = static_cast<const BufferLoadNode*>(node.get());
  p->Print(op->buffer->data);
  p->Print(op->indices);
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<BufferStoreNode>([](const ObjectRef& node, IRPrinter* p) {
  auto* op = static_cast<const BufferStoreNode*>(node.get());
  p->PrintIndent();
  p->Print(op->buffer->data);
  p->Print(op->indices);
  p->stream << " = ";
  p->Print(op->value);
  p->stream << '\n';
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<AnnotationNode>([](const ObjectRef& node, IRPrinter* p) {
  auto* op = static_cast<const AnnotationNode*>(node.get());
  p->stream << op->attr_key << ": ";
  p->Print(op->value);
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<TensorRegionNode>([](const ObjectRef& node, IRPrinter* p) {
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

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<LoopNode>([](const ObjectRef& node, IRPrinter* p) {
  auto* op = static_cast<const LoopNode*>(node.get());

  // print loop and annotations
  p->PrintIndent();
  p->stream << "for ";
  p->Print(op->loop_var);
  p->stream << " = ";
  p->Print(op->min);
  p->stream << " to ";
  p->Print(op->extent);
  if (op->annotations.size() > 0) {
    p->stream << " (attr: ";
    p->Print(op->annotations);
    p->stream << ")";
  }

  // print body
  p->stream << " {\n";
  p->indent += 2;
  p->Print(op->body);
  p->indent -= 2;
  p->PrintIndent();
  p->stream << "}\n";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<BlockNode>([](const ObjectRef& node, IRPrinter* p) {
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
    p->stream << "]=";
    p->Print(op->values[i]);
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
  if (!is_one(op->predicate)) {
    p->stream << " pred: ";
    p->Print(op->predicate);
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

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<BufferAllocateNode>([](const ObjectRef& node, IRPrinter* p) {
  auto* op = static_cast<const BufferAllocateNode*>(node.get());
  p->PrintIndent();
  p->stream << "BufferAllocate(";
  p->stream << op->buffer->name;
  p->Print(op->buffer->shape);
  p->stream << ", " << op->buffer->dtype;
  p->stream << ", \"" << op->scope << "\")\n";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<FunctionNode>([](const ObjectRef& node, IRPrinter* p) {
  auto* op = static_cast<const FunctionNode*>(node.get());
  p->PrintIndent();
  p->stream << "func " << op->name << "(";
  for (size_t i = 0; i < op->params.size(); ++i) {
    p->Print(op->params[i]);
    if (i != op->params.size() - 1) {
      p->stream << ", ";
    }
  }
  p->stream << ") {\n";
  p->indent += 2;
  p->Print(op->body);
  p->indent -= 2;
  p->PrintIndent();
  p->stream << "}\n";
});

TVM_REGISTER_NODE_TYPE(SeqStmtNode);
TVM_REGISTER_NODE_TYPE(TensorRegionNode);
TVM_REGISTER_NODE_TYPE(BufferLoadNode);
TVM_REGISTER_NODE_TYPE(BufferStoreNode);
TVM_REGISTER_NODE_TYPE(BufferAllocateNode);
TVM_REGISTER_NODE_TYPE(LoopNode);
TVM_REGISTER_NODE_TYPE(BlockNode);
TVM_REGISTER_NODE_TYPE(FunctionNode);

}  // namespace te
}  // namespace tvm
