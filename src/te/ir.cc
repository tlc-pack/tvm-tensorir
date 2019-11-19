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

#include "ir.h"
#include <tvm/api_registry.h>
#include <tvm/arithmetic.h>
#include <tvm/ir_pass.h>

namespace tvm {
namespace te {
using namespace ir;

Expr BufferLoad::make(DataType type, Var buffer_var, Array<Expr> indices) {
  NodePtr<BufferLoad> node = make_node<BufferLoad>();
  node->type = type;
  node->buffer_var = std::move(buffer_var);
  node->indices = std::move(indices);
  return Expr(node);
}

Stmt BufferStore::make(Var buffer_var, Expr value, Array<Expr> indices) {
  NodePtr<BufferStore> node = make_node<BufferStore>();
  node->buffer_var = std::move(buffer_var);
  node->value = std::move(value);
  node->indices = std::move(indices);
  return Stmt(node);
}

Stmt Block::make(Array<BlockVar> vars,
                 Array<TensorRegion> inputs,
                 Array<TensorRegion> outputs,
                 Stmt body,
                 Expr predicate,
                 Array<Annotation> annotations,
                 std::string tag) {
  NodePtr<Block> node = make_node<Block>();
  node->vars = std::move(vars);
  node->inputs = std::move(inputs);
  node->outputs = std::move(outputs);
  node->body = std::move(body);
  node->predicate = std::move(predicate);
  node->annotations = std::move(annotations);
  node->tag = std::move(tag);
  return Stmt(node);
}

TensorRegion TensorRegionNode::make(Var buffer, Array<Range> ranges) {
  NodePtr<TensorRegionNode> node = make_node<TensorRegionNode>();
  node->buffer = std::move(buffer);
  node->ranges = std::move(ranges);
  return TensorRegion(node);
}

Annotation AnnotationNode::make(std::string attr_key, Expr value) {
  NodePtr<AnnotationNode> node = make_node<AnnotationNode>();
  node->attr_key = std::move(attr_key);
  node->value = std::move(value);
  return Annotation(node);
}

Stmt Loop::make(Var loop_var,
                Expr min,
                Expr extent,
                LoopType loop_type,
                Array<Annotation> annotations,
                Stmt body) {
  NodePtr<Loop> node = make_node<Loop>();
  node->loop_var = std::move(loop_var);
  node->min = std::move(min);
  node->extent = std::move(extent);
  node->loop_type = std::move(loop_type);
  node->annotations = std::move(annotations);
  node->body = std::move(body);
  return Stmt(node);
}

BlockVar BlockVarNode::make(Var var, Expr value, LoopType type, Range range) {
  NodePtr<BlockVarNode> node = make_node<BlockVarNode>();
  node->var = std::move(var);
  node->value = std::move(value);
  node->type = std::move(type);
  node->range = std::move(range);
  return BlockVar(node);
}

Expr BufferBindNode::make(tvm::runtime::NDArray data,
                          tvm::Array<tvm::Expr> shape,
                          tvm::DataType type,
                          std::string name) {
  NodePtr<BufferBindNode> node = make_node<BufferBindNode>();
  node->data = std::move(data);
  node->shape = std::move(shape);
  node->type = std::move(type);
  node->name = std::move(name);
  return Expr(node);
}

std::ostream &operator<<(std::ostream &out, LoopType type) {
  switch (type) {
    case LoopType::kDataPar:out << "";
      break;
    case LoopType::kReduce:out << "reduce";
      break;
    case LoopType::kScan:out << "scan";
      break;
    case LoopType::kOpaque:out << "opaque";
      break;
  }
  return out;
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<BufferLoad>([](const ObjectRef &node, IRPrinter* p) {
  auto* op = static_cast<const BufferLoad*>(node.get());
  p->Print(op->buffer_var);
  p->Print(op->indices);
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<BufferStore>([](const ObjectRef &node, IRPrinter* p) {
  auto* op = static_cast<const BufferStore*>(node.get());
  p->PrintIndent();
  p->Print(op->buffer_var);
  p->Print(op->indices);
  p->stream << " = ";
  p->Print(op->value);
  p->stream << '\n';
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<AnnotationNode>([](const ObjectRef &node, IRPrinter* p) {
  auto* op = static_cast<const AnnotationNode*>(node.get());
  p->stream << op->attr_key << ": ";
  p->Print(op->value);
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<TensorRegionNode>([](const ObjectRef &node, IRPrinter* p) {
  auto* op = static_cast<const TensorRegionNode*>(node.get());
  p->Print(op->buffer);
  p->stream << "[";
  for (size_t i = 0; i < op->ranges.size(); ++i) {
    const auto &range = op->ranges[i];
    p->Print(range->min);
    p->stream << ":";
    p->Print(Simplify(range->min + range->extent));
    if (i != op->ranges.size() - 1) {
      p->stream << ", ";
    }
  }
  p->stream << "]";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<BlockVarNode>([](const ObjectRef &node, IRPrinter* p) {
  auto* op = static_cast<const BlockVarNode*>(node.get());
  if (op->type != LoopType::kDataPar) {
    p->stream << op->type << " ";
  }
  p->Print(op->var);
  p->stream << "[";
  p->Print(op->range->min);
  p->stream << ":";
  p->Print(op->range->min + op->range->extent);
  p->stream << "]=";
  p->Print(op->value);
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Loop>([](const ObjectRef &node, IRPrinter* p) {
  auto* op = static_cast<const Loop*>(node.get());

  // print loop and annotations
  p->PrintIndent();
  p->stream << "for ";
  p->Print(op->loop_var);
  p->stream << " = ";
  p->Print(op->min);
  p->stream << " to ";
  p->Print(op->extent);
  if (op->loop_type != LoopType::kDataPar) {
    p->stream << " (" << op->loop_type <<")";
  }
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
.set_dispatch<Block>([](const ObjectRef &node, IRPrinter* p) {
  auto* op = static_cast<const Block*>(node.get());

  // print block name and block vars
  p->PrintIndent();
  p->stream << "block " << op->tag << "(";
  for (size_t i = 0; i < op->vars.size(); ++i) {
    p->Print(op->vars[i]);
    if (i != op->vars.size() - 1) {
      p->stream << ", ";
    }
  }
  p->stream << ")";

  // print tensor region and annotations
  p->stream << " W: ";
  p->Print(op->outputs);
  p->stream << " R: ";
  p->Print(op->inputs);
  if (!is_one(op->predicate)) {
    p->stream << " pred: ";
    p->Print(op->predicate);
  }
  if (op->annotations.size() > 0) {
    p->stream << " attr: ";
    p->Print(op->annotations);
  }

  // print body
  p->stream << " {\n";
  p->indent += 2;
  p->Print(op->body);
  p->indent -= 2;
  p->PrintIndent();
  p->stream << "}\n";
});

TVM_REGISTER_NODE_TYPE(TensorRegionNode);
TVM_REGISTER_NODE_TYPE(BufferLoad);
TVM_REGISTER_NODE_TYPE(BufferStore);
TVM_REGISTER_NODE_TYPE(BlockVarNode);
TVM_REGISTER_NODE_TYPE(Loop);
TVM_REGISTER_NODE_TYPE(Block);
TVM_REGISTER_API("make.TensorRegion")
.set_body_typed(TensorRegionNode::make);
TVM_REGISTER_API("make.BufferLoad")
.set_body_typed(BufferLoad::make);
TVM_REGISTER_API("make.BufferStore")
.set_body_typed(BufferStore::make);
TVM_REGISTER_API("make.BlockVarNode")
.set_body_typed<BlockVar(Var, Expr, int, Range)>([](
    Var data, Expr value, int type, Range range) {
  return BlockVarNode::make(data,
                            value,
                            static_cast<LoopType>(type),
                            range);
});
TVM_REGISTER_API("make.Loop")
.set_body_typed<Stmt(Var, Expr, Expr, int, Array<Annotation>, Stmt)>([](
    Var loop_var, Expr min, Expr extent,
    int loop_type, Array<Annotation> annotations, Stmt body) {
  return Loop::make(loop_var,
                    min,
                    extent,
                    static_cast<LoopType>(loop_type),
                    annotations,
                    body);
});
TVM_REGISTER_API("make.TeBlock").
set_body_typed(Block::make);

} // namespace te
} // namespace tvm