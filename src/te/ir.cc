#include "ir.h"

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
                 Expr predicates,
                 Array<Annotation> annotations,
                 std::string tag) {
  NodePtr<Block> node = make_node<Block>();
  node->vars = std::move(vars);
  node->inputs = std::move(inputs);
  node->outputs = std::move(outputs);
  node->body = std::move(body);
  node->predicate = std::move(predicates);
  node->annotations = std::move(annotations);
  node->tag = std::move(tag);
  return Stmt(node);
}

TensorRegion TensorRegionNode::make(Tensor data, Array<Range> ranges) {
  NodePtr<TensorRegionNode> node = make_node<TensorRegionNode>();
  node->data = std::move(data);
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

BlockVar BlockVarNode::make(Var data, Expr value, LoopType type, Range range) {
  NodePtr<BlockVarNode> node = make_node<BlockVarNode>();
  node->data = std::move(data);
  node->value = std::move(value);
  node->type = std::move(type);
  node->range = std::move(range);
  return BlockVar(node);
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
  p->Print(op->data);
  p->Print(op->ranges);
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<BlockVarNode>([](const ObjectRef &node, IRPrinter* p) {
  auto* op = static_cast<const BlockVarNode*>(node.get());
  p->stream << op->type << " ";
  p->Print(op->data);
  p->Print(op->range);
  p->stream << "=";
  p->Print(op->value);
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Loop>([](const ObjectRef &node, IRPrinter* p) {
  auto* op = static_cast<const Loop*>(node.get());

  // print loop and annotations
  p->stream << "for ";
  p->Print(op->loop_var);
  p->stream << " = ";
  p->Print(op->min);
  p->stream << " to ";
  p->Print(op->extent);
  p->stream << "  (" << op->loop_type;
  if (op->annotations.size() > 0) {
    p->stream << ", attr: ";
    p->Print(op->annotations);
  }

  // print body
  p->stream << ") {\n";
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
  p->stream << "block " << op->tag << " (";
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

} // namespace te
} // namespace tvm