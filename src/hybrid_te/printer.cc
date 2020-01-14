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
 * \file hybrid_te/printer.cc
 * \brief Printer class to print Te IR to python syntax script
 */

#include <tvm/te/ir.h>
#include <tvm/ir.h>
#include <tvm/te/printer.h>
#include <tvm/ir_pass.h>

namespace tvm {
namespace te {
using namespace ir;

void TePrinter::Print(const ObjectRef& node) {
  if (node.as<StmtNode>()) {
    VisitStmt(Downcast<Stmt>(node));
  } else if (node.as<ExprNode>()) {
    VisitExpr(Downcast<Expr>(node));
  } else {
    // TODO
    static const FType& f = vtable();
    if (f.can_dispatch(node)) {
      f(node, this);
    } else {
      // default value, output type key and addr.
      stream << node->GetTypeKey() << "(" << node.get() << ")";
    }
  }
}

void TePrinter::PrintIndent() {
  for (int i = 0; i < indent; ++i) {
    stream << ' ';
  }
}

void TePrinter::VisitExpr_(const Variable* op) {
  this->stream << op->name_hint;
}

void TePrinter::VisitExpr_(const Add* op) {
  this->stream << '(';
  this->Print(op->a);
  this->stream << " + ";
  this->Print(op->b);
  this->stream << ')';
}

void TePrinter::VisitExpr_(const Sub* op) {
  this->stream << '(';
  this->Print(op->a);
  this->stream << " - ";
  this->Print(op->b);
  this->stream << ')';
}

void TePrinter::VisitExpr_(const Mul* op) {
  this->stream << '(';
  this->Print(op->a);
  this->stream << "*";
  this->Print(op->b);
  this->stream << ')';
}

void TePrinter::VisitExpr_(const Div* op) {
  this->stream << '(';
  this->Print(op->a);
  this->stream << "/";
  this->Print(op->b);
  this->stream << ')';
}

void TePrinter::VisitExpr_(const Mod* op) {
  this->stream << '(';
  this->Print(op->a);
  this->stream << " % ";
  this->Print(op->b);
  this->stream << ')';
}

void TePrinter::VisitExpr_(const FloorDiv* op) {
  this->stream << "floordiv(" << op->a << ", " << op->b << ")";
}

void TePrinter::VisitExpr_(const FloorMod* op) {
  this->stream << "floordiv(" << op->a << ", " << op->b << ")";
}

void TePrinter::VisitExpr_(const Min* op) {
  this->stream << "min(";
  this->Print(op->a);
  this->stream << ", ";
  this->Print(op->b);
  this->stream << ")";
}

void TePrinter::VisitExpr_(const Max* op) {
  this->stream << "max(";
  this->Print(op->a);
  this->stream << ", ";
  this->Print(op->b);
  this->stream << ")";
}

void TePrinter::VisitExpr_(const EQ* op) {
  this->stream << '(';
  this->Print(op->a);
  this->stream << " == ";
  this->Print(op->b);
  this->stream << ')';
}

void TePrinter::VisitExpr_(const NE* op) {
  this->stream << '(';
  this->Print(op->a);
  this->stream << " != ";
  this->Print(op->b);
  this->stream << ')';
}

void TePrinter::VisitExpr_(const LT* op) {
  this->stream << '(';
  this->Print(op->a);
  this->stream << " < ";
  this->Print(op->b);
  this->stream << ')';
}

void TePrinter::VisitExpr_(const LE* op) {
  this->stream << '(';
  this->Print(op->a);
  this->stream << " <= ";
  this->Print(op->b);
  this->stream << ')';
}

void TePrinter::VisitExpr_(const GT* op) {
  this->stream << '(';
  this->Print(op->a);
  this->stream << " > ";
  this->Print(op->b);
  this->stream << ')';
}

void TePrinter::VisitExpr_(const GE* op) {
  this->stream << '(';
  this->Print(op->a);
  this->stream << " >= ";
  this->Print(op->b);
  this->stream << ')';
}

void TePrinter::VisitExpr_(const And* op) {
  this->stream << '(';
  this->Print(op->a);
  this->stream << " && ";
  this->Print(op->b);
  this->stream << ')';
}

void TePrinter::VisitExpr_(const Or* op) {
  this->stream << '(';
  this->Print(op->a);
  this->stream << " || ";
  this->Print(op->b);
  this->stream << ')';
}

void TePrinter::VisitExpr_(const IntImm* op) {
  if (op->dtype.bits() == 32) {
    this->stream << op->value;
  } else {
    this->stream << op->dtype << "(" << op->value << ")";
  }
}

void TePrinter::VisitExpr_(const UIntImm* op) {
  this->stream << op->dtype << "(" << op->value << ")";
}

void TePrinter::VisitExpr_(const FloatImm* op) {
  this->stream << op->dtype << "(" << op->value << ")";
}

void TePrinter::VisitExpr_(const StringImm* op) {
  auto& stream = this->stream;
  stream << '"';
  for (unsigned char c : op->value) {
    if (c >= ' ' && c <= '~' && c != '\\' && c != '"') {
      stream << c;
    } else {
      stream << '\\';
      switch (c) {
        case '"':
          stream << '"';
          break;
        case '\\':
          stream << '\\';
          break;
        case '\t':
          stream << 't';
          break;
        case '\r':
          stream << 'r';
          break;
        case '\n':
          stream << 'n';
          break;
        default:
          const char* hex_digits = "0123456789ABCDEF";
          stream << 'x' << hex_digits[c >> 4] << hex_digits[c & 0xf];
      }
    }
  }
  stream << '"';
}

void TePrinter::VisitExpr_(const te::BufferLoadNode* op) {
  this->Print(op->buffer->data);
  this->Print(op->indices);
}

void TePrinter::VisitStmt_(const SeqStmtNode* op) {
  for (Stmt stmt : op->seq) {
    this->Print(stmt);
  }
}

void TePrinter::VisitStmt_(const Evaluate* op) {
  this->PrintIndent();
  this->Print(op->value);
  this->stream << "\n";
}

void TePrinter::VisitStmt_(const te::BlockNode* op) {
  // print block name and block vars
  this->PrintIndent();
  this->stream << "with block({";
  for (size_t i = 0; i < op->iter_vars.size(); ++i) {
    const auto& iter_var = op->iter_vars[i];
    this->Print(iter_var->var);
    this->stream << "(";
    this->Print(iter_var->dom->min);
    this->stream << ", ";
    this->Print(iter_var->dom->min + iter_var->dom->extent);
    if (iter_var->iter_type != kDataPar) {
      std::string str;
      switch (iter_var->iter_type) {
        case kCommReduce:
          str = "reduce";
          break;
        case kOrdered:
          str = "ordered";
          break;
        case kOpaque:
          str = "opaque";
          break;
        default:
          str = "unknown";
          break;
      }
      this->stream << ", iter_type=\"" << str << "\"";
    }
    this->stream << "): ";
    this->Print(op->values[i]);
    if (i != op->iter_vars.size() - 1) {
      this->stream << ", ";
    }
  }
  this->stream << "}";

  // print tensor region and annotations
  this->stream << ", writes=";
  this->Print(op->writes);
  this->stream << ", reads=";
  this->Print(op->reads);
  if (!is_one(op->predicate)) {
    this->stream << ", predicate=";
    this->Print(op->predicate);
  }
  if (!op->annotations.empty()) {
    this->stream << ", annotations=";
    this->Print(op->annotations);
  }
  this->stream << ", name=\"" << op->tag << "\")";
  // print body
  this->stream << ": \n";
  this->indent += 2;
  for (const auto& allocate : op->allocations) {
    this->Print(allocate);
  }
  this->Print(op->body);
  this->indent -= 2;
}

void TePrinter::VisitStmt_(const te::LoopNode* op) {
  // print loop and annotations
  this->PrintIndent();
  this->stream << "for ";
  this->Print(op->loop_var);
  this->stream << " in range(";
  this->Print(op->min);
  this->stream << ", ";
  this->Print(Simplify(op->min + op->extent));
  this->stream << ")";

  // print body
  this->stream << ": \n";
  this->indent += 2;
  this->Print(op->body);
  this->indent -= 2;
}

void TePrinter::VisitStmt_(const te::BufferAllocateNode* op) {
  this->PrintIndent();
  this->stream << op->buffer->name;
  this->stream << " = buffer_allocate(";
  this->stream << '(';
  for (size_t i = 0; i < op->buffer->shape.size(); ++i) {
    if (i != 0) {
      this->stream << ", ";
    }
    this->Print(op->buffer->shape[i]);
  }
  this->stream << ')';
  this->stream << ", \"" << op->buffer->dtype << "\"";
  this->stream << ", \"" << op->buffer->name << "\"";
  this->stream << ", \"" << op->scope << "\")\n";
}

void TePrinter::VisitStmt_(const te::BufferStoreNode* op) {
  this->PrintIndent();
  this->Print(op->buffer->data);
  this->Print(op->indices);
  this->stream << " = ";
  this->Print(op->value);
  this->stream << '\n';
}

TVM_STATIC_IR_FUNCTOR(TePrinter, vtable)
.set_dispatch<FunctionNode>([](const ObjectRef& node, TePrinter* p) {
  auto* op = static_cast<const FunctionNode*>(node.get());
  p->PrintIndent();
  p->stream << "def " << op->name << "(";
  for (size_t i = 0; i < op->params.size(); ++i) {
    p->Print(op->params[i]);
    if (i != op->params.size() - 1) {
      p->stream << ", ";
    }
  }
  p->stream << "): \n";
  p->indent += 2;

  // print buffer_bind
  for (auto it = op->buffer_map.begin(); it != op->buffer_map.end(); ++it) {
    p->PrintIndent();
    p->stream << (*it).second->name << " = buffer_bind(";
    p->Print((*it).first);
    p->stream << ", (";
    for (size_t i = 0; i < (*it).second->shape.size(); ++i) {
      if (i != 0) {
        p->stream << ", ";
      }
      p->Print((*it).second->shape[i]);
    }
    p->stream << ')';
    p->stream << ", \"" << (*it).second->dtype << "\"";
    p->stream << ", \"" << (*it).second->name << "\")\n";
  }

  // print body
  p->Print(op->body);
  p->indent -= 2;
  p->PrintIndent();
  p->stream << "\n";
});

TVM_STATIC_IR_FUNCTOR(TePrinter, vtable)
.set_dispatch<TensorRegionNode>([](const ObjectRef& node, TePrinter* p) {
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

TVM_STATIC_IR_FUNCTOR(TePrinter, vtable)
.set_dispatch<AnnotationNode>([](const ObjectRef& node, TePrinter* p) {
  auto* op = static_cast<const AnnotationNode*>(node.get());
  p->stream << op->attr_key << ": ";
  p->Print(op->value);
});

TVM_STATIC_IR_FUNCTOR(TePrinter, vtable)
.set_dispatch<ArrayNode>([](const ObjectRef& node, TePrinter* p) {
  auto* op = static_cast<const ArrayNode*>(node.get());
  p->stream << '[';
  for (size_t i = 0; i < op->data.size(); ++i) {
    if (i != 0) {
      p->stream << ", ";
    }
    p->Print(op->data[i]);
  }
  p->stream << ']';
});

TePrinter::FType& TePrinter::vtable() {
  static FType inst;
  return inst;
}

} // namespace te
} // namespace tvm