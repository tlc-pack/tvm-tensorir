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
#include <tvm/runtime/registry.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/node/serialization.h>


namespace tvm {
namespace te {
using namespace ir;

class TextMetaDataContext {
 public:
  std::string GetMetaNode(const ObjectRef& node) {
    auto it = meta_repr_.find(node);
    if (it != meta_repr_.end()) {
      return it->second;
    }
    std::string type_key = node->GetTypeKey();
    CHECK(!type_key.empty());
    Array<ObjectRef>& mvector = meta_data_[type_key];
    auto index = static_cast<int64_t>(mvector.size());
    mvector.push_back(node);
    std::ostringstream doc;
    doc << "meta[" << type_key << "][" << index << "]";
    meta_repr_[node] = doc.str();
    return meta_repr_[node];
  }

  std::string GetMetaSection() const {
    if (meta_data_.empty()) return std::string();
    return SaveJSON(Map<std::string, ObjectRef>(meta_data_.begin(), meta_data_.end()));
  }

  bool empty() const {
    return meta_data_.empty();
  }

 private:
  std::unordered_map<std::string, Array<ObjectRef> > meta_data_;
  std::unordered_map<ObjectRef, std::string, ObjectHash, ObjectEqual> meta_repr_;
};

class TePrinter :
      public StmtExprVisitor {
 public:
  /*! \brief The output stream */
  std::ostream& stream;
  /*! \brief The indentation level */
  int indent{0};

  explicit TePrinter(std::ostream& stream) // NOLINT(*)
      : stream(stream) {}
  /*! \brief Print the node */
  TVM_DLL void Print(const ObjectRef& node);
  /*! \brief Print indent to the stream  */
  TVM_DLL void PrintIndent();

  // Allow registration to be printer.
  using FType = NodeFunctor<void(const ObjectRef&, TePrinter*)>;
  static FType& vtable();

 private:
  /*! \brief meta data context */
  TextMetaDataContext meta_;

  void VisitExpr_(const Variable* op) override;
  void VisitExpr_(const Add* op) override;
  void VisitExpr_(const Sub* op) override;
  void VisitExpr_(const Mul* op) override;
  void VisitExpr_(const Div* op) override;
  void VisitExpr_(const Mod* op) override;
  void VisitExpr_(const FloorDiv* op) override;
  void VisitExpr_(const FloorMod* op) override;
  void VisitExpr_(const Min* op) override;
  void VisitExpr_(const Max* op) override;
  void VisitExpr_(const EQ* op) override;
  void VisitExpr_(const NE* op) override;
  void VisitExpr_(const LT* op) override;
  void VisitExpr_(const LE* op) override;
  void VisitExpr_(const GT* op) override;
  void VisitExpr_(const GE* op) override;
  void VisitExpr_(const And* op) override;
  void VisitExpr_(const Or* op) override;
  void VisitExpr_(const IntImm* op) override;
  void VisitExpr_(const UIntImm* op) override;
  void VisitExpr_(const FloatImm* op) override;
  void VisitExpr_(const StringImm* op) override;
  void VisitExpr_(const te::BufferLoadNode* op) override;
  void VisitExprDefault_(const Object* op) override;

  void VisitStmt_(const SeqStmtNode* op) override;
  void VisitStmt_(const Evaluate* op) override;
  void VisitStmt_(const te::BlockNode* op) override;
  void VisitStmt_(const te::LoopNode* op) override;
  void VisitStmt_(const te::BufferAllocateNode* op) override;
  void VisitStmt_(const te::BufferStoreNode* op) override;
  void VisitStmtDefault_(const Object* op) override;
};

void TePrinter::Print(const ObjectRef& node) {
  if (node.as<StmtNode>()) {
    VisitStmt(Downcast<Stmt>(node));
  } else if (node.as<ExprNode>()) {
    VisitExpr(Downcast<Expr>(node));
  } else {
    static const FType& f = vtable();
    if (f.can_dispatch(node)) {
      f(node, this);
    } else {
      this->stream << this->meta_.GetMetaNode(node);
    }
  }
}

void TePrinter::PrintIndent() {
  for (int i = 0; i < indent; ++i) {
    stream << ' ';
  }
}

void TePrinter::VisitExprDefault_(const Object* op) {
  this->stream << this->meta_.GetMetaNode(GetRef<ObjectRef>(op));
}

void TePrinter::VisitStmtDefault_(const Object* op) {
  this->stream << this->meta_.GetMetaNode(GetRef<ObjectRef>(op));
}

void TePrinter::VisitExpr_(const Variable* op) {
  this->stream << op->name_hint;
}

#define TVM_DECLARE_TEPRINTER_BINOP(OpName, OpString)                                \
  void TePrinter::VisitExpr_(const OpName* op) {                                     \
    this->stream << '(';                                                             \
    this->Print(op->a);                                                              \
    this->stream << OpString;                                                        \
    this->Print(op->b);                                                              \
    this->stream << ')';                                                             \
  }                                                                                  \

TVM_DECLARE_TEPRINTER_BINOP(Add, " + ")
TVM_DECLARE_TEPRINTER_BINOP(Sub, " - ")
TVM_DECLARE_TEPRINTER_BINOP(Mul, "*")
TVM_DECLARE_TEPRINTER_BINOP(Div, " / ")
TVM_DECLARE_TEPRINTER_BINOP(Mod, " % ")
TVM_DECLARE_TEPRINTER_BINOP(EQ, " == ")
TVM_DECLARE_TEPRINTER_BINOP(NE, " != ")
TVM_DECLARE_TEPRINTER_BINOP(LT, " < ")
TVM_DECLARE_TEPRINTER_BINOP(LE, " <= ")
TVM_DECLARE_TEPRINTER_BINOP(GT, " > ")
TVM_DECLARE_TEPRINTER_BINOP(GE, " >= ")
TVM_DECLARE_TEPRINTER_BINOP(And, " and ")
TVM_DECLARE_TEPRINTER_BINOP(Or, " or ")

void TePrinter::VisitExpr_(const FloorDiv* op) {
  this->stream << "floordiv(";
  this->Print(op->a);
  this->stream<< ", ";
  this->Print(op->b);
  this->stream<< ")";
}

void TePrinter::VisitExpr_(const FloorMod* op) {
  this->stream << "floormod(";
  this->Print(op->a);
  this->stream<< ", ";
  this->Print(op->b);
  this->stream<< ")";
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
    this->stream << "):";
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
  this->stream << ":\n";
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
  this->stream << ":\n";
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
  auto* op = node.as<FunctionNode>();
  p->PrintIndent();
  p->stream << "def " << op->name << "(";
  for (size_t i = 0; i < op->params.size(); ++i) {
    p->Print(op->params[i]);
    if (i != op->params.size() - 1) {
      p->stream << ", ";
    }
  }
  p->stream << "):\n";
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
  auto* op = node.as<TensorRegionNode>();
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
  auto* op = node.as<AnnotationNode>();
  p->stream << op->attr_key << ": ";
  p->Print(op->value);
});

TVM_STATIC_IR_FUNCTOR(TePrinter, vtable)
.set_dispatch<ArrayNode>([](const ObjectRef& node, TePrinter* p) {
  auto* op = node.as<ArrayNode>();
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

TVM_REGISTER_GLOBAL("hybrid_te.AsText")
.set_body_typed<std::string(Function)>(
  [](Function function) {
      std::ostringstream os;
      TePrinter(os).Print(function);
      return os.str();
  });

}  // namespace te
}  // namespace tvm
