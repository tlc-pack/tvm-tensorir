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
 * \file hybrid_tir/printer.cc
 * \brief Printer class to print Te IR to python syntax script
 */

#include <tvm/runtime/registry.h>
#include <tvm/tir/ir.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/node/serialization.h>
#include "doc.h"


namespace tvm {
namespace tir {

class TextMetaDataContext {
 public:
  /*!
   * \brief Get text representation of meta node.
   * \param node The node to be converted to meta node.
   * \return A string representation of the meta node.
   */
  Doc GetMetaNode(const ObjectRef& node) {
    auto it = meta_repr_.find(node);
    if (it != meta_repr_.end()) {
      return it->second;
    }
    std::string type_key = node->GetTypeKey();
    CHECK(!type_key.empty());
    Array<ObjectRef>& mvector = meta_data_[type_key];
    auto index = static_cast<int64_t>(mvector.size());
    mvector.push_back(node);
    Doc doc;
    doc << "meta[" << type_key << "][" << index << "]";
    meta_repr_[node] = doc;
    return meta_repr_[node];
  }

  Doc GetMetaSection() const {
    if (meta_data_.empty()) return Doc();
    return Doc(SaveJSON(Map<std::string, ObjectRef>(meta_data_.begin(), meta_data_.end())));
  }

  bool empty() const {
    return meta_data_.empty();
  }

 private:
  std::unordered_map<std::string, Array<ObjectRef> > meta_data_;
  std::unordered_map<ObjectRef, Doc, ObjectHash, ObjectEqual> meta_repr_;
};

class TePrinter :
      public StmtFunctor<Doc(const Stmt&)>,
      public ExprFunctor<Doc(const PrimExpr&)>{
 public:
  explicit TePrinter(runtime::TypedPackedFunc<std::string(Stmt)> annotate) : annotate_(annotate) {}
  /*! \brief Print the node */
  TVM_DLL Doc Print(const ObjectRef& node);

  // Allow registration to be printer.
  using FType = NodeFunctor<Doc(const ObjectRef&, TePrinter*)>;
  static FType& vtable();

 private:
  /*! \brief additional comment function */
  runtime::TypedPackedFunc<std::string(Stmt)> annotate_;
  /*! \brief meta data context */
  TextMetaDataContext meta_;

  Doc VisitExpr_(const VarNode* op) override;
  Doc VisitExpr_(const AddNode* op) override;
  Doc VisitExpr_(const SubNode* op) override;
  Doc VisitExpr_(const MulNode* op) override;
  Doc VisitExpr_(const DivNode* op) override;
  Doc VisitExpr_(const ModNode* op) override;
  Doc VisitExpr_(const FloorDivNode* op) override;
  Doc VisitExpr_(const FloorModNode* op) override;
  Doc VisitExpr_(const MinNode* op) override;
  Doc VisitExpr_(const MaxNode* op) override;
  Doc VisitExpr_(const EQNode* op) override;
  Doc VisitExpr_(const NENode* op) override;
  Doc VisitExpr_(const LTNode* op) override;
  Doc VisitExpr_(const LENode* op) override;
  Doc VisitExpr_(const GTNode* op) override;
  Doc VisitExpr_(const GENode* op) override;
  Doc VisitExpr_(const AndNode* op) override;
  Doc VisitExpr_(const OrNode* op) override;
  Doc VisitExpr_(const IntImmNode* op) override;
  Doc VisitExpr_(const FloatImmNode* op) override;
  Doc VisitExpr_(const StringImmNode* op) override;
  Doc VisitExpr_(const BufferLoadNode* op) override;
  Doc VisitExprDefault_(const Object* op) override;

  Doc VisitStmt_(const SeqStmtNode* op) override;
  Doc VisitStmt_(const EvaluateNode* op) override;
  Doc VisitStmt_(const BlockNode* op) override;
  Doc VisitStmt_(const LoopNode* op) override;
  Doc VisitStmt_(const BufferAllocateNode* op) override;
  Doc VisitStmt_(const BufferStoreNode* op) override;
  Doc VisitStmtDefault_(const Object* op) override;

  /*!
  * \brief Print additional info about expr in comment.
  * \param expr The expression.
  */
  Doc PrintOptionalInfo(const Stmt& stmt) {
    Doc doc;
    // default annotations
    if (annotate_ != nullptr) {
      std::string annotated_stmt = annotate_(stmt);
      if (!annotated_stmt.empty()) {
        doc << "# " << annotated_stmt << PrintNewLine();
      }
    }
    return doc;
  }
};

Doc TePrinter::Print(const ObjectRef& node) {
  if (node.as<StmtNode>()) {
    return PrintOptionalInfo(Downcast<Stmt>(node)) << VisitStmt(Downcast<Stmt>(node));
  } else if (node.as<PrimExprNode>()) {
    return VisitExpr(Downcast<PrimExpr>(node));
  } else {
    static const FType& f = vtable();
    if (f.can_dispatch(node)) {
      return f(node, this);
    } else {
      return this->meta_.GetMetaNode(node);
    }
  }
}

Doc TePrinter::VisitExprDefault_(const Object* op) {
  return this->meta_.GetMetaNode(GetRef<ObjectRef>(op));
}

Doc TePrinter::VisitStmtDefault_(const Object* op) {
  return this->meta_.GetMetaNode(GetRef<ObjectRef>(op));
}

Doc TePrinter::VisitExpr_(const VarNode* op) {
  return Doc(op->name_hint);
}

#define TVM_DECLARE_TEPRINTER_BINOP(OpName, OpString)               \
  Doc TePrinter::VisitExpr_(const OpName* op) {                     \
    Doc doc;                                                        \
    doc << '(' << Print(op->a) << OpString << Print(op->b) << ")";  \
    return doc;                                                     \
  }                                                                 \

TVM_DECLARE_TEPRINTER_BINOP(AddNode, " + ")
TVM_DECLARE_TEPRINTER_BINOP(SubNode, " - ")
TVM_DECLARE_TEPRINTER_BINOP(MulNode, "*")
TVM_DECLARE_TEPRINTER_BINOP(DivNode, " / ")
TVM_DECLARE_TEPRINTER_BINOP(ModNode, " % ")
TVM_DECLARE_TEPRINTER_BINOP(EQNode, " == ")
TVM_DECLARE_TEPRINTER_BINOP(NENode, " != ")
TVM_DECLARE_TEPRINTER_BINOP(LTNode, " < ")
TVM_DECLARE_TEPRINTER_BINOP(LENode, " <= ")
TVM_DECLARE_TEPRINTER_BINOP(GTNode, " > ")
TVM_DECLARE_TEPRINTER_BINOP(GENode, " >= ")
TVM_DECLARE_TEPRINTER_BINOP(AndNode, " and ")
TVM_DECLARE_TEPRINTER_BINOP(OrNode, " or ")

Doc TePrinter::VisitExpr_(const FloorDivNode* op) {
  Doc doc;
  doc << "floordiv(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TePrinter::VisitExpr_(const FloorModNode* op) {
  Doc doc;
  doc << "floormod(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TePrinter::VisitExpr_(const MinNode* op) {
  Doc doc;
  doc << "min(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TePrinter::VisitExpr_(const MaxNode* op) {
  Doc doc;
  doc << "max(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TePrinter::VisitExpr_(const IntImmNode* op) {
  return PrintConstScalar<int64_t>(op->dtype, &(op->value));
}

Doc TePrinter::VisitExpr_(const FloatImmNode* op) {
  return PrintConstScalar<double>(op->dtype, &(op->value));
}

Doc TePrinter::VisitExpr_(const StringImmNode* op) {
  return PrintString(op->value);
}

Doc TePrinter::VisitExpr_(const BufferLoadNode* op) {
  Doc doc;
  doc << Print(op->buffer->data) << Print(op->indices);
  return doc;
}

Doc TePrinter::VisitStmt_(const SeqStmtNode* op) {
  std::vector<Doc> stmts;
  for (Stmt stmt : op->seq) {
    stmts.push_back(Print(stmt));
  }
  return PrintSep(stmts, PrintNewLine());
}

Doc TePrinter::VisitStmt_(const EvaluateNode* op) {
  return Print(op->value);
}

Doc TePrinter::VisitStmt_(const BlockNode* op) {
  // print block name and block vars
  Doc doc;
  doc << "with block({";
  for (size_t i = 0; i < op->iter_vars.size(); ++i) {
    const auto& iter_var = op->iter_vars[i];
    doc << Print(iter_var->var);
    doc << "(";
    doc << Print(iter_var->dom->min);
    doc << ", ";
    doc << Print(iter_var->dom->min + iter_var->dom->extent);
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
      doc << ", iter_type=\"" << str << "\"";
    }
    doc << "):";
    doc << Print(op->values[i]);
    if (i != op->iter_vars.size() - 1) {
      doc << ", ";
    }
  }
  doc << "}";

  // print tensor region and annotations
  doc << ", writes=" << Print(op->writes);
  doc << ", reads=" << Print(op->reads);
  if (!is_one(op->predicate)) {
    doc << ", predicate=" << Print(op->predicate);
  }
  if (!op->annotations.empty()) {
    doc << ", annotations=" << Print(op->annotations);
  }
  doc << ", name=" << PrintString(op->tag) <<  "):";
  // print body
  Doc body;
  body << PrintNewLine();
  for (const auto& allocate : op->allocations) {
    body << Print(allocate) << PrintNewLine();
  }
  body << Print(op->body);
  doc << Indent(2, body);
  return doc;
}

Doc TePrinter::VisitStmt_(const LoopNode* op) {
  Doc doc;
  // print loop and annotations
  doc << "for " << Print(op->loop_var);
  doc << " in range(" << Print(op->min) << ", " << Print(Simplify(op->min + op->extent));
  doc << "):";

  // print body
  Doc body;
  body << PrintNewLine() << Print(op->body);
  doc << Indent(2, body);
  return doc;
}

Doc TePrinter::VisitStmt_(const BufferAllocateNode* op) {
  Doc doc;
  doc << op->buffer->name;
  doc << " = buffer_allocate(";
  doc << '(';
  for (size_t i = 0; i < op->buffer->shape.size(); ++i) {
    if (i != 0) {
      doc << ", ";
    }
    doc << Print(op->buffer->shape[i]);
  }
  doc << ')';
  doc << ", " << PrintString(PrintDType(op->buffer->dtype).str());
  doc << ", " << PrintString(op->buffer->name);
  doc << ", " << PrintString(op->scope) << ")";
  return doc;
}

Doc TePrinter::VisitStmt_(const BufferStoreNode* op) {
  Doc doc;
  doc << Print(op->buffer->data) << Print(op->indices);
  doc << " = " << Print(op->value);
  return doc;
}

TVM_STATIC_IR_FUNCTOR(TePrinter, vtable)
.set_dispatch<FunctionNode>([](const ObjectRef& node, TePrinter* p) {
  auto* op = node.as<FunctionNode>();
  Doc doc;
  doc << "def " << op->name << "(";
  for (size_t i = 0; i < op->params.size(); ++i) {
    doc << p->Print(op->params[i]);
    if (i != op->params.size() - 1) {
      doc << ", ";
    }
  }
  doc << "):";

  Doc body;
  body << PrintNewLine();
  // print buffer_bind
  for (auto it = op->buffer_map.begin(); it != op->buffer_map.end(); ++it) {
    body << (*it).second->name << " = buffer_bind(";
    body << p->Print((*it).first);
    body << ", (";
    for (size_t i = 0; i < (*it).second->shape.size(); ++i) {
      if (i != 0) {
        body << ", ";
      }
      body << p->Print((*it).second->shape[i]);
    }
    body << ')';
    body << ", " << PrintString(PrintDType((*it).second->dtype).str());
    body << ", " << PrintString((*it).second->name);
    body << ")" << PrintNewLine();
  }

  // print body
  body << p->Print(op->body);
  doc << Indent(2, body);
  return doc;
});

TVM_STATIC_IR_FUNCTOR(TePrinter, vtable)
.set_dispatch<TensorRegionNode>([](const ObjectRef& node, TePrinter* p) {
  auto* op = node.as<TensorRegionNode>();
  Doc doc;
  doc << p->Print(op->buffer->data) << "[";
  for (size_t i = 0; i < op->region.size(); ++i) {
    const auto& range = op->region[i];
    doc << p->Print(range->min) << ":" << p->Print(range->min + range->extent);
    if (i != op->region.size() - 1) {
      doc << ", ";
    }
  }
  doc << "]";
  return doc;
});

TVM_STATIC_IR_FUNCTOR(TePrinter, vtable)
.set_dispatch<AnnotationNode>([](const ObjectRef& node, TePrinter* p) {
  auto* op = node.as<AnnotationNode>();
  Doc doc;
  doc << op->attr_key << ": " << p->Print(op->value);
  return doc;
});

TVM_STATIC_IR_FUNCTOR(TePrinter, vtable)
.set_dispatch<ArrayNode>([](const ObjectRef& node, TePrinter* p) {
  auto* op = node.as<ArrayNode>();
  Doc doc;
  doc << '[';
  for (size_t i = 0; i < op->data.size(); ++i) {
    if (i != 0) {
      doc << ", ";
    }
    doc << p->Print(op->data[i]);
  }
  doc << ']';
  return doc;
});

TePrinter::FType& TePrinter::vtable() {
  static FType inst;
  return inst;
}

TVM_REGISTER_GLOBAL("hybrid_tir.AsText")
.set_body_typed<std::string(const Function&)>(
    [](const Function& function) {
      return TePrinter(nullptr).Print(function).str() + "\n";
    });

}  // namespace te
}  // namespace tvm
