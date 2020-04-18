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
 * \file printer/tir_hybrid_printer.cc
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
#include <tvm/ir/module.h>
#include "doc.h"
#include "meta_data.h"

namespace tvm {
namespace tir {

class TIRHybridPrinter :
    public StmtFunctor<Doc(const Stmt&)>,
    public ExprFunctor<Doc(const PrimExpr&)> {
 public:
  explicit TIRHybridPrinter(bool show_meta,
                            runtime::TypedPackedFunc<std::string(Stmt)> annotate = nullptr)
      : show_meta_(show_meta), annotate_(annotate) {}

  /*! \brief comm_reducer map */
  std::unordered_map<const CommReducerNode*, int> reducer_map;

  /*! \brief Print the node */
  TVM_DLL Doc Print(const ObjectRef& node);

  // Allow registration to be printer.
  using FType = NodeFunctor<Doc(const ObjectRef&, TIRHybridPrinter*)>;
  static FType& vtable();

  /*!
   * \brief special method to render vectors of docs with a separator
   * \param vec vector of docs
   * \param sep separator
   */
  static Doc PrintSep(const std::vector<Doc>& vec, const Doc& sep) {
    Doc seq;
    if (vec.size() != 0) {
      seq = vec[0];
      for (size_t i = 1; i < vec.size(); i++) {
        seq << sep << vec[i];
      }
    }
    return seq;
  }

  /*!
   * \brief dump meta info
   * \return Doc with meta info
   */
  Doc DumpMeta() {
    if (show_meta_) {
      return Doc::Text("__tvm_meta__ = ")
          << (meta_.empty() ? Doc::Text("None") : meta_.GetMetaSection());
    } else {
      return Doc::Text("");
    }
  }

  /*!
   * \brief Entry point of printer
   * \return Doc
   */
  Doc PrintFinal(const ObjectRef& functions) {
    if (functions.as<IRModuleNode>()) {
      return Print(functions);
    } else if (functions.as<FunctionNode>()) {
      return Print(functions) << Doc::NewLine() << DumpMeta();
    } else {
      return Doc::Text("");
    }
  }

 private:
  /*! \brief whether show meta data */
  bool show_meta_;
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
  Doc VisitExpr_(const ReductionNode* op) override;
  Doc VisitExprDefault_(const Object* op) override;

  Doc VisitStmt_(const SeqStmtNode* op) override;
  Doc VisitStmt_(const EvaluateNode* op) override;
  Doc VisitStmt_(const BlockRealizeNode* op) override;
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
        doc << "# " << annotated_stmt << Doc::NewLine();
      }
    }
    return doc;
  }

  /*!
  * \brief special method to print out data type
  * \param dtype The data type
  */
  static Doc PrintDType(DataType dtype) {
    return Doc::Text(runtime::DLDataType2String(dtype));
  }

  /*!
   * \brief special method to print out const scalar
   * \param dtype The data type
   * \param data The pointer to hold the data.
   */
  template <typename T>
  static Doc PrintConstScalar(DataType dtype, const T* data) {
    Doc doc;
    std::ostringstream os;
    os << data[0];
    if (dtype == DataType::Int(32)) {
      doc << Doc::Text(os.str());
    } else {
      doc << PrintDType(dtype) << "(" << Doc::Text(os.str()) << ")";
    }
    return doc;
  }
};

Doc TIRHybridPrinter::Print(const ObjectRef& node) {
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

Doc TIRHybridPrinter::VisitExprDefault_(const Object* op) {
  return this->meta_.GetMetaNode(GetRef<ObjectRef>(op));
}

Doc TIRHybridPrinter::VisitStmtDefault_(const Object* op) {
  return this->meta_.GetMetaNode(GetRef<ObjectRef>(op));
}

Doc TIRHybridPrinter::VisitExpr_(const VarNode* op) {
  return Doc::Text(op->name_hint);
}

#define TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(OpName, OpString)      \
  Doc TIRHybridPrinter::VisitExpr_(const OpName* op) {              \
    Doc doc;                                                        \
    doc << '(' << Print(op->a) << OpString << Print(op->b) << ")";  \
    return doc;                                                     \
  }                                                                 \

TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(AddNode, " + ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(SubNode, " - ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(MulNode, "*")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(DivNode, " / ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(ModNode, " % ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(EQNode, " == ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(NENode, " != ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(LTNode, " < ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(LENode, " <= ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(GTNode, " > ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(GENode, " >= ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(AndNode, " and ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(OrNode, " or ")

Doc TIRHybridPrinter::VisitExpr_(const FloorDivNode* op) {
  Doc doc;
  doc << "floordiv(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TIRHybridPrinter::VisitExpr_(const FloorModNode* op) {
  Doc doc;
  doc << "floormod(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TIRHybridPrinter::VisitExpr_(const MinNode* op) {
  Doc doc;
  doc << "min(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TIRHybridPrinter::VisitExpr_(const MaxNode* op) {
  Doc doc;
  doc << "max(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TIRHybridPrinter::VisitExpr_(const IntImmNode* op) {
  return PrintConstScalar<int64_t>(op->dtype, &(op->value));
}

Doc TIRHybridPrinter::VisitExpr_(const FloatImmNode* op) {
  return PrintConstScalar<double>(op->dtype, &(op->value));
}

Doc TIRHybridPrinter::VisitExpr_(const StringImmNode* op) {
  return Doc::StrLiteral(op->value);
}

Doc TIRHybridPrinter::VisitExpr_(const BufferLoadNode* op) {
  Doc doc;
  doc << Print(op->buffer->data) << Print(op->indices);
  return doc;
}

Doc TIRHybridPrinter::VisitExpr_(const ReductionNode* op) {
  Doc doc;
  doc << "reduction(" << Print(op->update) << ", " << Print(op->init) << ")";
  return doc;
}

Doc TIRHybridPrinter::VisitStmt_(const SeqStmtNode* op) {
  std::vector<Doc> stmts;
  for (Stmt stmt : op->seq) {
    stmts.push_back(Print(stmt));
  }
  return PrintSep(stmts, Doc::NewLine());
}

Doc TIRHybridPrinter::VisitStmt_(const EvaluateNode* op) {
  return Print(op->value);
}

Doc TIRHybridPrinter::VisitStmt_(const BlockRealizeNode* op) {
  const BlockNode* block_op = (op->block).as<BlockNode>();
  // print block name and block vars
  Doc doc;
  doc << "with block({";
  for (size_t i = 0; i < block_op->iter_vars.size(); ++i) {
    const auto& iter_var = block_op->iter_vars[i];
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
    doc << Print(op->binding_values[i]);
    if (i != block_op->iter_vars.size() - 1) {
      doc << ", ";
    }
  }
  doc << "}";

  // print tensor region and annotations
  doc << ", writes=" << Print(block_op->writes);
  doc << ", reads=" << Print(block_op->reads);
  if (!is_one(op->predicate)) {
    doc << ", predicate=" << Print(op->predicate);
  }
  if (!block_op->annotations.empty()) {
    doc << ", annotations=" << Print(block_op->annotations);
  }
  doc << ", name=" << Doc::StrLiteral(block_op->tag) << "):";
  // print body
  Doc body;
  body << Doc::NewLine();
  for (const auto& allocate : block_op->allocations) {
    body << Print(allocate) << Doc::NewLine();
  }
  body << Print(block_op->body);
  doc << Doc::Indent(4, body);
  return doc;
}

Doc TIRHybridPrinter::VisitStmt_(const LoopNode* op) {
  Doc doc;
  // print loop and annotations
  doc << "for " << Print(op->loop_var);
  doc << " in range(" << Print(op->min) << ", " << Print(Simplify(op->min + op->extent));
  if (!op->annotations.empty()) {
    doc << ", annotation = {";
    for (size_t i = 0; i < op->annotations.size(); ++i) {
      if (i != 0) {
        doc << ", ";
      }
      doc << "\"" << op->annotations[i]->attr_key << "\":" << Print(op->annotations[i]->value);
    }
    doc << "}";
  }
  doc << "):";

  // print body
  Doc body;
  body << Doc::NewLine() << Print(op->body);
  doc << Doc::Indent(4, body);
  return doc;
}

Doc TIRHybridPrinter::VisitStmt_(const BufferAllocateNode* op) {
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
  doc << ", " << Doc::StrLiteral(PrintDType(op->buffer->dtype).str());
  doc << ", " << Doc::StrLiteral(op->scope) << ")";
  return doc;
}

Doc TIRHybridPrinter::VisitStmt_(const BufferStoreNode* op) {
  Doc doc;
  doc << Print(op->buffer->data) << Print(op->indices);
  doc << " = " << Print(op->value);
  return doc;
}

TVM_STATIC_IR_FUNCTOR(TIRHybridPrinter, vtable)
.set_dispatch<IRModuleNode>([](const ObjectRef& node, TIRHybridPrinter* p) {
  auto* op = node.as<IRModuleNode>();
  Doc doc;
  doc << "class Module:";

  Doc body;
  body << Doc::NewLine();
  std::vector<Doc> functions;
  for (auto it = op->functions.begin(); it != op->functions.end(); ++it) {
    if ((*it).second.as<FunctionNode>()) {
      functions.push_back(p->Print((*it).second));
    }
  }
  body << TIRHybridPrinter::PrintSep(functions, Doc::NewLine() << Doc::NewLine());
  body << Doc::NewLine() << p->DumpMeta();
  doc << Doc::Indent(4, body);
  return doc;
});

TVM_STATIC_IR_FUNCTOR(TIRHybridPrinter, vtable)
.set_dispatch<FunctionNode>([](const ObjectRef& node, TIRHybridPrinter* p) {
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
  body << Doc::NewLine();
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
    body << ", " << Doc::StrLiteral(runtime::DLDataType2String((*it).second->dtype));
    body << ")" << Doc::NewLine();
  }

  // print body
  body << p->Print(op->body);
  doc << Doc::Indent(4, body);
  return doc;
});

TVM_STATIC_IR_FUNCTOR(TIRHybridPrinter, vtable)
.set_dispatch<TensorRegionNode>([](const ObjectRef& node, TIRHybridPrinter* p) {
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

TVM_STATIC_IR_FUNCTOR(TIRHybridPrinter, vtable)
.set_dispatch<AnnotationNode>([](const ObjectRef& node, TIRHybridPrinter* p) {
  auto* op = node.as<AnnotationNode>();
  Doc doc;
  doc << op->attr_key << ": " << p->Print(op->value);
  return doc;
});

TVM_STATIC_IR_FUNCTOR(TIRHybridPrinter, vtable)
.set_dispatch<ArrayNode>([](const ObjectRef& node, TIRHybridPrinter* p) {
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

TIRHybridPrinter::FType& TIRHybridPrinter::vtable() {
  static FType inst;
  return inst;
}

TVM_REGISTER_GLOBAL("tir.hybrid.AsHybrid")
.set_body_typed<std::string(const ObjectRef&, bool)>(
[](const ObjectRef& functions, bool show_meta) {
  CHECK(functions.as<FunctionNode>() != nullptr || functions.as<IRModuleNode>() != nullptr);
  return TIRHybridPrinter(show_meta).PrintFinal(functions).str() + "\n";
});

}  // namespace tir
}  // namespace tvm
