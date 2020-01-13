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
 * \file tvm/te/printer.h
 * \brief Printer class to print Te IR to python syntax script
 */

#include <tvm/te/ir.h>
#include <tvm/ir.h>
#include <tvm/ir_functor_ext.h>

namespace tvm {
namespace te {

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

  void VisitOther(const FunctionNode* op);
  void VisitOther(const TensorRegionNode* op);
  void VisitOther(const AnnotationNode* op);
  void VisitOther(const ArrayNode* op);

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

  void VisitStmt_(const SeqStmtNode* op) override;
  void VisitStmt_(const Evaluate* op) override;
  void VisitStmt_(const te::BlockNode* op) override;
  void VisitStmt_(const te::LoopNode* op) override;
  void VisitStmt_(const te::BufferAllocateNode* op) override;
  void VisitStmt_(const te::BufferStoreNode* op) override;
  void VisitExpr(const Expr& e) override;
};

} // namespace te
} // namespace tvm