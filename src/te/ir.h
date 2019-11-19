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
 * \file tvm/te/ir.h
 * \brief Additional high level nodes in the TensorIR
 */
#pragma once

#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/operation.h>

namespace tvm {
namespace te {
using namespace ir;

/*!
 * \brief Load value from the high dimension buffer.
 *
 * \code
 *
 *  value = buffer[i, j];
 *
 * \endcode
 * \sa BufferLoad
 */
class BufferLoad : public ExprNode {
 public:
  /*! \brief The buffer variable. */
  Var buffer_var;
  /*! \brief The indices location to be loaded. */
  Array<Expr> indices;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer_var", &buffer_var);
    v->Visit("indices", &indices);
  }

  TVM_DLL static Expr make(DataType type,
                           Var buffer_var,
                           Array<Expr> indices);

  static constexpr const char* _type_key = "BufferStore";
  TVM_DECLARE_NODE_TYPE_INFO(BufferLoad, ExprNode);
};

/*!
 * \brief Store value to the high dimension buffer.
 *
 * \code
 *
 *  buffer[i, j] = value;
 *
 * \endcode
 * \sa BufferLoad
 */
class BufferStore : public StmtNode {
 public:
  /*! \brief The buffer variable. */
  Var buffer_var;
  /*! \brief The value to be stored. */
  Expr value;
  /*! \brief The indices location to be stored. */
  Array<Expr> indices;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer_var", &buffer_var);
    v->Visit("value", &value);
    v->Visit("indices", &indices);
  }

  TVM_DLL static Stmt make(Var buffer_var,
                           Expr value,
                           Array<Expr> indices);

  static constexpr const char* _type_key = "BufferStore";
  TVM_DECLARE_NODE_TYPE_INFO(BufferStore, StmtNode);
};

/*! \brief Additional annotation of for loop. */
enum class LoopType : int {
  /*! \brief data parallel. */
      kDataPar = 0,
  /*! \brief reduce loop. */
      kReduce = 1,
  /*! \brief scan loop. */
      kScan = 2,
  /*! \brief opaque loop. */
      kOpaque = 3
};

/*!
 * \brief A loop annotation node to show attribute to the loop
 */
class Annotation;
class AnnotationNode : public Node {
 public:
  /*! \brief the type key of the attribute */
  std::string attr_key;
  /*! \brief The attribute value, value is well defined at current scope. */
  Expr value;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("attr_key", &attr_key);
    v->Visit("value", &value);
  }

  TVM_DLL static Annotation make(std::string attr_key, Expr value);

  static constexpr const char* _type_key = "te.AnnotationNode";
  TVM_DECLARE_NODE_TYPE_INFO(AnnotationNode, Node);
};

class Annotation : public NodeRef {
 public:
  TVM_DEFINE_NODE_REF_METHODS(Annotation, NodeRef, AnnotationNode)
};

/*!
 * \brief A for loop, with annotations and loop type.
 *
 * \code
 *
 *  for loop_var = min to min+extent (loop_type,
 *    attr: [attr_key0: attr_value0, ..., attr_key_m: attr_value_m]) {
 *    // body
 *  }
 *
 * \endcode
 */
class Loop : public StmtNode {
 public:
  /*! \brief The loop variable. */
  Var loop_var;
  /*! \brief The minimum value of iteration. */
  Expr min;
  /*! \brief The extent of the iteration. */
  Expr extent;
  /*! \brief The type of the for loop. */
  LoopType loop_type;
  /*! \brief Loop annotations. */
  Array<Annotation> annotations;
  /*! \brief The body of the for loop. */
  Stmt body;

  TVM_DLL static Stmt make(Var loop_var,
                           Expr min,
                           Expr extent,
                           LoopType loop_type,
                           Array<Annotation> annotations,
                           Stmt body);

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("loop_var", &loop_var);
    v->Visit("min", &min);
    v->Visit("extent", &extent);
    v->Visit("loop_type", &loop_type);
    v->Visit("annotations", &annotations);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "te.Loop";
  TVM_DECLARE_NODE_TYPE_INFO(Loop, StmtNode);
};

/*!
 * \brief A block variable with value, iteration type and required ranges
 */
class BlockVar;
class BlockVarNode : public Node {
 public:
  /*! \brief The variable of the block var. */
  Var data;
  /*! \brief The value of the block var. */
  Expr value;
  /*! \brief The required iteration type of the block var. */
  LoopType type;
  /*! \brief The required ranges the block var. */
  Range range;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("data", &data);
    v->Visit("value", &value);
    v->Visit("type", &type);
    v->Visit("range", &range);
  }

  TVM_DLL static BlockVar make(Var data, Expr value, LoopType type, Range range);

  static constexpr const char* _type_key = "te.TensorRegion";
  TVM_DECLARE_NODE_TYPE_INFO(BlockVarNode, Node);
};

class BlockVar : public NodeRef {
 public:
  TVM_DEFINE_NODE_REF_METHODS(BlockVar, NodeRef, BlockVarNode);
};

/*!
 * \brief A sub-region of a specific tensor.
 */
class TensorRegion;
class TensorRegionNode : public Node {
 public:
  /*! \brief The tensor of the tensor region. */
  Tensor data;
  /*! \brief The ranges array of the tensor region. */
  Array<Range> ranges;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("data", &data);
    v->Visit("ranges", &ranges);
  }

  TVM_DLL static TensorRegion make(Tensor data, Array<Range> ranges);

  static constexpr const char* _type_key = "te.TensorRegion";
  TVM_DECLARE_NODE_TYPE_INFO(TensorRegionNode, Node);
};

class TensorRegion : public NodeRef {
 public:
  TVM_DEFINE_NODE_REF_METHODS(TensorRegion, NodeRef, TensorRegionNode);
};

/*!
 * \brief A block is the basic schedule unit in tensor expression
 * \code
 *
 *  block name(iter_type %v0 = expr0, ... iter_type %v_n = expr_n)
 *  W: [tensor_0[start:end]], ..., tensor_p[start:end]]
 *  R: [tensor_0[start:end]], ..., tensor_q[start:end]]
 *  pred: predicate expr
 *  attr: [attr_key0: attr_value0, ..., attr_key_m: attr_value_m] {
 *   // body
 *  }
 *
 * \endcode
 */
class Block : public StmtNode {
 public:
  /*! \brief The variables of the block. */
  Array<BlockVar> vars;
  /*! \brief The input tensor region of the block. */
  Array<TensorRegion> inputs;
  /*! \brief The output tensor region of the block. */
  Array<TensorRegion> outputs;
  /*! \brief The body of the block. */
  Stmt body;
  /*! \brief The predicates of the block. */
  Expr predicate;
  /*! \brief The annotation of the block. */
  Array<Annotation> annotations;
  /*! \brief The tag of the block. */
  std::string tag;

  TVM_DLL static Stmt make(Array<BlockVar> vars,
                           Array<TensorRegion> inputs,
                           Array<TensorRegion> outputs,
                           Stmt body,
                           Expr predicates,
                           Array<Annotation> annotations,
                           std::string tag);

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("body", &body);
    v->Visit("vars", &vars);
    v->Visit("inputs", &inputs);
    v->Visit("outputs", &outputs);
    v->Visit("predicate", &predicate);
    v->Visit("annotations", &annotations);
    v->Visit("tag", &tag);
  }

  static constexpr const char* _type_key = "te.Block";
  TVM_DECLARE_NODE_TYPE_INFO(Block, StmtNode);
};

} // namespace te
} // namespace tvm
