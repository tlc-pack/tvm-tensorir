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
 * \file tvm/include/te/ir.h
 * \brief Additional high level nodes in the TensorIR
 */
#ifndef TVM_TE_IR_H_
#define TVM_TE_IR_H_

#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/operation.h>
#include <string>

namespace tvm {
namespace ir {
class IRSubstitue;
}  // namespace ir

namespace te {
using namespace ir;

// forward declare class.
class SeqStmt;
class Schedule;
class ScheduleMutator;

/*! \brief The container of seq statement. */
class SeqStmtNode : public StmtNode {
 public:
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("seq", &seq);
  }
  /*! \return get the size of the sequence */
  size_t size() const {
    return seq.size();
  }
  /*!
   * \brief Get the index-th element in the sequence.
   */
  Stmt operator[](size_t index) const {
    return seq[index];
  }

  static constexpr const char* _type_key = "SeqStmt";
  TVM_DECLARE_NODE_TYPE_INFO(SeqStmtNode, StmtNode);

  /*!
   * \brief Hide seq for now, use SeqStmt.operator[] for iteration.
   * \note We may change the SeqStmtNode data structure later
   *       to directly contain the array content.
   */
  Array<Stmt> seq;
};

/*! \brief Sequence statement. */
class SeqStmt : public Stmt {
 public:
  /*!
   * \brief Construct SeqStmt.
   * \param seq The sequence.
   */
  explicit SeqStmt(Array<Stmt> seq);

  /*! \return get the size of the sequence */
  size_t size() const {
    return operator->()->size();
  }
  /*!
   * \brief Get the index-th element in the sequence.
   */
  Stmt operator[](size_t index) const {
    return (*(operator->()))[index];
  }

  TVM_DEFINE_NODE_REF_METHODS(SeqStmt, Stmt, SeqStmtNode);
};

/*!
 * \brief Load value from the high dimension buffer.
 *
 * \code
 *
 *  value = buffer[i, j];
 *
 * \endcode
 * \sa BufferStore
 */
class BufferLoad;
class BufferLoadNode : public ExprNode {
 public:
  /*! \brief The buffer variable. */
  Buffer buffer;
  /*! \brief The indices location to be loaded. */
  Array<Expr> indices;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer", &buffer);
    v->Visit("indices", &indices);
  }

  static constexpr const char* _type_key = "BufferLoad";
  TVM_DECLARE_NODE_TYPE_INFO(BufferLoadNode, ExprNode);
};

class BufferLoad : public Expr {
 public:
  explicit BufferLoad(DataType type,
                      Buffer buffer,
                      Array<Expr> indices);
  TVM_DEFINE_NODE_REF_METHODS(BufferLoad, Expr, BufferLoadNode);
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
class BufferStore;
class BufferStoreNode : public StmtNode {
 public:
  /*! \brief The buffer variable. */
  Buffer buffer;
  /*! \brief The value to be stored. */
  Expr value;
  /*! \brief The indices location to be stored. */
  Array<Expr> indices;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer", &buffer);
    v->Visit("value", &value);
    v->Visit("indices", &indices);
  }

  static constexpr const char* _type_key = "BufferStore";
  TVM_DECLARE_NODE_TYPE_INFO(BufferStoreNode, StmtNode);
};

class BufferStore : public Stmt {
 public:
  explicit BufferStore(Buffer buffer,
                       Expr value,
                       Array<Expr> indices);
  TVM_DEFINE_NODE_REF_METHODS(BufferStore, Stmt, BufferStoreNode);
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

  static constexpr const char* _type_key = "te.AnnotationNode";
  TVM_DECLARE_NODE_TYPE_INFO(AnnotationNode, Node);
};

class Annotation : public NodeRef {
 public:
  explicit Annotation(std::string attr_key, Expr value);
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
class Loop;
class LoopNode : public StmtNode {
 public:
  /*! \brief The loop variable. */
  Var loop_var;
  /*! \brief The minimum value of iteration. */
  Expr min;
  /*! \brief The extent of the iteration. */
  Expr extent;
  /*! \brief Loop annotations. */
  Array<Annotation> annotations;
  /*! \brief The body of the for loop. */
  Stmt body;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("loop_var", &loop_var);
    v->Visit("min", &min);
    v->Visit("extent", &extent);
    v->Visit("annotations", &annotations);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "Loop";
  TVM_DECLARE_NODE_TYPE_INFO(LoopNode, StmtNode);
};

class Loop : public Stmt {
 public:
  explicit Loop(Var loop_var,
                Expr min,
                Expr extent,
                Array<Annotation> annotations,
                Stmt body);

  TVM_DEFINE_NODE_REF_METHODS(Loop, Stmt, LoopNode);

 private:
  friend Schedule;
  friend ir::IRSubstitue;
  friend ScheduleMutator;
  /*! \brief mutate the node
   * Node that the mutate can be only used in schedule
   * and will be replaced later
   */
};

/*!
 * \brief A sub-region of a specific tensor.
 */
class TensorRegion;
class TensorRegionNode : public Node {
 public:
  /*! \brief The tensor of the tensor region. */
  Buffer buffer;
  /*! \brief The region array of the tensor region. */
  Array<Range> region;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer", &buffer);
    v->Visit("region", &region);
  }

  static constexpr const char* _type_key = "TensorRegion";
  TVM_DECLARE_NODE_TYPE_INFO(TensorRegionNode, Node);
};

class TensorRegion : public NodeRef {
 public:
  explicit TensorRegion(Buffer buffer, Array<Range> region);

  TVM_DEFINE_NODE_REF_METHODS(TensorRegion, NodeRef, TensorRegionNode);
};

/*!
 * \brief Allocate a new buffer in TE
 * \code
 *
 * BufferAllocate(buffer[shape], type)
 *
 * \endcode
 */
class BufferAllocate;
class BufferAllocateNode : public StmtNode {
 public:
  /*! \brief The buffer to be allocated. */
  Buffer buffer;
  std::string scope;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer", &buffer);
    v->Visit("scope", &scope);
  }

  static constexpr const char* _type_key = "BufferAllocate";
  TVM_DECLARE_NODE_TYPE_INFO(BufferAllocateNode, StmtNode);
};

class BufferAllocate : public Stmt {
 public:
  explicit BufferAllocate(Buffer buffer, std::string scope);

  TVM_DEFINE_NODE_REF_METHODS(BufferAllocate, Stmt, BufferAllocateNode);
};

/*!
 * \brief A block is the basic schedule unit in tensor expression
 * \code
 *
 *  block name(iter_type %v0[start:end] = expr0, ...,
 *  iter_type %v_n[start:end] = expr_n)
 *  W: [tensor_0[start:end]], ..., tensor_p[start:end]]
 *  R: [tensor_0[start:end]], ..., tensor_q[start:end]]
 *  pred: predicate expr
 *  attr: [attr_key0: attr_value0, ..., attr_key_m: attr_value_m] {
 *   // body
 *  }
 *
 * \endcode
 */
class Block;
class BlockNode : public StmtNode {
 public:
  /*! \brief The variables of the block. */
  Array<IterVar> iter_vars;
  /*! \brief The corresponding value of the iter vars. */
  Array<Expr> values;
  /*! \brief The read tensor region of the block. */
  Array<TensorRegion> reads;
  /*! \brief The write tensor region of the block. */
  Array<TensorRegion> writes;
  /*! \brief The body of the block. */
  Stmt body;
  /*! \brief The predicates of the block. */
  Expr predicate;
  /*! \brief The buffer allocated in the block. */
  Array<BufferAllocate> allocations;
  /*! \brief The annotation of the block. */
  Array<Annotation> annotations;
  /*! \brief The tag of the block. */
  std::string tag;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("body", &body);
    v->Visit("iter_vars", &iter_vars);
    v->Visit("reads", &reads);
    v->Visit("writes", &writes);
    v->Visit("predicate", &predicate);
    v->Visit("allocations", &allocations);
    v->Visit("annotations", &annotations);
    v->Visit("tag", &tag);
  }

  static constexpr const char* _type_key = "TeBlock";
  TVM_DECLARE_NODE_TYPE_INFO(BlockNode, StmtNode);
};

class Block : public Stmt {
 public:
  Block(Array<IterVar> iter_vars,
        Array<Expr> values,
        Array<TensorRegion> reads,
        Array<TensorRegion> writes,
        Stmt body,
        Expr predicate,
        Array<BufferAllocate> allocations,
        Array<Annotation> annotations,
        std::string tag);

  TVM_DEFINE_NODE_REF_METHODS(Block, Stmt, BlockNode);

  /*! \brief mutate the node
    * Node that the mutate can be only used in schedule
    * and will be replaced later
    */
 private:
  friend Schedule;
  friend ir::IRSubstitue;
  friend ScheduleMutator;
};

/*!
 * \brief A function in TE
 * \code
 *
 *  func func_name(var_0, ..., var_n) {
 *    // body
 *  }
 *
 * \endcode
 */
// TODO(siyuan): add matches in the text format.
class Function;
class FunctionNode : public Node {
 public:
  /*! \brief Function parameters */
  Array<Var> params;
  /*! \brief Parameter shape and type constraints */
  Map<Var, Buffer> buffer_map;
  /*! \brief Function body */
  Stmt body;
  /*! \brief Function name */
  std::string name;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("params", &params);
    v->Visit("buffer_map", &buffer_map);
    v->Visit("body", &body);
    v->Visit("name", &name);
  }

  static constexpr const char* _type_key = "TeFunction";
  TVM_DECLARE_NODE_TYPE_INFO(FunctionNode, Node);
};

class Function : public NodeRef {
 public:
  explicit Function(Array<Var> params,
                    Map<Var, Buffer> buffer_map,
                    std::string name,
                    Stmt body);

  TVM_DEFINE_NODE_REF_METHODS(Function, NodeRef, FunctionNode);

 private:
  friend Schedule;
  FunctionNode* Mutable() {
    return static_cast<FunctionNode*>(data_.get());
  }
};

}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_IR_H_
