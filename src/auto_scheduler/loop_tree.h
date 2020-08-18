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
#ifndef SRC_AUTO_SCHEDULER_LOOP_TREE_H_
#define SRC_AUTO_SCHEDULER_LOOP_TREE_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>

#include <string>

namespace tvm {
namespace auto_scheduler {

/**************** Define Iterator ****************/

/*! \brief The type of an iterator. */
enum class IterKind : int {
  /*! \brief Data parallel iterator. */
  kDataPar = 0,
  /*! \brief Reduction iterator. */
  kReduction = 1,
  /*! \brief Fused spatial and reduction iterator. */
  kMixed = 2,
  /*! \brief Special iterator. (e.g. virtual root iterator) */
  kSpecial = 3,
};

/*!
 * \brief Converts IterKind to string
 * \param kind The IterKind to be converted
 * \return The corresponding string
 */
inline std::string IterKind2String(IterKind kind) {
  static std::string results[] = {"data_par", "reduce", "mixed", "special"};
  return results[static_cast<int>(kind)];
}

/*! \brief The type of the annotation of an iterator. */
enum class IterAnnotation : int {
  /*! \brief This iterator has no annotation. */
  kNone = 0,
  /*! \brief This iterator has been unrolled. */
  kUnroll = 1,
  /*! \brief This iterator has been vectorized. */
  kVectorize = 2,
  /*! \brief This iterator has been paralleld. */
  kParallel = 3,
  /*! \brief This iterator has been bind to virtual thread. */
  kVThread = 4,
  /*! \brief This iterator has been bind to blockIdx.x. */
  kBlockX = 5,
  /*! \brief This iterator has been bind to blockIdx.y. */
  kBlockY = 6,
  /*! \brief This iterator has been bind to blockIdx.z. */
  kBlockZ = 7,
  /*! \brief This iterator has been bind to threadIdx.x. */
  kThreadX = 8,
  /*! \brief This iterator has been bind to threadIdx.y. */
  kThreadY = 9,
  /*! \brief This iterator has been bind to threadIdx.z. */
  kThreadZ = 10,
  /*! \brief This iterator has been mapped with a tensorize intrinsic. */
  kTensorized = 11,
};

/*!
 * \brief Converts IterAnnotation to string
 * \param annotation The IterAnnotation to be converted
 * \return The corresponding string
 */
inline std::string IterAnnotation2String(IterAnnotation annotation) {
  static std::string results[] = {
      "none",       "unroll",     "vectorize",   "parallel",    "vthread",     "blockIdx.x",
      "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z", "tensorized",
  };
  return results[static_cast<int>(annotation)];
}

/*!
 * \brief A loop iterator
 * Similar to tvm::IterVar in `include/tvm/tir/expr.h`
 */
class IteratorNode : public Object {
 public:
  /*! \brief The name of this iterator. */
  String name;
  /*! \brief The extent of this iterator. */
  PrimExpr extent;
  /*! \brief The iterator type of this iterator. */
  IterKind kind;
  /*! \brief The annotation type of this iterator. */
  IterAnnotation annotation;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("extent", &extent);
    v->Visit("kind", &kind);
    v->Visit("annotation", &annotation);
  }

  static constexpr const char* _type_key = "auto_scheduler.Iterator";
  TVM_DECLARE_FINAL_OBJECT_INFO(IteratorNode, Object);
};

/*!
 * \brief Managed reference to IteratorNode
 */
class Iterator : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of this iterator.
   * \param extent The extent of this iterator.
   * \param kind The iterator type of this iterator.
   * \param annotation The annotation type of this iterator.
   */
  Iterator(String name, PrimExpr extent, IterKind kind, IterAnnotation annotation);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Iterator, ObjectRef, IteratorNode);
};

/**************** Define MetaIR ****************/

/*!
 * \brief The base node of the loop tree,
 * recording the parent and siblings of a node in the tree
 */
class MetaIRNode : public Object {
 public:
  /*! \brief Parent of the node */
  mutable const MetaIRNode* parent;
  /*! \brief Left sibling of the node */
  mutable const MetaIRNode* left_sibling;
  /*! \brief Right sibling of the node */
  mutable const MetaIRNode* right_sibling;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "auto_scheduler.MetaIR";
  TVM_DECLARE_BASE_OBJECT_INFO(MetaIRNode, Object);
};

/*! \brief Managed reference to MetaIRNode */
class MetaIR : public ObjectRef {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(MetaIR, ObjectRef, MetaIRNode);

 protected:
  /*! \brief Constructor. The node should never be constructed directly. */
  MetaIR() = default;
};

/**************** Define LeafStmt ****************/

/*!
 * \brief The category that the leaf statement falls into
 */
enum class LeafStmtKind : int {
  /*! \brief The leaf statement is BufferStore */
  kBufferStore,
  /*! \brief The leaf statement is ReduceStep */
  kReduceStep,
};

/*!
 * \brief Converts LeafStmtKind to string
 * \param kind The LeafStmtKind to be converted
 * \return A string, the conversion result
 */
inline std::string LeafStmtKind2String(LeafStmtKind kind) {
  static std::string results[] = {"BufferStore", "ReduceStep"};
  return results[static_cast<int>(kind)];
}

/*! \brief The leaf statement in the loop tree */
class LeafStmtNode : public MetaIRNode {
 public:
  /*! \brief The category the statement is in */
  LeafStmtKind kind;
  /*! \brief The buffer and index the statement writes */
  tir::BufferLoad write;
  /*! \brief The buffers and indices the statement reads */
  Array<tir::BufferLoad> reads;
  /*! \brief The original to the TIR statement it corresponds to */
  tir::Stmt stmt;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("kind", &kind);
    v->Visit("write", &write);
    v->Visit("reads", &reads);
    v->Visit("stmt", &stmt);
  }

  static constexpr const char* _type_key = "auto_scheduler.LeafStmt";
  TVM_DECLARE_FINAL_OBJECT_INFO(LeafStmtNode, MetaIRNode);
};

/*! \brief Managed reference to LeafStmtNode */
class LeafStmt : public MetaIR {
 public:
  /*!
   * \brief Constructor from TIR statement
   * \param stmt The TIR statement
   */
  explicit LeafStmt(const tir::Stmt& stmt);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(LeafStmt, MetaIR, LeafStmtNode);
};

/**************** Define LoopTree ****************/

/*! \brief A node in loop tree */
class LoopTreeNode : public MetaIRNode {
 public:
  /*! \brief Iterators in the block */
  mutable Array<Iterator> iters;
  /*! \brief The corresponding BlockRealize node in TIR */
  mutable Optional<tir::BlockRealize> block_realize;
  /*! \brief Children of the node, can be LoopTree or tir::Stmt */
  mutable Array<MetaIR> children;
  /*!
   * \brief Converts the LoopTreeNode to human readable string format
   * \return The human readable string format
   */
  String ToString() const;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("iters", &iters);
    v->Visit("block_realize", &block_realize);
    v->Visit("children", &children);
  }

  static constexpr const char* _type_key = "auto_scheduler.LoopTree";
  TVM_DECLARE_FINAL_OBJECT_INFO(LoopTreeNode, MetaIRNode);
};

/*! \brief Managed reference to LoopTreeNode */
class LoopTree : public MetaIR {
 public:
  /*!
   * \brief Constructor of LoopTree
   * \param children Children of the node to be constructed
   * \param iters Iterators of the root of the node
   * \param block_realize The corresponding BlockRealize node in TIR
   */
  LoopTree(Array<Iterator> iters, Optional<tir::BlockRealize> block_realize,
           Array<MetaIR> children);
  /*!
   * \brief Construct a LoopTree from PrimFunc
   * \param func The PrimFunc
   * \return The loop tree constructed
   */
  static LoopTree FromPrimFunc(const tir::PrimFunc& func);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(LoopTree, MetaIR, LoopTreeNode);
};

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // SRC_AUTO_SCHEDULER_LOOP_TREE_H_
