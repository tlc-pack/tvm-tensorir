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
#ifndef SRC_AUTO_SCHEDULER_LOOP_TREE_H_ /* TODO(@junrushao1994): guard name convention */
#define SRC_AUTO_SCHEDULER_LOOP_TREE_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>

#include <string>

namespace tvm {
namespace auto_scheduler {

/*! \brief The type of an iterator. */
enum class IterKind : int {
  /*! \brief Spatial iterator. */
  kSpatial = 0,
  /*! \brief Reduction iterator. */
  kReduction = 1,
  /*! \brief Fused spatial and reduction iterator. */
  kMixed = 2,
  /*! \brief Special iterator. (e.g. virtual root iterator) */
  kSpecial = 3
};

/*!
 * \brief Converts IterKind to string
 * \param kind The IterKind to be converted
 * \return The corresponding string
 */
inline std::string IterKind2String(IterKind kind) {
  static std::string results[] = {
      "space",
      "reduce",
      "mixed",
  };
  return results[static_cast<int>(kind)];
}

/*! \brief The type of an iterator's annotation. */
enum class IterAnnotation : int {
  /*! \brief This iterator has no annotation. */
  kNone = 0,
  /*! \brief This iterator has been unrolled. */
  kUnroll = 1,
  /*! \brief This iterator has been vectorized. */
  kVectorize = 2,
  /*! \brief This iterator has been paralleld. */
  kParallel = 3,
  /*! \brief This iterator has been bind to vthread. */
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
  /*! \brief The min value of this iterator. */
  PrimExpr min;
  /*! \brief The extent of this iterator. */
  PrimExpr extent;
  /*! \brief The iterator type of this iterator. */
  IterKind kind;
  /*! \brief The annotation type of this iterator. */
  IterAnnotation annotation;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("min", &min);
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
   * \param min The min value of this iterator.
   * \param extent The exntent of this iterator.
   * \param kind The iterator type of this iterator.
   * \param annotation The annotation type of this iterator.
   */
  Iterator(String name, PrimExpr min, PrimExpr extent, IterKind kind, IterAnnotation annotation);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Iterator, ObjectRef, IteratorNode);
};

/*! \brief A node in loop tree */
class LoopTreeNode : public Object {
 public:
  /*! \brief Children of the node, can be LoopTree or tir::Stmt */
  Array<ObjectRef> children;
  /*! \brief Iterators in the block */
  Array<Iterator> iters;
  /*! \brief The corresponding BlockRealize node in TIR */
  const tir::BlockRealizeNode* block_realize;
  /*!
   * \brief Converts the LoopTreeNode to human readable string format\
   * \return The human readable string format
   */
  String ToString() const;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("children", &children);
    v->Visit("iters", &iters);
    // We didn't visit block_realize
  }

  static constexpr const char* _type_key = "auto_scheduler.LoopTree";
  TVM_DECLARE_FINAL_OBJECT_INFO(LoopTreeNode, Object);

 private:
  /*! \brief Helper class to convert LoopTree to string */
  class Stringifier;
};

/*! \brief Managed reference to LoopTreeNode */
class LoopTree : public ObjectRef {
 public:
  /*!
   * \brief Constructor of LoopTree
   * \param children Children of the node to be constructed
   * \param iters Iterators of the root of the node
   * \param block_realize The corresponding BlockRealize node in TIR
   */
  LoopTree(Array<ObjectRef> children, Array<Iterator> iters,
           const tir::BlockRealizeNode* block_realize);
  /*!
   * \brief Construct a LoopTree from PrimFunc
   * \param func The PrimFunc
   * \return The loop tree constructed
   */
  static LoopTree FromPrimFunc(const tir::PrimFunc& func);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(LoopTree, ObjectRef, LoopTreeNode);
};

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // SRC_AUTO_SCHEDULER_LOOP_TREE_H_
