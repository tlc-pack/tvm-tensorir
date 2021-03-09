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
 * \brief This file defines two pillar data structure for TensorIR scheduling
 *
 * 1) StmtSRef, aka sref: An object that references to schedulable statements in the TensorIR.
 * The schedulable elements of the TensorIR are blocks and for loops, so there are two kinds of
 * srefs, block sref and loop sref.
 * An sref also records the sref to the closest schedulable statement in its ancestors as its
 * parent. The parent-relationship of StmtSRefs form a tree, named sref tree.
 *
 * 2) BlockScope. In the sref tree, each block sref has its correpsonding block scope.
 * A scope is a contiguous subtree of the sref tree, whose root and leaves are block srefs, and
 * the internal nodes are loop srefs. Those leaf blocks are called child blocks of the root.
 * A BlockScope object records the producer-consuer relationships between the child blocks inside
 * the scope, and therefore could provide property checks for a leaf block, e.g. whether a block is
 * dominant, complete, reduction, etc.
 *
 * \sa StmtSRefNode
 * \sa StmtSRef
 * \sa BlockScopeNode
 * \sa BlockScope
 */
#ifndef TVM_TIR_SCHEDULE_BLOCK_SCOPE_H_
#define TVM_TIR_SCHEDULE_BLOCK_SCOPE_H_

#include <tvm/tir/stmt.h>

#include <unordered_map>

namespace tvm {
namespace tir {

/*! \brief An object that references to schedulable statements in the TensorIR */
class StmtSRefNode : public runtime::Object {
 public:
  /*! \brief The corresponding stmt node, can be either block or for loop. */
  const StmtNode* stmt;
  /*! \brief The parent sref. */
  StmtSRefNode* parent;
  /*! \brief The location in an array if the parent of the stmt contains multiple children. */
  int64_t seq_index;
  /*! \brief If true, the block bindings are semi-affine maps. */
  bool binding_valid;

  void VisitAttrs(AttrVisitor* v) {}

  /*!
   * \brief Get the referenced statement with proper type checking.
   * It serves the same purpose as `ObjectRef::as`, but does not acquire strong reference to `stmt`
   * \tparam StmtType The type that `this->stmt` to be downcasted to
   * \return nullptr if type check fails, otherwise the type casted from `this->stmt`
   */
  template <typename StmtType>
  const StmtType* GetStmt() const {
    if (stmt != nullptr && stmt->IsInstance<StmtType>()) {
      return static_cast<const StmtType*>(stmt);
    } else {
      return nullptr;
    }
  }

  static constexpr const char* _type_key = "tir.StmtSRef";
  TVM_DECLARE_FINAL_OBJECT_INFO(StmtSRefNode, Object);
};

/*!
 * \brief Managed reference to StmtSRefNode
 * \sa StmtSRefNode
 */
class StmtSRef : public runtime::ObjectRef {
 public:
  /*!
   * \brief The constructor
   * \param stmt The corresponding stmt node, can be either block or for loop.
   * \param parent The parent sref.
   * \param seq_index The location in an array if the parent of the stmt contains multiple children.
   * \param binding_valid If true, the block bindings are semi-affine maps.
   */
  explicit StmtSRef(const StmtNode* stmt, StmtSRefNode* parent, int64_t seq_index,
                    bool binding_valid);
  /*! \return The mutable pointer to the StmtSRefNode */
  StmtSRefNode* get() const { return static_cast<StmtSRefNode*>(data_.get()); }
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(StmtSRef, ObjectRef, StmtSRefNode);

 public:
  /*!
   * \return A special StmtSRef, which doesn't point to any stmt in the AST,
   * only serving as a "mark" to hint compute-at to do the work of compute-inline
   */
  static StmtSRef InlineMark();
  /*!
   * \return A special StmtSRef, which doesn't point to any stmt in the AST,
   * only serving as a "mark" to hint compute-at to do nothing
   */
  static StmtSRef RootMark();
};

/*! \brief Type of dependency */
enum class DepKind : int32_t {
  kRAW = 0,
  kWAW = 1,
  kWAR = 2,
  kOpaque = 3,
};

/*! \brief An edge representing certain types of dependency, e.g. read-after-write */
class DependencyNode : public runtime::Object {
 public:
  /*! \brief The destination block */
  StmtSRef dst;
  /*! \brief The dependency kind */
  DepKind kind;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dst", &dst);
    v->Visit("kind", &kind);
  }

  static constexpr const char* _type_key = "tir.Dependency";
  TVM_DECLARE_FINAL_OBJECT_INFO(DependencyNode, Object);
};

/*!
 * \brief Managed reference to DependencyNode
 * \sa DependencyNode
 */
class Dependency : public runtime::ObjectRef {
 public:
  /*! \brief Constructor */
  explicit Dependency(StmtSRef dst, DepKind type);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Dependency, ObjectRef, DependencyNode);
};

/*! \brief An object recording the producer-consumer dependency between child blocks of a scope */
class BlockScopeNode : public runtime::Object {
 public:
  /*! \brief The forward dependency edges of the block */
  std::unordered_map<StmtSRef, Array<Dependency>, ObjectPtrHash, ObjectPtrEqual> src2deps;
  /*! \brief The backward dependency edges of the block */
  std::unordered_map<StmtSRef, Array<Dependency>, ObjectPtrHash, ObjectPtrEqual> dst2deps;
  /*! \brief The mapping from the buffer to the blocks who write it */
  std::unordered_map<Buffer, Array<StmtSRef>, ObjectPtrHash, ObjectPtrEqual> buffer_writers;

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "tir.BlockScope";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockScopeNode, runtime::Object);

 public:
  /******** DependencyNode ********/
  /*!
   * \brief Get all blocks the block depends on
   * \param block_sref The queried block
   * \return The predecessors edges
   */
  TVM_DLL Array<Dependency> GetDepsBySrc(const StmtSRef& block_sref) const;
  /*!
   * \brief Get all blocks that depends on the block
   * \param block_sref The queried block
   * \return The successor edges
   */
  TVM_DLL Array<Dependency> GetDepsByDst(const StmtSRef& block_sref) const;

  /******** Property of a block ********/
  /*!
   * \brief Check whether the block is a dominate block under the scope
   * \note A block is complete iff the block is the only producer
   * for each tensor it produces.
   * \param block_sref The query block
   * \return Whether is a dominate block
   */
  TVM_DLL bool IsDominate(const StmtSRef& block_sref) const;
  /*!
   * \brief Check whether the block is a complete block under the scope
   * \note A block is complete iff the block is the only producer
   * for each tensor it produces and its args must be data parallel.
   * Also, the block can not read its output buffer.
   * \param block The query block
   * \return Whether is a complete block
   */
  TVM_DLL bool IsComplete(const StmtSRef& block_sref) const;
  /*!
   * \brief Check whether the block is a reduction block under the scope
   * \note A block is reduction iff the block is the only producer
   * for each tensor it produces, its args must be data parallel/reduce
   * \param block The query block
   * \return Whether is a complete block
   */
  TVM_DLL bool IsReduction(const StmtSRef& block_sref) const;

  /******** Inter-block properties ********/
  /*!
   * \brief Check whether a subtree satisfies the one-way fine-grained data flow check
   * \details Suppose a loop tree has several blocks on the leaves.
   * We can sort them by DFS order as B1, B2, ...., Bn.
   * The subtree satisfies compact data flow if
   * - All the blocks are complete/reduction
   * - Bi doesn't read the buffers that Bi+1, Bi+2, ... Bn will write
   * - Suppose Bi reads Bj's output buffer(j < i) and Loop k is the LCA of Bi and
   * Bj, Bj's output region covers Bi's input under Loop k
   * \param subtree_sref The subtree to be checked
   * \param child_blocks The schedule that the scope is in
   * \return A boolean indicating if the subtree satisfies the one-way fine-grained data flow check
   * \note Condition 2 and 3 are global condition of a schedulable IR,
   * so it is omitted in the check.
   */
  TVM_DLL bool IsCompactDataFlow(const StmtSRef& subtree_sref,
                                 const Array<StmtSRef>& child_blocks) const;
  /*!
   * \brief Check the merged block of init_block and update_block is a reduction block
   * \param init_sref the query init block
   * \param update_sref the query update block
   * \return Whether the merged block of init_block and update_block is a reduction block
   */
  TVM_DLL bool CanMergeReduction(const StmtSRef& init_sref, const StmtSRef& update_sref) const;
};

/*!
 * \brief Managed reference to BlockScopeNode
 * \sa BlockScopeNode
 */
class BlockScope : public runtime::ObjectRef {
 public:
  /*! \brief The constructor creating an empty block scope. */
  TVM_DLL BlockScope();
  /*!
   * \brief Create the block scope given the leaf blocks
   * \param leaf_block_srefs The srefs to the leaf blocks, from left to right
   */
  TVM_DLL BlockScope(const Array<StmtSRef>& leaf_block_srefs);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BlockScope, ObjectRef, BlockScopeNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_BLOCK_SCOPE_H_
