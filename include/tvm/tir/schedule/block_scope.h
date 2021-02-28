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
#ifndef TVM_TIR_SCHEDULE_BLOCK_SCOPE_H_
#define TVM_TIR_SCHEDULE_BLOCK_SCOPE_H_

#include <tvm/tir/stmt.h>

#include <unordered_map>

namespace tvm {
namespace tir {

// TODO(@junrushao1994): Change std::unordered_map to Map

/*! \brief The container of stmt schedulable ref. */
class StmtSRefNode : public runtime::Object {
 public:
  /*! \brief The corresponding stmt node */
  const StmtNode* stmt;
  /*! \brief The parent sref */
  StmtSRefNode* parent;
  /*! \brief The location in an array if parent contains SeqStmt. */
  int64_t seq_index;
  /*! \brief Whether the loop bindings are validatable */
  bool binding_valid;

  /*!
   * \brief Get the referenced statement with type checking. It serves the same purpose as
   * ObjectRef::as, but does not require strong reference to `stmt`
   * \tparam StmtType The type that `this->stmt` is assumed to be
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

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "tir.StmtSRef";
  TVM_DECLARE_FINAL_OBJECT_INFO(StmtSRefNode, Object);
};

/*!
 * \brief The stmt schedulable ref.
 */
class StmtSRef : public runtime::ObjectRef {
 public:
  explicit StmtSRef(const StmtNode* stmt, StmtSRefNode* parent, int64_t seq_index,
                    bool binding_valid);
  StmtSRefNode* get() const { return static_cast<StmtSRefNode*>(data_.get()); }
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(StmtSRef, ObjectRef, StmtSRefNode);
};

enum class DepType : int {
  kRAW = 0,
  kWAW = 1,
  kWAR = 2,
  kOpaque = 3,
};

class DepEdgeNode : public runtime::Object {
 public:
  /*! \brief The destination block */
  StmtSRef dst;
  /*! \brief The dependency type */
  DepType type;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dst", &dst);
    v->Visit("type", &type);
  }

  static constexpr const char* _type_key = "tir.DepEdge";
  TVM_DECLARE_FINAL_OBJECT_INFO(DepEdgeNode, Object);
};

class DepEdge : public runtime::ObjectRef {
 public:
  explicit DepEdge(StmtSRef dst, DepType type);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(DepEdge, ObjectRef, DepEdgeNode);
};

/*!
 * \brief Dependency Graph that stores read/write dependency between Blocks
 * \note It is not a traditional and complete dependency graph, but only a
 *       dependency hint. If there is an edge from A to B, iff B writes at
 *       least one of the read tensors of A. That's means B must produce the
 *       necessary element (but not all the element) before A under the Lowest
 *       Common Ancestor (LCA) Loop of the A and B.
 */
class BlockScopeNode : public runtime::Object {
 public:
  /*! \brief The forward dependency edges of the block */
  std::unordered_map<StmtSRef, Array<DepEdge>, ObjectPtrHash, ObjectPtrEqual> forward_edges;
  /*! \brief The backward dependency edges of the block */
  std::unordered_map<StmtSRef, Array<DepEdge>, ObjectPtrHash, ObjectPtrEqual> backward_edges;
  /*! \brief The mapping from the buffer to the blocks who write it */
  std::unordered_map<Buffer, Array<StmtSRef>, ObjectPtrHash, ObjectPtrEqual> buffer_writers;

  void VisitAttrs(AttrVisitor* v) {}

  /*!
   * \brief Add a dependency edge.
   * \param from The departure of the edge
   * \param to The destination of the edge
   */
  void AddEdge(const StmtSRef& from, const StmtSRef& to, DepType type);
  /*!
   * \brief get all blocks that this block dependent on.
   * \param block_sref The query block
   * \return The successor blocks
   */
  Array<DepEdge> GetSuccessors(const StmtSRef& block_sref) const;
  /*!
   * \brief Get all blocks that are dependent on block.
   * \param block_sref The query block
   * \return The predecessors blocks
   */
  Array<DepEdge> GetPredecessors(const StmtSRef& block_sref) const;
  /*!
   * \brief Check whether the block is a dominate block under the scope
   * \note A block is complete iff the block is the only producer
   *       for each tensor it produces.
   * \param block_sref The query block
   * \return Whether is a dominate block
   */
  bool IsDominate(const StmtSRef& block_sref) const;
  /*!
   * \brief Check whether the block is a complete block under the scope
   * \note A block is complete iff the block is the only producer
   *       for each tensor it produces and its args must be data parallel.
   *       Also, the block can not read its output buffer.
   * \param block The query block
   * \return Whether is a complete block
   */
  bool IsComplete(const StmtSRef& block_sref) const;
  /*!
   * \brief Check whether the block is a reduction block under the scope
   * \note A block is reduction iff the block is the only producer
   *       for each tensor it produces, its args must be data parallel/reduce
   * \param block The query block
   * \return Whether is a complete block
   */
  bool IsReduction(const StmtSRef& block_sref) const;
  /*!
   * \brief Check whether a subtree satisfies the one-way fine-grained data flow check
   * \details Suppose a loop tree has several blocks on the leaves.
   *          We can sort them by DFS order as B1, B2, ...., Bn.
   *          The subtree satisfies compact data flow if
   *          - All the blocks are complete/reduction
   *          - Bi doesn't read the buffers that Bi+1, Bi+2, ... Bn will write
   *          - Suppose Bi reads Bj's output buffer(j < i) and Loop k is the LCA of Bi and
   *            Bj, Bj's output region covers Bi's input under Loop k
   * \param subtree_sref The subtree to be checked
   * \param child_blocks The schedule that the scope is in
   * \return A boolean indicating if the subtree satisfies the one-way fine-grained data flow check
   * \note Condition 2 and 3 are global condition of a schedulable IR,
   *       so it is omitted in the check.
   */
  bool IsCompactDataFlow(const StmtSRef& subtree_sref, const Array<StmtSRef>& child_blocks) const;

  /*!
   * \brief Check the merged block of init_block and update_block is a reduction block
   * \param init_sref the query init block
   * \param update_sref the query update block
   * \return Whether the merged block of init_block and update_block is a reduction block
   */
  bool CanMergeReduction(const StmtSRef& init_sref, const StmtSRef& update_sref) const;
  /*!
   * \brief Declare a new child block, update the `buffer_writes`, `buffer_readers` and the
   *        dependency graph
   * \param child_sref The child block to be added
   * \param buffer_readers Maps a buffer to a list of blocks that reads it
   */
  void AddChildBlock(
      const StmtSRef& child_sref,
      std::unordered_map<Buffer, Array<StmtSRef>, ObjectPtrHash, ObjectPtrEqual>* buffer_readers);
  /*!
   * \brief Declare a new child block, update the `buffer_writes`, `buffer_readers` and the
   *        dependency graph
   * \param child_sref The child block to be added
   * \param buffer_readers Maps a buffer to a list of blocks that reads it
   */
  void ReplaceChildBlock(const StmtSRef& old_sref, const StmtSRef& new_sref);

  static constexpr const char* _type_key = "tir.BlockScope";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockScopeNode, Object);
};

class BlockScope : public runtime::ObjectRef {
 public:
  /*! \brief Constructor */
  BlockScope();

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(BlockScope, ObjectRef, BlockScopeNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_BLOCK_SCOPE_H_
