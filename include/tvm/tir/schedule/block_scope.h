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
 * \file tvm/tir/schedule/block_scope.h
 * \brief Definition of two pillar data structure for TensorIR scheduling: StmtSRef, BlockScope.
 * \sa StmtSRefNode
 * \sa BlockScopeNode
 */
#ifndef TVM_TIR_SCHEDULE_BLOCK_SCOPE_H_
#define TVM_TIR_SCHEDULE_BLOCK_SCOPE_H_

#include <tvm/tir/stmt.h>

#include <unordered_map>

namespace tvm {
namespace tir {

/*!
 * \brief An object that refers to schedulable elements (block/for-loop) in TensorIR, aka "sref".
 *
 * Glossary
 * - Block sref: An StmtSref that points to a TensorIR block.
 * - Loop sref: An StmtSRef that points to a TensorIR for loop.
 * - Parent sref: The parent sref of an sref is the block/loop sref that points to its closest
 * schedulable statement.
 * - Root sref: Sref to the root block. Every sref has exactly one parent sref except for root sref.
 * - Sref tree: The parent-children-relationship of srefs form a tree, uniquely determined by the
 * TensorIR AST.
 */
class StmtSRefNode : public Object {
 public:
  /*!
   * \brief The block/for stmt the object refers to
   * \note Non-owned reference (raw pointer) is used here, so that we can perform copy-on-write
   * optimization on statements when possible. The strong reference is held in the ScheduleState
   */
  const StmtNode* stmt;
  /*! \brief The parent sref. */
  StmtSRefNode* parent;
  /*!
   * \brief The location in an array if the parent of the stmt contains multiple children.
   * -1 if the parent does not contain multiple children.
   */
  int64_t seq_index;
  /*! \brief If true, the block bindings are quasi-affine maps. */
  bool affine_block_binding;

  void VisitAttrs(AttrVisitor* v) {
    // `stmt` is not visited
    // `parent` is not visited
    v->Visit("seq_index", &seq_index);
    v->Visit("affine_block_binding", &affine_block_binding);
  }

  static constexpr const char* _type_key = "tir.StmtSRef";
  TVM_DECLARE_FINAL_OBJECT_INFO(StmtSRefNode, Object);

  /*!
   * \brief Reset the object inplace
   * \param stmt The new `StmtSRefNode::stmt`
   * \param parent The new `StmtSRefNode::parent`
   * \param seq_index The new `StmtSRefNode::seq_index`
   */
  void Reset(const StmtNode* new_stmt, StmtSRefNode* new_parent, int64_t new_seq_index) {
    this->stmt = new_stmt;
    this->parent = new_parent;
    this->seq_index = new_seq_index;
  }

  /*!
   * \brief Get the referenced statement with proper type checking.
   * It serves the same purpose as `ObjectRef::as`, but does not acquire strong reference to `stmt`
   * \tparam StmtType The type that `this->stmt` to be downcasted to. Preassumably
   * tvm::tir::BlockNode or tvm::tir::ForNode
   * \return nullptr if type check fails, otherwise the casted result for `this->stmt`
   */
  template <typename StmtType>
  const StmtType* GetStmt() const {  // TODO
    if (stmt != nullptr && stmt->IsInstance<StmtType>()) {
      return static_cast<const StmtType*>(stmt);
    } else {
      return nullptr;
    }
  }
};

/*!
 * \brief Managed reference to StmtSRefNode
 * \sa StmtSRefNode
 */
class StmtSRef : public ObjectRef {
 public:
  /*!
   * \brief The constructor
   * \param stmt The corresponding stmt node, can be either block or for loop.
   * \param parent The parent sref.
   * \param seq_index The location in an array if the parent of the stmt contains multiple children.
   * -1 if the parent does not contain multiple children.
   * \param affine_block_binding If true, the block bindings are quasi-affine maps.
   */
  TVM_DLL explicit StmtSRef(const StmtNode* stmt, StmtSRefNode* parent, int64_t seq_index,
                            bool affine_block_binding);
  /*! \return The mutable pointer to the StmtSRefNode */
  StmtSRefNode* get() const { return static_cast<StmtSRefNode*>(data_.get()); }

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(StmtSRef, ObjectRef, StmtSRefNode);

 public:
  /*!
   * \return A special StmtSRef, which doesn't point to any stmt in the AST,
   * only serving as a "mark" to hint compute-at to do the work of compute-inline
   */
  TVM_DLL static StmtSRef InlineMark();
  /*!
   * \return A special StmtSRef, which doesn't point to any stmt in the AST,
   * only serving as a "mark" to hint compute-at to do nothing
   */
  TVM_DLL static StmtSRef RootMark();
};

/*!
 * \brief Type of dependency. Right now we have 4 types of dependencies
 * 1) Read-after-write (kRAW)
 * 2) Write-after-write (kWAW)
 * 3) Write-after-read (kWAR)
 * 4) Opaque dependency (kOpaque)
 */
enum class DepKind : int32_t {
  kRAW = 0,
  kWAW = 1,
  kWAR = 2,
  kOpaque = 3,
};

/*! \brief An edge representing certain types of dependency, e.g. read-after-write */
class DependencyNode : public Object {
 public:
  /*! \brief The source of the dependency relation */
  StmtSRef src;
  /*! \brief The destination of the dependency relation */
  StmtSRef dst;
  /*! \brief The dependency kind */
  DepKind kind;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("src", &src);
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
class Dependency : public ObjectRef {
 public:
  /*! \brief Constructor */
  TVM_DLL explicit Dependency(StmtSRef src, StmtSRef dst, DepKind kind);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Dependency, ObjectRef, DependencyNode);
};

/*!
 * \brief An object corresponds to each block sref in the sref tree,
 * which tracks the producer-consumer dependency between blocks.
 *
 * Glossary:
 * - Block scope: A contiguous subtree of the sref tree, rooted at each block sref, where:
 *   - scope root: a block sref
 *   - internal srefs: loop srefs
 *   - scope leaves: block srefs
 * - Child block: The scope leaf blocks under the scope root or a specific internal sref
 */
class BlockScopeNode : public Object {
 public:
  /*! \brief Lookup table for the `src` of dependencies */
  std::unordered_map<StmtSRef, Array<Dependency>, ObjectPtrHash, ObjectPtrEqual> src2deps;
  /*! \brief Lookup table for the `dst` of dependencies */
  std::unordered_map<StmtSRef, Array<Dependency>, ObjectPtrHash, ObjectPtrEqual> dst2deps;
  /*! \brief The mapping from the buffer to the blocks who write it */
  std::unordered_map<Buffer, Array<StmtSRef>, ObjectPtrHash, ObjectPtrEqual> buffer_writers;

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "tir.BlockScope";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockScopeNode, Object);

 public:
  /******** Dependency ********/
  /*!
   * \brief Get all dependencies whose `src` equals `src`
   * \param src The queried block
   * \return The dependencies
   */
  TVM_DLL Array<Dependency> GetDepsBySrc(const StmtSRef& src) const;
  /*!
   * \brief Get all dependencies whose `dst` equals `dst`
   * \param dst The queried block
   * \return The dependencies
   */
  TVM_DLL Array<Dependency> GetDepsByDst(const StmtSRef& dst) const;

  /******** Property of a block ********/
  /*!
   * \brief Check whether the block is a complete block under the scope
   * \param block_sref The block to be checked
   * \return A boolean indicating if the block is a complete block
   * \note Definition of a complete block:
   * 1) dominant: the block is the only writer of its output, which dominates the reader of
   * its output buffers
   * 2) all block vars are data parallel
   * 3) no overlap between the buffers it reads and writes
   */
  TVM_DLL bool IsComplete(const StmtSRef& block_sref) const;
  /*!
   * \brief Check whether the block is a reduction block under the scope
   * \param block_sref The block to be checked
   * \return A boolean indicating if the block is a reduction block
   * \note Definition of a reduction block:
   * 1) dominant: the block is the only writer of its output, which dominates the reader of
   * its output buffers
   * 2) all block vars are data parallel or reduction
   * 3) block bindings are quasi-affine expressions
   * 4) has the init statement
   * 5) reduction block vars are not used to index output buffers
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
class BlockScope : public ObjectRef {
 public:
  /*! \brief The constructor creating an empty block scope. */
  TVM_DLL BlockScope();
  /*!
   * \brief Create the block scope given the leaf blocks
   * \param child_block_srefs The srefs to the leaf blocks, from left to right
   */
  TVM_DLL BlockScope(const Array<StmtSRef>& child_block_srefs);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BlockScope, ObjectRef, BlockScopeNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_BLOCK_SCOPE_H_
