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

#ifndef TVM_TIR_SCHEDULE_H_
#define TVM_TIR_SCHEDULE_H_

#include <tvm/tir/function.h>
#include <tvm/tir/scope.h>

#include <unordered_map>

namespace tvm {
namespace tir {

class Schedule;
class Buffer;

class ScheduleNode : public Object {
 public:
  /*! \brief The function to be scheduled */
  PrimFunc func;
  /*! \brief The root of schedulable reference tree */
  StmtSRef root;
  /*!
   * \brief The mapping from stmt to its schedulable reference node
   * \note This is a hint to improve mutation efficiency
   */
  std::unordered_map<const StmtNode*, StmtSRef> stmt2ref;
  /*! \brief The block scopes of each block */
  std::unordered_map<StmtSRef, Scope, ObjectPtrHash, ObjectPtrEqual> scopes;
  /*! \brief The reducer list for reduction pattern matching */
  Array<CommReducer> reducers;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("func", &func);
    v->Visit("root", &root);
    // `stmt2ref` is not visited
    // `scopes` is not visited
    v->Visit("reducers", &reducers);
  }

  /*!
   * \brief Replace part of AST with new statement
   * \param src_sref The sref of the statement to be replaced
   * \param tgt_stmt The statement to be replaced to
   * \param block_reuse Maps an new block (replaced to) back to an old block (to be replaced),
   * and enforces reuse of srefs between them (rather than create new srefs)
   * i.e. after being replaced, the sref that points to the old block will point to the new one
   * \note `loop_reuse` will be automatically detected via loop vars
   */
  void Replace(const StmtSRef& src_sref, const Stmt& tgt_stmt,
               const Map<Block, Block>& block_reuse);

  /*!
   * \brief Get block from its tag
   * \param tag The query tag
   * \return the block schedulable reference list
   */
  Array<StmtSRef> GetBlock(const String& tag) const;

  /*!
   * \brief Get block from its output tensor
   * \param buffer The query buffer
   * \param scope The scope of interest
   * \return the block schedulable reference list
   */
  Array<StmtSRef> GetBlock(const Buffer& buffer, StmtSRef scope) const;

  /*!
   * \brief Get all blocks in the scope
   * \param scope The block scope
   * \return the block schedulable reference list
   */
  Array<StmtSRef> Blocks(StmtSRef scope) const;

  /*!
   * \brief Get direct child blocks of the given block
   * \param parent_sref The block scope
   * \return An array of StmtSRef that are children of the parent
   */
  Array<StmtSRef> GetChildBlocks(const StmtSRef& parent_sref) const;

  /*!
   * \brief Get loops of the block
   * \param block The query block
   * \return the loop sref list
   */
  Array<StmtSRef> GetAxes(const StmtSRef& block) const;

  /*!
   * \brief Get the parent block sref of the given sref
   * \param sref The queried node
   * \return The sref to the parent block
   */
  StmtSRef GetParentBlockSRef(const StmtSRef& sref) const;

  /*!
   * \brief Get the scope of the parent block
   * \param sref The queried node
   * \return The scope of the parent block
   */
  Scope GetParentScope(const StmtSRef& sref) const;

  /*!
   * \brief Fuse two consecutive loops of one computation.
   * \param outer_sref The outer loop
   * \param inner_sref The inner loop
   * \return The fused loop
   */
  StmtSRef fuse(const StmtSRef& outer_sref, const StmtSRef& inner_sref);

  /*!
   * \brief Split a specified loop into two loops by factor.
   * \param loop_sref The loop to be split
   * \param nparts The extent of the new outer loop
   * \param factor The extent of the new inner loop
   * \return The loops after splitting
   */
  Array<StmtSRef> split(const StmtSRef& loop_sref, const PrimExpr& nparts, const PrimExpr& factor);

  /*!
   * \brief Move the block under the loop and regenerate the loops to cover the producing region.
   * \param block_sref The block to be moved
   * \param loop_sref The target loop
   * \param preserve_trivial_loop Keep the trivial loops whose extent is 1
   */
  void compute_at(const StmtSRef& block_sref, const StmtSRef& loop_sref,
                  bool preserve_trivial_loop = false);

  /*!
   * \brief Move the block under the loop and regenerate the loops to cover the producing region.
   * \param block_sref The block to be moved
   * \param loop_sref The target loop
   * \param preserve_trivial_loop Keep the trivial loops whose extent is 1
   */
  void reverse_compute_at(const StmtSRef& block_sref, const StmtSRef& loop_sref,
                          bool preserve_trivial_loop = false);

  /*!
   * \brief Make the block inline
   * \param block_sref The sref of the block
   */
  void compute_inline(const StmtSRef& block_sref);

  /*!
   * \brief Make the block inline
   * \param block_sref The sref of block
   */
  void reverse_compute_inline(const StmtSRef& block_sref);

  /*!
   * \brief vectorize a loop
   * \param loop_sref the loop to be vectorized
   */
  void vectorize(const StmtSRef& loop_sref);

  /*!
   * \brief parallelize a loop
   * \param loop_sref the loop to be paralleled
   */
  void parallel(const StmtSRef& loop_sref);

  /*!
   * \brief parallel a loop
   * \param loop_sref the loop to be paralleled
   */
  void bind(const StmtSRef& loop_sref, const IterVar& thread);

  /*!
   * \brief unroll a loop
   * \param loop_sref the loop to be unrolled
   */
  void unroll(const StmtSRef& loop_sref);

  /*!
   * \brief reorder a list of loops
   * \param order the order of loops
   */
  void reorder(const Array<StmtSRef>& order);

  /*!
   * \brief Decompose reduction block_sref into init&update blocks
   * \param block_sref the reduction block_sref
   * \param loop_sref the position where init block_sref will be
   * \return the sref of init block
   */
  StmtSRef decompose_reduction(const StmtSRef& block_sref, const Optional<StmtSRef>& loop_sref);

  /*!
   * \brief Merge init and reduction block into reduction block
   * \param init_sref the init block
   * \param update_sref the update block
   */
  void merge_reduction(const StmtSRef& init_sref, const StmtSRef& update_sref);

  /*!
   * \brief Create a cache read of original tensor for readers.
   * \param block_sref The consumer of the buffer
   * \param i The index of the buffer in block's read region
   * \param storage_scope The storage scope
   */
  StmtSRef cache_read(StmtSRef block_sref, int i, const String& storage_scope);

  /*!
   * \brief Create a cache write of original tensor, before storing into tensor.
   * \param block_sref The producer of the buffer
   * \param i The index of the buffer in block's write region
   * \param storage_scope The storage scope
   */
  StmtSRef cache_write(StmtSRef block_sref, int i, const String& storage_scope);

  /*!
   * \brief make subtree rooted by loop_sref into a block
   * \param loop_sref the subtree root
   * \return the loop_sref of new block
   */
  StmtSRef blockize(const StmtSRef& loop_sref, const String& exec_scope);

  /*!
   * \brief Tensorize the computation enclosed by loop with tensor_intrin
   * \param loop_sref the loop/block to be tensorized
   * \param intrinsic the tensor intrinsic
   */
  void tensorize(const StmtSRef& loop_sref, const TensorIntrin& intrinsic);

  /*!
   * \brief Register a reducer pattern
   * \param comm_reducer the reducer pattern to be registered
   */
  void register_reducer(const CommReducer& comm_reducer);

  /*!
   * \brief rfactor a reduction block using loop
   * \param loop_sref the loop outside block we want to do rfactor
   * \param factor_axis the position where the new axis is placed
   * \return the sref of new block
   */
  StmtSRef rfactor(const StmtSRef& loop_sref, int factor_axis);

  /*!
   * \brief add annotation to a loop
   * \param loop_sref the loop of interest
   * \param pragma_type the attribute key
   * \param pragma_value the attribute value
   */
  void pragma(const StmtSRef& loop_sref, const String& pragma_type, const PrimExpr& pragma_value);

  /*!
   * \brief add double_buffer annotation to a complete block
   * \param block_sref the block of interest
   */
  void double_buffer(const StmtSRef& block_sref);

  /*!
   * \brief validate sref tree and scope information
   */
  bool ValidateSRef() const;

  /*!
   * \brief Check the region cover for the single consumer block
   */
  static void ValidateHierarchy(const PrimFunc& f);

  static constexpr const char* _type_key = "tir.Schedule";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleNode, Object);

 private:
  /*!
   * \brief Help function for checking and mutating loops to do parallel computation
   *        For now it is only used for vectorize, bind and parallel
   * \param loop_sref the loop to be annotated
   * \param annotation the annotation
   */
  void ParallelCompute(const StmtSRef& loop_sref, const Annotation& annotation);

  /*!
   * \brief Validate Tir, now the ValidateLoops pass contains the following checks
   *        1) loop binding validation: a set of binding expressions is valid if and only if
   *          1.  vi=i, vj=j, vk=k ... (one loop_var binds exactly one block_var)
   *          2.  if f is a legal binding and g is the binding after we applying `split` on f,
   *          then g is legal
   *          3.  if f is a legal binding and g is the binding after we applying `fuse` on f,
   *          then g is legal
   *        2) region cover check: Suppose B is a RAW predecessor of C, Loop k is the LCA of B and
   *          C, then B's output region covers C's input region under Loop k
   */
  void ValidateLoops();

  /*!
   * \brief Check the region cover for the single consumer block
   */
  bool ValidateRegionCover(const StmtSRef& consumer) const;

  friend class Schedule;
  friend class ScheduleHelper;
};

class Schedule : public ObjectRef {
 public:
  /*!
   * \brief Construct a schedule from a PrimFunc
   * \param func The PrimFunc to be created
   */
  explicit Schedule(PrimFunc func);

  /*!
   * \brief Constructor
   * \param func The function to be scheduled
   * \param root brief The root of schedulable reference tree
   * \param stmt2ref The block scopes of each block
   * \param scopes The mapping from stmt to its schedulable reference node
   */
  explicit Schedule(PrimFunc func, StmtSRef root,
                    std::unordered_map<const StmtNode*, StmtSRef> stmt2ref,
                    std::unordered_map<StmtSRef, Scope, ObjectPtrHash, ObjectPtrEqual> scopes);

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Schedule, ObjectRef, ScheduleNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_H_
