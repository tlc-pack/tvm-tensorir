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
#include <tvm/ir/attrs.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/function.h>
#include <tvm/tir/scope.h>
#include <tvm/tir/stmt_sref.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tir {

class Schedule;
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

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("func", &func);
    v->Visit("root", &root);
  }

  /*!
   * \brief Create a new schedule
   * \param function The function to be scheduled
   * \return The schedule
   */
  static Schedule Create(PrimFunc function);

  /*!
   * \brief replace part of AST with new stmt
   * \param ref The schedulable reference of the old stmt
   * \param target The new stmt
   * \param block_sref_map The Sref remapping of blocks
   */
  void Replace(StmtSRef ref, Stmt target,
               Map<Block, Block> block_sref_map = NullValue<Map<Block, Block> >());

  /*!
   * \brief Get block from its tag
   * \param scope The block scope
   * \param tag The query tag
   * \return the block schedulable reference list
   */
  Array<StmtSRef> GetBlock(const std::string& tag, StmtSRef scope = StmtSRef()) const;

  /*!
   * \brief Get block from its output tensor
   * \param scope The block scope
   * \param buffer The query buffer
   * \return the block schedulable reference list
   */
  Array<StmtSRef> GetBlock(const Buffer& buffer, StmtSRef scope = StmtSRef()) const;

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
  Array<StmtSRef> GetLoopsInScope(const StmtSRef& block) const;

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
   */
  void compute_at(const StmtSRef& block_sref, const StmtSRef& loop_sref);

  /*!
   * \brief Move the block under the loop and regenerate the loops to cover the producing region.
   * \param block_sref The block to be moved
   * \param loop_sref The target loop
   */
  void reverse_compute_at(const StmtSRef& block_sref, const StmtSRef& loop_sref);

  /*!
   * \brief Make the block inline
   * \param block_sref The sref of the block
   */
  void compute_inline(const StmtSRef& block_sref);

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
  StmtSRef decompose_reduction(const StmtSRef& block_sref, const StmtSRef& loop_sref);

  /*!
   * \brief Merge init and reduction block into reduction block
   * \param init_sref the init block
   * \param update_sref the update block
   */
  void merge_reduction(const StmtSRef& init_sref, const StmtSRef& update_sref);

  /*!
   * \brief Create a cache read of original tensor for readers.
   * \param buffer The buffer
   * \param storage_scope The storage scope
   */
  StmtSRef cache_read(const Buffer& buffer, const std::string& storage_scope);

  /*!
   * \brief Create a cache write of original tensor, before storing into tensor.
   * \param buffer The buffer
   * \param storage_scope The storage scope
   */
  StmtSRef cache_write(const Buffer& buffer, const std::string& storage_scope);

  /*!
   * \brief make subtree rooted by sref into a block
   * \param sref the subtree root
   * \return the sref of new block
   */
  StmtSRef blockize(const StmtSRef& sref, const String& exe_scope = "");

  /*!
   * \brief Tensorize the computation enclosed by loop with tensor_intrin
   * \param sref the loop/block to be tensorized
   * \param intrinsic the tensor intrinsic
   */
  void tensorize(const StmtSRef& sref, const TensorIntrin& intrinsic);

  /*!
   * \brief Register a reducer pattern
   * \param comm_reducer the reducer pattern to be registered
   */
  void register_reducer(const CommReducer& comm_reducer);

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
   * \brief Update the sref to make it point to new Block/Loop
   * \param sref The outdated sref
   * \param stmt The new stmt
   */
  void UpdateSRef(StmtSRefNode* sref, const Stmt& stmt);

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

  /*! \brief The reducer list for reduction pattern matching */
  std::vector<CommReducer> reducers_;
};

class Schedule : public ObjectRef {
 public:
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
