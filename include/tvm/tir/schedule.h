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
#include <tvm/tir/ir.h>
#include <tvm/ir/attrs.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/stmt_sref.h>
#include <tvm/tir/scope.h>
#include <utility>
#include <vector>
#include <string>
#include <unordered_map>

namespace tvm {
namespace tir {

class Schedule;
class ScheduleNode : public Object {
 public:
  /*! \brief The function to be scheduled */
  Function func;
  /*! \brief The root of schedulable reference tree */
  StmtSRef root;
  /*!
   * \brief The mapping from stmt to its schedulable reference node
   * \note This is a hint to improve mutation efficiency
   * */
  std::unordered_map<const StmtNode*, StmtSRef> stmt2ref;
  /*! \brief The block scopes of each block */
  std::unordered_map<StmtSRef, Scope, ObjectHash, ObjectEqual> scopes_;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("func", &func);
    v->Visit("root", &root);
  }

  /*!
   * \brief Create a new schedule
   * \param function The function to be scheduled
   * \return The schedule
   */
  static Schedule Create(Function function);

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
   * \brief Get loops of the block
   * \param block The query block
   * \return the loop sref list
   */
  Array<StmtSRef> GetLoopsInScope(const StmtSRef& block) const;

  /*!
   * \brief Get the scope of the schedulable reference
   * \param node The queried node
   * \return the block scope reference
   */
  StmtSRef GetScope(StmtSRef node) const;

  /*!
   * \brief fuse two consecutive loops of one computation.
   * \param outer The outer loop
   * \param inner The inner loop
   * \return the fused loop
   */
  StmtSRef fuse(const StmtSRef& outer, const StmtSRef& inner);

  /*!
   * \brief split a specified loop into two loops by factor.
   * \param node The loop to be split
   * \param factor The split factor
   * \return the loops after splitting
   */
  Array<StmtSRef> split(const StmtSRef& node, const PrimExpr& nparts, const PrimExpr& factor);

  /*!
   * \brief Move the block under the loop and regenerate the
   *        loops to cover the producing region.
   * \param block_sref The block to be moved
   * \param loop_sref The target loop
   * \return the regenerated loops
   * */
  void compute_at(const StmtSRef& block_sref, const StmtSRef& loop_sref);

  /*!
   * \brief vectorize a loop
   * \param node the loop to be vectorized
   */
  void vectorize(const StmtSRef& node);

  /*!
   * \brief unroll a loop
   * \param node the loop to be unrolled
   */
  void unroll(const StmtSRef& node);

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
   * \brief Register a reducer pattern
   * \param comm_reducer the reducer pattern to be registered
   */
  void register_reducer(const CommReducer& comm_reducer);

  /*!
   * \brief validate sref tree and scope information
   */
  bool ValidateSRef() const;

  static constexpr const char* _type_key = "tir.Schedule";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleNode, Object);

 private:
  /*! \brief The reducer list for reduction pattern matching */
  std::vector<CommReducer> reducers_;

  /*!
 * \brief Update the sref to make it point to new Block/Loop
 * \param sref The outdated sref
 * \param stmt The new stmt
 */
  void UpdateSRef(StmtSRefNode* sref, const Stmt& stmt);
  /*!
   * \brief Check the region cover for the single consumer block
   */
  bool CheckRegionCover(const StmtSRef& consumer) const;
  /*!
   * \brief Check whether a sub_tree satisfies the one-way fine-grained data flow check
   * \details Suppose a loop tree has several blocks on the leaves.
   *          We can sort them by DFS order as B1, B2, ...., Bn.
   *          The subtree satisfies compact data flow if
   *          - All the blocks are complete
   *          - Bi doesn't read the buffers that Bi+1, Bi+2, ... Bn will write
   *          - Suppose Bi reads Bj's output buffer(j < i) and Loop k is the LCA of Bi and
   *            Bj, Bj's output region covers Bi's input under Loop k
   * \note Condition 2 and 3 are global condition of a schedulable IR,
   *       so it is omitted in the check.
   */
  bool IsCompactDataFlow(const StmtSRef& sub_tree) const;
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
   * \param func the TirFunction to be validated
   */
  void ValidateLoops(Function function);
};

class Schedule : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Schedule, ObjectRef, ScheduleNode);

  ScheduleNode* operator->() {
    return static_cast<ScheduleNode*>(ObjectRef::get_mutable());
  }
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_H_
