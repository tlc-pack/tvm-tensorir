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
 * \file tvm/tir/schedule/state.h
 * \brief This file defines ScheduleState, the core data structure of TensorIR scheduling.
 */
#ifndef TVM_TIR_SCHEDULE_STATE_H_
#define TVM_TIR_SCHEDULE_STATE_H_

#include <tvm/ir/module.h>
#include <tvm/tir/schedule/block_scope.h>

#include <unordered_map>
#include <utility>

namespace tvm {
namespace tir {

class PrimFunc;

struct BlockInfo {
  /*! \brief Property of a block scope rooted at the block, storing dependencies in the scope */
  BlockScope scope{nullptr};
  /*! \brief Property of a block, indicating if the block binding is quasi-affine */
  bool affine_binding{false};
  /*!
   * \brief Property of a block, indicating if each of the block's read regions is
   * produced by its producers
   */
  bool region_cover{false};
  /*!
   * \brief Property of a block scope root at the block, indicaiting if the scope is an equivalence
   * of a stage pipeline. Conditions:
   * 1) The region cover property holds for every of it child blocks
   * 2) No write-after-read dependency
   */
  bool stage_pipeline{false};

  BlockInfo() = default;

  explicit BlockInfo(BlockScope scope, bool affine_binding, bool region_cover, bool stage_pipeline)
      : scope(std::move(scope)),         //
        affine_binding(affine_binding),  //
        region_cover(region_cover),      //
        stage_pipeline(stage_pipeline) {}
};

/*!
 * \brief The state of scheduling, which exposes a `Replace` method as
 * the primary resort for all the scheduling primitives to manipulate the TensorIR.
 *
 * The data structure contains the following
 * 1) The AST being scheduled (mod)
 * 2) The sref tree of schedulable statements
 * 3) A reverse mapping from the AST nodes to that in the sref tree (stmt2ref)
 * 4) The dependency information of each block scope (block_info)
 * 5) A debug flag, if set, extra checking is enabled (debug_mode)
 */
class ScheduleStateNode : public Object {
 public:
  /*! \brief The AST of the module being scheduled */
  IRModule mod;
  /*!
   * \brief Mapping from a block sref to the block scope at it,
   * tracking the dependency inside the block scope
   */
  std::unordered_map<StmtSRef, BlockInfo, ObjectPtrHash, ObjectPtrEqual> block_info;
  /*! \brief The reverse mapping from block/for-loop to their corresponding srefs */
  std::unordered_map<const StmtNode*, StmtSRef> stmt2ref;
  /*!
   * \brief In debug mode, we do extra correctness checking after each replacement.
   * 1) If `debug_mode & 1`: verify the sref tree
   * 2) If `debug_mode & 2`: verify the affine_binding in block info
   */
  int debug_mode;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("mod", &mod);
    // `block_info` is not visited
    // `stmt2ref` is not visited
    v->Visit("debug_mode", &debug_mode);
  }
  /*!
   * \brief Replace the part of the AST, as being pointed to by `src_sref`,
   * with a specific statement `tgt_stmt`, and maintain the sref tree accordingly.
   * Replace will try to perform copy on write as much as possible when the ScheduleState holds
   * the only copy to the IRModule and IR nodes.
   *
   * Only 3 types of replacements are allowed: from `src_sref->stmt` to `tgt_stmt`.
   * 1) Block -> Block
   * 2) Loop -> Loop
   * 3) Loop -> BlockRealize
   *
   * \param src_sref The sref to the statement to be replaced
   * \param tgt_stmt The statement to be replaced to
   * \param block_sref_reuse Maps an old block (to be replaced in the subtree under
   * `src_sref->stmt`) to a new block (replaced to, in the subtree under `tgt_stmt`), and enforces
   * reuse of srefs between them (rather than create new srefs) i.e. after being replaced, the sref
   * that points to the old block will point to the new one
   * \note The reuse of loop srefs are detected automatically according to the reuse of loop vars.
   */
  TVM_DLL void Replace(const tir::StmtSRef& src_sref, const Stmt& tgt_stmt,
                       const Map<Block, Block>& block_sref_reuse);

  /*! \brief Trigger the verification depending on `debug_mode` */
  TVM_DLL void DebugVerify() const;

  static constexpr const char* _type_key = "tir.ScheduleState";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleStateNode, Object);

  /******** Property of blocks ********/
  /*! \brief Returns the BlockInfo correpsonding to the block sref */
  TVM_DLL BlockInfo GetBlockInfo(const StmtSRef& block_sref) const;
  /*!
   * \brief Get the BlockScope correpsonding to the block sref
   * \param block_sref The block sref to be retrieved
   * \return The corresponding BlockScope
   */
  TVM_DLL BlockScope GetBlockScope(const StmtSRef& block_sref) const {
    return GetBlockInfo(block_sref).scope;
  }
  /*!
   * \brief Check a cached flag indicating if the specific block has quasi-affine bindings
   * \param block_sref The block sref to be checked
   * \return A boolean flag indicating if the block has quasi-affine bindings
   */
  TVM_DLL bool IsAffineBlockBinding(const StmtSRef& block_sref) const {
    return GetBlockInfo(block_sref).affine_binding;
  }
  /*!
   * \brief Check a cached flag indicating if each of the specific consumer block's read region
   * is fully produced by its producers
   * \param block_sref The block sref to be checked
   * \param consumer_block_sref The specific consumer block
   * \return A boolean flag indicating if the block has quasi-affine bindings
   */
  TVM_DLL bool IsRegionCoveredConsumer(const StmtSRef& consumer_block_sref) const {
    return GetBlockInfo(consumer_block_sref).region_cover;
  }
};

/*!
 * \brief Managed reference to ScheduleStateNode
 * \sa ScheduleStateNode
 */
class ScheduleState : public ObjectRef {
 public:
  /*!
   * \brief Construct a schedule state from an IRModule
   * \param mod The IRModule to be scheduled
   * \param debug_mode When turned on, additional checks will be performed after each mutation
   */
  TVM_DLL explicit ScheduleState(IRModule mod, int debug_mode = 0);
  /*!
   * \brief Construct a schedule state from a PrimFunc
   * \param func The PrimFunc to be scheduled. A new IRModule will be created with
   * this specific PrimFunc as "main" as the module to be scheduled
   * \param debug_mode When turned on, additional checks will be performed after each mutation
   */
  TVM_DLL explicit ScheduleState(PrimFunc func, int debug_mode = 0);

  ScheduleStateNode* get() { return static_cast<ScheduleStateNode*>(data_.get()); }

  const ScheduleStateNode* get() const {
    return static_cast<const ScheduleStateNode*>(data_.get());
  }

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScheduleState, ObjectRef, ScheduleStateNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_STATE_H_