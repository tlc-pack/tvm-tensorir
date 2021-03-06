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
#ifndef SRC_META_SCHEDULE_ANALYSIS_H_
#define SRC_META_SCHEDULE_ANALYSIS_H_

#include <vector>

#include "../tir/schedule/analysis.h"
#include "../tir/schedule/utils.h"
#include "./schedule.h"
#include "./search.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Check if
 * 1) number of blocks vars equals to number of loop vars
 * 2) each block var is bound to a loop var directly
 * 3) the order is preserved, i.e. the i-th block var is the i-th loop var
 * \param self The TIR schedule class
 * \param block_sref The block to be analyzed
 * \return A boolean indicating if the block binding is trivial
 */
TVM_DLL bool IsTrivialBinding(const tir::ScheduleState& self, const tir::StmtSRef& block_sref);

/*!
 * \brief Check if a block is the direct children of the root block
 * \param self The TIR schedule class
 * \param block_sref The block to be analyzed
 * \return A boolean flag indicating if the block is the subroot block
 */
TVM_DLL bool IsSubrootBlock(const tir::ScheduleState& self, const tir::StmtSRef& block_sref);

/*!
 * \brief Check if a block has no child block
 * \param self The TIR schedule class
 * \param block_sref The block to be analyzed
 * \return A boolean flag indicating if the block is a leaf block
 */
TVM_DLL bool IsLeafBlock(const tir::ScheduleState& self, const tir::StmtSRef& block_sref);

/*!
 * \brief Return the IterVarType of each block var
 * \param self The TIR schedule class
 * \param block_sref The block to be analyzed
 * \return An array of integers, the IterVarTypes corresponding to each block var in order
 * \sa tir::IterVarType
 */
TVM_DLL Array<Integer> GetBlockVarTypes(const tir::ScheduleState& self,
                                        const tir::StmtSRef& block_sref);

/*!
 * \brief Check if the iter types of all block vars are data parallel
 * \param self The TIR schedule class
 * \param block_sref The block to be analyzed
 * \return A boolean indicating if the block is spatial
 */
TVM_DLL bool IsSpatial(const tir::ScheduleState& self, const tir::StmtSRef& block_sref);

/*!
 * \brief Check if a block is output block
 * \param self The TIR schedule class
 * \param block_sref The block to be analyzed
 * \return A boolean flag indicating if it is an output block
 */
TVM_DLL bool IsOutputBlock(const tir::ScheduleState& self, const tir::StmtSRef& block_sref);

/*!
 * \brief Count the number of occurrence of an operator, i.e. tir.exp
 * \param self The TIR schedule class
 * \param block_sref The block to be analyzed
 * \param op The operator to be counted
 * \return An integer indicating the number of its occurrence
 */
TVM_DLL int CountOp(const tir::ScheduleState& self, const tir::StmtSRef& block_sref, const Op& op);

/*!
 * \brief Check if there is any branch in the given block, which includes
 * 1) block predicate
 * 2) if-then-else statement
 * 3) select expression
 * 4) if-then-else operator
 * \param self The TIR schedule class
 * \param block_sref The block to be analyzed
 * \return A boolean indicating there is at least a branch in the given block
 */
TVM_DLL bool HasBranch(const tir::ScheduleState& self, const tir::StmtSRef& block_sref);

/*!
 * \brief Check whether the producer and consumer matches in elementwise way.
 * Assuming consumer_sref is the only consumer of producer_sref.
 * \param self The meta schedule class
 * \param producer_sref The producer block
 * \param consumer_sref The consumer block
 * \return A boolean flag indicating if they match
 */
TVM_DLL bool IsElementWiseMatch(const tir::ScheduleState& self, const tir::StmtSRef& producer_sref,
                                const tir::StmtSRef& consumer_sref);

/*!
 * \brief Check if a block needs multi-level tiling
 * \param self The TIR schedule class
 * \param block_sref The block to be analyzed
 * \return A boolean flag indicating if the block needs multi-level tiling
 */
TVM_DLL bool NeedsMultiLevelTiling(const tir::ScheduleState& self, const tir::StmtSRef& block_sref);

/*!
 * \brief Check if a block can be inlined
 * \param self The TIR schedule class
 * \param block_sref The block to be analyzed
 * \return A boolean flag indicating if the block needs multi-level tiling
 */
TVM_DLL bool IsStrictlyInlineable(const tir::ScheduleState& self, const tir::StmtSRef& block_sref);

/*! \brief Necessary information used for tensorization */
class TensorizeInfoNode : public Object {
 public:
  /*! \brief Maps block loops to desc loops */
  Map<tir::StmtSRef, tir::For> loop_map;
  /*! \brief Maps loops in desc to its index, outer to inner */
  Map<tir::For, Integer> desc_loop_indexer;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("loop_map", &loop_map);
    v->Visit("desc_loop_indexer", &desc_loop_indexer);
  }

  static constexpr const char* _type_key = "meta_schedule.analysis.TensorizeInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(TensorizeInfoNode, Object);
};

/*!
 * \brief Managed reference to TensorizeInfoNode
 * \sa TensorizeInfoNode
 */
class TensorizeInfo : public ObjectRef {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TensorizeInfo, ObjectRef, TensorizeInfoNode);
};

/*!
 * \brief Check if the given block can be tensorized, and in the meantime gather the necessary
 * information for tensorization
 * \param self The TIR schedule
 * \param block_sref The block to be analyzed
 * \param desc_func The target function for tensorization
 * \return The necessary information used for tensorization, or NullOpt if the block cannot be
 * tensorized
 */
TVM_DLL Optional<TensorizeInfo> GetTensorizeLoopMapping(const tir::ScheduleState& self,
                                                        const tir::StmtSRef& block_sref,
                                                        const tir::PrimFunc& desc_func);

/*!
 * \brief Count the floating point operations of a PrimFunc
 * \param func The PrimFunc to be counted
 * \return The number of floating point operations
 */
TVM_DLL double CountFlop(const tir::PrimFunc& func);

/*!
 * \brief Calculate the product of extent of all spatial and reduction loop axes.
 * \param self The TIR schedule
 * \param block_sref The block to be analyzed
 * \return A pair indicating the cumulative length of spacial and reduction loop axes. Or (-1, -1)
 *         if some loops are dynamic or with type other than kDataPar and kCommReduce.
 */
TVM_DLL std::pair<int64_t, int64_t> GetCumulativeSpaceAndReductionLength(
    const tir::ScheduleState& self, const tir::StmtSRef& block_sref);

/*!
 * \brief Check if the block needs rfactor. The conditions are:
 *          1. The block is a reduction block and has trivial binding.
 *          2. Every the loop axis out side the block must be either spatial axis or reduction axis.
 *          3. There is at least one reduction loop.
 *          4. The outside loops are continuous, and the body of the innermost loop is exactly
 *             the block.
 *          5. The outside loops are not dynamic.
 *          6. a. For blocks which need MultiLevelTiling, don't perform rfactor if we have enough
 *                parallelism on spatial loops
 *             b. For other blocks, always try to perform rfactor.
 * \param task The search task
 * \param self The TIR schedule
 * \param block_sref The block to be analyzed
 * \return A boolean indicating if it needs rfactor
 */
TVM_DLL bool NeedsRFactor(const tir::ScheduleState& self, const tir::StmtSRef& block_sref,
                          const SearchTask& task, const int& max_jobs_per_core,
                          std::atomic<int>* warned_num_cores_missing);

/*!
 * \brief Check if the block has its cache-write block
 * \param sch The TIR schedule
 * \param block_rv The block to be analyzed
 * \param i The index of the buffer in block's write region
 * \return A boolean indicating if it has cache-write block
 */
TVM_DLL bool HasCacheWriteBlock(const Schedule& sch, const BlockRV& block_rv, const int& i);

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_ANALYSIS_H_
