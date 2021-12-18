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
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

class CrossThreadReductionNode : public ScheduleRuleNode {
 public:
  // Inherited from ScheduleRuleNode
  void InitializeWithTuneContext(const TuneContext& context) final {
    ICHECK(context->target.defined());
    Target target = context->target.value();

    Optional<Integer> opt_max_threads_per_block = target->GetAttr<Integer>("max_threads_per_block");
    Optional<Integer> opt_warp_size = target->GetAttr<Integer>("thread_warp_size");
    CHECK(opt_max_threads_per_block.defined())
        << "ValueError: Target does not have attribute \"max_threads_per-block\"";
    CHECK(opt_warp_size.defined())
        << "ValueError: Target does not have attribute \"thread_warp_size\"";

    max_threads_per_block = opt_max_threads_per_block.value()->value;
    warp_size = opt_warp_size.value()->value;
  }

  // Inherited from ScheduleRuleNode
  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) final {
    // Step 0. Check the conditions of this rule.
    const tir::StmtSRef& block_sref = sch->GetSRef(block_rv);
    if (!NeedsRFactorOrCrossThreadReduction(sch->state(), block_sref, max_threads_per_block,
                                            warp_size)) {
      return {sch};
    }

    // Step 1. Make a copy of the original schedule. The new copy is used for scheduling.
    tir::Schedule tmp_sch = sch->Copy();
    tmp_sch->Seed(sch->ForkSeed());

    // Step 2. Check the opportunity for block fusion. We say "fusible", if we can compute-at the
    // block to its consumers. We want to fuse as much as possible because it results in
    // significantly faster schedule.
    bool fusible = false;
    // `target_loop` is the loop position where the input block will be computed at.
    tir::LoopRV target_loop{nullptr};
    // `target_block` is the consumer block that we want to compute-at the input block to.
    tir::BlockRV target_block{nullptr};

    std::tie(fusible, target_loop, target_block) = GetComputeTargetLoopAndBlock(tmp_sch, block_rv);

    // Step 3. Try block fusion.
    if (fusible) {
      ICHECK(target_block.defined());
      ICHECK(target_loop.defined());

      // Step 3.1. If the outer loops of `target_block` haven't been bound to threadIdx, we should
      // first bound the innermost outer loop of `target_block` to threadIdx. Possibly we need to
      // split the loop before binding.
      if (!InThreadScope(tmp_sch, target_block)) {
        const Array<tir::LoopRV>& split_res =
            tmp_sch->Split(tmp_sch->GetLoops(target_block).back(), {NullOpt, Integer(warp_size)});
        tmp_sch->Bind(split_res[1], "threadIdx.x");
      }
      // Step 3.2. Do the compute-at.
      tmp_sch->ComputeAt(block_rv, target_loop, /*preserve_unit_loops=*/true);
      // Step 3.3. Set the storage scope of the output buffer to shared memory.
      tmp_sch->SetScope(block_rv, /*buffer_index=*/0, /*storage_scope=*/"shared");
    }

    // Step 4. Reorder the loop axes if reduction loops are not innermost. After the reordering,
    // fuse all the reduction loops.
    size_t num_spatial_loops;
    tir::LoopRV fused_reduce_loop;
    ReorderAndFuseReductionLoops(tmp_sch, block_rv, &fused_reduce_loop, &num_spatial_loops);
    // Step 5. Split the fused reduction loop and bind the inner one to threadIdx.
    const Array<tir::LoopRV>& split_res =
        tmp_sch->Split(fused_reduce_loop, {NullOpt, Integer(warp_size)});
    tmp_sch->Bind(split_res[1], "threadIdx.x");

    return {tmp_sch, sch};
  }

 private:
  /*!
   * \brief Check whether the input block is in thread scope, i.e., some of its outer loop is
   * bound to threadIdx.
   * \param sch The TensorIR schedule
   * \param block The block to be checked
   * \return A boolean indicating whether the block is in thread scope.
   */
  bool InThreadScope(const tir::Schedule& sch, const tir::BlockRV& block) {
    const Array<tir::LoopRV>& axes = sch->GetLoops(block);
    for (const tir::LoopRV& loop_rv : axes) {
      const tir::For& loop = sch->Get(loop_rv);
      if (!loop->thread_binding.defined()) {
        continue;
      }
      if (std::string(loop->thread_binding.value()->thread_tag).substr(0, 9) == "threadIdx") {
        return true;
      }
    }
    return false;
  }

  /*!
   * \brief Get the compute-at target loop and the first block under the target loop.
   * \param sch The TensorIR schedule
   * \param block_rv The block whose compute-at target loop is queried
   * \return A tuple consisting of
   * 1. a boolean indicating whether the block can be computed at some target loop (a.k.a. fusible);
   * 2. the compute-at target loop when fusible, or a null loop random variable;
   * 3. the first block under the target loop when fusible, or a null block random variable.
   */
  std::tuple<bool, tir::LoopRV, tir::BlockRV> GetComputeTargetLoopAndBlock(
      const tir::Schedule& sch, const tir::BlockRV& block_rv) {
    // Step 1. Get all the consumers of the input block.
    Array<tir::BlockRV> consumers = sch->GetConsumers(block_rv);

    // Step 2. If the block has no consumer or the first consumer needs multi-level tiling, it is
    // not fusible.
    if (consumers.empty() || tir::NeedsMultiLevelTiling(sch->state(), sch->GetSRef(consumers[0]))) {
      return std::make_tuple(false, tir::LoopRV{nullptr}, tir::BlockRV{nullptr});
    }

    // Step 3. Calculate the lowest common ancestor of all the consumers.
    // - If the lowest common ancestor is a block, either there is only one consumer, or the LCA is
    //   the scope block, and thereby the target block is the first consumer;
    // - If the lowest common ancestor is a loop, the target block is also the first consumer.
    const tir::StmtSRef& lca_sref =
        tir::GetSRefLowestCommonAncestor(tir::BlockRVs2BlockSRefs(sch, consumers));

    // Step 4. Get the outer loops of the target block, and get the compute-at position index.
    Array<tir::LoopRV> tgt_block_loops = sch->GetLoops(consumers[0]);
    int pos = GetComputePosition(sch, sch->GetLoops(block_rv), tgt_block_loops, lca_sref);

    // Step 5. A negative position index means not fusible, and vice-versa.
    if (pos < 0) {
      return std::make_tuple(false, tir::LoopRV{nullptr}, tir::BlockRV{nullptr});
    } else {
      return std::make_tuple(true, tgt_block_loops[pos], consumers[0]);
    }
  }

  /*!
   * \brief Get the compute-at position index of the input block, according to
   * 1. the loops outside the input block;
   * 2. the loops outside the target block;
   * 3. the lowest common ancestor of all the consumers of the input block.
   * \param sch The TensorIR schedule
   * \param block_loops The loops outside the input block
   * \param tgt_block_loops The loops outside the target block
   * \param lca_sref The lowest common ancestor of all the consumers of the input block
   * \return The compute-at position index of the input block
   */
  int GetComputePosition(const tir::Schedule& sch, const Array<tir::LoopRV>& block_loops,
                         const Array<tir::LoopRV>& tgt_block_loops, const tir::StmtSRef& lca_sref) {
    int n_block_loop = static_cast<int>(block_loops.size());
    int n_tgt_block_loop = static_cast<int>(tgt_block_loops.size());

    for (int i = 0; i < n_block_loop && i < n_tgt_block_loop; ++i) {
      if (tir::GetLoopIterType(sch->GetSRef(block_loops[i])) != tir::IterVarType::kDataPar) {
        return i - 1;
      } else if (sch->GetSRef(tgt_block_loops[i]).same_as(lca_sref)) {
        // If the lowest common ancestor is a loop, the compute location of the input block should
        // not be deeper than the LCA loop.
        return i;
      }
    }
    return std::min(n_block_loop, n_tgt_block_loop) - 1;
  }

 public:
  /*! \brief The maximum number of threads allowed in a thread block */
  int max_threads_per_block;
  /*! \brief The number of threads per warp */
  int warp_size;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("max_threads_per_block", &max_threads_per_block);
    v->Visit("warp_size", &warp_size);
  }

  static constexpr const char* _type_key = "meta_schedule.CrossThreadReduction";
  TVM_DECLARE_FINAL_OBJECT_INFO(CrossThreadReductionNode, ScheduleRuleNode);
};

ScheduleRule ScheduleRule::CrossThreadReduction() {
  ObjectPtr<CrossThreadReductionNode> n = make_object<CrossThreadReductionNode>();
  return ScheduleRule(n);
}

TVM_REGISTER_NODE_TYPE(CrossThreadReductionNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleCrossThreadReduction")
    .set_body_typed(ScheduleRule::CrossThreadReduction);

}  // namespace meta_schedule
}  // namespace tvm
