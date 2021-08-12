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
#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt_functor.h>

#include "../../tir/schedule/analysis.h"
#include "../search.h"
#include "../utils.h"
#include "./postproc.h"
#include "./search_rule.h"

namespace tvm {
namespace meta_schedule {

/********** PostOrderApply **********/

/*! \brief Search space that is specified by applying rules in post-DFS order */
class PostOrderApplyNode : public SearchSpaceNode {
 public:
  /*! \brief The rules to be applied */
  Array<SearchRule> stages;
  /*! \brief The postprocessors */
  Array<Postproc> postprocs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("stages", &stages);
    v->Visit("postprocs", &postprocs);
  }
  /*! \brief Default destructor */
  ~PostOrderApplyNode() = default;

  /*!
   * \brief Apply postprocessors onto the schedule
   * \param task The search task
   * \param sch The schedule to be postprocessed
   * \param rand_state The random state for sampling
   */
  bool Postprocess(const SearchTask& task, const Schedule& sch,
                   tir::TRandState* rand_state) override;
  /*!
   * \brief Sample a schedule out of the search space
   * \param task The search task to be sampled from
   * \return The schedule sampled
   */
  Schedule SampleSchedule(const SearchTask& task, tir::TRandState* rand_state) override;
  /*!
   * \brief Get support of the search space
   * \param task The search task to be sampled from
   * \return An array with a single element returned from SampleSchedule
   * \sa PostOrderApplyNode::SampleSchedule
   */
  Array<Schedule> GetSupport(const SearchTask& task, tir::TRandState* rand_state) override;

  static constexpr const char* _type_key = "meta_schedule.PostOrderApply";
  TVM_DECLARE_FINAL_OBJECT_INFO(PostOrderApplyNode, SearchSpaceNode);
};

/*!
 * \brief Managed reference to PostOrderApplyNode
 * \sa PostOrderApplyNode
 */
class PostOrderApply : public SearchSpace {
 public:
  /*!
   * \brief Constructor
   * \param stages The rules to be applied
   */
  explicit PostOrderApply(Array<SearchRule> stages, Array<Postproc> postprocs);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PostOrderApply, SearchSpace, PostOrderApplyNode);
};

/********** Constructor **********/

PostOrderApply::PostOrderApply(Array<SearchRule> stages, Array<Postproc> postprocs) {
  ObjectPtr<PostOrderApplyNode> n = make_object<PostOrderApplyNode>();
  n->stages = std::move(stages);
  n->postprocs = std::move(postprocs);
  data_ = std::move(n);
}

/********** Sampling **********/

bool PostOrderApplyNode::Postprocess(const SearchTask& task, const Schedule& sch,
                                     tir::TRandState* rand_state) {
  sch->EnterPostproc();
  for (const Postproc& postproc : postprocs) {
    if (!postproc->Apply(task, sch, rand_state)) {
      return false;
    }
  }
  return true;
}

Schedule PostOrderApplyNode::SampleSchedule(const SearchTask& task, tir::TRandState* rand_state) {
  Array<Schedule> support = GetSupport(task, rand_state);
  ICHECK(!support.empty()) << "ValueError: Found null support";
  int i = tir::SampleInt(rand_state, 0, support.size());
  return support[i];
}

/*! \brief Collecting all the non-root blocks */
class BlockCollector : public tir::StmtVisitor {
 public:
  /*! \brief Constructor */
  explicit BlockCollector(const tir::Schedule& sch) : sch_(sch) {
    const auto* realize = GetOnlyFunc(sch->mod())->body.as<tir::BlockRealizeNode>();
    root_block_ = realize->block.get();
  }

  /*! \brief Entry point */
  Array<tir::StmtSRef> Run() {
    VisitStmt(GetOnlyFunc(sch_->mod())->body);
    Array<tir::StmtSRef> result = std::move(result_);
    return result;
  }

 private:
  void VisitStmt_(const tir::BlockNode* block) override {
    if (block != root_block_) {
      result_.push_back(sch_->GetSRef(block));
    }
    this->VisitStmt(block->body);
  }

  /*! \brief The schedule to be collected */
  const tir::Schedule& sch_;
  /*! \brief Result of collection */
  Array<tir::StmtSRef> result_;
  /*! \brief The */
  const tir::BlockNode* root_block_;
};

Array<Schedule> PostOrderApplyNode::GetSupport(const SearchTask& task,
                                               tir::TRandState* rand_state) {
  using ScheduleAndUnvisitedBlocks = std::pair<Schedule, Array<tir::StmtSRef>>;

  Array<Schedule> curr{
      Schedule::Traced(/*mod=*/IRModule({{GlobalVar("main"), task->workload}}),
                       /*seed=*/tir::ForkSeed(rand_state),
                       /*debug_mode=*/false,
                       /*error_render_level=*/tir::ScheduleErrorRenderLevel::kDetail)};
  for (const SearchRule& rule : stages) {
    std::vector<ScheduleAndUnvisitedBlocks> stack;
    stack.reserve(curr.size());
    for (const Schedule& sch : curr) {
      stack.emplace_back(sch, BlockCollector(sch).Run());
    }
    Array<Schedule> next;
    while (!stack.empty()) {
      // get the stack.top()
      Schedule sch = stack.back().first;
      Array<tir::StmtSRef> unvisited = stack.back().second;
      stack.pop_back();
      // if all blocks are visited
      if (unvisited.empty()) {
        next.push_back(sch);
        continue;
      }
      // otherwise, get the last block that is not visited
      tir::StmtSRef block_sref = unvisited.back();
      unvisited.pop_back();
      if (block_sref->stmt != nullptr) {
        const auto* block = block_sref->StmtAs<tir::BlockNode>();
        ICHECK(block) << "TypeError: Expects BlockNode, but gets: "
                      << block_sref->stmt->GetTypeKey();
        // TODO(@junrushao1994): replace this quick hack
        if (!tir::GetBlocks(sch->state(), block->name_hint).empty()) {
          // apply the rule to the block
          Array<Schedule> applied =
              rule->Apply(task, sch, /*block=*/sch->GetBlock(block->name_hint));
          // append the newly got schedules to the top of the stack
          for (const Schedule& sch : applied) {
            stack.emplace_back(sch, unvisited);
          }
          continue;
        }
      }
      stack.emplace_back(sch, unvisited);
    }
    curr = next;
  }
  // Simplify the trace
  Array<Schedule> result;
  result.reserve(curr.size());
  for (const Schedule& sch : curr) {
    Trace trace = sch->trace().value()->Simplified(/*remove_postproc=*/true);
    Schedule new_sch =
        Schedule::Traced(/*mod=*/IRModule({{GlobalVar("main"), task->workload}}),
                         /*seed=*/tir::ForkSeed(rand_state),
                         /*debug_mode=*/false,
                         /*error_render_level=*/tir::ScheduleErrorRenderLevel::kDetail);
    trace->ApplyToSchedule(new_sch, /*remove_postproc=*/true);
    result.push_back(new_sch);
  }
  return result;
}

/********** FFI **********/

struct Internal {
  /*!
   * \brief Constructor of PostOrderApply
   * \param rule The rule to be applied
   * \param postprocs The postprocessors
   * \return The PostOrderApply constructed
   * \sa PostOrderApply::PostOrderApply
   */
  static PostOrderApply New(Array<SearchRule> stages, Array<Postproc> postprocs) {
    return PostOrderApply(stages, postprocs);
  }
};

TVM_REGISTER_NODE_TYPE(PostOrderApplyNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostOrderApply").set_body_typed(Internal::New);

}  // namespace meta_schedule
}  // namespace tvm
