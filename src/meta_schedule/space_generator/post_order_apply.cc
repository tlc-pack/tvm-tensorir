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

/*!
 * \brief Design Space Generator that generates design spaces by applying schedule rules to blocks
 *  in post-DFS order.
 * */
class PostOrderApplyNode : public SpaceGeneratorNode {
 public:
  using TRandState = support::LinearCongruentialEngine::TRandState;

  /*! \brief The module to be tuned. */
  IRModule mod_{nullptr};
  /*! \brief The random state. -1 means using random number. */
  TRandState rand_state_ = -1;
  /*! \brief The schedule rules to be applied in order. */
  Array<ScheduleRule> sch_rules_{nullptr};

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `mod_` is not visited
    // `rand_state_` is not visited
    // `sch_rules_` is not visited
  }

  void InitializeWithTuneContext(const TuneContext& tune_context) final {
    this->mod_ = tune_context->mod.value();
    this->rand_state_ = ForkSeed(&tune_context->rand_state);
    this->sch_rules_ = tune_context->sch_rules;
    for (const ScheduleRule& sch_rule : sch_rules_) {
      sch_rule->InitializeWithTuneContext(tune_context);
    }
  }

  Array<tir::Schedule> GenerateDesignSpace(const IRModule& mod) final {
    using ScheduleAndUnvisitedBlocks = std::pair<tir::Schedule, Array<tir::StmtSRef>>;

    tir::Schedule sch = tir::Schedule::Traced(        //
        this->mod_,                                   //
        /*rand_state=*/ForkSeed(&this->rand_state_),  //
        /*debug_mode=*/0,                             //
        /*error_render_level=*/tir::ScheduleErrorRenderLevel::kNone);
    std::vector<ScheduleAndUnvisitedBlocks> stack{
        ScheduleAndUnvisitedBlocks(sch, BlockCollector(sch).Run())};
    Array<tir::Schedule> result;

    while (!stack.empty()) {
      // get the stack.top()
      tir::Schedule sch = stack.back().first;
      Array<tir::StmtSRef> unvisited = stack.back().second;
      stack.pop_back();
      // if all blocks are visited
      if (unvisited.empty()) {
        result.push_back(sch);
        continue;
      }
      // otherwise, get the last block that is not visited
      tir::StmtSRef block_sref = unvisited.back();
      unvisited.pop_back();
      if (block_sref->stmt != nullptr) {
        const auto* block = block_sref->StmtAs<tir::BlockNode>();
        ICHECK(block) << "TypeError: Expects BlockNode, but gets: "
                      << block_sref->stmt->GetTypeKey();

        Array<tir::Schedule> current{sch};
        for (ScheduleRule sch_rule : sch_rules_) {
          // apply the rule to the block
          Array<tir::Schedule> applied;
          for (const tir::Schedule& sch : current) {
            // TODO(@zxybazh, @junrushao1994): Check the condition
            if (!tir::GetBlocks(sch->state(), block->name_hint, "main").empty()) {
              Array<tir::Schedule> tmp =
                  sch_rule->Apply(sch, /*block=*/sch->GetBlock(block->name_hint));
              applied.insert(applied.end(), tmp.begin(), tmp.end());
            }
          }
          current = std::move(applied);
        }
        for (const tir::Schedule& sch : current) {
          stack.emplace_back(sch, unvisited);
        }
      }
    }
    return result;
  }
  static constexpr const char* _type_key = "meta_schedule.PostOrderApply";
  TVM_DECLARE_FINAL_OBJECT_INFO(PostOrderApplyNode, SpaceGeneratorNode);
};

SpaceGenerator SpaceGenerator::PostOrderApply() {
  ObjectPtr<PostOrderApplyNode> n = make_object<PostOrderApplyNode>();
  return SpaceGenerator(n);
}

TVM_REGISTER_NODE_TYPE(PostOrderApplyNode);
TVM_REGISTER_GLOBAL("meta_schedule.SpaceGeneratorPostOrderApply")
    .set_body_typed(SpaceGenerator::PostOrderApply);

}  // namespace meta_schedule
}  // namespace tvm
