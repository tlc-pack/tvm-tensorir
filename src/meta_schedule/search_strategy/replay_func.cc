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

/*! \brief A search strategy that generates measure candidates using space generator. */
class ReplayFuncNode : public SearchStrategyNode {
 public:
  using TRandState = support::LinearCongruentialEngine::TRandState;

  /*! \brief The state of the search strategy. */
  struct State {
    /*! \brief The search strategy itself */
    ReplayFuncNode* self;
    /*! \brief The design spaces. */
    Array<tir::Schedule> design_spaces;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int st;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int ed;

    explicit State(ReplayFuncNode* self, Array<tir::Schedule> design_spaces)
        : self(self), design_spaces(design_spaces), st(0), ed(self->num_trials_per_iter) {}

    inline Optional<Array<MeasureCandidate>> GenerateMeasureCandidates();
    inline void NotifyRunnerResults(const Array<RunnerResult>& results);
  };

  /*! \brief The number of trials per iteration. */
  int num_trials_per_iter;
  /*! \brief The number of total trials. */
  int num_trials_total;
  /*! \brief The space generator for measure candidates generation. */
  SpaceGenerator space_generator{nullptr};

  /*! \brief The module to be tuned. */
  Array<IRModule> mod_{nullptr};
  /*! \brief The metadata of the function arguments. */
  Array<ArgInfo> args_info_{nullptr};
  /*! \brief The number of threads to use. -1 means using logical cpu number. */
  int num_threads_ = -1;
  /*! \brief The random state. -1 means using random number. */
  TRandState rand_state_ = -1;
  /*! \brief The state of the search strategy. */
  std::unique_ptr<State> state_ = nullptr;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("num_trials_per_iter", &num_trials_per_iter);
    v->Visit("num_trials_total", &num_trials_total);
    v->Visit("space_generator", &space_generator);
    // `mod_` is not visited
    // `args_info_` is not visited
    // `num_threads_` is not visited
    // `rand_state_` is not visited
    // `state_` is not visited
  }

  static constexpr const char* _type_key = "meta_schedule.ReplayFunc";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReplayFuncNode, SearchStrategyNode);

  void InitializeWithTuneContext(const TuneContext& tune_context) final {
    CHECK(tune_context->num_threads > 0) << "Number of threads has to be larger than 0.";
    this->num_threads_ = tune_context->num_threads;
    this->mod_.reserve(this->num_threads_);
    this->args_info_ = ArgInfo::FromPrimFunc(FindEntryFunc(tune_context->mod.value()));
    this->rand_state_ = ForkSeed(&tune_context->rand_state);
    this->space_generator->InitializeWithTuneContext(tune_context);
    this->state_.reset();
  }

  void PreTuning(const Array<tir::Schedule>& design_spaces) final {
    this->mod_.clear();
    for (int i = 0; i < this->num_threads_; i++) {
      this->mod_.push_back(DeepCopyIRModule(
          design_spaces[tir::SampleInt(&this->rand_state_, 0, design_spaces.size())]->mod()));
    }
    ICHECK(!design_spaces.empty());
    ICHECK(this->state_ == nullptr);
    this->state_ = std::make_unique<State>(this, design_spaces);
  }

  void PostTuning() final {
    ICHECK(this->state_ != nullptr);
    this->state_.reset();
  }

  Optional<Array<MeasureCandidate>> GenerateMeasureCandidates() final {
    ICHECK(this->state_ != nullptr);
    return this->state_->GenerateMeasureCandidates();
  }

  void NotifyRunnerResults(const Array<RunnerResult>& results) final {
    ICHECK(this->state_ != nullptr);
    this->state_->NotifyRunnerResults(results);
  }
};

inline Optional<Array<MeasureCandidate>> ReplayFuncNode::State::GenerateMeasureCandidates() {
  if (st >= self->num_trials_total) {
    return NullOpt;
  }
  ed = std::min(ed, self->num_trials_total);
  ICHECK_LT(st, ed);
  std::vector<TRandState> per_thread_rand_state = ForkSeed(&self->rand_state_, self->num_threads_);
  Array<MeasureCandidate> per_task_result(ed - st, MeasureCandidate{nullptr});
  auto f_worker = [this, &per_thread_rand_state, &per_task_result](int thread_id,
                                                                   int task_id) -> void {
    TRandState& rand_state = per_thread_rand_state[thread_id];
    Array<tir::Schedule> schs = self->space_generator->GenerateDesignSpace(self->mod_[thread_id]);
    per_task_result.Set(task_id, MeasureCandidate(schs[tir::SampleInt(&rand_state, 0, schs.size())],
                                                  self->args_info_));
  };
  support::parallel_for_dynamic(0, ed - st, self->num_threads_, f_worker);
  return per_task_result;
}

inline void ReplayFuncNode::State::NotifyRunnerResults(const Array<RunnerResult>& results) {
  st += self->num_trials_per_iter;
  ed += self->num_trials_per_iter;
}

SearchStrategy SearchStrategy::ReplayFunc(int num_trials_per_iter,  //
                                          int num_trials_total,     //
                                          const SpaceGenerator& space_generator) {
  ObjectPtr<ReplayFuncNode> n = make_object<ReplayFuncNode>();
  n->num_trials_per_iter = num_trials_per_iter;
  n->num_trials_total = num_trials_total;
  n->space_generator = space_generator;
  return SearchStrategy(n);
}

TVM_REGISTER_NODE_TYPE(ReplayFuncNode);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategyReplayFunc")
    .set_body_typed(SearchStrategy::ReplayFunc);

}  // namespace meta_schedule
}  // namespace tvm
