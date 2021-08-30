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
#include <tvm/support/parallel_for.h>
#include <tvm/tir/schedule/schedule.h>

#include "../search_strategy.h"
#include "../tune_context.h"

namespace tvm {
namespace meta_schedule {

class ReplayTraceNode : public SearchStrategyNode {
 public:
  struct State {
    Array<tir::Trace> design_spaces;
    int i;

    explicit State(Array<tir::Trace> design_spaces, int i) : design_spaces(design_spaces), i(i) {}
  };

  int num_trials_per_iter;
  int num_trials_total;

  Optional<IRModule> mod = NullOpt;
  int num_threads = -1;
  std::unique_ptr<State> state = nullptr;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("num_trials_per_iter", &num_trials_per_iter);
    v->Visit("num_trials_total", &num_trials_total);
    v->Visit("mod", &mod);
    v->Visit("num_thread", &num_threads);
    // `state` is not visited
  }

  static constexpr const char* _type_key = "meta_schedule.ReplayTraceNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReplayTraceNode, Object);

 public:
  void InitializeWithTuneContext(const TuneContext& tune_context) final {
    this->mod = tune_context->mod;
    this->num_threads = tune_context->num_threads;
  }

  void PreTuning(const Array<tir::Trace>& design_spaces) final {
    ICHECK(this->mod.defined());
    ICHECK(this->state == nullptr);
    this->state = std::make_unique<State>(design_spaces, 0);
  }

  void PostTuning() final {
    ICHECK(this->state != nullptr);
    this->state.reset();
  }

  Optional<runtime::Array<IRModule>> GenerateMeasureCandidates() final {
    ICHECK(this->state != nullptr);
    ICHECK(this->mod.defined());
    int st = this->state->i;
    int ed = std::min(st + num_trials_per_iter, num_trials_total);
    if (st >= num_trials_total) {
      return NullOpt;
    }

    std::vector<int64_t> seeds_per_thread(this->num_threads, 0);
    runtime::Array<IRModule> results(ed - st, IRModule{nullptr});
    auto f_worker = [this, &seeds_per_thread, &results](int thread_id, int task_id) -> void {
      int64_t& seed = seeds_per_thread[thread_id];
      tir::Trace trace = this->state->design_spaces[0];  // need rand
      tir::Trace new_trace = tir::Trace(trace->insts, {});
      tir::Schedule sch =
          tir::Schedule::Traced(this->mod.value(),  //
                                /*seed=*/111,       // need fork from `seed`
                                /*debug_mode=*/0,   //
                                /*error_render_level=*/tir::ScheduleErrorRenderLevel::kNone);
      new_trace->ApplyToSchedule(sch, /*remove_postproc=*/true);
      // AWAIT: postproc
      results.Set(task_id, sch->mod());
    };
    support::parallel_persist_for(0, ed - st, f_worker, this->num_threads);
    return results;
  }

  void NotifyRunnerResults(const Array<RunnerResult>& results) final {
    ICHECK(this->state != nullptr);
    this->state->i += results.size();
  }
};

SearchStrategy SearchStrategy::ReplayTrace(int num_trials_per_iter, int num_trials_total) {
  ObjectPtr<ReplayTraceNode> n = make_object<ReplayTraceNode>();
  n->num_trials_per_iter = num_trials_per_iter;
  n->num_trials_total = num_trials_total;
  return SearchStrategy(n);
}

TVM_REGISTER_NODE_TYPE(ReplayTraceNode);
TVM_REGISTER_GLOBAL("meta_schedule.search_strategy.ReplayTrace")
    .set_body_typed(SearchStrategy::ReplayTrace);

}  // namespace meta_schedule
}  // namespace tvm
