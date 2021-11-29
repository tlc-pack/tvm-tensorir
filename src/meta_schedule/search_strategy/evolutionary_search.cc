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

/**************** Data Structure ****************/

/*!
 * \brief A heap with a size up-limit. If overflow happens, it evicted the worst items.
 * \note It maintains a min heap in terms of `CachedTrace::score`. Therefore, when
 * overflow happens, the element evicted is the one with the min `CachedTrace::score`.
 * As time goes, the elements in the heap are going to be larger.
 */
class SizedHeap {
  /*! \brief The comparator class, used by `std::push_heap` and `std::pop_heap` */
  struct Comparator {
    bool operator()(const CachedTrace& a, const CachedTrace& b) const { return a.score > b.score; }
  };

 public:
  /*!
   * \brief Constructor
   * \param size_limit The up-limit of the heap size
   */
  explicit SizedHeap(int size_limit) : size_limit(size_limit) { heap.reserve(size_limit); }

  /*!
   * \brief Push the specific item to the heap if its key did not appears in the heap
   * \param item The item to be pushed
   */
  void Push(const CachedTrace& item) {
    if (!in_heap.insert(item.repr).second) {
      return;
    }
    int size = heap.size();
    if (size < size_limit) {
      // Heap is not full, just push
      heap.emplace_back(item);
      std::push_heap(heap.begin(), heap.end(), Comparator());
    } else if (Comparator()(item, heap.front())) {
      // if the item is better than the worst one in the heap, we can safely kick it out
      std::pop_heap(heap.begin(), heap.end(), Comparator());
      heap.back() = item;
      std::push_heap(heap.begin(), heap.end(), Comparator());
    }
    // Otherwise, the item is worse than any other element in the heap
  }

  /*! \brief Up-limit of the heap size */
  int size_limit;
  /*! \brief The heap, the worse the topper */
  std::vector<CachedTrace> heap;
  /*! \brief The traces that are in the heap */
  std::unordered_set<String> in_heap;
};

/*!
 * \brief A search strategy that generates measure candidates using evolutionary search.
 * \note The algorithm:
 *
 * Loop until #measured >= total_measures:
 *   init =
 *      pick top `k = population *      init_measured_ratio ` from measured
 *      pick     `k = population * (1 - init_measured_ratio)` random selected from search space
 *   best = generate `population` states with the cost model,
 *          starting from `init`,
 *          using mutators,
 *          and return the top-n states during the search,
 *          where `n = num_measures_per_iter`
 *   chosen = pick top `k = num_measures_per_iter * (1 - eps_greedy)` from `best`
 *            pick     `k = num_measures_per_iter *      eps_greedy ` from `init`
 *   do the measurement on `chosen` & update the cost model
 */
class EvolutionarySearchNode : public SearchStrategyNode {
 public:
  using TRandState = support::LinearCongruentialEngine::TRandState;

  /*! \brief The state of the search strategy. */
  struct State {
    /*! \brief The search strategy itself */
    EvolutionarySearchNode* self;
    /*! \brief The design spaces. */
    Array<tir::Schedule> design_spaces;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int st;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int ed;

    explicit State(EvolutionarySearchNode* self, Array<tir::Schedule> design_spaces)
        : self(self), design_spaces(design_spaces), st(0), ed(self->num_trials_per_iter) {}

    /*!
     * \brief Sample the initial population from previous measured results and randomly generated
     *  traces via trace replaying.
     * \return The initial population of traces sampled.
     */
    inline Array<tir::Trace> SampleInitPopulation();
    /*!
     * \brief Evolve the initial population using mutators and samplers.
     * \param inits The initial population of traces sampled.
     * \return The evolved traces from initial population.
     */
    inline Array<tir::Trace> EvolveWithCostModel(const Array<tir::Trace>& inits);
    /*!
     * \brief Pick final candidates from the given initial population and bests of evolved ones.
     * \param inits The initial population of traces sampled.
     * \param bests The best candidates predicted from evolved traces.
     * \return The final picked candidates with a ratio of both.
     */
    inline Array<tir::Trace> PickWithEpsGreedy(const Array<tir::Trace>& inits,
                                               const Array<tir::Trace>& bests);
    /*!
     * \brief Assemble measure candidates from the given candidate traces.
     * \param traces The picked candidate traces.
     * \return The assembled measure candidates.
     */
    inline Array<MeasureCandidate> AssembleCandidates(const Array<tir::Trace>& picks);
    inline Optional<Array<MeasureCandidate>> GenerateMeasureCandidates();
    inline void NotifyRunnerResults(const Array<RunnerResult>& results);
  };

  /*! \brief The number of trials per iteration. */
  int num_trials_per_iter;
  /*! \brief The number of total trials. */
  int num_trials_total;
  /*! \brief THe population size in the evolutionary search.*/
  int population;

  /*! \brief The target for the workload. */
  Target target_{nullptr};
  /*! \brief The tuning context of the evolutionary search strategy. */
  TuneContext tune_context_{nullptr};
  /*! \brief The mutators to be used. */
  Array<Mutator> mutators_{nullptr};
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

  /*** Configuration: the initial population ***/
  /*! \brief The ratio of measured states used in the initial population */
  double init_measured_ratio;

  /*** Configuration: evolution ***/
  /*! \brief The number of iterations performed by generic algorithm. */
  int genetic_algo_iters;
  /*! \brief The probability to perform mutation */
  double p_mutate;
  /*! \brief Mutators and their probability mass */
  Map<Mutator, FloatImm> mutator_probs{nullptr};
  /*! \brief A Database for selecting useful candidates. */
  Database database{nullptr};
  /*! \brief A cost model helping to explore the search space */
  CostModel cost_model{nullptr};
  /*! \brief The batch of measure candidates generated for measurement. */
  Array<MeasureCandidate> candidates{nullptr};

  /*** Configuration: pick states for measurement ***/
  /*! \brief The ratio of measurements to use randomly sampled states. */
  double eps_greedy;

  /*!
   * Helpers
   * Note that the use of trace cache could be multi-threaded.
   */
  mutable std::unordered_map<tir::Trace, CachedTrace, ObjectPtrHash, ObjectPtrEqual> trace_cache_;
  mutable std::mutex trace_cache_mutex_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `tune_context_` is not visited
    // `target_` is not visited
    // `mod_` is not visited
    // `args_info_` is not visited
    // `num_threads_` is not visited
    // `rand_state_` is not visited
    // `state_` is not visited

    /*** Configuration: global ***/
    v->Visit("num_trials_total", &num_trials_total);
    v->Visit("num_trials_per_iter", &num_trials_per_iter);
    v->Visit("population", &population);
    /*** Configuration: the initial population ***/
    v->Visit("init_measured_ratio", &init_measured_ratio);
    /*** Configuration: evolution ***/
    v->Visit("genetic_algo_iters", &genetic_algo_iters);
    v->Visit("p_mutate", &p_mutate);
    v->Visit("mutator_probs", &mutator_probs);
    v->Visit("cost_model", &cost_model);
    /*** Configuration: pick states for measurement ***/
    v->Visit("eps_greedy", &eps_greedy);
    /*** Helpers ***/
    // Not visited: `trace_cache_`
    // Not visited: `trace_cache_mutex_`
  }

  /*!
   * \brief Add the cached trace into the trace_cache_
   * \param cached_trace The cached_trace to be added
   */
  void _AddCachedTrace(const CachedTrace& cached_trace) const {
    // Todo(@zxybazh): Avoid redundent traces
    std::unique_lock<std::mutex> lock(this->trace_cache_mutex_);
    trace_cache_.emplace(GetRef<tir::Trace>(cached_trace.trace), cached_trace);
  }

  /*!
   * \brief Retrieve the cached trace given the trace
   * \param trace The trace to be retrieved
   * \return The cached trace
   */
  CachedTrace _GetCachedTrace(const tir::Trace& trace) const {
    auto iter = trace_cache_.find(trace);
    ICHECK(iter != trace_cache_.end());
    return iter->second;
  }

  static constexpr const char* _type_key = "meta_schedule.EvolutionarySearch";
  TVM_DECLARE_FINAL_OBJECT_INFO(EvolutionarySearchNode, SearchStrategyNode);

  void InitializeWithTuneContext(const TuneContext& tune_context) final {
    CHECK(tune_context.defined()) << "TuneContext must be defined!";
    CHECK(tune_context->num_threads > 0) << "Number of threads has to be larger than 0.";
    CHECK(tune_context->mutators.defined()) << "Mutators must be defined!";
    CHECK(tune_context->target.defined()) << "Target must be defined!";
    this->tune_context_ = tune_context;
    this->target_ = tune_context->target.value();
    this->mutators_ = tune_context->mutators.value();
    this->num_threads_ = tune_context->num_threads;

    this->mod_.reserve(this->num_threads_);
    for (int i = 0; i < this->num_threads_; i++) {
      this->mod_.push_back(DeepCopyIRModule(tune_context->mod.value()));
    }

    this->args_info_ = ArgInfo::FromPrimFunc(FindEntryFunc(tune_context->mod.value()));
    this->rand_state_ = ForkSeed(&tune_context->rand_state);
    this->state_.reset();
  }

  void PreTuning(const Array<tir::Schedule>& design_spaces) final {
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

inline Array<tir::Trace> EvolutionarySearchNode::State::SampleInitPopulation() {
  self->trace_cache_.clear();
  std::vector<tir::Trace> results;
  results.reserve(self->population);
  // Threading RNG
  std::vector<TRandState> per_thread_rand_state = ForkSeed(&self->rand_state_, self->num_threads_);
  // Pick measured states
  int num_measured = self->population * self->init_measured_ratio;
  for (TuningRecord record :
       self->database->GetTopK(self->database->CommitWorkload(self->mod_[0]), num_measured)) {
    results.push_back(record->trace);
  }

  auto f_proc_measured = [this, &results, &per_thread_rand_state](int thread_id,
                                                                  int trace_id) -> void {
    TRandState& rand_state = per_thread_rand_state[trace_id];
    const tir::Trace& trace = results[trace_id];
    if (Optional<tir::Schedule> opt_sch =
            meta_schedule::ReplayTrace(trace, self->mod_[trace_id], &rand_state)) {
      tir::Schedule sch = opt_sch.value();
      self->_AddCachedTrace(CachedTrace{trace.get(), sch, Repr(sch), -1.0});
    } else {
      LOG(FATAL) << "ValueError: Cannot postprocess the trace:\n" << trace;
      throw;
    }
  };
  support::parallel_for_dynamic(0, results.size(), self->num_threads_, f_proc_measured);

  // Pick unmeasured states
  std::atomic<int> tot_fail_ct(0);
  std::atomic<int> success_ct(0);
  auto f_proc_unmeasured = [this, &results, &per_thread_rand_state, &tot_fail_ct, &success_ct](
                               int thread_id, int trace_id) -> void {
    TRandState& rand_state = per_thread_rand_state[trace_id];
    for (;;) {
      int design_space_index = tir::SampleInt(&rand_state, 0, design_spaces.size());
      tir::Trace trace = design_spaces[design_space_index]->trace().value();
      Map<tir::Instruction, ObjectRef> decisions;
      try {
        if (Optional<tir::Schedule> opt_sch =
                meta_schedule::ReplayTrace(trace, self->mod_[trace_id], &rand_state)) {
          tir::Schedule sch = opt_sch.value();
          tir::Trace old_trace = sch->trace().value();
          tir::Trace trace(old_trace->insts, old_trace->decisions);
          self->_AddCachedTrace(CachedTrace{trace.get(), sch, Repr(sch), -1.0});
          results[trace_id] = std::move(trace);
          success_ct++;
          break;
        } else {
          tot_fail_ct++;
        }
      } catch (const dmlc::Error& e) {
        tot_fail_ct++;
      }
      if (success_ct > 64) {  // Todo(@junru): Why 64? Add to constructor.
        break;
      }
    }
  };
  num_measured = results.size();
  results.resize(self->population, tir::Trace(nullptr));
  support::parallel_for_dynamic(num_measured, self->population, self->num_threads_,
                                f_proc_unmeasured);
  std::vector<tir::Trace> pruned_results;
  for (const tir::Trace& result : results) {
    if (result.defined()) {
      pruned_results.push_back(result);
    }
  }
  // LOG(INFO) << "fail count: " << tot_fail_ct;
  return pruned_results;
}

Array<tir::Trace> EvolutionarySearchNode::State::EvolveWithCostModel(
    const Array<tir::Trace>& inits) {
  // The heap to record best schedule, we do not consider schedules that are already measured
  // Also we use `in_heap` to make sure items in the heap are de-duplicated
  SizedHeap heap(self->num_trials_per_iter);
  // Threading RNG
  std::vector<TRandState> per_thread_rand_state = ForkSeed(&self->rand_state_, self->num_threads_);
  std::vector<std::function<int()>> thread_trace_samplers(self->num_threads_);
  std::vector<std::function<Optional<Mutator>()>> thread_mutator_samplers(self->num_threads_);
  std::vector<int> trace_used;
  std::mutex trace_used_mutex;
  // Prepare search queues
  std::vector<CachedTrace> sch_curr;
  std::vector<CachedTrace> sch_next;
  sch_curr.reserve(self->population);
  sch_next.reserve(self->population);
  for (const tir::Trace& trace : inits) {
    sch_curr.push_back(self->_GetCachedTrace(trace));
  }
  // Main loop: (genetic_algo_iters + 1) times
  for (int iter = 0;; ++iter) {
    // Predict running time with the cost model,
    // and put the schedules with the predicted perf to the heap
    std::vector<double> scores =
        PredictNormalizedScore(sch_curr, self->tune_context_, self->cost_model, self->args_info_);
    for (int i = 0, n = sch_curr.size(); i < n; ++i) {
      CachedTrace& entry = sch_curr[i];
      entry.score = scores[i];
      if (!self->database->GetTopK(self->database->CommitWorkload(entry.sch->mod()), 1).size()) {
        heap.Push(entry);
      }
    }
    // Discontinue once it reaches end of search
    if (iter == self->genetic_algo_iters) {
      break;
    }
    // Set threaded samplers, with probability from predicated normalized throughputs
    for (int i = 0; i < self->num_threads_; ++i) {
      TRandState& rand_state = per_thread_rand_state[i];
      thread_trace_samplers[i] = MakeMultinomial(rand_state, scores);
      thread_mutator_samplers[i] =
          MakeMutatorSampler(self->p_mutate, self->mutator_probs, rand_state);
    }
    trace_used = std::vector<int>(scores.size(), 0);
    // The worker function
    auto f_find_candidate = [&per_thread_rand_state, &thread_trace_samplers,
                             &thread_mutator_samplers, &trace_used, &trace_used_mutex, &sch_curr,
                             &sch_next, this](int thread_id, int i) {
      // Prepare samplers
      TRandState& rand_state = per_thread_rand_state[thread_id];
      const std::function<int()>& trace_sampler = thread_trace_samplers[thread_id];
      const std::function<Optional<Mutator>()>& mutator_sampler =
          thread_mutator_samplers[thread_id];
      // Loop until success
      int max_retry_cnt = 10;
      int retry_cnt = 0;
      for (;;) {
        int trace_idx = trace_sampler();
        const CachedTrace& cached_trace = sch_curr[trace_idx];
        if (Optional<Mutator> opt_mutator = mutator_sampler()) {
          // Decision: mutate
          Mutator mutator = opt_mutator.value();
          if (Optional<tir::Trace> opt_new_trace =
                  mutator->Apply(GetRef<tir::Trace>(cached_trace.trace))) {
            tir::Trace new_trace = opt_new_trace.value();
            if (Optional<tir::Schedule> opt_sch =
                    ReplayTrace(new_trace, self->mod_[i], &rand_state)) {
              tir::Schedule sch = opt_sch.value();
              CachedTrace new_cached_trace{new_trace.get(), sch, Repr(sch), -1.0};
              self->_AddCachedTrace(new_cached_trace);
              sch_next[i] = new_cached_trace;
              break;
            }
          }
        } else {
          // Decision: do not mutate
          std::unique_lock<std::mutex> lock(trace_used_mutex);
          if (!trace_used[trace_idx]) {
            trace_used[trace_idx] = 1;
            sch_next[i] = cached_trace;
            break;
          }
        }
        retry_cnt++;
        if (retry_cnt >= max_retry_cnt) {
          sch_next[i] = cached_trace;
          break;
        }
      }
    };
    sch_next.clear();
    sch_next.resize(self->population);
    support::parallel_for_dynamic(0, self->population, 1, f_find_candidate);
    sch_curr.clear();
    sch_curr.swap(sch_next);
  }
  // Return the best states from the heap, sorting from higher score to lower ones
  std::sort(heap.heap.begin(), heap.heap.end(), CachedTrace::Compare);
  Array<tir::Trace> results;
  results.reserve(self->num_trials_per_iter);
  for (const CachedTrace& item : heap.heap) {
    results.push_back(GetRef<tir::Trace>(item.trace));
  }
  /* Logging
    constexpr int kNumScoresPerLine = 16;
    std::ostringstream os;
    int n = heap.heap.size();
    for (int st = 0; st < n; st += kNumScoresPerLine) {
      os << std::endl;
      int ed = std::min(st + kNumScoresPerLine, n);
      os << "[" << (st + 1) << " : " << ed << "]:\t";
      for (int i = st; i < ed; ++i) {
        if (i != st) {
          os << "  ";
        }
        os << std::fixed << std::setprecision(4) << heap.heap[i].score;
      }
    }
    LOG(INFO) << "Scores of the best " << n << " candidates:" << os.str();
  */
  return results;
}

Array<tir::Trace> EvolutionarySearchNode::State::PickWithEpsGreedy(const Array<tir::Trace>& inits,
                                                                   const Array<tir::Trace>& bests) {
  int num_rands = self->num_trials_per_iter * self->eps_greedy;
  int num_bests = self->num_trials_per_iter - num_rands;
  std::vector<int> rands =
      tir::SampleWithoutReplacement(&self->rand_state_, inits.size(), inits.size());
  Array<tir::Trace> results;
  results.reserve(self->num_trials_per_iter);
  for (int i = 0, i_bests = 0, i_rands = 0; i < self->num_trials_per_iter; ++i) {
    bool has_best = i_bests < static_cast<int>(bests.size());
    bool has_rand = i_rands < static_cast<int>(rands.size());
    // Pick a schedule
    Optional<tir::Trace> trace{NullOpt};
    // If needs `bests`, then prefer `bests`
    if (i < num_bests) {
      if (has_best) {
        trace = bests[i_bests++];
      } else if (has_rand) {
        trace = inits[rands[i_rands++]];
      } else {
        break;
      }
    } else {
      // Else prefer `rands`
      if (has_rand) {
        trace = inits[rands[i_rands++]];
      } else if (has_best) {
        trace = bests[i_bests++];
      } else {
        break;
      }
    }
    results.push_back(trace.value());
  }
  return results;
}

inline Array<MeasureCandidate> EvolutionarySearchNode::State::AssembleCandidates(
    const Array<tir::Trace>& picks) {
  Array<MeasureCandidate> measure_inputs;
  measure_inputs.reserve(picks.size());
  for (const tir::Trace& pick : picks) {
    CachedTrace trace = self->_GetCachedTrace(pick);
    measure_inputs.push_back(MeasureCandidate(trace.sch, self->args_info_));
  }
  return measure_inputs;
}

inline Optional<Array<MeasureCandidate>>
EvolutionarySearchNode::State::GenerateMeasureCandidates() {
  if (st >= self->num_trials_total) {
    self->candidates = Array<MeasureCandidate>(nullptr);
    return NullOpt;
  }
  if (ed > self->num_trials_total) {
    self->num_trials_per_iter += self->num_trials_total - ed;
    ed = self->num_trials_total;
  }
  ICHECK_LT(st, ed);

  // new parts
  Array<tir::Trace> inits = SampleInitPopulation();
  Array<tir::Trace> bests = EvolveWithCostModel(inits);
  Array<tir::Trace> picks = PickWithEpsGreedy(inits, bests);
  self->candidates = AssembleCandidates(picks);
  return self->candidates;
}

inline void EvolutionarySearchNode::State::NotifyRunnerResults(const Array<RunnerResult>& results) {
  // We need to assume the candidates' order are not changed in runner.
  ICHECK(self->candidates.defined() && self->candidates.size() == results.size());
  st += results.size();
  ed += results.size();
  int i = 0;
  for (const RunnerResult& result : results) {
    // Todo: Update to database measure callback
    if (result->error_msg.defined() || !result->run_secs.defined()) continue;
    self->database->CommitTuningRecord(TuningRecord(
        /*trace=*/self->candidates[i]->sch->trace().value(),         //
        /*run_secs=*/result->run_secs.value(),                       //
        /*workload=*/self->database->CommitWorkload(self->mod_[0]),  //
        /*target=*/self->target_,                                    //
        /*args_info=*/self->candidates[i]->args_info));
    // Todo: Update to cost model measure callback
    self->cost_model->Update(self->tune_context_, self->candidates, results);
    i++;
  }
}

SearchStrategy SearchStrategy::EvolutionarySearch(int num_trials_per_iter,               //
                                                  int num_trials_total,                  //
                                                  int population,                        //
                                                  double init_measured_ratio,            //
                                                  int genetic_algo_iters,                //
                                                  double p_mutate,                       //
                                                  double eps_greedy,                     //
                                                  Map<Mutator, FloatImm> mutator_probs,  //
                                                  Database database,                     //
                                                  CostModel cost_model) {
  ObjectPtr<EvolutionarySearchNode> n = make_object<EvolutionarySearchNode>();
  n->num_trials_per_iter = num_trials_per_iter;
  n->num_trials_total = num_trials_total;
  n->population = population;
  n->init_measured_ratio = init_measured_ratio;
  n->genetic_algo_iters = genetic_algo_iters;
  n->p_mutate = p_mutate;
  n->eps_greedy = eps_greedy;
  n->mutator_probs = mutator_probs;
  n->database = database;
  n->cost_model = cost_model;
  return SearchStrategy(n);
}

TVM_REGISTER_NODE_TYPE(EvolutionarySearchNode);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategyEvolutionarySearch")
    .set_body_typed(SearchStrategy::EvolutionarySearch);

}  // namespace meta_schedule
}  // namespace tvm
