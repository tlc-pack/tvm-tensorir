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

#define TVM_META_SCHEDULE_CHECK_PROB_RANGE(p, name)                               \
  CHECK(0.0 <= (p) && (p) <= 1.0) << "ValueError: name should be within [0, 1], " \
                                  << "but get `" << #p << " = " << (p) << '\'';

namespace tvm {
namespace meta_schedule {

/**************** Data Structure ****************/

/*!
 * \brief The struct to store schedule, trace and its score.
 * \note The trace is available by visiting the schedule's trace method.
 */
struct CachedTrace {
  /*! \brief The schedule the trace creates. */
  tir::Schedule sch{nullptr};
  /*! \brief The normalized score, the higher the better. */
  double score;

  /*! \brief Default constructor. */
  CachedTrace() = default;
  /*!
   * \brief Constructor from Schedule and score.
   * \param sch The given Schedule, which can be used to obtain the trace.
   * \param score The predicted normalized score, -1.0 if score is not assigned yet.
   */
  explicit CachedTrace(const tir::Schedule& sch, double score) : sch(sch), score(score) {}
  /*! \brief Reload the operator < for CachedTrace. */
  friend inline bool operator<(const CachedTrace& lhs, const CachedTrace& rhs) {
    return lhs.score > rhs.score;
  }
};

/*!
 * \brief A heap with a size up-limit. If overflow happens, it evicted the worst items.
 * \note It maintains a min heap in terms of `CachedTrace::score`. Therefore, when
 * overflow happens, the element evicted is the one with the min `CachedTrace::score`.
 * As time goes, the elements in the heap are going to be larger.
 */
class SizedHeap {
 public:
  struct IRModuleSHash {
    IRModule mod;
    size_t shash;
  };

  struct IRModuleSHashHash {
    size_t operator()(const IRModuleSHash& hash) const { return hash.shash; }
  };

  struct IRModuleSHashEqual {
    bool operator()(const IRModuleSHash& lhs, const IRModuleSHash& rhs) const {
      return lhs.shash == rhs.shash && StructuralEqual()(lhs.mod, rhs.mod);
    }
  };
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
    if (!in_heap.insert(IRModuleSHash{item.sch->mod(), StructuralHash()(item.sch->mod())}).second) {
      return;
    }
    int size = heap.size();
    if (size < size_limit) {
      // Heap is not full, just push
      heap.emplace_back(item);
      std::push_heap(heap.begin(), heap.end());
    } else if (item < heap.front()) {
      // if the item is better than the worst one in the heap, we can safely kick it out
      std::pop_heap(heap.begin(), heap.end());
      heap.back() = item;
      std::push_heap(heap.begin(), heap.end());
    }
    // Otherwise, the item is worse than any other element in the heap
  }

  /*! \brief Up-limit of the heap size */
  int size_limit;
  /*! \brief The heap, the worse the topper */
  std::vector<CachedTrace> heap;
  /*! \brief The traces that are in the heap */
  std::unordered_set<IRModuleSHash, IRModuleSHashHash, IRModuleSHashEqual> in_heap;
};

struct PerThreadData {
  IRModule mod;
  TRandState rand_state;
  std::function<int32_t()> trace_sampler;
  std::function<Optional<Mutator>()> mutator_sampler;

  /*! \brief Default constructor. */
  PerThreadData() = default;
  explicit PerThreadData(const IRModule& mod, TRandState* rand_state)
      : mod(mod), rand_state(ForkSeed(rand_state)) {}

  /*!
   * \brief Create a sampler function that picks mutators according to the mass function
   * \param rand_state The random state for sampling
   * \return The sampler created
   */
  inline std::function<Optional<Mutator>()> MakeMutatorSampler(
      double p_mutate, const Map<Mutator, FloatImm>& mutator_probs,
      support::LinearCongruentialEngine::TRandState* rand_state) {
    std::vector<Optional<Mutator>> mutators;
    std::vector<double> masses;
    mutators.push_back(NullOpt);
    masses.push_back(1.0 - p_mutate);
    double total_mass_mutator = 0.0;
    if (p_mutate > 0) {
      for (const auto& kv : mutator_probs) {
        const Mutator& mutator = kv.first;
        double mass = kv.second->value;
        CHECK_GE(mass, 0.0) << "ValueError: Probability of mutator '" << mutator
                            << "' is ill-formed: " << mass;
        total_mass_mutator += mass;
        mutators.push_back(kv.first);
        masses.push_back(mass * p_mutate);
      }
    }
    // Normalize the sum to 1.0
    if (total_mass_mutator == 0.0) {
      masses[0] = 1.0;
      for (int i = 1, n = masses.size(); i < n; ++i) {
        masses[i] = 0.0;
      }
    } else if (total_mass_mutator != 1.0) {
      for (int i = 1, n = masses.size(); i < n; ++i) {
        masses[i] /= total_mass_mutator;
      }
    }
    return [idx_sampler = tir::MakeMultinomialSampler(rand_state, masses),
            mutators = std::move(mutators)]() -> Optional<Mutator> {
      int i = idx_sampler();
      return mutators[i];
    };
  }

  /*!
   * \brief Set the value for the trace and mutator samplers per thread.
   * \param scores The predicted score for the given samples.
   * \param p_mutate The probability of mutation.
   * \param mutator_probs The probability of each mutator as a dict.
   */
  void Set(const std::vector<double>& scores, double p_mutate,
           const Map<Mutator, FloatImm>& mutator_probs) {
    trace_sampler = tir::MakeMultinomialSampler(&rand_state, scores);
    mutator_sampler = MakeMutatorSampler(p_mutate, mutator_probs, &rand_state);
  }
};

struct ConcurrentBitmask {
  /*! The bit width. */
  static constexpr const int kBitWidth = 64;
  /*! \brief The size of the concurrent bitmask. */
  int size;
  /*! \brief The bitmasks. */
  std::vector<uint64_t> bitmask;
  /*! \brief The mutexes, one per kBitWidth(64 here) bitmasks. */
  std::vector<std::mutex> mutexes;

  /*!
   * \brief Constructor
   * \param n The total slots managed by the concurrent bitmask.
   */
  explicit ConcurrentBitmask(int n)
      : size((n + kBitWidth - 1) / kBitWidth), bitmask(size, 0), mutexes(size) {}
  /*!
   * \brief Query and mark the given index if not visited before.
   * \param x The index to concurrently check if used. If not, mark as used.
   * \return Whether the index has been used before.
   */
  bool QueryAndMark(int x) {
    std::unique_lock<std::mutex> lock(mutexes[x / kBitWidth]);
    constexpr uint64_t one = 1;
    if (bitmask[x / kBitWidth] & (one << (x % kBitWidth))) {
      return false;
    } else {
      bitmask[x / kBitWidth] |= one << (x % kBitWidth);
      return true;
    }
  }
};

/**************** Util Functions ****************/

/*!
 * \brief Assemble measure candidates from the given candidate traces.
 * \param traces The picked candidate traces.
 * \return The assembled measure candidates.
 */
inline Array<MeasureCandidate> AssembleCandidates(const std::vector<CachedTrace>& picks,
                                                  const Array<ArgInfo>& args_info) {
  Array<MeasureCandidate> measure_inputs;
  measure_inputs.reserve(picks.size());
  for (const CachedTrace& pick : picks) {
    measure_inputs.push_back(MeasureCandidate(pick.sch, args_info));
  }
  return measure_inputs;
}

/*!
 * \brief Predict the normalized score of each candidate.
 * \param candidates The candidates for prediction
 * \param task The search task
 * \param space The search space
 * \return The normalized score in the prediction
 */
inline std::vector<double> PredictNormalizedScore(const std::vector<CachedTrace>& cached_traces,
                                                  const TuneContext& tune_context,
                                                  const CostModel& cost_model,
                                                  const Array<ArgInfo>& args_info) {
  ICHECK(cached_traces.size() > 0)
      << "Candidates given for score prediction can not be empty list!";
  std::vector<double> scores =
      cost_model->Predict(tune_context, AssembleCandidates(cached_traces, args_info));
  // Normalize the score
  // TODO(@junrushao1994): use softmax + temperature to replace simple normalization to [0.0, +oo)
  for (double& score : scores) {
    score = std::max(0.0, score);
  }
  return scores;
}

/**************** Evolutionary Search ****************/

// TODO(@zxybazh): Early stopping for small search space, including deduplication.
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
 *
 */
class EvolutionarySearchNode : public SearchStrategyNode {
 public:
  /*! \brief The state of the search strategy. */
  struct State {
    /*! \brief The search strategy itself */
    EvolutionarySearchNode* self;
    /*! \brief The design spaces. Decisions are not used so traces only. */
    Array<tir::Trace> design_spaces;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int st;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int ed;

    explicit State(EvolutionarySearchNode* self, Array<tir::Trace> design_spaces)
        : self(self), design_spaces(design_spaces), st(0), ed(self->num_trials_per_iter) {}

    /*!
     * \brief Pick up best candidates from database.
     * \param num The number of traces to produce.
     * \return The picked best candidates.
     */
    inline std::vector<CachedTrace> PickBestFromDatabase(int num);
    /*!
     * \brief Sample the initial population from previous measured results and randomly generated
     *  traces via trace replaying.
     * \param num The number of traces to produce.
     * \return The initial population of traces sampled.
     */
    inline std::vector<CachedTrace> SampleInitPopulation(int num);
    /*!
     * \brief Pick final candidates from the given initial population and bests of evolved ones.
     * \param measured Measured samples from database.
     * \param unmeasured Unmeasured samples from replaying traces from design space.
     * \return The merged results, excluding undefined samples.
     */
    inline std::vector<CachedTrace> MergeSamples(const std::vector<CachedTrace>& measured,
                                                 const std::vector<CachedTrace>& unmeasured);
    /*!
     * \brief Evolve the initial population using mutators and samplers.
     * \param inits The initial population of traces sampled.
     * \param num The number of traces to produce.
     * \return The evolved traces from initial population.
     */
    inline std::vector<CachedTrace> EvolveWithCostModel(const std::vector<CachedTrace>& inits,
                                                        int num);
    /*!
     * \brief Pick final candidates from the given initial population and bests of evolved ones.
     * \param inits The initial population of traces sampled.
     * \param bests The best candidates predicted from evolved traces.
     * \param num The number of traces to produce.
     * \return The final picked candidates with a ratio of both.
     */
    inline std::vector<CachedTrace> PickWithEpsGreedy(const std::vector<CachedTrace>& inits,
                                                      const std::vector<CachedTrace>& bests,
                                                      int num);
    inline Optional<Array<MeasureCandidate>> GenerateMeasureCandidates();
    inline void NotifyRunnerResults(const TuneContext& tune_context,
                                    const Array<MeasureCandidate>& measure_candidates,
                                    const Array<RunnerResult>& results);
  };

  /*! \brief The tuning context of the evolutionary search strategy. */
  const TuneContextNode* tune_context_{nullptr};
  /*! \brief The target for the workload. */
  Target target_{nullptr};
  /*! \brief The metadata of the function arguments. */
  Array<ArgInfo> args_info_{nullptr};
  /*! \brief A Database for selecting useful candidates. */
  Database database_{nullptr};
  /*! \brief A cost model helping to explore the search space */
  CostModel cost_model_{nullptr};
  /*! \brief The postprocessors. */
  Array<Postproc> postprocs_{nullptr};
  /*! \brief Mutators and their probability mass */
  Map<Mutator, FloatImm> mutator_probs_{nullptr};
  /*! \brief The number of threads to use. To be initialized with TuneContext. */
  int num_threads_;
  /*! \brief The random state. To be initialized with TuneContext. */
  TRandState rand_state_;
  /*! \brief Pre thread data including module to be tuned and random state. */
  std::vector<PerThreadData> per_thread_data_;
  /*! \brief The state of the search strategy. */
  std::unique_ptr<State> state_ = nullptr;
  /*! \brief The token registered for the given workload in database. */
  Workload token_{nullptr};

  /*** Configuration: global ***/
  /*! \brief The number of trials per iteration. */
  int num_trials_per_iter;
  /*! \brief The number of total trials. */
  int num_trials_total;

  /*** Configuration: the initial population ***/
  /*! \brief The population size in the evolutionary search. */
  int population;
  /*! \brief The ratio of measured states used in the initial population */
  double init_measured_ratio;
  /*! \brief The maximum number to fail trace replaying. */
  int max_replay_fail_cnt;

  /*** Configuration: evolution ***/
  /*! \brief The number of iterations performed by generic algorithm. */
  int genetic_algo_iters;
  /*! \brief The maximum number to try evolving the given trace. */
  int max_evolve_fail_cnt;
  /*! \brief The probability to perform mutation */
  double p_mutate;

  /*** Configuration: pick states for measurement ***/
  /*! \brief The ratio of measurements to use randomly sampled states. */
  double eps_greedy;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `tune_context_` is not visited
    // `target_` is not visited
    // `args_info_` is not visited
    // `database` is not visited
    // `cost_model` is not visited
    // `postprocs` is not visited
    // `mutator_probs_` is not visited
    // `num_threads` is not visited
    // `rand_state_` is not visited
    // `per_thread_data_` is not visited
    // `state_` is not visited

    /*** Configuration: global ***/
    v->Visit("num_trials_total", &num_trials_total);
    v->Visit("num_trials_per_iter", &num_trials_per_iter);
    /*** Configuration: the initial population ***/
    v->Visit("population", &population);
    v->Visit("init_measured_ratio", &init_measured_ratio);
    v->Visit("max_replay_fail_cnt", &max_replay_fail_cnt);
    /*** Configuration: evolution ***/
    v->Visit("genetic_algo_iters", &genetic_algo_iters);
    v->Visit("max_evolve_fail_cnt", &max_evolve_fail_cnt);
    v->Visit("p_mutate", &p_mutate);
    /*** Configuration: pick states for measurement ***/
    v->Visit("eps_greedy", &eps_greedy);
  }

  static constexpr const char* _type_key = "meta_schedule.EvolutionarySearch";
  TVM_DECLARE_FINAL_OBJECT_INFO(EvolutionarySearchNode, SearchStrategyNode);

  void InitializeWithTuneContext(const TuneContext& tune_context) final {
    CHECK(tune_context.defined()) << "TuneContext must be defined!";
    CHECK(tune_context->num_threads > 0) << "Number of threads has to be larger than 0.";
    CHECK(tune_context->target.defined()) << "Target must be defined!";

    this->tune_context_ = tune_context.get();
    this->target_ = tune_context->target.value();
    this->args_info_ = ArgInfo::FromPrimFunc(FindEntryFunc(tune_context->mod.value()));
    this->mutator_probs_ = tune_context->mutator_probs;
    this->postprocs_ = tune_context->postprocs;
    this->num_threads_ = tune_context->num_threads;
    this->rand_state_ = ForkSeed(&tune_context->rand_state);
    this->cost_model_ = tune_context->task_scheduler->cost_model.value();
    this->database_ = tune_context->task_scheduler->database;
    this->token_ = this->database_->CommitWorkload(tune_context->mod.value());
    this->per_thread_data_.reserve(this->num_threads_);
    for (int i = 0; i < this->num_threads_; i++) {
      this->per_thread_data_.push_back(
          PerThreadData(DeepCopyIRModule(tune_context->mod.value()), &this->rand_state_));
    }
    this->state_.reset();
  }

  void PreTuning(const Array<tir::Schedule>& design_spaces) final {
    ICHECK(!design_spaces.empty());
    ICHECK(this->state_ == nullptr);
    // Change to traces
    Array<tir::Trace> design_space_traces;
    design_space_traces.reserve(design_spaces.size());
    for (const tir::Schedule& space : design_spaces) {
      design_space_traces.push_back(space->trace().value()->Simplified(true));
    }
    this->state_ = std::make_unique<State>(this, design_space_traces);
  }

  void PostTuning() final {
    ICHECK(this->state_ != nullptr);
    this->state_.reset();
  }

  Optional<Array<MeasureCandidate>> GenerateMeasureCandidates() final {
    ICHECK(this->state_ != nullptr);
    return this->state_->GenerateMeasureCandidates();
  }

  void NotifyRunnerResults(const TuneContext& tune_context,
                           const Array<MeasureCandidate>& measure_candidates,
                           const Array<RunnerResult>& results) final {
    ICHECK(this->state_ != nullptr);
    this->state_->NotifyRunnerResults(tune_context, measure_candidates, results);
  }
};

inline std::vector<CachedTrace> EvolutionarySearchNode::State::PickBestFromDatabase(int num) {
  std::vector<tir::Trace> measured_traces;
  measured_traces.reserve(num);
  Array<TuningRecord> top_records = self->database_->GetTopK(self->token_, num);
  for (TuningRecord record : top_records) {
    measured_traces.push_back(record->trace);
  }
  int actual_num = measured_traces.size();
  std::vector<CachedTrace> results(actual_num);
  auto f_proc_measured = [this, &measured_traces, &results](int thread_id, int trace_id) -> void {
    TRandState& rand_state = self->per_thread_data_[thread_id].rand_state;
    const IRModule& mod = self->per_thread_data_[thread_id].mod;
    tir::Trace trace = measured_traces[trace_id];
    if (Optional<tir::Schedule> opt_sch =
            meta_schedule::ApplyTrace(mod, trace, &rand_state, self->postprocs_)) {
      tir::Schedule sch = opt_sch.value();
      results[trace_id] = CachedTrace(sch, -1.0);
    } else {
      LOG(FATAL) << "ValueError: Cannot postprocess the trace:\n" << trace;
      throw;
    }
  };
  support::parallel_for_dynamic(0, actual_num, self->num_threads_, f_proc_measured);
  return results;
}

inline std::vector<CachedTrace> EvolutionarySearchNode::State::SampleInitPopulation(int num) {
  // Pick unmeasured states
  std::vector<CachedTrace> results(num);
  auto f_proc_unmeasured = [this, &results](int thread_id, int trace_id) -> void {
    TRandState& rand_state = self->per_thread_data_[thread_id].rand_state;
    const IRModule& mod = self->per_thread_data_[thread_id].mod;
    CachedTrace& result = results[trace_id];
    for (int fail_ct = 0; fail_ct < self->max_replay_fail_cnt; fail_ct++) {
      int design_space_index = tir::SampleInt(&rand_state, 0, design_spaces.size());
      tir::Trace trace = design_spaces[design_space_index];
      if (Optional<tir::Schedule> opt_sch =
              // replay trace, i.e., remove decisions
          ApplyTrace(mod, tir::Trace(trace->insts, {}), &rand_state, self->postprocs_)) {
        tir::Schedule sch = opt_sch.value();
        result = CachedTrace(sch, -1.0);
        break;
      }
    }
    if (!result.sch.defined()) {
      LOG(FATAL) << "Sample-Init-Population failed over the maximum limit!";
    }
  };
  support::parallel_for_dynamic(0, num, self->num_threads_, f_proc_unmeasured);
  return results;
}

inline std::vector<CachedTrace> EvolutionarySearchNode::State::MergeSamples(
    const std::vector<CachedTrace>& measured, const std::vector<CachedTrace>& unmeasured) {
  ICHECK(measured.size() + unmeasured.size() == self->population)
      << "Num of total init samples does not equal to population size!";
  std::vector<CachedTrace> inits;
  inits.reserve(self->population);
  inits.insert(inits.end(), measured.begin(), measured.end());
  inits.insert(inits.end(), unmeasured.begin(), unmeasured.end());
  return inits;
}

std::vector<CachedTrace> EvolutionarySearchNode::State::EvolveWithCostModel(
    const std::vector<CachedTrace>& inits, int num) {
  // The heap to record best schedule, we do not consider schedules that are already measured
  // Also we use `in_heap` to make sure items in the heap are de-duplicated
  SizedHeap heap(num);

  // Prepare search queues
  std::vector<CachedTrace> sch_curr;
  std::vector<CachedTrace> sch_next;
  sch_curr.reserve(self->population);
  sch_next.reserve(self->population);
  for (const CachedTrace& ctrace : inits) {
    sch_curr.push_back(ctrace);
  }
  // Main loop: (genetic_algo_iters + 1) times
  for (int iter = 0;; ++iter) {
    // Predict normalized score with the cost model,
    std::vector<double> scores = PredictNormalizedScore(
        sch_curr, GetRef<TuneContext>(self->tune_context_), self->cost_model_, self->args_info_);
    for (int i = 0, n = sch_curr.size(); i < n; ++i) {
      CachedTrace& entry = sch_curr[i];
      entry.score = scores[i];
      if (!self->database_->HasWorkload(entry.sch->mod())) {
        heap.Push(entry);
      }
    }
    // Discontinue once it reaches end of search
    if (iter == self->genetic_algo_iters) {
      break;
    }
    // Set threaded samplers, with probability from predicated normalized throughputs
    for (int i = 0; i < self->num_threads_; ++i) {
      self->per_thread_data_[i].Set(scores, self->p_mutate, self->mutator_probs_);
    }
    ConcurrentBitmask cbmask(scores.size());
    // The worker function
    auto f_find_candidate = [&cbmask, &sch_curr, &sch_next, this](int thread_id, int trace_id) {
      // Prepare samplers
      TRandState& rand_state = self->per_thread_data_[thread_id].rand_state;
      const IRModule& mod = self->per_thread_data_[thread_id].mod;
      const std::function<int()>& trace_sampler = self->per_thread_data_[thread_id].trace_sampler;
      const std::function<Optional<Mutator>()>& mutator_sampler =
          self->per_thread_data_[thread_id].mutator_sampler;
      CachedTrace& result = sch_next[trace_id];
      // Loop until success
      for (int retry_cnt = 0; retry_cnt < self->max_evolve_fail_cnt; retry_cnt++) {
        int sampled_trace_id = trace_sampler();
        const CachedTrace& ctrace = sch_curr[sampled_trace_id];
        if (Optional<Mutator> opt_mutator = mutator_sampler()) {
          // Decision: mutate
          Mutator mutator = opt_mutator.value();
          if (Optional<tir::Trace> opt_new_trace =
                  mutator->Apply(ctrace.sch->trace().value(), &rand_state)) {
            tir::Trace new_trace = opt_new_trace.value();
            if (Optional<tir::Schedule> opt_sch =
                    ApplyTrace(mod, new_trace, &rand_state, self->postprocs_)) {
              // note that sch's trace is different from new_trace
              // because it contains post-processing information
              result = CachedTrace(opt_sch.value(), -1.0);
              break;
            }
          }
        } else if (cbmask.QueryAndMark(sampled_trace_id)) {
          // Decision: do not mutate
          result = ctrace;
          break;
        }
        // if retry count exceeds the limit, the result should be just ctrace
        if (retry_cnt + 1 == self->max_evolve_fail_cnt) {
          sch_next[trace_id] = ctrace;
        }
      }
    };
    sch_next.clear();
    sch_next.resize(self->population);
    support::parallel_for_dynamic(0, self->population, self->num_threads_, f_find_candidate);
    sch_curr.clear();
    sch_curr.swap(sch_next);
  }
  // Return the best states from the heap, sorting from higher score to lower ones
  std::sort(heap.heap.begin(), heap.heap.end());
  std::vector<CachedTrace> results;
  results.reserve(num);
  for (const CachedTrace& item : heap.heap) {
    results.push_back(item);
  }

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
  return results;
}

std::vector<CachedTrace> EvolutionarySearchNode::State::PickWithEpsGreedy(
    const std::vector<CachedTrace>& unmeasured, const std::vector<CachedTrace>& bests, int num) {
  int num_rands = num * self->eps_greedy;
  int num_bests = num - num_rands;
  std::vector<int> rands =
      tir::SampleWithoutReplacement(&self->rand_state_, unmeasured.size(), unmeasured.size());
  std::vector<CachedTrace> results;
  results.reserve(num);
  for (int i = 0, i_bests = 0, i_rands = 0; i < num; ++i) {
    bool has_best = i_bests < static_cast<int>(bests.size());
    bool has_rand = i_rands < static_cast<int>(rands.size());
    // Pick a schedule
    CachedTrace ctrace;
    // If needs `bests`, then prefer `bests`
    if (i < num_bests) {
      if (has_best) {
        ctrace = bests[i_bests++];
      } else if (has_rand) {
        ctrace = unmeasured[rands[i_rands++]];
      } else {
        break;
      }
    } else {
      // Else prefer `rands`
      if (has_rand) {
        ctrace = unmeasured[rands[i_rands++]];
      } else if (has_best) {
        ctrace = bests[i_bests++];
      } else {
        break;
      }
    }
    results.push_back(ctrace);
  }
  return results;
}

inline Optional<Array<MeasureCandidate>>
EvolutionarySearchNode::State::GenerateMeasureCandidates() {
  if (st >= self->num_trials_total) {
    return NullOpt;
  }
  int sample_num = self->num_trials_per_iter;
  if (ed > self->num_trials_total) {
    sample_num = self->num_trials_total - st;
    ed = self->num_trials_total;
  }
  ICHECK_LT(st, ed);

  std::vector<CachedTrace> measured =
      PickBestFromDatabase(self->population * self->init_measured_ratio);
  std::vector<CachedTrace> unmeasured = SampleInitPopulation(self->population - measured.size());
  std::vector<CachedTrace> inits = MergeSamples(measured, unmeasured);
  std::vector<CachedTrace> bests = EvolveWithCostModel(inits, sample_num);
  std::vector<CachedTrace> picks = PickWithEpsGreedy(unmeasured, bests, sample_num);
  return AssembleCandidates(picks, self->args_info_);
}

inline void EvolutionarySearchNode::State::NotifyRunnerResults(
    const TuneContext& tune_context, const Array<MeasureCandidate>& measure_candidates,
    const Array<RunnerResult>& results) {
  st += results.size();
  ed += results.size();
}

SearchStrategy SearchStrategy::EvolutionarySearch(int num_trials_per_iter,     //
                                                  int num_trials_total,        //
                                                  int population,              //
                                                  int max_replay_fail_cnt,     //
                                                  double init_measured_ratio,  //
                                                  int genetic_algo_iters,      //
                                                  int max_evolve_fail_cnt,     //
                                                  double p_mutate,             //
                                                  double eps_greedy) {
  ObjectPtr<EvolutionarySearchNode> n = make_object<EvolutionarySearchNode>();
  n->num_trials_per_iter = num_trials_per_iter;
  n->num_trials_total = num_trials_total;
  n->population = population;
  n->max_replay_fail_cnt = max_replay_fail_cnt;
  TVM_META_SCHEDULE_CHECK_PROB_RANGE(init_measured_ratio, "Initial measured ratio");
  n->init_measured_ratio = init_measured_ratio;
  n->genetic_algo_iters = genetic_algo_iters;
  n->max_evolve_fail_cnt = max_evolve_fail_cnt;
  TVM_META_SCHEDULE_CHECK_PROB_RANGE(p_mutate, "Mutation probability");
  n->p_mutate = p_mutate;
  TVM_META_SCHEDULE_CHECK_PROB_RANGE(eps_greedy, "Greedy pick probability");
  n->eps_greedy = eps_greedy;
  return SearchStrategy(n);
}

TVM_REGISTER_NODE_TYPE(EvolutionarySearchNode);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategyEvolutionarySearch")
    .set_body_typed(SearchStrategy::EvolutionarySearch);

}  // namespace meta_schedule
}  // namespace tvm
