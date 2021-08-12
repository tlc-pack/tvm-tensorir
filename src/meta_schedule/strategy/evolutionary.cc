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
#include <tvm/node/serialization.h>
#include <tvm/support/parallel_for.h>

#include <mutex>   // NOLINT(build/c++11)
#include <thread>  // NOLINT(build/c++11)

#include "../cost_model.h"
#include "../database.h"
#include "../measure.h"
#include "../search.h"
#include "../utils.h"
#include "./mutator.h"

namespace tvm {
namespace meta_schedule {

/*
 * TODO(@junrushao1994): Items left undone
 * 1) early stopping if the best schedule was obtained long long ago
 * 2) check if space is fully explored
 * 3) check if the failure count of mutations and stop
 */

/********** Evolutionary **********/

/*! \brief The postprocessed built of a trace */
struct CachedTrace {
  /*! \brief The trace */
  const tir::TraceNode* trace;
  /*! \brief The schedule the trace creates */
  Schedule sch;
  /*! \brief The string representation of the schedule */
  String repr;
  /*! \brief The normalized throughput, the higher the better */
  double throughput;

  static bool Compare(const CachedTrace& lhs, const CachedTrace& rhs) {
    return lhs.throughput > rhs.throughput;
  }
};

/*!
 * \brief Evolutionary Search
 *
 * The algorithm:
 *
 * Loop until #measured >= total_measures:
 *   init =
 *      pick top `k = population *      init_measured_ratio ` from measured
 *      pick     `k = population * (1 - init_measured_ratio)` from random support
 *   best = generate `population` states with the cost model,
 *          starting from `init`,
 *          using mutators,
 *          and return the top-n states during the search,
 *          where `n = num_measures_per_iter`
 *   chosen = pick top `k = num_measures_per_iter * (1 - eps_greedy)` from `best`
 *            pick     `k = num_measures_per_iter *      eps_greedy ` from `init`
 *   do the measurement on `chosen` & update the cost model
 */
class EvolutionaryNode : public SearchStrategyNode {
 public:
  /*** Configuration: global ***/
  /*! \brief The maximum number of measurements performed by genetic algorithm */
  int total_measures;
  /*! \brief The number of measures to be performed in each iteration */
  int num_measures_per_iteration;
  /*! \brief The population size in the evolutionary search */
  int population;
  /*! \brief A table storing all states that have been measured */
  Database database;

  /*** Configuration: the initial population ***/
  /*! \brief The ratio of measured states used in the initial population */
  double init_measured_ratio;

  /*** Configuration: evolution ***/
  /*! \brief The number of iterations performed by generic algorithm. */
  int genetic_algo_iters;
  /*! \brief The probability to perform mutation */
  double p_mutate;
  /*! \brief Mutators and their probability mass */
  Map<Mutator, FloatImm> mutator_probs;
  /*! \brief A cost model helping to explore the search space */
  CostModel cost_model;

  /*** Configuration: pick states for measurement ***/
  /*! \brief The ratio of measurements to use randomly sampled states. */
  double eps_greedy;

  /*** Helpers ***/
  mutable std::unordered_map<Trace, CachedTrace, ObjectPtrHash, ObjectPtrEqual> trace_cache_;
  mutable std::mutex trace_cache_mutex_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    /*** Configuration: global ***/
    v->Visit("total_measures", &total_measures);
    v->Visit("num_measures_per_iteration", &num_measures_per_iteration);
    v->Visit("population", &population);
    v->Visit("database", &database);
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
   * \brief Explore the search space and find the best schedule
   * \param task The search task
   * \param space The search space
   * \param measurer The measurer that builds, runs and profiles sampled programs
   * \param rand_state The random state for sampling
   * \param verbose Whether or not in verbose mode
   * \return The best schedule found, NullOpt if no valid schedule is found
   */
  Optional<Schedule> Search(const SearchTask& task, const SearchSpace& space,
                            const ProgramMeasurer& measurer, tir::TRandState* rand_state,
                            int verbose) override;

  /********** Stages in evolutionary search **********/

  /*!
   * \brief Sample the initial population from both the support and previously measured traces
   * pick top `k = population *      init_measured_ratio ` from measured
   * pick     `k = population * (1 - init_measured_ratio)` from random support
   * \param support The support to be sampled from
   * \param task The search task
   * \param space The search space
   * \param rand_state The random state for sampling
   * \return The generated samples, all of which are not post-processed
   */
  Array<Trace> SampleInitPopulation(const Array<Schedule>& support, const SearchTask& task,
                                    const SearchSpace& space, tir::TRandState* rand_state);

  /*!
   * \brief Perform evolutionary search using genetic algorithm with the cost model
   * \param inits The initial population
   * \param task The search task
   * \param space The search space
   * \param rand_state The random state for sampling
   * \return An array of schedules, the sampling result
   */
  Array<Trace> EvolveWithCostModel(const Array<Trace>& inits, const SearchTask& task,
                                   const SearchSpace& space, tir::TRandState* rand_state);

  /*!
   * \brief Pick a batch of samples for measurement with epsilon greedy
   * \param inits The initial population used when picking random states
   * \param bests The best populations according to the cost model when picking top states
   * \param task The search task
   * \param space The search space
   * \param rand_state The random state for sampling
   * \return A list of schedules, result of epsilon-greedy sampling
   */
  Array<Trace> PickWithEpsGreedy(const Array<Trace>& inits, const Array<Trace>& bests,
                                 const SearchTask& task, const SearchSpace& space,
                                 tir::TRandState* rand_state);

  /*!
   * \brief Make measurements and update the cost model
   * \param task The search task
   * \param schedules The schedules to be measured
   * \param measurer The measurer
   * \param verbose A boolean flag for verbosity
   * \return A list of MeasureResult for measurements
   */
  Array<MeasureResult> MeasureAndUpdateCostModel(const SearchTask& task, const Array<Trace>& picks,
                                                 const ProgramMeasurer& measurer, int verbose);

  static constexpr const char* _type_key = "meta_schedule.Evolutionary";
  TVM_DECLARE_FINAL_OBJECT_INFO(EvolutionaryNode, SearchStrategyNode);

 private:
  // Helper functions
  friend class Evolutionary;

  /*!
   * \brief Fork a random state into `n` random states
   * \param n The number of random states to be forked
   * \param rand_state The random state for sampling
   * \return A list of random states, the result of forking
   */
  static std::vector<tir::TRandState> ForkRandStates(int n, tir::TRandState* rand_state) {
    std::vector<tir::TRandState> result;
    result.reserve(n);
    for (int i = 0; i < n; ++i) {
      result.emplace_back(tir::ForkSeed(rand_state));
    }
    return result;
  }

  static std::vector<tir::PrimFunc> ForkWorkload(int n, const tir::PrimFunc& workload) {
    std::vector<tir::PrimFunc> result;
    result.reserve(n);
    for (int i = 0; i < n; i++) {
      auto deep_copy = Downcast<tir::PrimFunc>(LoadJSON(SaveJSON(workload)));
      result.push_back(deep_copy);
    }
    return result;
  }

  /*!
   * \brief Replay the trace and do postprocessing
   */
  static Optional<Schedule> ReplayTrace(const Trace& trace, const SearchTask& task,
                                        const SearchSpace& space, tir::TRandState* rand_state,
                                        const tir::PrimFunc& workload) {
    Schedule sch = Schedule::Traced(/*mod=*/IRModule({{GlobalVar("main"), workload}}),
                                    /*seed=*/tir::ForkSeed(rand_state),
                                    /*debug_mode=*/false,
                                    /*error_render_level=*/tir::ScheduleErrorRenderLevel::kDetail);
    trace->ApplyToSchedule(sch, /*remove_postproc=*/true);
    if (!space->Postprocess(task, sch, rand_state)) {
      return NullOpt;
    }
    return sch;
  }

  /*!
   * \brief Create a sampler function that picks mutators according to the mass function
   * \param rand_state The random state for sampling
   * \return The sampler created
   */
  static std::function<Optional<Mutator>()> MakeMutatorSampler(
      double p_mutate, const Map<Mutator, FloatImm>& mutator_probs, tir::TRandState* rand_state) {
    CHECK(0.0 <= p_mutate && p_mutate <= 1.0)  //
        << "ValueError: Probability should be within [0, 1], "
        << "but get `p_mutate = " << p_mutate << '\'';
    std::vector<Optional<Mutator>> mutators;
    std::vector<double> masses;
    mutators.push_back(NullOpt);
    masses.push_back(1.0 - p_mutate);
    double total_mass_mutator = 0.0;
    for (const auto& kv : mutator_probs) {
      const Mutator& mutator = kv.first;
      double mass = kv.second->value;
      CHECK_GE(mass, 0.0) << "ValueError: Probability of mutator '" << mutator->name
                          << "' is ill-formed: " << mass;
      total_mass_mutator += mass;
      mutators.push_back(kv.first);
      masses.push_back(mass * p_mutate);
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
    auto idx_sampler = tir::MakeMultinomial(rand_state, masses);
    return [idx_sampler = std::move(idx_sampler),
            mutators = std::move(mutators)]() -> Optional<Mutator> {
      int i = idx_sampler();
      return mutators[i];
    };
  }

  /*!
   * \brief Add the cached trace into the trace_cache_
   * \param cached_trace The cached_trace to be added
   */
  void AddCachedTrace(const CachedTrace& cached_trace) const {
    std::unique_lock<std::mutex> lock(this->trace_cache_mutex_);
    trace_cache_.emplace(GetRef<Trace>(cached_trace.trace), cached_trace);
  }

  /*!
   * \brief Retrieve the cached trace given the trace
   * \param trace The trace to be retrieved
   * \return The cached trace
   */
  CachedTrace GetCachedTrace(const Trace& trace) const {
    auto iter = trace_cache_.find(trace);
    ICHECK(iter != trace_cache_.end());
    return iter->second;
  }

  /*!
   * \brief Predict the normalized throughput of each candidate.
   * \param candidates The candidates for prediction
   * \param task The search task
   * \param space The search space
   * \return The normalized throughput in the prediction
   */
  std::vector<double> PredictNormalizedThroughput(const std::vector<CachedTrace>& candidates,
                                                  const SearchTask& task) const {
    Array<Schedule> schs;
    schs.reserve(candidates.size());
    for (const CachedTrace& entry : candidates) {
      schs.push_back(entry.sch);
    }
    std::vector<double> scores = cost_model->Predict(task, schs);
    // Normalize the score
    // TODO(@junrushao1994): use softmax + temperature to replace simple normalization to [0.0, +oo)
    for (double& score : scores) {
      score = std::max(0.0, score);
    }
    return scores;
  }
};

/*!
 * \brief Managed refernce to EvolutionaryNode
 * \sa EvolutionaryNode
 */
class Evolutionary : public SearchStrategy {
 public:
  /*!
   * \brief Constructor
   * \param total_measures The maximum number of measurements performed by genetic algorithm
   * \param num_measures_per_iteration The number of measures to be performed in each iteration
   * \param population The population size for evolutionary search
   * \param init_measured_ratio The ratio of measured states used in the initial population
   * \param genetic_algo_iters The number of iterations performed by generic algorithm
   * \param p_mutate The probability to perform mutation
   * \param mutator_probs Mutators and their probability mass
   * \param cost_model A cost model helping to explore the search space
   * \param eps_greedy The percentage of measurements to use randomly sampled states
   */
  explicit Evolutionary(
      /*** Configuration: global ***/
      int total_measures, int num_measures_per_iteration, int population,
      /*** Configuration: the initial population ***/
      double init_measured_ratio,
      /*** Configuration: evolution ***/
      int genetic_algo_iters, double p_mutate, Map<Mutator, FloatImm> mutator_probs,
      CostModel cost_model,
      /*** Configuration: pick states for measurement ***/
      double eps_greedy);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Evolutionary, SearchStrategy, EvolutionaryNode);
};

/**************** Data Structure ****************/

/*!
 * \brief A heap with a size up-limit. If overflow happens, it evicted the worst items.
 * \note It maintains a min heap in terms of `CachedTrace::throughput`. Therefore, when
 * overflow happens, the element evicted is the one with the min `CachedTrace::throughput`.
 * As time goes, the elements in the heap are going to be larger.
 */
class SizedHeap {
  /*! \brief The comparator class, used by `std::push_heap` and `std::pop_heap` */
  struct Comparator {
    bool operator()(const CachedTrace& a, const CachedTrace& b) const {
      return a.throughput > b.throughput;
    }
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

/********** Constructor **********/

Evolutionary::Evolutionary(int total_measures, int num_measures_per_iteration, int population,
                           double init_measured_ratio, int genetic_algo_iters, double p_mutate,
                           Map<Mutator, FloatImm> mutator_probs, CostModel cost_model,
                           double eps_greedy) {
  // Doing some sanity checks
  CHECK_LE(num_measures_per_iteration, population)
      << "ValueError: requires `num_measures_per_iteration <= population`";
  {
    tir::TRandState rand_state = 42;
    EvolutionaryNode::MakeMutatorSampler(p_mutate, mutator_probs, &rand_state);
  }
  ObjectPtr<EvolutionaryNode> n = make_object<EvolutionaryNode>();
  n->total_measures = total_measures;
  n->num_measures_per_iteration = num_measures_per_iteration;
  n->population = population;
  n->database = InMemoryDB(NullOpt);
  n->init_measured_ratio = init_measured_ratio;
  n->genetic_algo_iters = genetic_algo_iters;
  n->p_mutate = p_mutate;
  n->mutator_probs = std::move(mutator_probs);
  n->cost_model = std::move(cost_model);
  n->eps_greedy = eps_greedy;
  data_ = std::move(n);
}

/********** Search **********/

Optional<Schedule> EvolutionaryNode::Search(const SearchTask& task, const SearchSpace& space,
                                            const ProgramMeasurer& measurer,
                                            tir::TRandState* rand_state, int verbose) {
  Array<Schedule> support = space->GetSupport(task, rand_state);
  int iter = 1;
  for (int num_measured = 0; num_measured < this->total_measures; ++iter) {
    LOG(INFO) << "Evolutionary search: Iteration #" << iter << " | Measured: " << num_measured
              << "/" << this->total_measures;
    // `inits`: Sampled initial population, whose size is at most `this->population`
    LOG(INFO) << "Sampling initial population...";
    Array<Trace> inits = SampleInitPopulation(support, task, space, rand_state);
    LOG(INFO) << "Initial population size: " << inits.size();
    // `bests`: The best schedules according to the cost mode when explore the space using mutators
    LOG(INFO) << "Evolving...";
    Array<Trace> bests = EvolveWithCostModel(inits, task, space, rand_state);
    // Pick candidates with eps greedy
    LOG(INFO) << "Picking with epsilon greedy where epsilon = " << eps_greedy;
    Array<Trace> picks = PickWithEpsGreedy(inits, bests, task, space, rand_state);
    // Run measurement, update cost model
    LOG(INFO) << "Sending " << picks.size() << " samples for measurement";
    Array<MeasureResult> results = MeasureAndUpdateCostModel(task, picks, measurer, verbose);
    num_measured += results.size();
  }
  return measurer->GetBest(task);
}

Array<Trace> EvolutionaryNode::SampleInitPopulation(const Array<Schedule>& support,
                                                    const SearchTask& task,
                                                    const SearchSpace& space,
                                                    tir::TRandState* global_rand_state) {
  trace_cache_.clear();
  std::vector<Trace> results;
  results.reserve(this->population);
  // Threading RNG
  int num_threads = std::thread::hardware_concurrency();
  std::vector<tir::TRandState> thread_rand_states = ForkRandStates(num_threads, global_rand_state);
  std::vector<tir::PrimFunc> thread_workloads = ForkWorkload(num_threads, task->workload);
  // Pick measured states
  int num_measured = this->population * this->init_measured_ratio;
  for (const Database::Entry& entry : database->GetTopK(num_measured, task)) {
    results.push_back(entry.trace.value());
  }
  auto f_proc_measured = [this, &results, &thread_rand_states, &task, &space, thread_workloads](
                             int thread_id, int i) -> void {
    tir::TRandState* rand_state = &thread_rand_states[thread_id];
    const Trace& trace = results[i];
    if (Optional<Schedule> opt_sch =
            ReplayTrace(trace, task, space, rand_state, thread_workloads[thread_id])) {
      Schedule sch = opt_sch.value();
      this->AddCachedTrace(CachedTrace{trace.get(), sch, Repr(sch), -1.0});
    } else {
      LOG(FATAL) << "ValueError: Cannot postprocess the trace:\n" << trace;
      throw;
    }
  };
  support::parallel_persist_for(0, results.size(), f_proc_measured);
  // Pick unmeasured states
  std::atomic<int> tot_fail_ct(0);
  std::atomic<int> success_ct(0);
  auto f_proc_unmeasured = [this, &results, &thread_rand_states, &tot_fail_ct, &task, &space,
                            &support, &success_ct, thread_workloads](int thread_id, int i) -> void {
    tir::TRandState* rand_state = &thread_rand_states[thread_id];
    for (;;) {
      Trace support_trace = support[tir::SampleInt(rand_state, 0, support.size())]->trace().value();
      Map<Instruction, ObjectRef> decisions;
      try {
        if (Optional<Schedule> opt_sch =
                ReplayTrace(Trace(support_trace->insts, decisions), task, space, rand_state,
                            thread_workloads[thread_id])) {
          Schedule sch = opt_sch.value();
          Trace old_trace = sch->trace().value();
          Trace trace(old_trace->insts, old_trace->decisions);
          this->AddCachedTrace(CachedTrace{trace.get(), sch, Repr(sch), -1.0});
          results[i] = std::move(trace);
          success_ct++;
          break;
        } else {
          tot_fail_ct++;
        }
      } catch (const dmlc::Error& e) {
        tot_fail_ct++;
      }
      if (success_ct > 64) {
        break;
      }
    }
  };
  num_measured = results.size();
  results.resize(this->population, Trace(nullptr));
  support::parallel_persist_for(num_measured, this->population, f_proc_unmeasured);
  std::vector<Trace> pruned_results;
  for (const auto& result : results) {
    if (result.defined()) {
      pruned_results.push_back(result);
    }
  }
  LOG(INFO) << "fail count: " << tot_fail_ct;
  return pruned_results;
}

Array<Trace> EvolutionaryNode::EvolveWithCostModel(const Array<Trace>& inits,
                                                   const SearchTask& task, const SearchSpace& space,
                                                   tir::TRandState* global_rand_state) {
  // The heap to record best schedule, we do not consider schedules that are already measured
  // Also we use `in_heap` to make sure items in the heap are de-duplicated
  SizedHeap heap(this->num_measures_per_iteration);
  // Threading RNG
  int num_threads = std::thread::hardware_concurrency();
  std::vector<tir::TRandState> thread_rand_states = ForkRandStates(num_threads, global_rand_state);
  std::vector<tir::PrimFunc> thread_workloads = ForkWorkload(num_threads, task->workload);
  std::vector<std::function<int()>> thread_trace_samplers(num_threads);
  std::vector<std::function<Optional<Mutator>()>> thread_mutator_samplers(num_threads);
  std::vector<int> trace_used;
  std::mutex trace_used_mutex;
  auto f_set_sampler = [this, num_threads, &thread_rand_states, &thread_trace_samplers,
                        &thread_mutator_samplers, &trace_used](const std::vector<double>& scores) {
    for (int i = 0; i < num_threads; ++i) {
      tir::TRandState* rand_state = &thread_rand_states[i];
      thread_trace_samplers[i] = tir::MakeMultinomial(rand_state, scores);
      thread_mutator_samplers[i] =
          MakeMutatorSampler(this->p_mutate, this->mutator_probs, rand_state);
    }
    trace_used = std::vector<int>(scores.size(), 0);
  };
  // Prepare search queues
  std::vector<CachedTrace> sch_curr;
  std::vector<CachedTrace> sch_next;
  sch_curr.reserve(this->population);
  sch_next.reserve(this->population);
  for (const Trace& trace : inits) {
    sch_curr.push_back(GetCachedTrace(trace));
  }
  // Main loop: (genetic_algo_iters + 1) times
  for (int iter = 0;; ++iter) {
    // Predict running time with the cost model,
    // and put the schedules with the predicted perf to the heap
    std::vector<double> scores = this->PredictNormalizedThroughput(sch_curr, task);
    for (int i = 0, n = sch_curr.size(); i < n; ++i) {
      CachedTrace& entry = sch_curr[i];
      entry.throughput = scores[i];
      if (!database->Has(entry.repr, task)) {
        heap.Push(entry);
      }
    }
    // Discontinue once it reaches end of search
    if (iter == genetic_algo_iters) {
      break;
    }
    // Set threaded samplers, with probability from predicated normalized throughputs
    f_set_sampler(scores);
    // The worker function
    auto f_find_candidate = [&thread_rand_states, &thread_trace_samplers, &thread_mutator_samplers,
                             &trace_used, &trace_used_mutex, &sch_curr, &sch_next, &task, &space,
                             thread_workloads, this](int thread_id, int i) {
      // Prepare samplers
      tir::TRandState* rand_state = &thread_rand_states[thread_id];
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
          if (Optional<Trace> opt_new_trace =
                  mutator->Apply(task, GetRef<Trace>(cached_trace.trace), rand_state)) {
            Trace new_trace = opt_new_trace.value();
            if (Optional<Schedule> opt_sch =
                    ReplayTrace(new_trace, task, space, rand_state, thread_workloads[thread_id])) {
              Schedule sch = opt_sch.value();
              CachedTrace new_cached_trace{new_trace.get(), sch, Repr(sch), -1.0};
              this->AddCachedTrace(new_cached_trace);
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
    sch_next.resize(this->population);
    support::parallel_persist_for(0, this->population, f_find_candidate);
    sch_curr.clear();
    sch_curr.swap(sch_next);
  }
  // Return the best states from the heap, sorting from higher throughput to lower ones
  std::sort(heap.heap.begin(), heap.heap.end(), CachedTrace::Compare);
  Array<Trace> results;
  results.reserve(this->num_measures_per_iteration);
  for (const CachedTrace& item : heap.heap) {
    results.push_back(GetRef<Trace>(item.trace));
  }
  {
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
        os << std::fixed << std::setprecision(4) << heap.heap[i].throughput;
      }
    }
    LOG(INFO) << "Scores of the best " << n << " candidates:" << os.str();
  }
  return results;
}

Array<Trace> EvolutionaryNode::PickWithEpsGreedy(const Array<Trace>& inits,
                                                 const Array<Trace>& bests, const SearchTask& task,
                                                 const SearchSpace& space,
                                                 tir::TRandState* rand_state) {
  int num_rands = this->num_measures_per_iteration * this->eps_greedy;
  int num_bests = this->num_measures_per_iteration - num_rands;
  std::vector<int> rands = tir::SampleWithoutReplacement(rand_state, inits.size(), inits.size());
  Array<Trace> results;
  results.reserve(this->num_measures_per_iteration);
  for (int i = 0, i_bests = 0, i_rands = 0; i < this->num_measures_per_iteration; ++i) {
    bool has_best = i_bests < static_cast<int>(bests.size());
    bool has_rand = i_rands < static_cast<int>(rands.size());
    // Pick a schedule
    Optional<Trace> trace{NullOpt};
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

Array<MeasureResult> EvolutionaryNode::MeasureAndUpdateCostModel(const SearchTask& task,
                                                                 const Array<Trace>& picks,
                                                                 const ProgramMeasurer& measurer,
                                                                 int verbose) {
  // Extract cached trace info
  std::vector<CachedTrace> cached_traces;
  cached_traces.reserve(picks.size());
  for (const Trace& trace : picks) {
    cached_traces.push_back(GetCachedTrace(trace));
  }
  // Assemble `measure_inputs`
  Array<MeasureInput> measure_inputs;
  measure_inputs.reserve(picks.size());
  for (const CachedTrace& trace : cached_traces) {
    measure_inputs.push_back(MeasureInput(task, trace.sch));
  }
  // Run and get `measure_results`
  Array<MeasureResult> measure_results =
      measurer->BatchMeasure(measure_inputs, measure_inputs.size(), verbose);
  // Record the measurement result
  ICHECK_EQ(measure_inputs.size(), measure_results.size());
  // Update the measure
  for (int i = 0, n = measure_inputs.size(); i < n; ++i) {
    const MeasureResult& measure_result = measure_results[i];
    const CachedTrace& trace = cached_traces[i];
    database->Add(trace.trace->Simplified(/*remove_postproc=*/true), trace.repr,
                  AsVector<FloatImm, double>(measure_result->costs), task);
  }
  // Update the cost model
  cost_model->Update(measure_inputs, measure_results);
  trace_cache_.clear();
  return measure_results;
}

/********** FFI **********/

struct Internal {
  /*!
   * \brief Constructor of Evolutionary
   * \param total_measures The maximum number of measurements performed by genetic algorithm
   * \param num_measures_per_iteration The number of measures to be performed in each iteration
   * \param population The population size for evolutionary search
   * \param init_measured_ratio The ratio of measured states used in the initial population
   * \param genetic_algo_iters The number of iterations performed by generic algorithm
   * \param p_mutate The probability to perform mutation
   * \param mutator_probs Mutators and their probability mass
   * \param cost_model A cost model helping to explore the search space
   * \param eps_greedy The percentage of measurements to use randomly sampled states
   * \return The Evolutionary constructed
   * \sa Evolutionary::Evolutionary
   */
  static Evolutionary New(int total_measures, int num_measures_per_iteration, int population,
                          double init_measured_ratio, int genetic_algo_iters, double p_mutate,
                          Map<Mutator, FloatImm> mutator_probs, CostModel cost_model,
                          double eps_greedy) {
    return Evolutionary(total_measures, num_measures_per_iteration, population, init_measured_ratio,
                        genetic_algo_iters, p_mutate, mutator_probs, cost_model, eps_greedy);
  }
  /*!
   * \brief Sample the initial population from the support
   * \param self The evolutionary seach class
   * \param support The support to be sampled from
   * \param num_samples The number of samples to be drawn
   * \param space The search space
   * \return The generated samples
   * \sa EvolutionaryNode::SampleInitPopulation
   */
  static Array<Trace> SampleInitPopulation(Evolutionary self, Array<Schedule> support,
                                           SearchTask task, SearchSpace space,
                                           Optional<Integer> seed) {
    tir::TRandState rand_state;
    if (seed.defined() && seed.value()->value > 0) {
      tir::RandEngine(&rand_state).Seed(seed.value()->value);
    } else {
      tir::RandEngine(&rand_state).Seed(std::random_device()());
    }
    return self->SampleInitPopulation(support, task, space, &rand_state);
  }
  /*!
   * \brief Perform evolutionary search using genetic algorithm with the cost model
   * \param self The evolutionary seach class
   * \param task The search task
   * \param inits The initial population
   * \param num_samples The number of samples to be drawn
   * \param space The search space
   * \return An array of schedules, the sampling result
   * \sa EvolutionaryNode::EvolveWithCostModel
   */
  static Array<Trace> EvolveWithCostModel(Evolutionary self, Array<Trace> inits, SearchTask task,
                                          SearchSpace space, Optional<Integer> seed) {
    tir::TRandState rand_state;
    if (seed.defined() && seed.value()->value > 0) {
      tir::RandEngine(&rand_state).Seed(seed.value()->value);
    } else {
      tir::RandEngine(&rand_state).Seed(std::random_device()());
    }
    return self->EvolveWithCostModel(inits, task, space, &rand_state);
  }

  /*!
   * \brief Pick a batch of samples for measurement with epsilon greedy
   * \param inits The initial population used when picking random states
   * \param bests The best populations according to the cost model when picking top states
   * \param space The search space
   * \return A list of schedules, result of epsilon-greedy sampling
   * \sa EvolutionaryNode::PickWithEpsGreedy
   */
  static Array<Trace> PickWithEpsGreedy(Evolutionary self, Array<Trace> inits, Array<Trace> bests,
                                        SearchTask task, SearchSpace space,
                                        Optional<Integer> seed) {
    tir::TRandState rand_state;
    if (seed.defined() && seed.value()->value > 0) {
      tir::RandEngine(&rand_state).Seed(seed.value()->value);
    } else {
      tir::RandEngine(&rand_state).Seed(std::random_device()());
    }
    return self->PickWithEpsGreedy(inits, bests, task, space, &rand_state);
  }

  /*!
   * \brief Make measurements and update the cost model
   * \param task The search task
   * \param schedules The schedules to be measured
   * \param measurer The measurer
   * \param verbose A boolean flag for verbosity
   * \return A list of MeasureResult for measurements
   * \sa EvolutionaryNode::MeasureAndUpdateCostModel
   */
  static Array<MeasureResult> MeasureAndUpdateCostModel(Evolutionary self, SearchTask task,
                                                        Array<Trace> schedules,
                                                        ProgramMeasurer measurer, int verbose) {
    return self->MeasureAndUpdateCostModel(task, schedules, measurer, verbose);
  }
};

TVM_REGISTER_NODE_TYPE(EvolutionaryNode);
TVM_REGISTER_GLOBAL("meta_schedule.Evolutionary").set_body_typed(Internal::New);
TVM_REGISTER_GLOBAL("meta_schedule.EvolutionarySampleInitPopulation")
    .set_body_typed(Internal::SampleInitPopulation);
TVM_REGISTER_GLOBAL("meta_schedule.EvolutionaryEvolveWithCostModel")
    .set_body_typed(Internal::EvolveWithCostModel);
TVM_REGISTER_GLOBAL("meta_schedule.EvolutionaryPickWithEpsGreedy")
    .set_body_typed(Internal::PickWithEpsGreedy);
TVM_REGISTER_GLOBAL("meta_schedule.EvolutionaryMeasureAndUpdateCostModel")
    .set_body_typed(Internal::MeasureAndUpdateCostModel);

}  // namespace meta_schedule
}  // namespace tvm
