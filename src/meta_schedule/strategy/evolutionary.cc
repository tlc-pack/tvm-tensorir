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
 * 1) early stopping
 * 2) check if space is fully explored
 * 3) report sampling failures in init population
 * 4) make sure schedule is still valid
 */

/********** Evolutionary **********/

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
 *          using mutators
 *   chosen = pick top `k = population * (1 - eps_greedy)` from `best`
 *            pick     `k = population *      eps_greedy ` from `init`
 *   do the measurement on `chosen` & update the cost model
 */
class EvolutionaryNode : public SearchStrategyNode {
 public:
  /*** Configuration: global ***/
  /*! \brief The maximum number of measurements performed by genetic algorithm */
  int total_measures;
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

  void VisitAttrs(tvm::AttrVisitor* v) {
    /*** Configuration: global ***/
    v->Visit("total_measures", &total_measures);
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
  }

  /*!
   * \brief Explore the search space and find the best schedule
   * \param task The search task
   * \param space The search space
   * \param measurer The measurer that builds, runs and profiles sampled programs
   * \param sampler The random number sampler
   * \param verbose Whether or not in verbose mode
   * \return The best schedule found, NullOpt if no valid schedule is found
   */
  Optional<Schedule> Search(const SearchTask& task, const SearchSpace& space,
                            const ProgramMeasurer& measurer, Sampler* sampler,
                            int verbose) override;

  void Init(const SearchTask& task) override;

  /********** Stages in evolutionary search **********/

  /*!
   * \brief Sample the initial population from the support
   * \param support The support to be sampled from
   * \param sampler The random number sampler
   * \return The generated samples
   */
  Array<Schedule> SampleInitPopulation(const Array<Schedule>& support, Sampler* sampler);

  /*!
   * \brief Perform evolutionary search using genetic algorithm with the cost model
   * \param task The search task
   * \param inits The initial population
   * \param sampler The random number sampler
   * \return An array of schedules, the sampling result
   */
  Array<Schedule> EvolveWithCostModel(const SearchTask& task, const Array<Schedule>& inits,
                                      Sampler* sampler);

  /*!
   * \brief Pick a batch of samples for measurement with epsilon greedy
   * \param inits The initial population used when picking random states
   * \param bests The best populations according to the cost model when picking top states
   * \param space The search space
   * \param sampler The random number sampler
   * \return A list of schedules, result of epsilon-greedy sampling
   */
  Array<Schedule> PickWithEpsGreedy(const SearchTask& task, const Array<Schedule>& inits,
                                    const Array<Schedule>& bests, const SearchSpace& space,
                                    Sampler* sampler);

  /*!
   * \brief Make measurements and update the cost model
   * \param task The search task
   * \param schedules The schedules to be measured
   * \param measurer The measurer
   * \param verbose A boolean flag for verbosity
   * \return A list of MeasureResult for measurements
   */
  Array<MeasureResult> MeasureAndUpdateCostModel(const SearchTask& task,
                                                 const Array<Schedule>& schedules,
                                                 const ProgramMeasurer& measurer, int verbose);

  static constexpr const char* _type_key = "meta_schedule.Evolutionary";
  TVM_DECLARE_FINAL_OBJECT_INFO(EvolutionaryNode, SearchStrategyNode);

 private:
  // Helper functions

  /*!
   * \brief Create a sampler function that picks mutators according to the mass function
   * \param sampler The source of randomness
   * \return The sampler created
   */
  std::function<Optional<Mutator>()> MakeMutatorSampler(Sampler* sampler) const {
    std::vector<Optional<Mutator>> mutators;
    std::vector<double> mass;
    mutators.push_back(NullOpt);
    mass.push_back(1.0 - p_mutate);
    for (const auto& kv : mutator_probs) {
      mutators.push_back(kv.first);
      mass.push_back(kv.second->value * p_mutate);
    }
    auto idx_sampler = sampler->MakeMultinomial(mass);
    return [idx_sampler = std::move(idx_sampler),
            mutators = std::move(mutators)]() -> Optional<Mutator> {
      int idx = idx_sampler();
      return mutators[idx];
    };
  }

  /*!
   * \brief Predict the normalized throughput of each candidate.
   * Write the prediction to each `Database::Entry` and also return a copy
   * \param task The search task
   * \param candidates The candidates for prediction
   * \return The normalized throughput in the prediction
   */
  std::vector<double> PredictWithCostModel(const SearchTask& task,
                                           std::vector<Database::Entry>* candidates) const {
    Array<Schedule> schs;
    schs.reserve(candidates->size());
    for (const Database::Entry& entry : *candidates) {
      schs.push_back(entry.sch);
    }
    std::vector<double> result = cost_model->Predict(task, schs);
    int i = 0;
    for (Database::Entry& entry : *candidates) {
      entry.time = result[i++];
    }
    return result;
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
      int total_measures, int population,
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
 * \brief A heap with a size up-limit. If outflow happens, it evicted the worst items.
 * \note It maintains a min heap in terms of `Database::Entry::time`. Therefore, when
 * overflow happens, the element evicted is the one with the min `Database::Entry::time`.
 * As time goes, the elements in the heap are going to be larger.
 * It makes sense only when the heap is used to store predictions of the normalized running time,
 * where `time` is normalized throughput, i.e. the larger the better
 */
class SizedHeap {
  /*! \brief The comparator class, used by `std::push_heap` and `std::pop_heap` */
  struct Comparator {
    bool operator()(const Database::Entry& a, const Database::Entry& b) const {
      return a.time > b.time;
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
  void Push(const Database::Entry& item) {
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
  std::vector<Database::Entry> heap;
};

/********** Constructor **********/

Evolutionary::Evolutionary(int total_measures, int population, double init_measured_ratio,
                           int genetic_algo_iters, double p_mutate,
                           Map<Mutator, FloatImm> mutator_probs, CostModel cost_model,
                           double eps_greedy) {
  ObjectPtr<EvolutionaryNode> n = make_object<EvolutionaryNode>();
  n->total_measures = total_measures;
  n->population = population;
  n->database = InMemoryDB();
  n->init_measured_ratio = init_measured_ratio;
  n->genetic_algo_iters = genetic_algo_iters;
  n->p_mutate = p_mutate;
  n->mutator_probs = std::move(mutator_probs);
  n->cost_model = std::move(cost_model);
  n->eps_greedy = eps_greedy;
  data_ = std::move(n);
}

/********** Search **********/

void EvolutionaryNode::Init(const SearchTask& task) {}

Optional<Schedule> EvolutionaryNode::Search(const SearchTask& task, const SearchSpace& space,
                                            const ProgramMeasurer& measurer, Sampler* sampler,
                                            int verbose) {
  Array<Schedule> support = space->GetSupport(task, sampler);
  for (int num_measured = 0; num_measured < this->total_measures;) {
    // `inits`: Sampled initial population, whose size is at most `this->population`
    Array<Schedule> inits = SampleInitPopulation(support, sampler);
    // `bests`: The best schedules according to the cost mode when explore the space using mutators
    Array<Schedule> bests = EvolveWithCostModel(task, inits, sampler);
    // Pick candidates with eps greedy
    Array<Schedule> picks = PickWithEpsGreedy(task, inits, bests, space, sampler);
    // Run measurement, update cost model
    Array<MeasureResult> measure_results =
        MeasureAndUpdateCostModel(task, picks, measurer, verbose);
    num_measured += measure_results.size();
  }
  return measurer->best_sch;
}

Array<Schedule> EvolutionaryNode::SampleInitPopulation(const Array<Schedule>& support,
                                                       Sampler* sampler) {
  int n = this->population;
  Array<Schedule> results;
  results.reserve(n);
  // Pick measured states
  int num_measured = n * this->init_measured_ratio;
  for (const Database::Entry& entry : database->GetTopK(num_measured)) {
    results.push_back(entry.sch);
  }
  // Pick unmeasured states
  for (int i = results.size(); i < n; ++i) {
    const Schedule& sch = support[sampler->SampleInt(0, support.size())];
    // TODO(@junrushao1994): re-sample could fail
    Schedule new_sch(sch->orig_func, Integer(sch->sampler.ForkSeed()));
    Trace(sch->trace->insts, {})->Apply(new_sch);
    results.push_back(new_sch);
  }
  return results;
}

Array<Schedule> EvolutionaryNode::EvolveWithCostModel(const SearchTask& task,
                                                      const Array<Schedule>& inits,
                                                      Sampler* sampler) {
  // The heap to record best schedules
  // We do not consider schedules that are already measured
  SizedHeap heap(population);
  std::unordered_set<String> in_heap;
  // Prepare the mutator sampler
  std::function<Optional<Mutator>()> mutator_sampler = MakeMutatorSampler(sampler);
  // Prepare search queues
  std::vector<Database::Entry> sch_curr;
  std::vector<Database::Entry> sch_next;
  sch_curr.reserve(population);
  sch_next.reserve(population);
  for (const Schedule& sch : inits) {
    sch_curr.push_back(Database::Entry{sch, Repr(sch), 0.0});
  }
  // Main loop: (genetic_algo_iters + 1) times
  for (int iter = 0;; ++iter, sch_curr.clear(), sch_curr.swap(sch_next)) {
    // Predict running time with the cost model
    // TODO(@junrushao1994): postproc required here
    std::vector<double> scores = this->PredictWithCostModel(task, &sch_curr);
    // Put the schedules with the predicted perf to the heap
    for (const Database::Entry& entry : sch_curr) {
      if (!database->Has(entry.repr)) {
        if (in_heap.insert(entry.repr).second) {
          heap.Push(entry);
        }
      }
    }
    // Discontinue once it reaches end of search
    if (iter == genetic_algo_iters) {
      break;
    }
    // Make sampler from sch_curr with scores predicted
    // Sample according to the score
    std::function<int()> sch_curr_sampler = sampler->MakeMultinomial(scores);
    sch_next.clear();
    for (int i = 0; i < population; ++i) {
      const Database::Entry& entry = sch_curr[sch_curr_sampler()];
      Optional<Mutator> opt_mutator = mutator_sampler();
      if (!opt_mutator.defined()) {
        // If we decide not to mutate
        sch_next.emplace_back(entry);
        continue;
      }
      // Apply the mutator
      Mutator mutator = opt_mutator.value();
      // N.B. The `MutatorNode::Apply` will not change the schedule itself inplace
      if (Optional<Schedule> opt_sch = mutator->Apply(task, entry.sch, sampler)) {
        Schedule sch = opt_sch.value();
        sch_next.emplace_back(Database::Entry{sch, Repr(sch), 0.0});
      } else {
        // If not successful, take a step back and redo
        --i;
      }
    }
  }
  // Return the best states from the heap
  std::sort(heap.heap.begin(), heap.heap.end(),
            [](const Database::Entry& a, const Database::Entry& b) -> bool {
              // `time` here is normalized throughput, i.e. the larger the better
              return a.time > b.time;
            });
  Array<Schedule> results;
  results.reserve(population);
  for (const Database::Entry& item : heap.heap) {
    results.push_back(item.sch);
  }
  return results;
}

Array<Schedule> EvolutionaryNode::PickWithEpsGreedy(const SearchTask& task,
                                                    const Array<Schedule>& inits,
                                                    const Array<Schedule>& bests,
                                                    const SearchSpace& space, Sampler* sampler) {
  int n = this->population;
  int num_rands = n * this->eps_greedy;
  int num_bests = n - num_rands;
  std::vector<int> rands = sampler->SampleWithoutReplacement(inits.size(), inits.size());
  Array<Schedule> results;
  results.reserve(n);
  for (int i = 0, i_bests = 0, i_rands = 0; i < n;) {
    bool has_best = i_bests < static_cast<int>(bests.size());
    bool has_rand = i_rands < static_cast<int>(rands.size());
    // Pick a schedule
    Schedule sch(nullptr);
    // If needs `bests`, then prefer `bests`
    if (i < num_bests) {
      if (has_best) {
        sch = bests[i_bests++];
      } else if (has_rand) {
        sch = inits[rands[i_rands++]];
      } else {
        break;
      }
    } else {
      // Else prefer `rands`
      if (has_rand) {
        sch = inits[rands[i_rands++]];
      } else if (has_best) {
        sch = bests[i_bests++];
      } else {
        break;
      }
    }
    // Postprocess the schedule
    if (space->Postprocess(task, sch, sampler)) {
      // Check if the schedule has been measured before
      // If not, it is the schedule we want to pick
      ++i;
      results.push_back(sch);
    }
  }
  return results;
}

Array<MeasureResult> EvolutionaryNode::MeasureAndUpdateCostModel(const SearchTask& task,
                                                                 const Array<Schedule>& schedules,
                                                                 const ProgramMeasurer& measurer,
                                                                 int verbose) {
  // Assemble `measure_inputs`
  Array<MeasureInput> measure_inputs;
  measure_inputs.reserve(schedules.size());
  for (const Schedule& sch : schedules) {
    measure_inputs.push_back(MeasureInput(task, sch));
  }
  // Run and get `measure_results`
  Array<MeasureResult> measure_results =
      measurer->BatchMeasure(measure_inputs, measure_inputs.size(), verbose);
  // Record the measurement result
  CHECK_EQ(measure_inputs.size(), measure_results.size());
  // Update the measure
  for (int i = 0, n = measure_inputs.size(); i < n; ++i) {
    const MeasureInput& measure_input = measure_inputs[i];
    const MeasureResult& measure_result = measure_results[i];
    const Schedule& sch = measure_input->sch;
    double avg_time_cost = FloatArrayMean(measure_result->costs);
    database->Add(sch, Repr(sch), avg_time_cost);
  }
  // Update the cost model
  cost_model->Update(measure_inputs, measure_results);
  return measure_results;
}

/********** FFI **********/

struct Internal {
  /*!
   * \brief Constructor of Evolutionary
   * \param total_measures The maximum number of measurements performed by genetic algorithm
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
  static Evolutionary New(int total_measures, int population, double init_measured_ratio,
                          int genetic_algo_iters, double p_mutate,
                          Map<Mutator, FloatImm> mutator_probs, CostModel cost_model,
                          double eps_greedy) {
    return Evolutionary(total_measures, population, init_measured_ratio, genetic_algo_iters,
                        p_mutate, mutator_probs, cost_model, eps_greedy);
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
  static Array<Schedule> EvolutionarySampleInitPopulation(Evolutionary self,
                                                          Array<Schedule> support,
                                                          Optional<Integer> seed) {
    Sampler seeded;
    if (seed.defined()) {
      seeded.Seed(seed.value());
    }
    return self->SampleInitPopulation(support, &seeded);
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
  static Array<Schedule> EvolutionaryEvolveWithCostModel(Evolutionary self, SearchTask task,
                                                         Array<Schedule> inits,
                                                         Optional<Integer> seed) {
    Sampler seeded;
    if (seed.defined()) {
      seeded.Seed(seed.value());
    }
    return self->EvolveWithCostModel(task, inits, &seeded);
  }

  /*!
   * \brief Pick a batch of samples for measurement with epsilon greedy
   * \param inits The initial population used when picking random states
   * \param bests The best populations according to the cost model when picking top states
   * \param space The search space
   * \return A list of schedules, result of epsilon-greedy sampling
   * \sa EvolutionaryNode::PickWithEpsGreedy
   */
  static Array<Schedule> EvolutionaryPickWithEpsGreedy(Evolutionary self, SearchTask task,
                                                       Array<Schedule> inits, Array<Schedule> bests,
                                                       SearchSpace space, Optional<Integer> seed) {
    Sampler seeded;
    if (seed.defined()) {
      seeded.Seed(seed.value());
    }
    return self->PickWithEpsGreedy(task, inits, bests, space, &seeded);
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
  static Array<MeasureResult> EvolutionaryMeasureAndUpdateCostModel(Evolutionary self,
                                                                    SearchTask task,
                                                                    Array<Schedule> schedules,
                                                                    ProgramMeasurer measurer,
                                                                    int verbose) {
    return self->MeasureAndUpdateCostModel(task, schedules, measurer, verbose);
  }
};

TVM_REGISTER_NODE_TYPE(EvolutionaryNode);
TVM_REGISTER_GLOBAL("meta_schedule.Evolutionary").set_body_typed(Internal::New);
TVM_REGISTER_GLOBAL("meta_schedule.EvolutionarySampleInitPopulation")
    .set_body_typed(Internal::EvolutionarySampleInitPopulation);
TVM_REGISTER_GLOBAL("meta_schedule.EvolutionaryEvolveWithCostModel")
    .set_body_typed(Internal::EvolutionaryEvolveWithCostModel);
TVM_REGISTER_GLOBAL("meta_schedule.EvolutionaryPickWithEpsGreedy")
    .set_body_typed(Internal::EvolutionaryPickWithEpsGreedy);
TVM_REGISTER_GLOBAL("meta_schedule.EvolutionaryMeasureAndUpdateCostModel")
    .set_body_typed(Internal::EvolutionaryMeasureAndUpdateCostModel);

}  // namespace meta_schedule
}  // namespace tvm
