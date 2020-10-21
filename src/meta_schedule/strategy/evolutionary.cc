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
#include "../measure.h"
#include "../search.h"
#include "../utils.h"
#include "./mutator.h"
#include "./postproc.h"

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

/*! \brief Evolutionary Search */
class EvolutionaryNode : public SearchStrategyNode {
 private:
  /*! \brief The measured states */
  struct MeasuredState {
    /*! \brief Running time */
    double time;
    /*! \brief The schedule to be measured */
    Schedule sch;
    /*! \brief Constructor */
    explicit MeasuredState(double time, const Schedule& sch) : time(time), sch(sch) {}
    /*! \brief Comparator */
    bool operator<(const MeasuredState& rhs) const { return time < rhs.time; }
  };

 public:
  /*! \brief The number of iterations of measurements performed by genetic algorithm */
  int num_measure_trials;
  /*! \brief The number of measurements in each batch */
  int num_measure_per_batch;
  /*! \brief The number of iterations performed by generic algorithm.*/
  int num_iters_in_genetic_algo;
  /*! \brief The percentage of measurements to use randomly sampled states. */
  double eps_greedy;
  /*! \brief The percentage of previously measured states used in the initial population */
  double use_measured_ratio;
  /*! \brief The population size for evolutionary search */
  int population;
  /*! \brief The probability to perform mutation */
  double p_mutate;
  /*! \brief Mutators and their probability mass */
  Map<Mutator, FloatImm> mutator_probs;
  /*! \brief A cost model helping to explore the search space */
  CostModel cost_model;
  /*! \brief Postprocessors */
  Array<Postproc> postprocs;
  /*! \brief A table storing all states that have been measured */
  SortedTable<String, MeasuredState> measured_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("num_measure_trials", &num_measure_trials);
    v->Visit("num_measure_per_batch", &num_measure_per_batch);
    v->Visit("num_measure_per_batch", &num_iters_in_genetic_algo);
    v->Visit("eps_greedy", &eps_greedy);
    v->Visit("use_measured_ratio", &use_measured_ratio);
    v->Visit("population", &population);
    v->Visit("p_mutate", &p_mutate);
    v->Visit("mutator_probs", &mutator_probs);
    v->Visit("cost_model", &cost_model);
    v->Visit("postprocs", &postprocs);
    // measured_ is not visited
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

  /********** Stages in evolutionary search **********/

  /*!
   * \brief Sample the initial population from the support
   * \param support The support to be sampled from
   * \param num_samples The number of samples to be drawn
   * \param sampler The random number sampler
   * \return The generated samples
   */
  Array<Schedule> SampleInitPopulation(const Array<Schedule>& support, int num_samples,
                                       Sampler* sampler);

  /*!
   * \brief Perform evolutionary search using genetic algorithm with the cost model
   * \param task The search task
   * \param inits The initial population
   * \param num_samples The number of samples to be drawn
   * \param sampler The random number sampler
   * \return An array of schedules, the sampling result
   */
  Array<Schedule> EvolveWithCostModel(const SearchTask& task, const Array<Schedule>& inits,
                                      int num_samples, Sampler* sampler);

  /*!
   * \brief Pick a batch of samples for measurement with epsilon greedy
   * \param inits The initial population used when picking random states
   * \param bests The best populations according to the cost model when picking top states
   * \param sampler The random number sampler
   * \return A list of schedules, result of epsilon-greedy sampling
   */
  Array<Schedule> PickWithEpsGreedy(const Array<Schedule>& inits, const Array<Schedule>& bests,
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
};

/*!
 * \brief Managed refernce to EvolutionaryNode
 * \sa EvolutionaryNode
 */
class Evolutionary : public SearchStrategy {
 public:
  /*!
   * \brief Constructor
   * \param num_measure_trials The number of iterations of measurements performed by genetic
   * algorithm
   * \param num_measure_per_batch The number of measurements in each batch
   * \param num_iters_in_genetic_algo The number of iterations performed by generic algorithm
   * \param eps_greedy The percentage of measurements to use randomly sampled states
   * \param use_measured_ratio The percentage of previously measured states used in the initial
   * population
   * \param population The population size for evolutionary search
   * \param p_mutate The probability to perform mutation
   * \param mutator_probs Mutators and their probability mass
   * \param cost_model A cost model helping to explore the search space
   * \param postprocs The postprocessors. If not present, use default built-in
   */
  explicit Evolutionary(int num_measure_trials, int num_measure_per_batch,
                        int num_iters_in_genetic_algo, double eps_greedy, double use_measured_ratio,
                        int population, double p_mutate, Map<Mutator, FloatImm> mutator_probs,
                        CostModel cost_model, Optional<Array<Postproc>> postprocs);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Evolutionary, SearchStrategy, EvolutionaryNode);
};

/********** Constructor **********/

Evolutionary::Evolutionary(int num_measure_trials, int num_measure_per_batch,
                           int num_iters_in_genetic_algo, double eps_greedy,
                           double use_measured_ratio, int population, double p_mutate,
                           Map<Mutator, FloatImm> mutator_probs, CostModel cost_model,
                           Optional<Array<Postproc>> postprocs) {
  ObjectPtr<EvolutionaryNode> n = make_object<EvolutionaryNode>();
  n->num_measure_trials = num_measure_trials;
  n->num_measure_per_batch = num_measure_per_batch;
  n->num_iters_in_genetic_algo = num_iters_in_genetic_algo;
  n->eps_greedy = eps_greedy;
  n->use_measured_ratio = use_measured_ratio;
  n->population = population;
  n->p_mutate = p_mutate;
  n->mutator_probs = std::move(mutator_probs);
  n->cost_model = std::move(cost_model);
  if (postprocs.defined()) {
    n->postprocs = postprocs.value();
  } else {
    n->postprocs = PostprocDefaults();
  }
  data_ = std::move(n);
}

/********** Search **********/

Optional<Schedule> EvolutionaryNode::Search(const SearchTask& task, const SearchSpace& space,
                                            const ProgramMeasurer& measurer, Sampler* sampler,
                                            int verbose) {
  measurer->Reset();
  Array<Schedule> support = space->GetSupport(task, sampler);
  for (int num_measured = 0; num_measured < num_measure_trials;) {
    // `inits`: Sampled initial population, whose size is at most `this->population`
    Array<Schedule> inits = SampleInitPopulation(support, population, sampler);
    // `bests`: The best schedules according to the cost mode when explore the space using mutators
    Array<Schedule> bests = EvolveWithCostModel(task, inits, num_measure_per_batch * 2, sampler);
    // Pick candidates with eps greedy
    Array<Schedule> picks = PickWithEpsGreedy(inits, bests, sampler);
    // Run measurement, update cost model
    Array<MeasureResult> measure_results =
        MeasureAndUpdateCostModel(task, picks, measurer, verbose);
    num_measured += measure_results.size();
  }
  return measurer->best_sch;
}

Array<Schedule> EvolutionaryNode::SampleInitPopulation(const Array<Schedule>& support,
                                                       int num_samples, Sampler* sampler) {
  int num_measured = num_samples * use_measured_ratio;
  Array<Schedule> results;
  results.reserve(num_samples);
  // Pick measured states
  std::vector<MeasuredState> measured = measured_.GetTopK(num_measured);
  for (const MeasuredState& state : measured) {
    results.push_back(state.sch);
  }
  // Pick unmeasured states
  for (int i = results.size(); i < num_samples; ++i) {
    int sample_index = sampler->SampleInt(0, support.size());
    const Schedule& sch = support[sample_index];
    Schedule new_sch = sch->Copy(sch->sampler.ForkSeed());
    new_sch->ReSample();  // TODO(@junrushao1994): deal with exceptions
    results.push_back(new_sch);
  }
  return results;
}

Array<Schedule> EvolutionaryNode::EvolveWithCostModel(const SearchTask& task,
                                                      const Array<Schedule>& inits, int num_samples,
                                                      Sampler* sampler) {
  // Extract weights of mutators
  std::function<Mutator()> mutator_sampler = nullptr;
  {
    std::vector<Mutator> mutators;
    std::vector<double> mutator_mass;
    for (const auto& kv : mutator_probs) {
      mutators.push_back(kv.first);
      mutator_mass.push_back(kv.second->value);
    }
    auto idx_sampler = sampler->MakeMultinomial(mutator_mass);
    mutator_sampler = [idx_sampler = std::move(idx_sampler),
                       mutators = std::move(mutators)]() -> Mutator {
      return mutators[idx_sampler()];
    };
  }
  // The heap to record best schedules
  /*! \brief Item in the heap */
  struct HeapItem {
    /*! \brief We de-duplicate using strings */
    using KeyType = String;
    /*! \brief The string key used for de-duplication */
    String key;
    /*! \brief The prediction score, the larger the better */
    double score;
    /*! \brief The schedule */
    Schedule sch;
    /*!
     * \brief Constructor
     * \param key The string key used for de-duplication
     * \param score The predicted score, the larger the better
     * \param sch The schedule
     */
    explicit HeapItem(const String& key, double score, const Schedule& sch)
        : key(key), score(score), sch(sch) {}
    /*! \brief Comparator */
    bool operator<(const HeapItem& rhs) const { return score > rhs.score; }
  };
  SizedHeap<HeapItem> heap(num_samples);
  // We do not consider schedules that are already measured
  heap.AddKeys(measured_.keys.begin(), measured_.keys.end());
  // Prepare search queues
  std::vector<Schedule> sch_curr(inits.begin(), inits.end());
  std::vector<Schedule> sch_next;
  sch_curr.reserve(population);
  sch_next.reserve(population);
  // Main loop: (num_iters_in_genetic_algo + 1) times
  for (int iter = 0;; ++iter, sch_curr.clear(), sch_curr.swap(sch_next)) {
    // Predict running time with the cost model
    std::vector<double> scores = AsVector<FloatImm, double>()(cost_model->Predict(task, sch_curr));
    // Put the predicted perf to the heap
    CHECK_EQ(scores.size(), sch_curr.size());
    for (int i = 0, n = scores.size(); i < n; ++i) {
      heap.Push(HeapItem(Repr(sch_curr[i]), scores[i], sch_curr[i]));
    }
    // Discontinue once it reaches end of search
    if (iter == num_iters_in_genetic_algo) {
      break;
    }
    // Make sampler from sch_curr with scores predicted
    // Sample according to the score
    std::function<int()> sch_curr_sampler = sampler->MakeMultinomial(scores);
    sch_next.clear();
    for (int i = 0; i < population; ++i) {
      const Schedule& sch = sch_curr[sch_curr_sampler()];
      if (sampler->SampleBernoulli(p_mutate)) {
        // with probability `p_mutate`, choose a mutator
        Mutator mutator = mutator_sampler();
        // apply the mutator
        if (Optional<Schedule> new_sch = mutator->Apply(task, sch, sampler)) {
          sch_next.emplace_back(new_sch.value());
        } else {
          // if not successful, take a step back and redo
          --i;
        }
      } else {
        sch_next.emplace_back(sch);
      }
    }
  }
  // Return the best states from the heap
  std::sort(heap.heap.begin(), heap.heap.end());
  Array<Schedule> results;
  results.reserve(num_samples);
  for (const HeapItem& item : heap.heap) {
    results.push_back(item.sch);
  }
  return results;
}

Array<Schedule> EvolutionaryNode::PickWithEpsGreedy(const Array<Schedule>& inits,
                                                    const Array<Schedule>& bests,
                                                    Sampler* sampler) {
  int n = this->num_measure_per_batch;
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
    // Check if the schedule has been measured before
    // If not, it is the schedule we want to pick
    String repr = Repr(sch);
    if (!measured_.Has(repr)) {
      ++i;
      measured_.Add(repr);
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
    double avg_time_cost = FloatArrayMean(measure_result->costs);
    measured_.Add(MeasuredState(avg_time_cost, measure_input->sch));
  }
  // Update the cost model
  cost_model->Update(measure_inputs, measure_results);
  return measure_results;
}

/********** FFI **********/

struct Internal {
  /*!
   * \brief Constructor of Evolutionary
   * \param num_measure_trials The number of iterations of measurements performed by genetic
   * algorithm
   * \param num_measure_per_batch The number of measurements in each batch
   * \param num_iters_in_genetic_algo The number of iterations performed by generic algorithm
   * \param eps_greedy The percentage of measurements to use randomly sampled states
   * \param use_measured_ratio The percentage of previously measured states used in the initial
   * population
   * \param population The population size for evolutionary search
   * \param p_mutate The probability to perform mutation
   * \param mutator_probs Mutators and their probability mass
   * \param cost_model A cost model helping to explore the search space
   * \return The Evolutionary constructed
   * \sa Evolutionary::Evolutionary
   */
  static Evolutionary New(int num_measure_trials, int num_measure_per_batch,
                          int num_iters_in_genetic_algo, double eps_greedy,
                          double use_measured_ratio, int population, double p_mutate,
                          Map<Mutator, FloatImm> mutator_probs, CostModel cost_model,
                          Optional<Array<Postproc>> postprocs) {
    return Evolutionary(num_measure_trials, num_measure_per_batch, num_iters_in_genetic_algo,
                        eps_greedy, use_measured_ratio, population, p_mutate, mutator_probs,
                        cost_model, postprocs);
  }
  /*!
   * \brief Sample the initial population from the support
   * \param self The evolutionary seach class
   * \param support The support to be sampled from
   * \param num_samples The number of samples to be drawn
   * \return The generated samples
   * \sa EvolutionaryNode::SampleInitPopulation
   */
  static Array<Schedule> EvolutionarySampleInitPopulation(Evolutionary self,
                                                          Array<Schedule> support, int num_samples,
                                                          Optional<Integer> seed) {
    Sampler seeded;
    if (seed.defined()) {
      seeded.Seed(seed.value());
    }
    return self->SampleInitPopulation(support, num_samples, &seeded);
  }
  /*!
   * \brief Perform evolutionary search using genetic algorithm with the cost model
   * \param self The evolutionary seach class
   * \param task The search task
   * \param inits The initial population
   * \param num_samples The number of samples to be drawn
   * \return An array of schedules, the sampling result
   * \sa EvolutionaryNode::EvolveWithCostModel
   */
  static Array<Schedule> EvolutionaryEvolveWithCostModel(Evolutionary self, SearchTask task,
                                                         Array<Schedule> inits, int num_samples,
                                                         Optional<Integer> seed) {
    Sampler seeded;
    if (seed.defined()) {
      seeded.Seed(seed.value());
    }
    return self->EvolveWithCostModel(task, inits, num_samples, &seeded);
  }

  /*!
   * \brief Pick a batch of samples for measurement with epsilon greedy
   * \param inits The initial population used when picking random states
   * \param bests The best populations according to the cost model when picking top states
   * \return A list of schedules, result of epsilon-greedy sampling
   * \sa EvolutionaryNode::PickWithEpsGreedy
   */
  static Array<Schedule> EvolutionaryPickWithEpsGreedy(Evolutionary self, Array<Schedule> inits,
                                                       Array<Schedule> bests,
                                                       Optional<Integer> seed) {
    Sampler seeded;
    if (seed.defined()) {
      seeded.Seed(seed.value());
    }
    return self->PickWithEpsGreedy(inits, bests, &seeded);
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
