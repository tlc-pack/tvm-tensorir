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

namespace tvm {
namespace meta_schedule {

/********** Mutator **********/

/*! \brief A mutation rule for the genetic algorithm  */
class MutatorNode : public Object {
 public:
  /*! \brief The probability weight of choosing this rule */
  double p;

  /*! \brief Base destructor */
  virtual ~MutatorNode() = default;

  /*!
   * \brief Mutate the schedule by applying the mutation
   * \param sch The schedule to be mutated
   * \param sampler The random number sampler
   * \return The new schedule after mutation, NullOpt if mutation fails
   */
  virtual Optional<Schedule> Apply(const Schedule& sch, Sampler* sampler) = 0;

  static constexpr const char* _type_key = "meta_schedule.Mutator";
  TVM_DECLARE_BASE_OBJECT_INFO(MutatorNode, Object);
};

/*!
 * \brief Managed refernce to MutatorNode
 * \sa MutatorNode
 */
class Mutator : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Mutator, ObjectRef, MutatorNode);
};

/********** Evolutionary **********/

/*!
 * \brief Get the string representation of a schedule
 * \param sch The schedule to be stringified
 * \return The string representation of a schedule
 */
String Repr(const Schedule& sch) {
  const auto* f = runtime::Registry::Get("hybrid.AsHybrid");
  CHECK(f) << "IndexError: global function \"hybrid.AsHybrid\" not found";
  String s = (*f)(sch->sch->func, false);
  return s;
}

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

class SizedHeap {
 public:
  explicit SizedHeap(int size_limit) : size_limit_(size_limit) { heap_.reserve(size_limit_); }

  void Push(double time, const Schedule& sch) {
    String str_repr = Repr(sch);
    if (in_heap_.count(str_repr)) {
      return;
    }
    int size = heap_.size();
    if (size < size_limit_) {
      // Heap is not full, just push
      heap_.emplace_back(time, sch);
      std::push_heap(heap_.begin(), heap_.end());
      in_heap_.insert(str_repr);
    } else if (time < heap_.front().time) {
      // if the time is better than the worse one in the heap, we can safely kick it out
      String old_str_repr = Repr(heap_.front().sch);
      in_heap_.erase(old_str_repr);
      in_heap_.insert(str_repr);
      std::pop_heap(heap_.begin(), heap_.end());
      heap_.back() = MeasuredState(time, sch);
      std::push_heap(heap_.begin(), heap_.end());
    }
    // Otherwise, the time is worse than any other element in the heap
  }

 private:
  int size_limit_;
  std::vector<MeasuredState> heap_;
  std::unordered_set<String> in_heap_;
};

/*! \brief A table storing all the states that have been measured */
class MeasuredStates {
 public:
  /*!
   * \brief Check if a string representation is in the table
   * \param str_repr The string representation to be checked
   * \return A boolean indicating if it is in the table
   */
  bool Has(const String& str_repr) const { return table_.count(str_repr); }

  /*!
   * \brief Add a string representation to the table
   * \param str_repr The string representation to be added
   */
  void Add(const String& str_repr) { table_.insert(str_repr); }

  /*!
   * \brief Add a schedule and its running time to the table
   * \param time The running time
   * \param sch The schedule
   */
  void Add(double time, const Schedule& sch) { states_.emplace(time, sch); }

  /*!
   * \brief Get the best k states
   * \param num_states The number of the best states to get
   * \return A list with length at most `num_states`
   */
  Array<Schedule> GetBestStates(int num_states) const {
    Array<Schedule> result;
    result.reserve(num_states);
    int i = 0;
    for (const MeasuredState& state : states_) {
      result.push_back(state.sch);
      if (++i >= num_states) {
        break;
      }
    }
    return result;
  }

 private:
  /*! \brief The table to store string representation */
  std::unordered_set<String> table_;
  /*! \brief The table to store measured states */
  std::multiset<MeasuredState> states_;
};

/*! \brief Evolutionary Search */
class EvolutionaryNode : public SearchStrategyNode {
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
  /*! \brief A list of mutations allowed to happen */
  Array<Mutator> mutators;

  CostModel cost_model;

  Sampler sampler_;

  MeasuredStates measured_;

  std::function<int()> mutator_sampler_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("num_measure_trials", &num_measure_trials);
    v->Visit("num_measure_per_batch", &num_measure_per_batch);
    v->Visit("num_measure_per_batch", &num_iters_in_genetic_algo);
    v->Visit("eps_greedy", &eps_greedy);
    v->Visit("use_measured_ratio", &use_measured_ratio);
    v->Visit("population", &population);
    v->Visit("p_mutate", &p_mutate);
    v->Visit("mutators", &mutators);
    v->Visit("cost_model", &cost_model);
    // sampler_ is not visited
    // measured_ is not visited
    // mutator_sampler_ is not visited
  }

  /*!
   * \brief Explore the search space and find the best schedule
   * \param task The search task
   * \param space The search space
   * \param measurer The measurer that builds, runs and profiles sampled programs
   * \param verbose Whether or not in verbose mode
   * \return The best schedule found, NullOpt if no valid schedule is found
   */
  Optional<Schedule> Search(const SearchTask& task, const SearchSpace& space,
                            const ProgramMeasurer& measurer, int verbose) override;

  static constexpr const char* _type_key = "meta_schedule.Evolutionary";
  TVM_DECLARE_FINAL_OBJECT_INFO(EvolutionaryNode, SearchStrategyNode);

 private:
  /*!
   * \brief Sample the initial population from the support
   * \param support The support to be sampled from
   * \return The generated samples
   */
  Array<Schedule> SampleInitPopulation(const Array<Schedule>& support);

  /*!
   * \brief Perform evolutionary search using genetic algorithm with the cost model
   * \param initial_population The initial population
   * \param num_samples The number of samples to be draw
   * \return An array of schedules, the sampling result
   */
  Array<Schedule> EvolveWithCostModel(const SearchTask& task,
                                      const Array<Schedule>& initial_population, int num_samples);

  /*!
   * \brief Construct measure inputs out of the candidate schedules
   * \param task The search task
   * \param bests The best scored candidate schedules
   * \param rands Some rand candidate schedules
   * \param batch_size Up-limit of the number of schedules to be picked
   * \param num_bests Up-limit of the number of schedules to be picked from bests
   * \param num_rands Up-limit of the number of schedules to be picked from rands
   * \return An array of MeasureInput to be picked
   */
  Array<MeasureInput> MakeMeasureInputs(const SearchTask& task, const Array<Schedule>& bests,
                                        const Array<Schedule>& rands, int batch_size, int num_bests,
                                        int num_rands);
};

/*!
 * \brief Managed refernce to EvolutionaryNode
 * \sa EvolutionaryNode
 */
class Evolutionary : public SearchStrategy {
 public:
  /*!
   * \brief Constructor
   * \param mutators A list of mutations allowed to happen
   */
  // TODO
  explicit Evolutionary(int num_measure_trials, int num_measure_per_batch,
                        int num_iters_in_genetic_algo, double eps_greedy, double use_measured_ratio,
                        int population, double p_mutate, Array<Mutator> mutators,
                        CostModel cost_model);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Evolutionary, SearchStrategy, EvolutionaryNode);
};

/********** Constructor **********/

Evolutionary::Evolutionary(int num_measure_trials, int num_measure_per_batch,
                           int num_iters_in_genetic_algo, double eps_greedy,
                           double use_measured_ratio, int population, double p_mutate,
                           Array<Mutator> mutators, CostModel cost_model) {
  // Extract weights of mutators
  std::vector<double> weights;
  for (const auto& mutator : mutators) {
    weights.push_back(mutator->p);
  }
  ObjectPtr<EvolutionaryNode> n = make_object<EvolutionaryNode>();
  n->num_measure_trials = num_measure_trials;
  n->num_measure_per_batch = num_measure_per_batch;
  n->num_iters_in_genetic_algo = num_iters_in_genetic_algo;
  n->eps_greedy = eps_greedy;
  n->use_measured_ratio = use_measured_ratio;
  n->population = population;
  n->p_mutate = p_mutate;
  n->mutators = std::move(mutators);
  n->cost_model = std::move(cost_model);
  n->mutator_sampler_ = n->sampler_.MakeMultinomial(weights);
  data_ = std::move(n);
}

/********** Search **********/

Optional<Schedule> EvolutionaryNode::Search(const SearchTask& task, const SearchSpace& space,
                                            const ProgramMeasurer& measurer, int verbose) {
  measurer->Reset();
  int num_eps_rand = this->eps_greedy * this->num_measure_per_batch;
  Array<Schedule> support = space->GetSupport(task);
  for (int num_measures = 0; num_measures < num_measure_trials;) {
    Array<Schedule> population = SampleInitPopulation(support);
    Array<Schedule> rands = sampler_.SampleWithReplacement(rands, num_eps_rand * 3);
    Array<Schedule> bests = EvolveWithCostModel(task, population, num_measure_per_batch * 2);
    int batch_size = std::min(num_measure_per_batch, num_measure_trials - num_measures);
    Array<MeasureInput> measure_inputs = MakeMeasureInputs(
        /*task=*/task, /*bests=*/bests, /*rands=*/rands, /*batch_size=*/batch_size,
        /*num_bests=*/num_measure_per_batch - num_eps_rand, /*num_rands=*/num_eps_rand);
  }
  return measurer->best_sch;
}

Array<Schedule> EvolutionaryNode::SampleInitPopulation(const Array<Schedule>& support) {
  Array<Schedule> results = measured_.GetBestStates(population * use_measured_ratio);
  results.reserve(population);
  for (int i = results.size(); i < population; ++i) {
    int sample_index = sampler_.SampleInt(0, support.size());
    Schedule sch = support[sample_index]->copy();
    sch->ReplayOnce();  // TODO(@junrushao1994): deal with exceptions
    results.push_back(sch);
  }
  return results;
}

Array<Schedule> EvolutionaryNode::EvolveWithCostModel(const SearchTask& task,
                                                      const Array<Schedule>& initial_population,
                                                      int num_samples) {
  // Prepare search queues
  std::vector<Schedule> sch_curr(initial_population.begin(), initial_population.end());
  std::vector<Schedule> sch_next;
  sch_curr.reserve(population);
  sch_next.reserve(population);
  // The heap to record best schedules
  SizedHeap heap(num_samples);
  // Main loop: (num_iters_in_genetic_algo + 1) times
  for (int iter = 0;; ++iter) {
    // Predict running time with the cost model
    std::vector<double> scores = cost_model->Predict(task, sch_curr);
    // Put the predicted perf to the heap
    CHECK_EQ(scores.size(), sch_curr.size());
    for (int i = 0, n = scores.size(); i < n; ++i) {
      heap.Push(scores[i], sch_curr[i]);
    }
    // Discontinue once it reaches end of search
    if (iter == num_iters_in_genetic_algo) {
      break;
    }
    // Make sampler from sch_curr with scores predicted
    std::function<int()> sch_curr_sampler = sampler_.MakeMultinomial(scores);
    // Sample according to the score
    for (int i = 0; i < population;) {
      const Schedule& sch = sch_curr[sch_curr_sampler()];
      if (!sampler_.SampleBernoulli(p_mutate)) {
        ++i;
        sch_next.emplace_back(sch);
        continue;
      }
      const Mutator& mutator = mutators[mutator_sampler_()];
      if (Optional<Schedule> new_sch = mutator->Apply(sch, &sampler_)) {
        ++i;
        sch_next.emplace_back(new_sch.value());
      }
    }
  }
}

Array<MeasureInput> EvolutionaryNode::MakeMeasureInputs(const SearchTask& task,
                                                        const Array<Schedule>& bests,
                                                        const Array<Schedule>& rands,
                                                        int batch_size, int num_bests,
                                                        int num_rands) {
  size_t i_bests = 0;
  size_t i_rands = 0;
  Array<MeasureInput> results;
  results.reserve(batch_size);
  for (int i = 0; i < batch_size;) {
    bool has_best = i_bests < bests.size();
    bool has_rand = i_rands < rands.size();
    // Pick a schedule
    Schedule sch(nullptr);
    if (i < num_bests) {
      if (has_best) {
        sch = bests[i_bests++];
      } else if (has_rand) {
        sch = rands[i_rands++];
      } else {
        break;
      }
    } else {
      if (has_rand) {
        sch = rands[i_rands++];
      } else if (has_best) {
        sch = bests[i_bests++];
      } else {
        break;
      }
    }
    // Check if the schedule has been measured before
    String repr = Repr(sch);
    if (measured_.Has(repr)) {
      continue;
    }
    // If not, it is the schedule we want to pick
    ++i;
    measured_.Add(repr);
    results.push_back(MeasureInput(task, sch));
  }
  return results;
}

/********** FFI **********/

struct Internal {
  /*!
   * \brief Constructor of Evolutionary
   * \param mutators A list of mutations allowed to happen
   * \return The Evolutionary constructed
   * \sa Evolutionary::Evolutionary
   */
  static Evolutionary New(int num_measure_trials, int num_measure_per_batch,
                          int num_iters_in_genetic_algo, double eps_greedy,
                          double use_measured_ratio, int population, double p_mutate,
                          Array<Mutator> mutators, CostModel cost_model) {
    // TODO: doc
    return Evolutionary(num_measure_trials, num_measure_per_batch, num_iters_in_genetic_algo,
                        eps_greedy, use_measured_ratio, population, p_mutate, mutators, cost_model);
  }
};

TVM_REGISTER_OBJECT_TYPE(MutatorNode);
TVM_REGISTER_NODE_TYPE(EvolutionaryNode);
TVM_REGISTER_GLOBAL("meta_schedule.Evolutionary").set_body_typed(Internal::New);

}  // namespace meta_schedule
}  // namespace tvm
