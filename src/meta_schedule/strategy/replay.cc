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
#include <tvm/support/parallel_for.h>

#include "../measure.h"
#include "../search.h"

namespace tvm {
namespace meta_schedule {

/********** Definition for Replay **********/

/*!
 * \brief A search strategy that just repeatedly replay the sampling process, do random sampling,
 * and picks the best from the results
 */
class ReplayNode : public SearchStrategyNode {
 public:
  /*! \brief Size of a batch for measurement */
  int batch_size;
  /*! \brief Number of iterations of replaying */
  int num_trials;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("batch_size", &batch_size);
    v->Visit("num_trials", &num_trials);
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
                            const ProgramMeasurer& measurer, Sampler* sampler,
                            int verbose) override;

  static constexpr const char* _type_key = "meta_schedule.Replay";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReplayNode, SearchStrategyNode);
};

/*!
 * \brief Managed refernce to ReplayNode
 * \sa ReplayNode
 */
class Replay : public SearchStrategy {
 public:
  /*!
   * \brief Constructor
   * \param batch_size Size of a batch for measurement
   * \param num_trials Number of iterations of replaying
   */
  explicit Replay(int batch_size, int num_trials);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Replay, SearchStrategy, ReplayNode);
};

/********** Constructor **********/

Replay::Replay(int batch_size, int num_trials) {
  ObjectPtr<ReplayNode> n = make_object<ReplayNode>();
  n->batch_size = batch_size;
  n->num_trials = num_trials;
  data_ = std::move(n);
}

/********** Search **********/

Optional<Schedule> ReplayNode::Search(const SearchTask& task, const SearchSpace& space,
                                      const ProgramMeasurer& measurer, Sampler* sampler,
                                      int verbose) {
  std::vector<Sampler> thread_samplers;
  std::vector<MeasureInput> thread_measure_inputs;
  thread_samplers.reserve(this->batch_size);
  thread_measure_inputs.reserve(this->batch_size);
  for (int i = 0; i < batch_size; ++i) {
    thread_samplers.emplace_back(sampler->ForkSeed());
    thread_measure_inputs.emplace_back(nullptr);
  }
  auto worker = [&task, &space, &thread_samplers, &thread_measure_inputs](int thread_id, int i) {
    Sampler* sampler = &thread_samplers[i];
    for (;;) {
      Schedule sch = space->SampleSchedule(task, sampler);
      if (space->Postprocess(task, sch, sampler)) {
        thread_measure_inputs[i] = MeasureInput(task, sch);
        break;
      }
    }
  };
  for (int st = 0; st < num_trials; st += batch_size) {
    int count = std::min(st + batch_size, num_trials) - st;
    support::parallel_persist_for(0, count, worker);
    Array<MeasureInput> measure_inputs{thread_measure_inputs.begin(),
                                       thread_measure_inputs.begin() + count};
    measurer->BatchMeasure(measure_inputs, count, verbose);
  }
  Optional<Schedule> res_opt = measurer->GetBest(task);
  if (!res_opt.defined()) return NullOpt;
  Schedule res = res_opt.value();
  Optional<Trace> trace_opt = res->trace();
  if (trace_opt.defined()) {
    // Check if the result schedule needs postprocing
    Trace trace = trace_opt.value();
    bool need_postproc = true;
    for (const Inst& inst : trace->insts) {
      if (inst->kind.same_as(InstKind::Get("EnterPostProc"))) {
        need_postproc = false;
        break;
      }
    }
    if (need_postproc) space->Postprocess(task, res, sampler);
  }
  return res;
}

/********** FFI **********/

struct Internal {
  /*!
   * \brief Constructor of Replay
   * \param batch_size Size of a batch for measurement
   * \param num_trials Number of iterations of replaying
   * \return The Replay constructed
   * \sa Replay::Replay
   */
  static Replay New(int batch_size, int num_trials) { return Replay(batch_size, num_trials); }
};

TVM_REGISTER_NODE_TYPE(ReplayNode);
TVM_REGISTER_GLOBAL("meta_schedule.Replay").set_body_typed(Internal::New);

}  // namespace meta_schedule
}  // namespace tvm
