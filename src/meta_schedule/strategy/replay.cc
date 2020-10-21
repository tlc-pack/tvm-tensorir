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
#include "../measure.h"
#include "../search.h"
#include "./postproc.h"

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
  int num_iterations;
  /*! \brief Postprocessors */
  Array<Postproc> postprocs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("batch_size", &batch_size);
    v->Visit("num_iterations", &num_iterations);
    v->Visit("postprocs", &postprocs);
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
   * \param num_iterations Number of iterations of replaying
   */
  explicit Replay(int batch_size, int num_iterations, Optional<Array<Postproc>> postprocs);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Replay, SearchStrategy, ReplayNode);
};

/********** Constructor **********/

Replay::Replay(int batch_size, int num_iterations, Optional<Array<Postproc>> postprocs) {
  ObjectPtr<ReplayNode> n = make_object<ReplayNode>();
  n->batch_size = batch_size;
  n->num_iterations = num_iterations;
  data_ = std::move(n);
}

/********** Search **********/

Optional<Schedule> ReplayNode::Search(const SearchTask& task, const SearchSpace& space,
                                      const ProgramMeasurer& measurer, Sampler* sampler,
                                      int verbose) {
  measurer->Reset();
  for (int iter_id = 0; iter_id < num_iterations;) {
    Array<MeasureInput> measure_inputs;
    measure_inputs.reserve(batch_size);
    for (int batch_id = 0; batch_id < batch_size && iter_id < num_iterations;
         ++batch_id, ++iter_id) {
      measure_inputs.push_back(MeasureInput(task, space->SampleSchedule(task, sampler)));
    }
    measurer->BatchMeasure(measure_inputs, this->batch_size, verbose);
  }
  return measurer->best_sch;
}

/********** FFI **********/

struct Internal {
  /*!
   * \brief Constructor of Replay
   * \param batch_size Size of a batch for measurement
   * \param num_iterations Number of iterations of replaying
   * \return The Replay constructed
   * \sa Replay::Replay
   */
  static Replay New(int batch_size, int num_iterations, Optional<Array<Postproc>> postprocs) {
    return Replay(batch_size, num_iterations, postprocs);
  }
};

TVM_REGISTER_NODE_TYPE(ReplayNode);
TVM_REGISTER_GLOBAL("meta_schedule.Replay").set_body_typed(Internal::New);

}  // namespace meta_schedule
}  // namespace tvm
