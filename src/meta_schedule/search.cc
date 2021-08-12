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

#include "./search.h"  // NOLINT(build/include)

#include "./analysis.h"
#include "./measure.h"
#include "./utils.h"

namespace tvm {
namespace meta_schedule {

/********** Constructor **********/

SearchTask::SearchTask(tir::PrimFunc workload, String task_name, Target target, Target target_host,
                       Optional<String> log_file) {
  ObjectPtr<SearchTaskNode> n = make_object<SearchTaskNode>();
  if (task_name == "") {
    n->task_name = "func" + std::to_string(StructuralHash()(workload));
  } else {
    n->task_name = std::move(task_name);
  }
  n->workload = std::move(workload);
  n->target = std::move(target);
  n->target_host = std::move(target_host);
  n->log_file = std::move(log_file);
  n->flop_ct = CountFlop(n->workload);
  data_ = std::move(n);
}

/********** Search **********/

/*!
 * \brief The entry function for auto tuning
 * \param task The search task
 * \param space The search space
 * \param strategy The search strategy
 * \param measurer The measurer that builds, runs and profiles sampled programs
 * \param seed The randon seed
 * \param verbose Flag for the verbose mode
 * \return The best schedule found, NullOpt if no valid schedule is found in the search space
 */
TVM_DLL Optional<Schedule> AutoTune(SearchTask task, SearchSpace space, SearchStrategy strategy,
                                    ProgramMeasurer measurer, Optional<Integer> seed, int verbose) {
  tir::TRandState rand_state;
  if (seed.defined() && seed.value()->value != -1) {
    tir::RandEngine(&rand_state).Seed(seed.value()->value);
  } else {
    tir::RandEngine(&rand_state).Seed(std::random_device()());
  }

  if (verbose) {
    LOG(INFO) << "Tuning for task: " << task;
  }
  space->Init(task);
  strategy->Init(task);
  measurer->Init(task);
  return strategy->Search(task, space, measurer, &rand_state, verbose);
}

/********** Printer **********/

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SearchTaskNode>([](const ObjectRef& obj, ReprPrinter* p) {
      const auto* n = static_cast<const SearchTaskNode*>(obj.get());
      p->stream << "SearchTask(task_name=" << n->task_name << ", flop_ct=" << std::fixed
                << n->flop_ct << "), workload:\n"
                << Repr(n->workload);
    });

/********** FFI **********/

struct Internal {
  /*!
   * \brief Wrap for SearchTask::SearchTask
   * \param func The function to be optimized
   * \param task_name Name of this search task
   * \param target The target to be built at
   * \param target_host The target host to be built at
   * \return SearchTask, the new object constructed
   * \sa SearchTask::SearchTask
   */
  static SearchTask SearchTaskNew(tir::PrimFunc func, String task_name, Target target,
                                  Target target_host, Optional<String> log_file) {
    return SearchTask(func, task_name, target, target_host, log_file);
  }
  /*!
   * \brief Apply postprocessors onto the schedule
   * \param space The search space
   * \param sch The schedule to be postprocessed
   * \param rand_state The random state for sampling
   * \return Whether postprocessing has succeeded
   * \sa SearchSpaceNode::Postprocess
   */
  static bool SearchSpacePostprocess(SearchSpace space, SearchTask task, Schedule sch,
                                     Optional<Integer> seed) {
    tir::TRandState rand_state;
    if (seed.defined() && seed.value()->value != -1) {
      tir::RandEngine(&rand_state).Seed(seed.value()->value);
    } else {
      tir::RandEngine(&rand_state).Seed(std::random_device()());
    }
    return space->Postprocess(task, sch, &rand_state);
  }
  /*!
   * \brief Sample a schedule out of the search space, calls SearchSpaceNode::SampleSchedule
   * \param space The specific space
   * \param task The search task to be sampled from
   * \return The schedule sampled
   * \sa SearchSpaceNode::SampleSchedule
   */
  static Schedule SearchSpaceSampleSchedule(SearchSpace space, SearchTask task,
                                            Optional<Integer> seed) {
    tir::TRandState rand_state;
    if (seed.defined() && seed.value()->value != -1) {
      tir::RandEngine(&rand_state).Seed(seed.value()->value);
    } else {
      tir::RandEngine(&rand_state).Seed(std::random_device()());
    }
    return space->SampleSchedule(task, &rand_state);
  }
  /*!
   * \brief Get support of the search space, calls SearchSpaceNode::GetSupport
   * \param space The specific space
   * \param task The search task to be sampled from
   * \return The support of the search space. Any point from the search space should along to one of
   * the traces returned
   * \sa SearchSpaceNode::GetSupport
   */
  static Array<Schedule> SearchSpaceGetSupport(SearchSpace space, SearchTask task,
                                               Optional<Integer> seed) {
    tir::TRandState rand_state;
    if (seed.defined() && seed.value()->value != -1) {
      tir::RandEngine(&rand_state).Seed(seed.value()->value);
    } else {
      tir::RandEngine(&rand_state).Seed(std::random_device()());
    }
    return space->GetSupport(task, &rand_state);
  }
  /*!
   * \brief Explore the search space and find the best schedule
   * \param strategy The strategy to explore the search space
   * \param task The search task
   * \param space The search space
   * \param measurer The measurer that builds, runs and profiles sampled programs
   * \param verbose Whether or not in verbose mode
   * \return The best schedule found, NullOpt if no valid schedule is found
   */
  static Optional<Schedule> SearchStrategySearch(SearchStrategy strategy, SearchTask task,
                                                 SearchSpace space, ProgramMeasurer measurer,
                                                 Optional<Integer> seed, int verbose) {
    tir::TRandState rand_state;
    if (seed.defined() && seed.value()->value != -1) {
      tir::RandEngine(&rand_state).Seed(seed.value()->value);
    } else {
      tir::RandEngine(&rand_state).Seed(std::random_device()());
    }
    return strategy->Search(task, space, measurer, &rand_state, verbose);
  }
};

TVM_REGISTER_NODE_TYPE(SearchTaskNode);
TVM_REGISTER_OBJECT_TYPE(SearchSpaceNode);
TVM_REGISTER_OBJECT_TYPE(SearchStrategyNode);

TVM_REGISTER_GLOBAL("meta_schedule.SearchTask").set_body_typed(Internal::SearchTaskNew);
TVM_REGISTER_GLOBAL("meta_schedule.SearchSpacePostprocess")
    .set_body_typed(Internal::SearchSpacePostprocess);
TVM_REGISTER_GLOBAL("meta_schedule.SearchSpaceSampleSchedule")
    .set_body_typed(Internal::SearchSpaceSampleSchedule);
TVM_REGISTER_GLOBAL("meta_schedule.SearchSpaceGetSupport")
    .set_body_typed(Internal::SearchSpaceGetSupport);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategySearch")
    .set_body_typed(Internal::SearchStrategySearch);
TVM_REGISTER_GLOBAL("meta_schedule.AutoTune").set_body_typed(AutoTune);

}  // namespace meta_schedule
}  // namespace tvm
