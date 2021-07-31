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

#include "../search.h"
#include "./postproc.h"

namespace tvm {
namespace meta_schedule {

using runtime::TypedPackedFunc;

/********** Definition for ScheduleFn **********/

/*! \brief Search space that is specified by a schedule function */
class ScheduleFnNode : public SearchSpaceNode {
 public:
  /*! \brief The schedule function used */
  TypedPackedFunc<void(Schedule)> sch_fn_;
  /*! \brief The postprocessors */
  Array<Postproc> postprocs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("postprocs", &postprocs);
    // sch_fn_ is not visited
  }
  /*! \brief Default destructor */
  ~ScheduleFnNode() = default;

  /*!
   * \brief Apply postprocessors onto the schedule
   * \param task The search task
   * \param sch The schedule to be postprocessed
   * \param rand_state The sampler's random state
   */
  bool Postprocess(const SearchTask& task, const Schedule& sch,
                   Sampler::TRandomState* rand_state) override;
  /*!
   * \brief Sample a schedule out of the search space
   * \param task The search task to be sampled from
   * \return The schedule sampled
   */
  Schedule SampleSchedule(const SearchTask& task, Sampler::TRandomState* rand_state) override;
  /*!
   * \brief Get support of the search space
   * \param task The search task to be sampled from
   * \return An array with a single element returned from SampleSchedule
   * \sa ScheduleFnNode::SampleSchedule
   */
  Array<Schedule> GetSupport(const SearchTask& task, Sampler::TRandomState* rand_state) override;

  static constexpr const char* _type_key = "meta_schedule.ScheduleFn";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleFnNode, SearchSpaceNode);
};

/*!
 * \brief Managed reference to ScheduleFnNode
 */
class ScheduleFn : public SearchSpace {
 public:
  /*!
   * Constructor
   * \param sch_fn The schedule function
   * \param postprocs The postprocessors
   */
  explicit ScheduleFn(PackedFunc sch_fn, Array<Postproc> postprocs);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScheduleFn, SearchSpace, ScheduleFnNode);
};

/********** Constructor **********/

ScheduleFn::ScheduleFn(PackedFunc sch_fn, Array<Postproc> postprocs) {
  ObjectPtr<ScheduleFnNode> n = make_object<ScheduleFnNode>();
  n->postprocs = std::move(postprocs);
  n->sch_fn_ = sch_fn;
  data_ = std::move(n);
}

/********** Sampling **********/

bool ScheduleFnNode::Postprocess(const SearchTask& task, const Schedule& sch,
                                 Sampler::TRandomState* rand_state) {
  sch->EnterPostproc();
  for (const Postproc& postproc : postprocs) {
    if (!postproc->Apply(task, sch, rand_state)) {
      return false;
    }
  }
  return true;
}

Schedule ScheduleFnNode::SampleSchedule(const SearchTask& task, Sampler::TRandomState* rand_state) {
  Schedule sch = Schedule::Traced(/*mod=*/IRModule({{GlobalVar("main"), task->workload}}),
                                  /*seed=*/Sampler(rand_state).ForkSeed(),
                                  /*debug_mode=*/false,
                                  /*error_render_level=*/tir::ScheduleErrorRenderLevel::kDetail);
  this->sch_fn_(sch);
  return sch;
}

Array<Schedule> ScheduleFnNode::GetSupport(const SearchTask& task,
                                           Sampler::TRandomState* rand_state) {
  return {SampleSchedule(task, rand_state)};
}

/********** FFI **********/

struct Internal {
  /*!
   * \brief Constructor of ScheduleFn
   * \param sch_fn The schedule function
   * \param postprocs The postprocessors
   * \return The ScheduleFn constructed
   * \sa ScheduleFn::ScheduleFn
   */
  static ScheduleFn New(PackedFunc sch_fn, Array<Postproc> postprocs) {
    return ScheduleFn(sch_fn, postprocs);
  }
};

TVM_REGISTER_NODE_TYPE(ScheduleFnNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleFn").set_body_typed(Internal::New);

}  // namespace meta_schedule
}  // namespace tvm
