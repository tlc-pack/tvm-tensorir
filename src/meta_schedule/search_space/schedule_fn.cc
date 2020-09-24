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

namespace tvm {
namespace meta_schedule {

using runtime::TypedPackedFunc;

/********** Definition for ScheduleFn **********/

/*! \brief Search space that is specified by a schedule function */
class ScheduleFnNode : public SearchSpaceNode {
 public:
  /*! \brief The schedule function used */
  TypedPackedFunc<void(Schedule)> sch_fn_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // sch_fn_ is not visited
  }
  /*! \brief Default destructor */
  ~ScheduleFnNode() = default;
  /*!
   * \brief Sample a schedule out of the search space
   * \param task The search task to be sampled from
   * \return The schedule sampled
   */
  Schedule SampleSchedule(const SearchTask& task) override;
  /*!
   * \brief Get support of the search space
   * \param task The search task to be sampled from
   * \return An array with a single element returned from SampleSchedule
   * \sa ScheduleFnNode::SampleSchedule
   */
  Array<Schedule> GetSupport(const SearchTask& task) override;

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
   */
  explicit ScheduleFn(PackedFunc sch_fn);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScheduleFn, SearchSpace, ScheduleFnNode);
};

/********** Constructor **********/

ScheduleFn::ScheduleFn(PackedFunc sch_fn) {
  ObjectPtr<ScheduleFnNode> n = make_object<ScheduleFnNode>();
  n->sch_fn_ = sch_fn;
  data_ = std::move(n);
}

/********** Sampling **********/

Schedule ScheduleFnNode::SampleSchedule(const SearchTask& task) {
  Schedule sch(task->func);
  this->sch_fn_(sch);
  return sch;
}

Array<Schedule> ScheduleFnNode::GetSupport(const SearchTask& task) {
  return {SampleSchedule(task)};
}

/********** FFI **********/

struct Internal {
  /*!
   * \brief Constructor of ScheduleFn
   * \param sch_fn The schedule function
   * \return The ScheduleFn constructed
   * \sa ScheduleFn::ScheduleFn
   */
  static ScheduleFn New(PackedFunc sch_fn) { return ScheduleFn(sch_fn); }
};

TVM_REGISTER_NODE_TYPE(ScheduleFnNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleFn").set_body_typed(Internal::New);

}  // namespace meta_schedule
}  // namespace tvm
