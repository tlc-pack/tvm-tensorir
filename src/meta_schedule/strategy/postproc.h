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
#ifndef SRC_META_SCHEDULE_STRATEGY_POSTPROC_H_
#define SRC_META_SCHEDULE_STRATEGY_POSTPROC_H_

#include "../schedule.h"
#include "../search.h"

namespace tvm {
namespace meta_schedule {

/********** Postproc **********/

/*! \brief A post processor, used for the search strategy for postprocess the schedule it gets */
class PostprocNode : public Object {
 public:
  /*! \brief The post-processor function */
  using FProc = runtime::TypedPackedFunc<bool(Schedule, void*)>;

  /*! \brief Name */
  String name;
  /*! \brief A packed function that applies the post-processor */
  FProc proc_;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("name", &name); }

  /*!
   * \brief Apply the postprocessor
   * \param sch The schedule to be processed
   * \param sampler The random number sampler
   * \return If the post-processing succeeds
   */
  bool Apply(const Schedule& sch, Sampler* sampler);

  static constexpr const char* _type_key = "meta_schedule.Postproc";
  TVM_DECLARE_BASE_OBJECT_INFO(PostprocNode, Object);
};

/*!
 * \brief Managed refernce to PostprocNode
 * \sa PostprocNode
 */
class Postproc : public ObjectRef {
 public:
  using FProc = PostprocNode::FProc;

  /*!
   * \brief Constructing with name and a packed function
   * \param name Name of the mutator
   * \param apply The application function
   */
  explicit Postproc(String name, FProc proc);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Postproc, ObjectRef, PostprocNode);
};

/*!
 * \brief Get the default post-processors
 * \return A list of post-processors
 */
Array<Postproc> PostprocDefaults();

/********** Built-in Post Processors **********/

/*!
 * \brief Creates a postprocessor that fuses the loops which are marked as "lazy_parallel",
 * and then parallelize the fused loop
 * \return The postprocessor
 */
TVM_DLL Postproc RewriteParallel();

/*!
 * \brief Creates a postprocessor that fuses the loops which are marked as "lazy_vectorize",
 * and then apply vectorization on the fused loop
 * \return The postprocessor
 */
TVM_DLL Postproc RewriteVectorize();

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_STRATEGY_POSTPROC_H_
