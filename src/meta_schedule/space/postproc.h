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
#ifndef SRC_META_SCHEDULE_SPACE_POSTPROC_H_
#define SRC_META_SCHEDULE_SPACE_POSTPROC_H_

#include "../analysis.h"
#include "../schedule.h"
#include "../search.h"

namespace tvm {
namespace meta_schedule {

/********** Postproc **********/

/*! \brief A post processor, used for the search strategy for postprocess the schedule it gets */
class PostprocNode : public Object {
 public:
  /*! \brief The post-processor function */
  using FProc = runtime::TypedPackedFunc<bool(SearchTask, Schedule, void*)>;

  /*! \brief Name */
  String name;
  /*! \brief A packed function that applies the post-processor */
  FProc proc_;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("name", &name); }

  /*!
   * \brief Apply the postprocessor
   * \param sch The schedule to be processed
   * \param rand_state The sampler's random state
   * \return If the post-processing succeeds
   */
  bool Apply(const SearchTask& task, const Schedule& sch, Sampler::TRandState* rand_state);

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

/********** Built-in Post Processors **********/

/*!
 * \brief Creates a postprocessor that applies parallelization, vectorization and auto unrolling
 * according to the annotation of each block
 * \return The postprocessor created
 */
TVM_DLL Postproc RewriteParallelizeVectorizeUnroll();

/*!
 * \brief Creates a postprocessor that decomposes reduction blocks
 * \return The postprocessor created
 */
TVM_DLL Postproc RewriteReductionBlock();

/*!
 * \brief Creates a postprocessor that matches the region that is marked as auto tensorized
 * \return The postprocessor created
 */
TVM_DLL Postproc RewriteTensorize(Array<tir::TensorIntrin> tensor_intrins);

/*!
 * \brief Creates a postprocessor that rewrites "lazy_cooperative_fetch" with the actual threadIdx
 * \return The postprocessor created
 */
TVM_DLL Postproc RewriteCooperativeFetch();

/*!
 * \brief Create a postprocessor that finds each block that is not bound to thread axes, and bind
 * them to `blockIdx.x` and `threadIdx.x`
 * \return The postprocessor created
 */
TVM_DLL Postproc RewriteUnboundBlocks();

/*!
 * \brief Create a postprocessor that checks if all loops are static
 * \return The postprocessor created
 */
TVM_DLL Postproc DisallowDynamicLoops();

/*!
 * \brief Creates a postprocessor that verifies if the GPU code is correct
 * \return The postprocessor created
 */
TVM_DLL Postproc VerifyGPUCode();

struct LayoutRewriteHint {
  std::vector<Integer> extents;
  std::vector<Integer> reorder;
};

TVM_DLL Postproc RewriteLayout();

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_SPACE_POSTPROC_H_
