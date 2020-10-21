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
#include "./postproc.h"  // NOLINT(build/include)

#include "../analysis.h"

namespace tvm {
namespace meta_schedule {

/********** Constructor **********/

Postproc::Postproc(String name, FProc proc) {
  ObjectPtr<PostprocNode> n = make_object<PostprocNode>();
  n->name = std::move(name);
  n->proc_ = std::move(proc);
  data_ = std::move(n);
}

/********** Postproc **********/

bool PostprocNode::Apply(const Schedule& sch, Sampler* sampler) { return proc_(sch, sampler); }

Array<Postproc> PostprocDefaults() { return {RewriteParallel(), RewriteVectorize()}; }

/********** Built-in Post-Processors **********/

Postproc RewriteParallel() {
  auto f_proc = [](Schedule self, void* _sampler) -> bool {
    Array<Array<tir::StmtSRef>> to_parallel = CollectAnnotatedLoops(self->sch, "lazy_parallel");
    for (const Array<tir::StmtSRef>& group : to_parallel) {
      for (const tir::StmtSRef& loop_sref : group) {
        const auto* loop = loop_sref->GetStmt<tir::LoopNode>();
        CHECK(loop) << "TypeError: Expects LoopNode, but gets: " << loop_sref->GetTypeKey();
        ObjectPtr<tir::LoopNode> new_loop = make_object<tir::LoopNode>(*loop);
        new_loop->annotations.clear();
        self->sch->Replace(loop_sref, tir::Loop(new_loop));
      }
      tir::StmtSRef fused = group[0];
      for (int i = 1, n = group.size(); i < n; ++i) {
        fused = self->sch->fuse(fused, group[i]);
      }
      self->sch->parallel(fused);
    }
    return true;
  };
  return Postproc("rewrite_parallel", f_proc);
}

Postproc RewriteVectorize() {
  auto f_proc = [](Schedule self, void* _sampler) -> bool {
    Array<Array<tir::StmtSRef>> to_vectorize = CollectAnnotatedLoops(self->sch, "lazy_vectorize");
    for (const Array<tir::StmtSRef>& group : to_vectorize) {
      for (const tir::StmtSRef& loop_sref : group) {
        const auto* loop = loop_sref->GetStmt<tir::LoopNode>();
        CHECK(loop) << "TypeError: Expects LoopNode, but gets: " << loop_sref->GetTypeKey();
        ObjectPtr<tir::LoopNode> new_loop = make_object<tir::LoopNode>(*loop);
        new_loop->annotations.clear();
        self->sch->Replace(loop_sref, tir::Loop(new_loop));
      }
      tir::StmtSRef fused = group[0];
      for (int i = 1, n = group.size(); i < n; ++i) {
        fused = self->sch->fuse(fused, group[i]);
      }
      self->sch->vectorize(fused);
    }
    return true;
  };
  return Postproc("rewrite_vectorize", f_proc);
}

/********** FFI **********/

struct Internal {
  /*!
   * \brief FFI function for PostProcNode::Apply
   * \sa PostProcNode::Apply
   */
  static bool Apply(Postproc self, Schedule sch, Optional<Integer> seed) {
    Sampler seeded;
    if (seed.defined()) {
      seeded.Seed(seed.value());
    }
    return self->Apply(sch, &seeded);
  }
};

TVM_REGISTER_NODE_TYPE(PostprocNode);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.Apply").set_body_typed(Internal::Apply);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.RewriteParallel").set_body_typed(RewriteParallel);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.RewriteVectorize").set_body_typed(RewriteVectorize);

}  // namespace meta_schedule
}  // namespace tvm
