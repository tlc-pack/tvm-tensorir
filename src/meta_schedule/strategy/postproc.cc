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

/********** RewriteParallel **********/

Postproc RewriteParallel() {
  auto f_proc = [](Schedule sch, void* _sampler) -> bool {
    Array<Array<tir::StmtSRef>> to_parallel = CollectAnnotatedLoops(sch->sch, "lazy_parallel");
    for (const Array<tir::StmtSRef>& group : to_parallel) {
      for (const tir::StmtSRef& loop_sref : group) {
        const auto* loop = loop_sref->GetStmt<tir::LoopNode>();
        CHECK(loop) << "TypeError: Expects LoopNode, but gets: " << loop_sref->GetTypeKey();
        ObjectPtr<tir::LoopNode> new_loop = make_object<tir::LoopNode>(*loop);
        new_loop->annotations.clear();
        sch->sch->Replace(loop_sref, tir::Loop(new_loop));
      }
      tir::StmtSRef fused = group[0];
      for (int i = 1, n = group.size(); i < n; ++i) {
        fused = sch->sch->fuse(fused, group[i]);
      }
      sch->sch->parallel(fused);
    }
    return true;
  };
  return Postproc("rewrite_parallel", f_proc);
}

/********** RewriteVectorize **********/

Postproc RewriteVectorize() {
  auto f_proc = [](Schedule sch, void* _sampler) -> bool {
    Array<Array<tir::StmtSRef>> to_vectorize = CollectAnnotatedLoops(sch->sch, "lazy_vectorize");
    for (const Array<tir::StmtSRef>& group : to_vectorize) {
      for (const tir::StmtSRef& loop_sref : group) {
        const auto* loop = loop_sref->GetStmt<tir::LoopNode>();
        CHECK(loop) << "TypeError: Expects LoopNode, but gets: " << loop_sref->GetTypeKey();
        ObjectPtr<tir::LoopNode> new_loop = make_object<tir::LoopNode>(*loop);
        new_loop->annotations.clear();
        sch->sch->Replace(loop_sref, tir::Loop(new_loop));
      }
      tir::StmtSRef fused = group[0];
      for (int i = 1, n = group.size(); i < n; ++i) {
        fused = sch->sch->fuse(fused, group[i]);
      }
      sch->sch->vectorize(fused);
    }
    return true;
  };
  return Postproc("rewrite_vectorize", f_proc);
}

/********** RewriteTensorize **********/

class PostprocRewriteTensorize {
 public:
  Array<tir::TensorIntrin> tensor_intrins;

  explicit PostprocRewriteTensorize(Array<tir::TensorIntrin> tensor_intrins)
      : tensor_intrins(tensor_intrins) {}

  Optional<tir::Block> FindAnnotatedBlock(const Schedule& sch) {
    Optional<tir::Block> result = NullOpt;
    tir::PreOrderVisit(sch->sch->func->body, [&result](const ObjectRef& obj) -> bool {
      if (const auto* block = obj.as<tir::BlockNode>()) {
        if (!block->annotations.empty()) {
          tir::Annotation ann = block->annotations[0];
          if (ann->attr_key == std::string(tir::attr::loop_type) &&
              Downcast<tir::StringImm>(ann->value)->value == "lazy_tensorize") {
            result = GetRef<tir::Block>(block);
            return false;
          }
        }
      }
      return true;
    });
    return result;
  }

  bool CanTensorize(const tir::Schedule& sch, const tir::StmtSRef& block_sref,
                    const tir::TensorIntrin& intrin) {
    Optional<TensorizeInfo> opt_tensorize_info =
        GetTensorizeLoopMapping(sch, block_sref, intrin->description);
    if (!opt_tensorize_info.defined()) {
      return false;
    }
    const auto* info = opt_tensorize_info.value().get();
    arith::Analyzer analyzer;
    for (const auto& kv : info->loop_map) {
      const tir::StmtSRef& block_loop_sref = kv.first;
      const auto* block_loop = block_loop_sref->GetStmt<tir::LoopNode>();
      const tir::Loop& desc_loop = kv.second;
      if (!analyzer.CanProve(block_loop->extent == desc_loop->extent)) {
        return false;
      }
    }
    return true;
  }

  bool Proc(const Schedule& sch) {
    while (Optional<tir::Block> opt_block = FindAnnotatedBlock(sch)) {
      tir::Block block = opt_block.value();
      tir::StmtSRef block_sref = sch->sch->stmt2ref.at(block.get());
      // Remove the annotation
      {
        ObjectPtr<tir::BlockNode> new_block = make_object<tir::BlockNode>(*block.get());
        new_block->annotations.clear();
        tir::Block new_block_obj = tir::Block(new_block);
        sch->sch->Replace(block_sref, new_block_obj, {{new_block_obj, block}});
        block = new_block_obj;
      }
      // Get the surrounding loops
      Array<tir::StmtSRef> loop_srefs = sch->sch->GetLoopsInScope(block_sref);
      // Decompose Reduction
      if (block->body->IsInstance<tir::ReduceStepNode>()) {
        sch->sch->decompose_reduction(block_sref, loop_srefs[0]);
      }
      // Tensorize
      for (const tir::TensorIntrin& intrin : tensor_intrins) {
        if (CanTensorize(sch->sch, block_sref, intrin)) {
          sch->sch->tensorize(loop_srefs[0], intrin);
          return true;
        }
      }
    }
    return false;
  }
};

Postproc RewriteTensorize(Array<tir::TensorIntrin> tensor_intrins) {
  auto f_proc = [tensor_intrins{std::move(tensor_intrins)}](Schedule self, void* _sampler) -> bool {
    return PostprocRewriteTensorize(tensor_intrins).Proc(self);
  };
  return Postproc("rewrite_tensorize", f_proc);
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
TVM_REGISTER_GLOBAL("meta_schedule.postproc.RewriteTensorize").set_body_typed(RewriteTensorize);

}  // namespace meta_schedule
}  // namespace tvm
