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
#include <tvm/tir/transform.h>

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
    arith::Analyzer analyzer;
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
      // Vectorize the loops whose extent is possible to be > 1
      // TODO(@junrushao1994): move the logic to meta schedule class
      const tir::LoopNode* loop = fused->GetStmt<tir::LoopNode>();
      CHECK(loop);
      if (!analyzer.CanProve(loop->extent <= 1)) {
        sch->sch->vectorize(fused);
      }
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

/********** RewriteCudaThreadBind **********/

inline tir::IterVar MakeThreadIdx(const String& name) {
  return tir::IterVar(Range(nullptr), tir::Var(name), tir::kThreadIndex, name);
}

class PostprocRewriteCudaThreadBind {
 public:
  /*! \brief Number of threads in a CUDA warp, should be 32 */
  int warp_size;

  explicit PostprocRewriteCudaThreadBind(int warp_size) : warp_size(warp_size) {}

  bool BindMultiLevelTiled(const Schedule& sch, const BlockRV& block_rv) const {
    arith::Analyzer analyzer;
    Array<LoopRV> loop_rvs = sch->GetAxes(block_rv);
    std::vector<tir::StmtSRef> loop_srefs;
    // The indices of `blockIdx.x` / `vthread` / `threadIdx.x` annotation
    std::vector<int> block_idx;
    std::vector<int> vthread_idx;
    std::vector<int> thread_idx;
    PrimExpr prod_extent = Integer(1);
    for (const LoopRV& loop_rv : loop_rvs) {
      int i = loop_srefs.size();
      // Evaluate to a TIR loop
      tir::StmtSRef loop_sref = sch->Eval(loop_rv);
      loop_srefs.push_back(loop_sref);
      const auto* loop = loop_sref->GetStmt<tir::LoopNode>();
      CHECK(loop) << "TypeError: Expects LoopNode, but gets: " << loop_sref->stmt->GetTypeKey();
      prod_extent = prod_extent * loop->extent;
      if (loop->annotations.empty()) {
        continue;
      }
      if (loop->annotations.size() != 1) {
        return false;
      }
      // Check the annotation
      if (loop->annotations[0]->attr_key == tir::attr::loop_type) {
        if (const auto* str_imm = loop->annotations[0]->value.as<tir::StringImmNode>()) {
          const String& ann = str_imm->value;
          if (ann == "lazy_blockIdx.x") {
            block_idx.push_back(i);
          } else if (ann == "lazy_vthread") {
            vthread_idx.push_back(i);
          } else if (ann == "lazy_threadIdx.x") {
            thread_idx.push_back(i);
          } else {
            return false;
          }
        }
      }
    }
    prod_extent = analyzer.Simplify(prod_extent);
    // Check if the block is annotated
    if (block_idx.empty()) {
      return false;
    }
    CHECK(!vthread_idx.empty());
    CHECK(!thread_idx.empty());
    // Remove the annotation on the loop
    for (const tir::StmtSRef& loop_sref : loop_srefs) {
      const auto* loop = loop_sref->GetStmt<tir::LoopNode>();
      CHECK(loop) << "TypeError: Expects LoopNode, but gets: " << loop_sref->stmt->GetTypeKey();
      ObjectPtr<tir::LoopNode> new_loop = make_object<tir::LoopNode>(*loop);
      new_loop->annotations.clear();
      sch->sch->Replace(loop_sref, tir::Loop(new_loop));
    }
    // Check if `prod_extent <= warp_size`
    if (analyzer.CanProve(prod_extent <= warp_size)) {
      // If so, only bind `threadIdx.x`
      std::vector<int> indices;
      indices.insert(indices.end(), block_idx.begin(), block_idx.end());
      indices.insert(indices.end(), vthread_idx.begin(), vthread_idx.end());
      indices.insert(indices.end(), thread_idx.begin(), thread_idx.end());
      std::sort(indices.begin(), indices.end());
      std::vector<LoopRV> to_fuse;
      for (int idx : indices) {
        to_fuse.push_back(loop_rvs[idx]);
      }
      LoopRV fused = sch->Fuse(to_fuse);
      sch->sch->bind(sch->Eval(fused), MakeThreadIdx("threadIdx.x"));
    } else {
      // Otherwise, bind `blockIdx.x`, `vthread` and `threadIdx.x`
      {
        // bind `blockIdx.x`
        std::vector<LoopRV> to_fuse;
        for (int idx : block_idx) {
          to_fuse.push_back(loop_rvs[idx]);
        }
        LoopRV fused = sch->Fuse(to_fuse);
        sch->sch->bind(sch->Eval(fused), MakeThreadIdx("blockIdx.x"));
      }
      {
        // bind `vthread`
        std::vector<LoopRV> to_fuse;
        for (int idx : vthread_idx) {
          to_fuse.push_back(loop_rvs[idx]);
        }
        LoopRV fused = sch->Fuse(to_fuse);
        sch->sch->bind(sch->Eval(fused), MakeThreadIdx("vthread"));
      }
      {
        // bind `threadIdx.x`
        std::vector<LoopRV> to_fuse;
        for (int idx : thread_idx) {
          to_fuse.push_back(loop_rvs[idx]);
        }
        LoopRV fused = sch->Fuse(to_fuse);
        sch->sch->bind(sch->Eval(fused), MakeThreadIdx("threadIdx.x"));
      }
    }
    return true;
  }

  bool BindSpatial(const Schedule& sch, const BlockRV& block_rv) const {
    Array<LoopRV> loop_rvs = sch->GetAxes(block_rv);
    tir::StmtSRef block_sref = sch->Eval(block_rv);
    Array<tir::StmtSRef> loop_srefs;
    for (const LoopRV& loop_rv : loop_rvs) {
      tir::StmtSRef loop_sref = sch->Eval(loop_rv);
      loop_srefs.push_back(loop_sref);
      const auto* loop = loop_sref->GetStmt<tir::LoopNode>();
      CHECK(loop) << "TypeError: Expects LoopNode, but gets: " << loop_sref->stmt->GetTypeKey();
      if (!loop->annotations.empty()) {
        return false;
      }
    }
    Array<Integer> loop_types = GetLoopType(sch->sch, block_sref, loop_srefs);
    int n_spatial = 0;
    for (const Integer& _loop_type : loop_types) {
      int loop_type = _loop_type;
      if (loop_type == tir::kDataPar) {
        ++n_spatial;
      } else {
        break;
      }
    }
    CHECK(n_spatial > 0) << "NotImplementedError: binding no spatial loops is not supported yet";
    LoopRV fused = sch->Fuse({loop_rvs.begin(), loop_rvs.begin() + n_spatial});
    Array<LoopRV> splits = sch->Split(fused, {NullOpt, Integer(32)});
    CHECK_EQ(splits.size(), 2);
    sch->sch->bind(sch->Eval(splits[0]), MakeThreadIdx("blockIdx.x"));
    sch->sch->bind(sch->Eval(splits[1]), MakeThreadIdx("threadIdx.x"));
    return true;
  }

  bool Proc(const Schedule& sch) const {
    Array<BlockRV> root_block_rvs = sch->GetRootBlocks();
    for (const BlockRV& block_rv : root_block_rvs) {
      if (BindMultiLevelTiled(sch, block_rv)) {
        continue;
      }
      if (BindSpatial(sch, block_rv)) {
        continue;
      }
    }
    return true;
  }
};

Postproc RewriteCudaThreadBind(int warp_size) {
  auto f_proc = [warp_size](Schedule sch, void* _sampler) -> bool {
    return PostprocRewriteCudaThreadBind(warp_size).Proc(sch);
  };
  return Postproc("rewrite_cuda_thread_bind", f_proc);
}

/********** VerifyGPUCode **********/

class PostprocVerifyGPUCode {
 public:
  int shared_memory_per_block;
  int registers_per_block;
  int max_threads_per_block;
  int vector_unit_bytes;

  explicit PostprocVerifyGPUCode(Target target)
      : shared_memory_per_block(ExtractFromTarget(target, "shared_memory_per_block")),
        registers_per_block(ExtractFromTarget(target, "registers_per_block")),
        max_threads_per_block(ExtractFromTarget(target, "max_threads_per_block")),
        vector_unit_bytes(ExtractFromTarget(target, "vector_unit_bytes")) {}

  static int ExtractFromTarget(const Target& target, const char* name) {
    if (Optional<Integer> v = target->GetAttr<Integer>(name)) {
      return v.value();
    }
    LOG(FATAL) << "AttributedError: \"shared_memory_per_block\" is not defined in the target";
    throw;
  }

  bool Proc(const Schedule& sch) const {
    static const constexpr int MAX_VTHREADS = 8;
    tir::transform::Sequential passes({
        // Phase 0
        tir::transform::InjectPrefetch(),
        tir::transform::BufferFlatten(),
        // Phase 1
        tir::transform::NarrowDataType(32),
        tir::transform::Simplify(),
        tir::transform::VectorizeLoop(true),
        tir::transform::InjectVirtualThread(),
        tir::transform::StorageRewrite(),
        tir::transform::Simplify(),
        tir::transform::VerifyGPUCode({
            {"max_shared_memory_per_block", shared_memory_per_block},
            {"max_local_memory_per_block", registers_per_block},
            {"max_threads_per_block", max_threads_per_block},
            {"max_vector_bytes", vector_unit_bytes},
            {"max_vthread", MAX_VTHREADS},
        }),
    });
    IRModule mod({{GlobalVar("main"), sch->sch->func}});
    mod = passes(std::move(mod));
    return true;
  }
};

Postproc VerifyGPUCode(Target target) {
  PostprocVerifyGPUCode postproc(target);
  auto f_proc = [postproc{std::move(postproc)}](Schedule sch, void* _sampler) -> bool {
    return postproc.Proc(sch);
  };
  return Postproc("verify_gpu_code", f_proc);
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
TVM_REGISTER_GLOBAL("meta_schedule.postproc.RewriteCudaThreadBind")
    .set_body_typed(RewriteCudaThreadBind);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.VerifyGPUCode").set_body_typed(VerifyGPUCode);

}  // namespace meta_schedule
}  // namespace tvm
