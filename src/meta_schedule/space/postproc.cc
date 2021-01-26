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

#include "../../tir/schedule/schedule_common.h"
#include "../analysis.h"
#include "../utils.h"

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

bool PostprocNode::Apply(const SearchTask& task, const Schedule& sch, Sampler* sampler) {
  return proc_(task, sch, sampler);
}

/********** RewriteTensorize **********/

class PostprocRewriteTensorize {
 public:
  Array<tir::TensorIntrin> tensor_intrins;

  explicit PostprocRewriteTensorize(Array<tir::TensorIntrin> tensor_intrins)
      : tensor_intrins(tensor_intrins) {}

  Optional<tir::StmtSRef> FindTensorized(const Schedule& sch) {
    Optional<tir::StmtSRef> result = NullOpt;
    tir::PreOrderVisit(sch->sch->func->body, [&result, &sch](const ObjectRef& obj) -> bool {
      if (const auto* block = obj.as<tir::BlockNode>()) {
        tir::StmtSRef block_sref = sch->sch->stmt2ref.at(block);
        if (HasAnn(block_sref, tir::attr::auto_tensorize, "1")) {
          result = block_sref;
          return false;
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
    while (Optional<tir::StmtSRef> opt_block_sref = FindTensorized(sch)) {
      tir::StmtSRef block_sref = opt_block_sref.value();
      // Remove the annotation
      DelAnn(sch->sch, block_sref, tir::attr::auto_tensorize);
      // Get the surrounding loops
      Array<tir::StmtSRef> loop_srefs = sch->sch->GetLoopsInScope(block_sref);
      // Decompose Reduction
      {
        const auto* block = block_sref->GetStmt<tir::BlockNode>();
        CHECK(block) << "TypeError: Expects BlockNode, but gets: "
                     << block_sref->stmt->GetTypeKey();
        if (block->body->IsInstance<tir::ReduceStepNode>()) {
          sch->sch->decompose_reduction(block_sref, loop_srefs[0]);
        }
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
  auto f_proc = [tensor_intrins{std::move(tensor_intrins)}](SearchTask task, Schedule self,
                                                            void* _sampler) -> bool {
    return PostprocRewriteTensorize(tensor_intrins).Proc(self);
  };
  return Postproc("rewrite_tensorize", f_proc);
}

/********** RewriteCooperativeFetch **********/

class PostprocRewriteCooperativeFetch {
 public:
  static std::function<bool(const tir::BlockNode* block)> MakeBlockFinder(const tir::Schedule& sch,
                                                                          int* idx) {
    return [&sch, idx](const tir::BlockNode* block) -> bool {
      const tir::StmtSRefNode* sref = sch->stmt2ref.at(block)->parent;
      for (int& i = *idx = 0; sref != nullptr; sref = sref->parent, ++i) {
        const tir::LoopNode* loop = sref->GetStmt<tir::LoopNode>();
        if (!loop) {
          break;
        }
        if (HasAnn(sch->stmt2ref.at(loop), tir::attr::loop_type, "lazy_cooperative_fetch")) {
          return true;
        }
      }
      return false;
    };
  }

  bool Proc(const Schedule& sch) const {
    int idx = 0;
    while (Optional<tir::StmtSRef> opt_block_sref =
               FindBlockSRef(sch->sch, MakeBlockFinder(sch->sch, &idx))) {
      // Extract block info
      tir::StmtSRef block_sref = opt_block_sref.value();
      const auto* block = block_sref->GetStmt<tir::BlockNode>();
      BlockRV block_rv = sch->GetBlock(block->tag);
      // Extract loop info
      Array<LoopRV> loop_rvs = sch->GetAxes(block_rv);
      int n_loops = loop_rvs.size();
      CHECK_LT(idx, n_loops);
      LoopRV loop_rv = loop_rvs[n_loops - 1 - idx];
      tir::StmtSRef loop_sref = sch->Eval(loop_rv);
      // Remove the annotation
      DelAnn(sch->sch, loop_sref, tir::attr::loop_type);
      // Find the threadIdx.x binding
      PrimExpr thread_idx_extent{nullptr};
      for (const tir::StmtSRefNode* sref = loop_sref->parent;; sref = sref->parent) {
        CHECK(sref) << "ValueError: Cannot find loops above with threadIdx.x";
        if (const tir::LoopNode* loop = sref->GetStmt<tir::LoopNode>()) {
          if (HasAnn(GetRef<tir::StmtSRef>(sref), tir::attr::loop_type, "threadIdx.x")) {
            CHECK(tir::is_zero(loop->min)) << "ValueError: Expect loops to start from 0, but gets: "
                                           << GetRef<tir::Loop>(loop);
            thread_idx_extent = loop->extent;
            break;
          }
        }
      }
      // Split the loop
      Array<LoopRV> split = sch->Split(loop_rv, {thread_idx_extent, NullOpt});
      CHECK_EQ(split.size(), 2);
      sch->Bind(split[0], "threadIdx.x");
    }
    return true;
  }
};

Postproc RewriteCooperativeFetch() {
  auto f_proc = [](SearchTask task, Schedule sch, void* _sampler) -> bool {
    return PostprocRewriteCooperativeFetch().Proc(sch);
  };
  return Postproc("rewrite_cooperative_fetch", f_proc);
}

/********** RewriteParallelizeVectorizeUnroll **********/

class PostprocRewriteParallelizeVectorizeUnroll {
 public:
  struct Parsed {
    int max_parallel_extent;
    int max_vectorize_extent;
    int unroll_explicit;
    int unroll_implicit;
    int num_parallel_loops;
    int num_vectorize_loops;
  };

  static bool HasSingleChild(const tir::StmtSRef& loop_sref) {
    const auto* loop = loop_sref->GetStmt<tir::LoopNode>();
    CHECK(loop) << "TypeError: Expects LoopNode, but gets: " << loop_sref->stmt->GetTypeKey();
    return !loop->body->IsInstance<tir::SeqStmtNode>();
  }

  static std::function<bool(const tir::BlockNode*)> MakeAnnParser(Parsed* parsed) {
    return [parsed](const tir::BlockNode* block) -> bool {
      bool found = false;
      *parsed = Parsed{-1, -1, -1, -1, -1, -1};
      for (const tir::Annotation& ann : block->annotations) {
        if (ann->attr_key == tir::attr::auto_parallel_extent) {
          found = true;
          if (const auto* str_imm = ann->value.as<tir::StringImmNode>()) {
            parsed->max_parallel_extent = std::atoi(str_imm->value.c_str());
          }
        } else if (ann->attr_key == tir::attr::auto_vectorize_extent) {
          found = true;
          if (const auto* str_imm = ann->value.as<tir::StringImmNode>()) {
            parsed->max_vectorize_extent = std::atoi(str_imm->value.c_str());
          }
        } else if (ann->attr_key == tir::attr::auto_unroll_explicit) {
          found = true;
          if (const auto* str_imm = ann->value.as<tir::StringImmNode>()) {
            parsed->unroll_explicit = std::atoi(str_imm->value.c_str());
          }
        } else if (ann->attr_key == tir::attr::auto_unroll_implicit) {
          found = true;
          if (const auto* str_imm = ann->value.as<tir::StringImmNode>()) {
            parsed->unroll_implicit = std::atoi(str_imm->value.c_str());
          }
        }
      }
      return found;
    };
  }

  static void RemoveParsedAnn(const tir::Schedule& sch, const tir::StmtSRef& block_sref,
                              const Parsed& parsed) {
    if (parsed.max_parallel_extent != -1) {
      DelAnn(sch, block_sref, tir::attr::auto_parallel_extent);
    }
    if (parsed.max_vectorize_extent != -1) {
      DelAnn(sch, block_sref, tir::attr::auto_vectorize_extent);
    }
    if (parsed.unroll_explicit != -1) {
      DelAnn(sch, block_sref, tir::attr::auto_unroll_explicit);
    }
    if (parsed.unroll_implicit != -1) {
      DelAnn(sch, block_sref, tir::attr::auto_unroll_implicit);
    }
  }

  static void AdjustParallelVectorize(const Schedule& sch, const tir::StmtSRef& block_sref,
                                      const Array<LoopRV>& loop_rvs, Parsed* parsed) {
    if (parsed->max_parallel_extent == -1 && parsed->max_vectorize_extent == -1) {
      return;
    }
    int n_loops = loop_rvs.size();
    // Extract loop_srefs, and calculate the iterator types
    Array<tir::StmtSRef> loop_srefs;
    std::vector<int> loop_types;
    {
      loop_srefs.reserve(n_loops);
      loop_types.reserve(n_loops);
      for (const LoopRV& loop_rv : loop_rvs) {
        loop_srefs.push_back(sch->Eval(loop_rv));
        loop_types.push_back(GetLoopIterType(sch->sch, loop_srefs.back()));
      }
    }
    // Calculate the parallelize extent
    if (parsed->max_parallel_extent != -1) {
      int max_extent = parsed->max_parallel_extent;
      int& num_fusible = parsed->num_parallel_loops = 0;
      int64_t prod_extent = 1;
      for (int i = 0; i < n_loops && loop_types[i] == tir::IterVarType::kDataPar; ++i) {
        const tir::StmtSRef& loop_sref = loop_srefs[i];
        // Check if the loop extent is valid
        Optional<Integer> extent = GetLoopIntExtent(loop_sref);
        if (!extent.defined()) {
          break;
        }
        // Then we can fuse it in
        ++num_fusible;
        // Check if we need to break
        prod_extent *= extent.value()->value;
        if (prod_extent > max_extent || !HasSingleChild(loop_sref)) {
          break;
        }
      }
      if (prod_extent == 1) {
        num_fusible = -1;
      }
    }
    // Calculate the vectorize extent
    if (parsed->max_vectorize_extent != -1) {
      int max_extent = parsed->max_vectorize_extent;
      int& num_fusible = parsed->num_vectorize_loops = 0;
      int64_t prod_extent = 1;
      for (int i = n_loops - 1; i >= 0 && loop_types[i] == tir::IterVarType::kDataPar; --i) {
        const tir::StmtSRef& loop_sref = loop_srefs[i];
        // Cannot fuse with a loop with multiple children
        if (!HasSingleChild(loop_sref)) {
          break;
        }
        // Check if the loop extent is valid
        Optional<Integer> extent = GetLoopIntExtent(loop_sref);
        if (!extent.defined()) {
          break;
        }
        // Check if the extent is still in a good range
        prod_extent *= extent.value()->value;
        if (prod_extent > max_extent) {
          break;
        }
        ++num_fusible;
      }
      if (prod_extent == 1) {
        num_fusible = -1;
      }
    }
    // Prefer num_vectorize to num_parallel
    if (parsed->num_parallel_loops != -1 && parsed->num_vectorize_loops != -1) {
      parsed->num_parallel_loops = std::min(parsed->num_parallel_loops,  //
                                            n_loops - parsed->num_parallel_loops);
    }
  }

  bool Proc(const Schedule& sch) const {
    Parsed parsed;
    while (Optional<tir::StmtSRef> opt_block_sref =
               FindBlockSRef(sch->sch, MakeAnnParser(&parsed))) {
      // Extract block info
      tir::StmtSRef block_sref = opt_block_sref.value();
      RemoveParsedAnn(sch->sch, block_sref, parsed);
      const auto* block = block_sref->GetStmt<tir::BlockNode>();
      BlockRV block_rv = sch->GetBlock(block->tag);
      // Extract loop info
      Array<LoopRV> loop_rvs = sch->GetAxes(block_rv);
      int n_loops = loop_rvs.size();
      if (n_loops == 0) {
        continue;
      }
      AdjustParallelVectorize(sch, block_sref, loop_rvs, &parsed);
      // Parallelize
      if (parsed.num_parallel_loops > 0) {
        LoopRV fused = sch->Fuse({loop_rvs.begin(), loop_rvs.begin() + parsed.num_parallel_loops});
        sch->Parallel(fused);
        for (int i = 0; i < parsed.num_parallel_loops; ++i) {
          loop_rvs.Set(i, fused);
        }
      }
      // Vectorize
      if (parsed.num_vectorize_loops > 0) {
        LoopRV fused = sch->Fuse({loop_rvs.end() - parsed.num_vectorize_loops, loop_rvs.end()});
        sch->Vectorize(fused);
        for (int i = n_loops - parsed.num_vectorize_loops; i < n_loops; ++i) {
          loop_rvs.Set(i, fused);
        }
      }
      // AutoUnroll
      if (parsed.unroll_explicit != -1 || parsed.unroll_implicit != -1) {
        CHECK(!(parsed.unroll_explicit != -1 && parsed.unroll_implicit != -1))
            << "ValueError: `auto_unroll_explicit` and `auto_unroll_explicit` cannot co-exist";
        int unroll_explicit = parsed.unroll_explicit != -1;
        int max_step = parsed.unroll_explicit + parsed.unroll_implicit + 1;
        LoopRV loop = loop_rvs[0];
        if (max_step > 0) {
          sch->MarkLoop(loop, "pragma_auto_unroll_max_step", Integer(max_step));
          sch->MarkLoop(loop, "pragma_unroll_explicit", Integer(unroll_explicit));
        }
      }
    }
    return true;
  }
};

Postproc RewriteParallelizeVectorizeUnroll() {
  auto f_proc = [](SearchTask task, Schedule sch, void* _sampler) -> bool {
    return PostprocRewriteParallelizeVectorizeUnroll().Proc(sch);
  };
  return Postproc("rewrite_parallelize_vectorize_unroll", f_proc);
}

/********** RewriteUnboundBlocks **********/

class PostprocRewriteUnboundBlocks {
 public:
  /*! \brief A helper class to find an unbound block */
  class UnboundBlockFinder : public tir::StmtExprVisitor {
   public:
    /*! \brief Find the first block that is not bound to any thread axes */
    static Optional<tir::StmtSRef> Find(const tir::Schedule& sch) {
      UnboundBlockFinder finder(sch);
      finder(GetRef<tir::Stmt>(sch->root->stmt));
      return finder.block_ == nullptr ? Optional<tir::StmtSRef>(NullOpt)
                                      : sch->stmt2ref.at(finder.block_);
    }

   private:
    explicit UnboundBlockFinder(const tir::Schedule& sch) : sch_(sch), block_(nullptr) {}

    void VisitStmt_(const tir::LoopNode* loop) override {
      if (block_) {
        return;
      }
      if (Optional<String> opt_ann = GetAnn(sch_->stmt2ref.at(loop), tir::attr::loop_type)) {
        String ann = opt_ann.value();
        if (ann == "threadIdx.x" || ann == "blockIdx.x" || ann == "vthread") {
          return;
        }
      }
      tir::StmtExprVisitor::VisitStmt_(loop);
    }

    void VisitStmt_(const tir::BlockNode* block) override {
      if (block == sch_->root->stmt) {
        tir::StmtExprVisitor::VisitStmt_(block);
      } else if (!block_) {
        block_ = block;
      }
    }

    const tir::Schedule& sch_;
    const tir::BlockNode* block_;
  };

  void BindThreadAxes(const Schedule& sch, const tir::StmtSRef& block_sref) const {
    // Extract the block
    const auto* block = block_sref->GetStmt<tir::BlockNode>();
    CHECK(block) << "TypeError: Expects Block, but gets: " << block_sref->stmt->GetTypeKey();
    BlockRV block_rv = sch->GetBlock(block->tag);
    // Extract loops
    Array<LoopRV> loop_rvs = sch->GetAxes(block_rv);
    // Find the outer spatial loops
    // TODO(@junrushao1994): check if each loop has only one children, otherwise we cannot fuse
    int n_spatial_loops = 0;
    for (const LoopRV& loop_rv : loop_rvs) {
      tir::IterVarType iter_type = GetLoopIterType(sch->sch, sch->Eval(loop_rv));
      if (iter_type != tir::kDataPar) {
        break;
      }
      ++n_spatial_loops;
    }
    CHECK_GT(n_spatial_loops, 0) << "ValueError: not supported when spatial loop doesn't exist";
    // Fuse the spatial loops
    LoopRV fused = sch->Fuse({loop_rvs.begin(), loop_rvs.begin() + n_spatial_loops});
    Array<LoopRV> splits = sch->Split(fused, {NullOpt, Integer(32)});
    CHECK_EQ(splits.size(), 2);
    sch->Bind(splits[0], "blockIdx.x");
    sch->Bind(splits[1], "threadIdx.x");
  }

  bool Proc(const SearchTask& task, const Schedule& sch) const {
    int warp_size = task->target->GetAttr<Integer>("thread_warp_size").value_or(Integer(-1));
    CHECK(warp_size != -1) << "ValueError: Target does not have attribute \"thread_warp_size\"";
    while (Optional<tir::StmtSRef> opt_block_sref = UnboundBlockFinder::Find(sch->sch)) {
      BindThreadAxes(sch, opt_block_sref.value());
    }
    return true;
  }
};

Postproc RewriteUnboundBlocks() {
  auto f_proc = [](SearchTask task, Schedule sch, void* _sampler) -> bool {
    return PostprocRewriteUnboundBlocks().Proc(task, sch);
  };
  return Postproc("rewrite_unbound_blocks", f_proc);
}

/********** RewriteReduceStep **********/

class PostprocRewriteReduceStep {
 public:
  class Finder : public tir::StmtVisitor {
   public:
    Finder() : result_(nullptr), stack_() {}

    static const tir::BlockNode* Find(const tir::Stmt& stmt) {
      Finder finder;
      finder.VisitStmt(stmt);
      return finder.result_;
    }

   private:
    void VisitStmt_(const tir::BlockNode* block) override {
      if (!result_) {
        stack_.push_back(block);
        tir::StmtVisitor::VisitStmt_(block);
        stack_.pop_back();
      }
    }

    void VisitStmt_(const tir::LoopNode* loop) override {
      if (!result_) {
        tir::StmtVisitor::VisitStmt_(loop);
      }
    }

    void VisitStmt_(const tir::ReduceStepNode* reduce_step) override {
      if (!result_) {
        result_ = stack_.back();
      }
    }

   private:
    const tir::BlockNode* result_ = nullptr;
    std::vector<const tir::BlockNode*> stack_;
  };

  bool Proc(const Schedule& sch) const {
    while (const tir::BlockNode* block = Finder::Find(sch->sch->func->body)) {
      BlockRV block_rv = sch->GetBlock(block->tag);
      Array<LoopRV> loop_rvs = sch->GetAxes(block_rv);
      int n_loops = loop_rvs.size();
      for (int i = 0; i < n_loops; ++i) {
        const LoopRV& loop_rv = loop_rvs[i];
        tir::StmtSRef loop_sref = sch->Eval(loop_rv);
        if (GetLoopIterType(sch->sch, loop_sref) != tir::kDataPar) {
          if (i >= 1) {
            sch->DecomposeReduction(block_rv, loop_rvs[i - 1]);
          }
          break;
        }
      }
    }
    return true;
  }
};

Postproc RewriteReduceStep() {
  auto f_proc = [](SearchTask task, Schedule sch, void* _sampler) -> bool {
    return PostprocRewriteReduceStep().Proc(sch);
  };
  return Postproc("rewrite_reduce_step", f_proc);
}

/********** VerifyGPUCode **********/

class PostprocVerifyGPUCode {
 public:
  static Integer Extract(const Target& target, const char* name) {
    if (Optional<Integer> v = target->GetAttr<Integer>(name)) {
      return v.value();
    }
    LOG(FATAL) << "AttributedError: \"" << name << "\" is not defined in the target";
    throw;
  }

  static tir::transform::Sequential MakePasses(const Target& target) {
    return tir::transform::Sequential(
        {tir::transform::InjectPrefetch(),       //
         tir::transform::BufferFlatten(),        //
         tir::transform::NarrowDataType(32),     //
         tir::transform::Simplify(),             //
         tir::transform::VectorizeLoop(true),    //
         tir::transform::InjectVirtualThread(),  //
         tir::transform::StorageRewrite(),       //
         tir::transform::Simplify(),             //
         tir::transform::VerifyGPUCode({
             {"max_shared_memory_per_block", Extract(target, "shared_memory_per_block")},
             {"max_local_memory_per_block", Extract(target, "registers_per_block")},
             {"max_threads_per_block", Extract(target, "max_threads_per_block")},
             {"max_vector_bytes", Extract(target, "vector_unit_bytes")},
             {"max_vthread", Integer(8)},
         })});
  }

  bool Proc(const SearchTask& task, const Schedule& sch) const {
    tir::transform::Sequential passes = MakePasses(task->target);
    IRModule mod({{GlobalVar("main"), sch->sch->func}});
    try {
      passes(std::move(mod));
    } catch (const dmlc::Error& e) {
      return false;
    }
    return true;
  }
};

Postproc VerifyGPUCode() {
  auto f_proc = [](SearchTask task, Schedule sch, void* _sampler) -> bool {
    return PostprocVerifyGPUCode().Proc(task, sch);
  };
  return Postproc("verify_gpu_code", f_proc);
}

/********** FFI **********/

struct Internal {
  /*!
   * \brief FFI function for PostProcNode::Apply
   * \sa PostProcNode::Apply
   */
  static bool Apply(Postproc self, SearchTask task, Schedule sch, Optional<Integer> seed) {
    Sampler seeded;
    if (seed.defined()) {
      seeded.Seed(seed.value());
    }
    return self->Apply(task, sch, &seeded);
  }
};

TVM_REGISTER_NODE_TYPE(PostprocNode);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.Apply").set_body_typed(Internal::Apply);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.RewriteTensorize").set_body_typed(RewriteTensorize);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.RewriteCooperativeFetch")
    .set_body_typed(RewriteCooperativeFetch);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.RewriteUnboundBlocks")
    .set_body_typed(RewriteUnboundBlocks);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.RewriteParallelizeVectorizeUnroll")
    .set_body_typed(RewriteParallelizeVectorizeUnroll);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.VerifyGPUCode").set_body_typed(VerifyGPUCode);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.RewriteReduceStep").set_body_typed(RewriteReduceStep);

}  // namespace meta_schedule
}  // namespace tvm
