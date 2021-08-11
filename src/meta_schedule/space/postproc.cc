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

#include <tvm/tir/analysis.h>
#include <tvm/tir/transform.h>

#include "../../relay/transforms/meta_schedule_layout_rewrite.h"
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

bool PostprocNode::Apply(const SearchTask& task, const Schedule& sch, tir::TRandState* rand_state) {
  return proc_(task, sch, rand_state);
}

/********** RewriteTensorize **********/

class PostprocRewriteTensorize {
 public:
  Array<tir::TensorIntrin> tensor_intrins;

  explicit PostprocRewriteTensorize(Array<tir::TensorIntrin> tensor_intrins)
      : tensor_intrins(tensor_intrins) {}

  Optional<tir::StmtSRef> FindTensorized(const Schedule& sch) {
    Optional<tir::StmtSRef> result = NullOpt;
    tir::PrimFunc func = GetOnlyFunc(sch->mod());
    tir::PreOrderVisit(
        func->body,
        [&result, &sch](const ObjectRef& obj) -> bool {
          if (const auto* block = obj.as<tir::BlockNode>()) {
            tir::StmtSRef block_sref = sch->GetSRef(block);
            if (HasAnn(block_sref, tir::attr::auto_tensorize, "1")) {
              result = block_sref;
              return false;
            }
          }
          return true;
        },
        /*visit_init_block=*/false);
    return result;
  }

  bool CanTensorize(const tir::Schedule& sch, const tir::StmtSRef& block_sref,
                    const tir::TensorIntrin& intrin) {
    Optional<TensorizeInfo> opt_tensorize_info =
        GetTensorizeLoopMapping(sch->state(), block_sref, intrin->description);
    if (!opt_tensorize_info.defined()) {
      return false;
    }
    const auto* info = opt_tensorize_info.value().get();
    arith::Analyzer analyzer;
    for (const auto& kv : info->loop_map) {
      const tir::StmtSRef& block_loop_sref = kv.first;
      const auto* block_loop = block_loop_sref->StmtAs<tir::ForNode>();
      const tir::For& desc_loop = kv.second;
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
      DelAnn(sch->state(), block_sref, tir::attr::auto_tensorize);
      // Get the surrounding loops
      Array<tir::StmtSRef> loop_srefs = tir::GetLoops(block_sref);
      // Decompose Reduction
      {
        // (TODO) bohan
      }
      // Tensorize
      for (const tir::TensorIntrin& intrin : tensor_intrins) {
        if (CanTensorize(sch, block_sref, intrin)) {
          tir::Tensorize(sch->state(), loop_srefs[0], intrin);
          return true;
        }
      }
    }
    return false;
  }
};

Postproc RewriteTensorize(Array<tir::TensorIntrin> tensor_intrins) {
  auto f_proc = [tensor_intrins{std::move(tensor_intrins)}](SearchTask task, Schedule self,
                                                            void* _rand_state) -> bool {
    return PostprocRewriteTensorize(tensor_intrins).Proc(self);
  };
  return Postproc("rewrite_tensorize", f_proc);
}

/********** RewriteCooperativeFetch **********/

class PostprocRewriteCooperativeFetch {
 public:
  static std::function<bool(const tir::BlockNode* block)> MakeBlockFinder(const tir::Schedule& sch,
                                                                          int* idx) {
    return [sch, idx](const tir::BlockNode* block) -> bool {
      const tir::StmtSRefNode* sref = sch->GetSRef(block)->parent;
      for (int& i = *idx = 0; sref != nullptr; sref = sref->parent, ++i) {
        const tir::ForNode* loop = sref->StmtAs<tir::ForNode>();
        if (!loop) {
          break;
        }
        if (HasAnn(sch->GetSRef(loop), tir::attr::loop_type, "lazy_cooperative_fetch")) {
          return true;
        }
      }
      return false;
    };
  }

  bool Proc(const Schedule& sch) const {
    int idx = 0;
    while (Optional<tir::StmtSRef> opt_block_sref =
               FindBlockSRef(sch->state(), MakeBlockFinder(sch, &idx))) {
      // Extract block info
      tir::StmtSRef block_sref = opt_block_sref.value();
      const auto* block = block_sref->StmtAs<tir::BlockNode>();
      BlockRV block_rv = sch->GetBlock(block->name_hint);
      // Extract loop info
      Array<LoopRV> loop_rvs = sch->GetLoops(block_rv);
      int n_loops = loop_rvs.size();
      CHECK_LT(idx, n_loops);
      LoopRV loop_rv = loop_rvs[n_loops - 1 - idx];
      tir::StmtSRef loop_sref = sch->GetSRef(loop_rv);
      // Remove the annotation
      DelAnn(sch->state(), loop_sref, tir::attr::loop_type);
      // Find the threadIdx.x binding
      PrimExpr thread_idx_extent{nullptr};
      for (const tir::StmtSRefNode* sref = loop_sref->parent;; sref = sref->parent) {
        ICHECK(sref) << "ValueError: Cannot find loops above with threadIdx.x";
        if (const tir::ForNode* loop = sref->StmtAs<tir::ForNode>()) {
          if (HasBinding(GetRef<tir::StmtSRef>(sref), "threadIdx.x")) {
            ICHECK(tir::is_zero(loop->min))
                << "ValueError: Expect loops to start from 0, but gets: " << GetRef<tir::For>(loop);
            thread_idx_extent = loop->extent;
            break;
          }
        }
      }
      // Split the loop
      Array<LoopRV> split = sch->Split(loop_rv, {NullOpt, thread_idx_extent});
      ICHECK_EQ(split.size(), 2);
      sch->Bind(split[1], "threadIdx.x");
    }
    return true;
  }
};

Postproc RewriteCooperativeFetch() {
  auto f_proc = [](SearchTask task, Schedule sch, void* _rand_state) -> bool {
    return PostprocRewriteCooperativeFetch().Proc(sch);
  };
  return Postproc("rewrite_cooperative_fetch", f_proc);
}

/********** RewriteParallelizeVectorizeUnroll **********/

class StrideExtractor : public tir::StmtExprVisitor {
 public:
  static int64_t Extract(const PrimExpr& expr, const tir::Var& var) {
    StrideExtractor extractor(var);
    extractor.VisitExpr(expr);
    return extractor.strides_[expr.get()];
  }

 private:
  explicit StrideExtractor(const tir::Var& var) : var_(var) {}

  void VisitExpr_(const tir::MulNode* node) final {
    StmtExprVisitor::VisitExpr_(node);

    if (const auto* a = node->a.as<IntImmNode>()) {
      if (strides_.count(node->b.get())) {
        strides_[node] = strides_[node->b.get()] * a->value;
      }
    } else if (const auto* b = node->b.as<IntImmNode>()) {
      if (strides_.count(node->a.get())) {
        strides_[node] = strides_[node->a.get()] * b->value;
      }
    }
  }

  void VisitExpr_(const tir::AddNode* node) final {
    StmtExprVisitor::VisitExpr_(node);
    int64_t stride_a, stride_b;
    if (strides_.count(node->a.get())) {
      stride_a = strides_[node->a.get()];
    } else {
      stride_a = INT64_MAX;
    }
    if (strides_.count(node->b.get())) {
      stride_b = strides_[node->b.get()];
    } else {
      stride_b = INT64_MAX;
    }
    if (stride_a != INT64_MAX || stride_b != INT64_MAX) {
      strides_[node] = std::min(stride_a, stride_b);
    }
  }

  void VisitExpr_(const tir::VarNode* node) final {
    if (node == var_.get()) {
      strides_[node] = 1;
    }
  }

  const tir::Var& var_;
  std::unordered_map<const PrimExprNode*, int64_t> strides_;
};

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

  static std::function<bool(const tir::BlockNode*)> MakeAnnParser(Parsed* parsed) {
    return [parsed](const tir::BlockNode* block) -> bool {
      bool found = false;
      *parsed = Parsed{-1, -1, -1, -1, -1, -1};
      for (const auto& ann : block->annotations) {
        if (ann.first == tir::attr::auto_parallel_extent) {
          found = true;
          if (const auto* str_imm = ann.second.as<tir::StringImmNode>()) {
            parsed->max_parallel_extent = std::atoi(str_imm->value.c_str());
          }
        } else if (ann.first == tir::attr::auto_vectorize_extent) {
          found = true;
          if (const auto* str_imm = ann.second.as<tir::StringImmNode>()) {
            parsed->max_vectorize_extent = std::atoi(str_imm->value.c_str());
          }
        } else if (ann.first == tir::attr::auto_unroll_explicit) {
          found = true;
          if (const auto* str_imm = ann.second.as<tir::StringImmNode>()) {
            parsed->unroll_explicit = std::atoi(str_imm->value.c_str());
          }
        } else if (ann.first == tir::attr::auto_unroll_implicit) {
          found = true;
          if (const auto* str_imm = ann.second.as<tir::StringImmNode>()) {
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
      DelAnn(sch->state(), block_sref, tir::attr::auto_parallel_extent);
    }
    if (parsed.max_vectorize_extent != -1) {
      DelAnn(sch->state(), block_sref, tir::attr::auto_vectorize_extent);
    }
    if (parsed.unroll_explicit != -1) {
      DelAnn(sch->state(), block_sref, tir::attr::auto_unroll_explicit);
    }
    if (parsed.unroll_implicit != -1) {
      DelAnn(sch->state(), block_sref, tir::attr::auto_unroll_implicit);
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
        loop_srefs.push_back(sch->GetSRef(loop_rv));
        loop_types.push_back(GetLoopIterType(sch->state(), loop_srefs.back()));
      }
    }
    // check the maximal number of axes that are vectorizable (contiguous memory access)
    tir::BlockRealize realize = tir::GetBlockRealize(block_sref);
    Array<tir::BufferRegion> buffer_access(realize->block->reads);
    buffer_access.insert(buffer_access.end(), realize->block->writes.begin(),
                         realize->block->writes.end());
    std::unordered_map<const tir::VarNode*, PrimExpr> binding_map;
    for (size_t i = 0; i < realize->iter_values.size(); i++) {
      binding_map[realize->block->iter_vars[i]->var.get()] = realize->iter_values[i];
    }
    int max_fusible = INT32_MAX;
    // for each block read/write, get the strides of the loop vars and find the fusible
    // (vectorizable) axes
    for (const tir::BufferRegion& access : buffer_access) {
      int fusible = 0;
      std::vector<int64_t> strides;
      // get strides for each loop var
      for (const tir::StmtSRef& loop_sref : loop_srefs) {
        int64_t stride = 0, buffer_stride = 1;
        const auto* var = loop_sref->StmtAs<tir::ForNode>();
        arith::Analyzer analyzer;
        for (int i = access->region.size() - 1; i >= 0; i--) {
          PrimExpr idx = analyzer.Simplify(tir::Substitute(access->region[i]->min, binding_map));
          int64_t coef = StrideExtractor::Extract(idx, var->loop_var);
          if (coef != 0) {
            stride = coef * buffer_stride;
            break;
          }
          buffer_stride *= access->buffer->shape[i].as<IntImmNode>()->value;
        }
        strides.push_back(stride);
      }
      int prev_used_iter = -1;
      // check the number of fusible loops
      for (int i = strides.size() - 1; i >= 0; i--) {
        if (strides[i] == 0) {
          // not used in the buffer access, safe to fuse
          fusible++;
          continue;
        } else if (prev_used_iter == -1) {
          // the stride of last axis is not 1 means the memory access is not contiguous
          if (strides[i] != 1) {
            break;
          }
          fusible++;
          prev_used_iter = i;
        } else {
          // contiguous memory access
          const auto* prev_loop = loop_srefs[prev_used_iter]->StmtAs<tir::ForNode>();
          int64_t prev_used_iter_extent = prev_loop->extent.as<IntImmNode>()->value;
          if (strides[i] == strides[prev_used_iter] * prev_used_iter_extent) {
            fusible++;
            prev_used_iter = i;
          } else {
            break;
          }
        }
      }
      max_fusible = std::min(max_fusible, fusible);
    }
    // Calculate the parallelize extent
    if (parsed->max_parallel_extent != -1) {
      int max_extent = parsed->max_parallel_extent;
      int& num_fusible = parsed->num_parallel_loops = 0;
      int64_t prod_extent = 1;
      for (int i = 0; i < n_loops && loop_types[i] == tir::IterVarType::kDataPar; ++i) {
        const tir::StmtSRef& loop_sref = loop_srefs[i];
        if (HasAnyAnn(loop_sref)) {
          break;
        }
        // Check if the loop extent is valid
        int64_t extent = GetLoopIntExtent(loop_sref);
        if (extent == -1) {
          break;
        }
        // Then we can fuse it in
        ++num_fusible;
        // Check if we need to break
        prod_extent *= extent;
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
      for (int i = n_loops - 1;
           i >= 0 && loop_types[i] == tir::IterVarType::kDataPar && num_fusible < max_fusible;
           --i) {
        const tir::StmtSRef& loop_sref = loop_srefs[i];
        if (HasAnyAnn(loop_sref)) {
          break;
        }
        // Cannot vectorize reduce axis
        if (GetLoopIterType(sch->state(), loop_sref) != tir::IterVarType::kDataPar) {
          break;
        }
        // Cannot fuse with a loop with multiple children
        if (!HasSingleChild(loop_sref)) {
          break;
        }
        // Check if the loop extent is valid
        int64_t extent = GetLoopIntExtent(loop_sref);
        if (extent == -1) {
          break;
        }
        // Check if the extent is still in a good range
        prod_extent *= extent;
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
                                            n_loops - parsed->num_vectorize_loops);
    }
  }

  bool Proc(const Schedule& sch) const {
    Parsed parsed;
    tir::BlockRV root_rv = sch->GetBlock("root");
    tir::StmtSRef root = sch->GetSRef(root_rv);
    bool find_ann = MakeAnnParser(&parsed)(sch->Get(root_rv).get());
    if (!find_ann) {
      return true;
    }
    RemoveParsedAnn(sch, root, parsed);
    for (BlockRV block_rv : sch->GetChildBlocks(root_rv)) {
      block_rv = sch->GetBlock(sch->Get(block_rv)->name_hint);
      tir::StmtSRef block_sref = sch->GetSRef(block_rv);
      Array<LoopRV> loop_rvs = sch->GetLoops(block_rv);
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
        ICHECK(!(parsed.unroll_explicit != -1 && parsed.unroll_implicit != -1))
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
  auto f_proc = [](SearchTask task, Schedule sch, void* _rand_state) -> bool {
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
      finder(GetOnlyFunc(sch->mod())->body);
      return finder.block_ == nullptr ? Optional<tir::StmtSRef>(NullOpt)
                                      : sch->GetSRef(finder.block_);
    }

   private:
    explicit UnboundBlockFinder(const tir::Schedule& sch)
        : sch_(sch), block_(nullptr), n_thread_binding_(0), n_block_binding_(0) {
      const auto* realize = GetOnlyFunc(sch->mod())->body.as<tir::BlockRealizeNode>();
      root_block_ = realize->block.get();
    }

    void VisitStmt_(const tir::ForNode* loop) override {
      if (block_) {
        return;
      }
      String ann;
      if (Optional<String> opt_ann = GetBinding(sch_->GetSRef(loop))) {
        ann = opt_ann.value();
        if (ann == "threadIdx.x" || ann == "threadIdx.y" || ann == "threadIdx.z") {
          ++n_thread_binding_;
        } else if (ann == "blockIdx.x" || ann == "blockIdx.y" || ann == "blockIdx.z") {
          ++n_block_binding_;
        }
      }
      if (!(n_thread_binding_ > 0 && n_block_binding_ > 0) && ann != "vthread") {
        tir::StmtExprVisitor::VisitStmt_(loop);
      }
      if (ann.defined()) {
        if (ann == "threadIdx.x" || ann == "threadIdx.y" || ann == "threadIdx.z") {
          --n_thread_binding_;
        } else if (ann == "blockIdx.x" || ann == "blockIdx.y" || ann == "blockIdx.z") {
          --n_block_binding_;
        }
      }
    }

    void VisitStmt_(const tir::BlockNode* block) override {
      if (block == root_block_) {
        tir::StmtExprVisitor::VisitStmt_(block);
      } else if (!block_) {
        block_ = block;
      }
    }

    const tir::Schedule& sch_;
    const tir::BlockNode* block_;
    const tir::BlockNode* root_block_;
    int n_thread_binding_;
    int n_block_binding_;
  };

  void BindThreadAxes(const Schedule& sch, const tir::StmtSRef& block_sref) const {
    // Extract the block
    const auto* block = block_sref->StmtAs<tir::BlockNode>();
    ICHECK(block) << "TypeError: Expects Block, but gets: " << block_sref->stmt->GetTypeKey();
    BlockRV block_rv = sch->GetBlock(block->name_hint);
    // Extract loops
    Array<LoopRV> loop_rvs = sch->GetLoops(block_rv);
    // Check whether the outer loops were bound to blockIdx or threadIdx
    Array<tir::StmtSRef> loop_srefs;
    bool has_block_binding = false;
    bool has_thread_binding = false;
    for (const LoopRV& loop_rv : loop_rvs) {
      tir::StmtSRef loop_sref = sch->GetSRef(loop_rv);
      loop_srefs.push_back(loop_sref);
      if (Optional<String> opt_ann = GetBinding(loop_sref)) {
        String ann = opt_ann.value();
        if (ann == "threadIdx.x" || ann == "threadIdx.y" || ann == "threadIdx.z") {
          has_thread_binding = true;
        } else if (ann == "blockIdx.x" || ann == "blockIdx.y" || ann == "blockIdx.z") {
          has_block_binding = true;
        }
      }
    }
    ICHECK(!(has_block_binding && has_thread_binding));
    CHECK(!(has_block_binding && !has_thread_binding))
        << "ValueError: Currently it is not allowed that a block has blockIdx outer loop but "
           "doesn't have threadIdx outer loop";
    // Find the outer spatial loops
    // TODO(@junrushao1994): check if each loop has only one children, otherwise we cannot fuse
    int n_spatial_loops = 0;
    for (const tir::StmtSRef& loop_sref : loop_srefs) {
      tir::IterVarType iter_type = GetLoopIterType(sch->state(), loop_sref);
      if (iter_type != tir::kDataPar || GetBinding(loop_sref).defined() ||
          (n_spatial_loops > 0 && loop_sref->seq_index != -1)) {
        break;
      }
      if (!HasSingleChild(loop_sref)) {
        n_spatial_loops++;
        break;
      }
      ++n_spatial_loops;
    }
    CHECK_GT(n_spatial_loops, 0) << "ValueError: not supported when spatial loop doesn't exist";
    // Fuse the spatial loops
    LoopRV fused = sch->Fuse({loop_rvs.begin(), loop_rvs.begin() + n_spatial_loops});
    if (!has_thread_binding) {
      Array<LoopRV> splits = sch->Split(fused, {NullOpt, Integer(32)});
      ICHECK_EQ(splits.size(), 2);
      sch->Bind(splits[0], "blockIdx.x");
      sch->Bind(splits[1], "threadIdx.x");
    } else {
      sch->Bind(fused, "blockIdx.x");
    }
  }

  bool Proc(const SearchTask& task, const Schedule& sch) const {
    int warp_size = task->target->GetAttr<Integer>("thread_warp_size").value_or(Integer(-1));
    ICHECK(warp_size != -1) << "ValueError: Target does not have attribute \"thread_warp_size\"";
    while (Optional<tir::StmtSRef> opt_block_sref = UnboundBlockFinder::Find(sch)) {
      BindThreadAxes(sch, opt_block_sref.value());
    }
    return true;
  }
};

Postproc RewriteUnboundBlocks() {
  auto f_proc = [](SearchTask task, Schedule sch, void* _rand_state) -> bool {
    return PostprocRewriteUnboundBlocks().Proc(task, sch);
  };
  return Postproc("rewrite_unbound_blocks", f_proc);
}

/********** RewriteReductionBlock **********/

class PostprocRewriteReductionBlock {
 public:
  /*!
   * \brief An auxiliary class to help find reduction blocks whose init block is going to be
   *        decomposed.
   */
  class Finder : public tir::StmtVisitor {
   public:
    static const tir::BlockNode* Find(const tir::Stmt& stmt) {
      Finder finder;
      finder.CollectBoundLoops(stmt);
      finder.VisitStmt(stmt);
      ICHECK(finder.result_ == nullptr || (finder.result_->init.defined() &&
                                           finder.AllReductionIterVarAreUnbound(finder.result_)));
      return finder.result_;
    }

   private:
    Finder() : bound_loop_vars_(), stack_(), result_(nullptr) {}

    /*!
     * \brief Collect all the loops inside `stmt` that are bound to threadIdx or blockIdx.
     * \param stmt The stmt to be inspected.
     */
    void CollectBoundLoops(const tir::Stmt& stmt) {
      tir::PreOrderVisit(stmt, [this](const ObjectRef& node) {
        if (const auto* loop = node.as<tir::ForNode>()) {
          std::string thread_tag =
              loop->thread_binding.defined() ? loop->thread_binding.value()->thread_tag : "";
          if (thread_tag.substr(0, 9) == "threadIdx" || thread_tag.substr(0, 8) == "blockIdx") {
            ICHECK(thread_tag == "threadIdx.x" || thread_tag == "threadIdx.y" ||
                   thread_tag == "threadIdx.z" || thread_tag == "blockIdx.x" ||
                   thread_tag == "blockIdx.y" || thread_tag == "blockIdx.z");
            bound_loop_vars_.insert(loop->loop_var.get());
          }
        }
        return false;
      });
    }

    /*!
     * \brief Check whether the two following conditions are both satisfied:
     *   1. the block has at least one reduction block var, and
     *   2. none of its reduction block var bindings is bound to threadIdx.
     * \param block_sref The block to be checked
     * \return A boolean indicating if it has at least one reduction block var
     */
    bool AllReductionIterVarAreUnbound(const tir::BlockNode* block) {
      bool has_reduction_var = false;
      CHECK(!stack_.empty() && stack_.back()->block.get() == block)
          << "ValueError: the block has outer BlockRealize or the outer BlockRealize doesn't match "
             "the block.";
      const tir::BlockRealize& block_realize = GetRef<tir::BlockRealize>(stack_.back());
      ICHECK_EQ(block_realize->iter_values.size(), block->iter_vars.size());
      for (int i = 0; i < static_cast<int>(block->iter_vars.size()); ++i) {
        const tir::IterVar& var = block->iter_vars[i];
        const PrimExpr& binding = block_realize->iter_values[i];
        if (var->iter_type == tir::kCommReduce &&
            tir::ExprUseVar(
                binding, [&](const tir::VarNode* node) { return bound_loop_vars_.count(node); })) {
          return false;
        }
      }
      return has_reduction_var;
    }

    void VisitStmt_(const tir::BlockNode* block) override {
      if (result_ != nullptr) {
        return;
      }
      /* 1. If a block doesn't have any bound reduction block var, there is no need to
       *    decompose reduction.
       * 2. If some of its reduction block var bindings are bound to threadIdx, this indicates
       *    that cross-thread-reduction is needed, and hence we should not decompose the init block.
       */
      if (block->init.defined() && AllReductionIterVarAreUnbound(block)) {
        result_ = block;
      } else {
        tir::StmtVisitor::VisitStmt_(block);
      }
    }

    void VisitStmt_(const tir::BlockRealizeNode* block_realize) override {
      if (result_ != nullptr) {
        return;
      }
      stack_.push_back(block_realize);
      tir::StmtVisitor::VisitStmt_(block_realize);
      ICHECK(!stack_.empty());
      stack_.pop_back();
    }

    /*! \brief A set recording all the bound loop vars. */
    std::unordered_set<const tir::VarNode*> bound_loop_vars_;
    /*! \brief A stack recording all the BlockRealizes along the visiting path. */
    std::vector<const tir::BlockRealizeNode*> stack_;
    /*!
     * \brief The result block which has at least one reduction block var and none of the block var
     *        bindings is bound to threadIdx (i.e., cross-thread-reduction is not needed).
     */
    const tir::BlockNode* result_;
  };

  bool Proc(const Schedule& sch) const {
    while (const tir::BlockNode* block = Finder::Find(GetOnlyFunc(sch->mod())->body)) {
      BlockRV block_rv = sch->GetBlock(block->name_hint);
      Array<LoopRV> loop_rvs = sch->GetLoops(block_rv);
      int n_loops = loop_rvs.size();
      for (int i = 0; i < n_loops; ++i) {
        const LoopRV& loop_rv = loop_rvs[i];
        tir::StmtSRef loop_sref = sch->GetSRef(loop_rv);
        tir::IterVarType type = GetLoopIterType(sch->state(), loop_sref);
        if (type == tir::kCommReduce || type == tir::kOpaque) {
          // Insert the initializing block above the first loop which is not data parallel.
          sch->DecomposeReduction(block_rv, loop_rvs[i]);
          break;
        }
      }
    }
    return true;
  }
};

Postproc RewriteReductionBlock() {
  auto f_proc = [](SearchTask task, Schedule sch, void* _rand_state) -> bool {
    return PostprocRewriteReductionBlock().Proc(sch);
  };
  return Postproc("rewrite_reduction_block", f_proc);
}

/********** VerifyGPUCode **********/

class PostprocDisallowDynamicLoops {
 public:
  bool Proc(const Schedule& sch) const {
    bool has_dyn_ext = false;
    auto f_visit = [&has_dyn_ext](const ObjectRef& obj) -> bool {
      if (has_dyn_ext) {
        return false;
      }
      if (const auto* loop = obj.as<tir::ForNode>()) {
        if (!loop->extent->IsInstance<IntImmNode>()) {
          has_dyn_ext = true;
          return false;
        }
      }
      return true;
    };
    tir::PreOrderVisit(GetOnlyFunc(sch->mod())->body, f_visit);
    return !has_dyn_ext;
  }
};

Postproc DisallowDynamicLoops() {
  auto f_proc = [](SearchTask task, Schedule sch, void* _rand_state) -> bool {
    return PostprocDisallowDynamicLoops().Proc(sch);
  };
  return Postproc("disallow_dynamic_loops", f_proc);
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

  static tir::transform::Sequential MakePasses() {
    return tir::transform::Sequential({
        tir::transform::InjectPrefetch(),       //
        tir::transform::AllreduceTransform(),   //
        tir::transform::BufferFlatten(),        //
        tir::transform::NarrowDataType(32),     //
        tir::transform::Simplify(),             //
        tir::transform::VectorizeLoop(true),    //
        tir::transform::InjectVirtualThread(),  //
        tir::transform::StorageRewrite(),       //
        tir::transform::Simplify()              //
    });
  }

  static bool VerifyGPU(const tir::PrimFunc& func, const Target& target) {
    Map<String, PrimExpr> constraints{
        {"max_shared_memory_per_block", Extract(target, "shared_memory_per_block")},
        {"max_local_memory_per_block", Extract(target, "registers_per_block")},
        {"max_threads_per_block", Extract(target, "max_threads_per_block")},
        {"max_vthread", Integer(8)},
        {"max_vector_bytes", Integer(16)}};
    return tir::VerifyGPUCode(func, constraints);
  }

  bool Proc(const SearchTask& task, const Schedule& sch) const {
    static tir::transform::Sequential passes = MakePasses();
    IRModule mod = sch->mod();
    try {
      mod = passes(std::move(mod));
    } catch (const dmlc::Error& e) {
      return false;
    }
    return VerifyGPU(GetOnlyFunc(mod), task->target);
  }
};

Postproc VerifyGPUCode() {
  auto f_proc = [](SearchTask task, Schedule sch, void* _rand_state) -> bool {
    return PostprocVerifyGPUCode().Proc(task, sch);
  };
  return Postproc("verify_gpu_code", f_proc);
}

/********** RewriteLayout **********/
class PostProcRewriteLayout {
 private:
  class IterVarMapCollector {
   public:
    explicit IterVarMapCollector(Array<arith::IterSplitExpr>& splitexprs)
        : splitexprs_(splitexprs) {}

    void CollectIterSplitExpr(const arith::IterSplitExpr& expr) {
      if (expr->source->source.as<tir::VarNode>()) {
        splitexprs_.push_back(
            arith::IterSplitExpr(expr->source, expr->lower_factor, expr->extent, expr->scale));
      } else if (const auto& op = expr->source->source.as<arith::IterSumExprNode>()) {
        CollectIterSumExpr(GetRef<arith::IterSumExpr>(op));
      }
    }

    void CollectIterSumExpr(const arith::IterSumExpr& expr) {
      for (const auto& arg : expr->args) {
        CollectIterSplitExpr(arg);
      }
    }

    // keep only the itersplitexpr and append them to the result array
    void Collect(const arith::IterMapExpr& expr) {
      if (const auto* op = expr.as<arith::IterSplitExprNode>()) {
        CollectIterSplitExpr(GetRef<arith::IterSplitExpr>(op));
      } else if (const auto* op = expr.as<arith::IterSumExprNode>()) {
        CollectIterSumExpr(GetRef<arith::IterSumExpr>(op));
      } else {
        LOG(FATAL);
      }
    }

   private:
    Array<arith::IterSplitExpr>& splitexprs_;
  };

  class IterVarResolver : public tir::StmtExprVisitor {
    void VisitStmt_(const tir::BlockRealizeNode* op) {
      realize_ = GetRef<tir::BlockRealize>(op);
      for (size_t i = 0; i < realize_->iter_values.size(); i++) {
        binding_map_.Set(realize_->block->iter_vars[i]->var, realize_->iter_values[i]);
      }
      VisitStmt(op->block->body);
    }

    // true if expr1's loop is above expr2's loop
    static bool compare(
        const arith::IterSplitExpr& expr1, const arith::IterSplitExpr& expr2,
        const std::unordered_map<tir::Var, int, ObjectPtrHash, ObjectPtrEqual>& loop_order) {
      tir::Var var1 = Downcast<tir::Var>(expr1->source->source);
      tir::Var var2 = Downcast<tir::Var>(expr2->source->source);
      if (loop_order.at(var1) < loop_order.at(var2)) {
        return true;
      } else if (loop_order.at(var1) == loop_order.at(var2)) {
        int factor1 = Downcast<IntImm>(expr1->lower_factor)->value;
        int factor2 = Downcast<IntImm>(expr2->lower_factor)->value;
        if (factor1 > factor2) {
          return true;
        }
      }
      return false;
    }

    void VisitExpr_(const tir::BufferLoadNode* op) {
      if (buffer_.same_as(op->buffer)) {
        ICHECK(!visited_) << "cannot rewrite a buffer with 2 or more accesses in the function";
        block_to_rewrite_ = realize_->block;
        visited_ = true;
        PrimExpr sum = 0;
        PrimExpr flatten_shape = 1;
        // get the flattened index
        for (size_t i = 0; i < op->indices.size(); i++) {
          sum *= op->buffer->shape[i];
          flatten_shape *= op->buffer->shape[i];
          tir::Var var = Downcast<tir::Var>(op->indices[i]);
          ICHECK(var.defined()) << "cannot handle irregular access pattern";
          auto bind = binding_map_.Get(var);
          ICHECK(bind.defined()) << "index " << i << " is not a block var";
          sum += bind.value();
        }
        arith::Analyzer analyzer;
        auto block_sref = sch_->GetSRef(realize_->block);
        auto loops = tir::GetLoops(block_sref);
        std::unordered_map<tir::Var, Range, ObjectPtrHash, ObjectPtrEqual> loop_vars;
        std::unordered_map<tir::Var, int, ObjectPtrHash, ObjectPtrEqual> loop_order;
        for (size_t i = 0; i < loops.size(); i++) {
          const auto* loop = loops[i]->StmtAs<tir::ForNode>();
          loop_vars.emplace(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
          loop_order.emplace(loop->loop_var, i);
        }
        // analyze the access pattern of the flattened index
        auto results = arith::DetectIterMap(Array<PrimExpr>{analyzer.Simplify(sum)}, loop_vars,
                                            realize_->predicate, true, &analyzer);
        Array<arith::IterSplitExpr> splitexprs;
        IterVarMapCollector collector(splitexprs);
        for (size_t i = 0; i < results.size(); i++) {
          collector.Collect(results[i]);
        }
        // get the extents for rewrite
        for (const auto& splitexpr : splitexprs) {
          IntImm extent = Downcast<IntImm>(splitexpr->extent);
          if (extent.defined()) {
            hint_.extents.push_back(extent->value);
          } else {
            LOG(FATAL) << "dynamic layout";
          }
        }
        hint_.reorder.resize(hint_.extents.size());
        // get the order for rewrite, which is consistent with the order of related loops
        for (size_t i = 0; i < hint_.extents.size(); i++) {
          int idx = 0;
          for (const auto& other : splitexprs) {
            if (compare(other, splitexprs[i], loop_order)) {
              idx++;
            }
          }
          hint_.reorder[idx] = Integer(i);
        }
      }
    }

    bool visited_ = false;
    tir::BlockRealize realize_;
    Map<tir::Var, PrimExpr> binding_map_{};
    const Schedule& sch_;
    const tir::Buffer& buffer_;
    LayoutRewriteHint& hint_;
    tir::Block block_to_rewrite_;

   public:
    explicit IterVarResolver(const Schedule& sch, const tir::Buffer& buffer,
                             LayoutRewriteHint& hint)
        : sch_(sch), buffer_(buffer), hint_(hint) {}

    tir::Block getBufferIterInfo(const tir::Stmt& body) {
      visited_ = false;
      VisitStmt(body);
      return block_to_rewrite_;
    }
  };

 public:
  class IndexRewriter : public tir::StmtExprMutator {
   public:
    IndexRewriter(const LayoutRewriteHint& hint, const tir::Buffer& buffer_to_rewrite,
                  const tir::Block& block, const tir::Buffer& new_buffer)
        : hint_(hint),
          buffer_to_rewrite_(buffer_to_rewrite),
          block_(block),
          new_buffer_(new_buffer) {}

    tir::Stmt Rewrite(const tir::Stmt& stmt) { return this->VisitStmt(stmt); }

    tir::Stmt VisitStmt_(const tir::BlockNode* op) final {
      tir::Block mutated_block = Downcast<tir::Block>(StmtMutator::VisitStmt_(op));
      if (GetRef<tir::Block>(op).same_as(block_)) {
        tir::BlockReadWriteCollector block_read_write_collector(mutated_block->alloc_buffers);
        block_read_write_collector(mutated_block->body);
        auto n = CopyOnWrite(mutated_block.operator->());
        n->reads = block_read_write_collector.reads();
        return tir::Block(n);
      }
      return std::move(mutated_block);
    }

    PrimExpr VisitExpr_(const tir::BufferLoadNode* op) final {
      if (op->buffer.same_as(buffer_to_rewrite_)) {
        PrimExpr sum = 0;
        for (size_t i = 0; i < op->indices.size(); i++) {
          sum *= op->buffer->shape[i];
          sum += op->indices[i];
        }
        Array<PrimExpr> r_new_indices;
        arith::Analyzer analyzer;
        for (int i = hint_.extents.size() - 1; i >= 0; i--) {
          r_new_indices.push_back(analyzer.Simplify(floormod(sum, hint_.extents[i])));
          sum = floordiv(sum, hint_.extents[i]);
        }
        Array<PrimExpr> new_indices(r_new_indices.rbegin(), r_new_indices.rend());
        Array<PrimExpr> reordered_indices;
        for (size_t i = 0; i < hint_.extents.size(); i++) {
          reordered_indices.push_back(new_indices[hint_.reorder[i]]);
        }

        return tir::BufferLoad(new_buffer_, reordered_indices);
      }
      return GetRef<PrimExpr>(op);
    }

   private:
    const LayoutRewriteHint& hint_;
    const tir::Buffer& buffer_to_rewrite_;
    const tir::Block& block_;
    const tir::Buffer& new_buffer_;
  };

 public:
  bool Proc(const Schedule& sch, const SearchTask& task) const {
    tir::PrimFunc func = GetOnlyFunc(sch->mod());
    auto buffer_to_rewrite = func->GetAttr("layout_free_placeholders", Array<tir::Var>());
    if (!buffer_to_rewrite.defined()) {
      return true;
    }
    // for each buffer with the annotation" layout_free_placeholders", do layout rewrite
    for (const auto& input_var : buffer_to_rewrite.value()) {
      tir::Buffer buffer = func->buffer_map.Get(input_var).value();
      LayoutRewriteHint hint;
      // Step 0: find the loops used in the buffer access and
      //       get the auxillary (`extent` and `reorder`) to do rewrite
      IterVarResolver resolver(sch, buffer, hint);
      const tir::Stmt& body = func->body;
      tir::Block block = resolver.getBufferIterInfo(body);
      Array<PrimExpr> new_shape;
      for (size_t i = 0; i < hint.extents.size(); i++) {
        new_shape.push_back(hint.extents[hint.reorder[i]]);
      }
      // Step 1: create a new buffer
      tir::Buffer new_buffer(buffer->data, buffer->dtype, new_shape, Array<PrimExpr>(),
                             buffer->elem_offset, buffer->name, buffer->data_alignment,
                             buffer->offset_factor, buffer->buffer_type);
      // Step 2: do the rewrite to the buffer access
      // the rule is as below:
      //      for example,
      //      let extents = [2, 3, 4], reorder = [0, 2, 1], and the shape of buffer A is (4, 6)
      //      then A[i, j] will be first rewritten to
      //      A'[(6 * i + j) / 12, (6 * i + j) / 4 % 3 , (6 * i + j) % 4] according to the
      //      `extents`, and then reordered to A'[(6 * i + j) / 12, (6 * i + j) % 4 , (6 * i + j) /
      //      4 % 3] according to `reorder`
      IndexRewriter rewriter(hint, buffer, block, new_buffer);
      sch->state()->Replace(sch->GetSRef(block), rewriter.Rewrite(block), {});
      // Step 3: update the buffer map
      Map<tir::Var, tir::Buffer> new_buffer_map(func->buffer_map);
      new_buffer_map.Set(input_var, new_buffer);
      tir::PrimFunc orig_func = GetOnlyFunc(sch->mod());
      auto new_func = orig_func.CopyOnWrite();
      new_func->buffer_map = std::move(new_buffer_map);
      sch->state()->mod = IRModule({{GlobalVar("main"), GetRef<tir::PrimFunc>(new_func)}});
      // Step 4: tell the relay builder how to rewrite the params
      std::lock_guard<std::mutex> lock(::tvm::relay::MetaScheduleLayoutRewriter::mutex);

      ::tvm::relay::MetaScheduleLayoutRewriter::global_layout_rewrite_queue.push_back(hint);
    }
    return true;
  }
};

Postproc RewriteLayout() {
  auto f_proc = [](SearchTask task, Schedule sch, void* _rand_state) -> bool {
    return PostProcRewriteLayout().Proc(sch, task);
  };
  return Postproc("rewrite_layout", f_proc);
}

/********** FFI **********/

struct Internal {
  /*!
   * \brief FFI function for PostProcNode::Apply
   * \sa PostProcNode::Apply
   */
  static bool Apply(Postproc self, SearchTask task, Schedule sch, Optional<Integer> seed) {
    tir::TRandState rand_state = std::random_device()();
    if (seed.defined()) {
      tir::RandEngine(&rand_state).Seed(seed.value());
    }
    return self->Apply(task, sch, &rand_state);
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
TVM_REGISTER_GLOBAL("meta_schedule.postproc.DisallowDynamicLoops")
    .set_body_typed(DisallowDynamicLoops);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.VerifyGPUCode").set_body_typed(VerifyGPUCode);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.RewriteReductionBlock")
    .set_body_typed(RewriteReductionBlock);
TVM_REGISTER_GLOBAL("meta_schedule.postproc.RewriteLayout").set_body_typed(RewriteLayout);

}  // namespace meta_schedule
}  // namespace tvm
