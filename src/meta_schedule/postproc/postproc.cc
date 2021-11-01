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
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

Postproc Postproc::PyPostproc(
    PyPostprocNode::FInitializeWithTuneContext f_initialize_with_tune_context,  //
    PyPostprocNode::FApply f_apply,                                             //
    PyPostprocNode::FAsString f_as_string) {
  ObjectPtr<PyPostprocNode> n = make_object<PyPostprocNode>();
  n->f_initialize_with_tune_context = std::move(f_initialize_with_tune_context);
  n->f_apply = std::move(f_apply);
  n->f_as_string = std::move(f_as_string);
  return Postproc(n);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PyPostprocNode>([](const ObjectRef& n, ReprPrinter* p) {
      const auto* self = n.as<PyPostprocNode>();
      ICHECK(self);
      PyPostprocNode::FAsString f_as_string = (*self).f_as_string;
      ICHECK(f_as_string != nullptr) << "PyPostproc's AsString method not implemented!";
      p->stream << f_as_string();
    });

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

class RewriteParallelizeVectorizeUnrollNode : public PostprocNode {
 public:
  void InitializeWithTuneContext(const TuneContext& context) final {}

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

  static void RemoveParsedAnn(const tir::Schedule& sch, const tir::BlockRV& block_rv,
                              const Parsed& parsed) {
    if (parsed.max_parallel_extent != -1) {
      sch->Unannotate(block_rv, tir::attr::auto_parallel_extent);
    }
    if (parsed.max_vectorize_extent != -1) {
      sch->Unannotate(block_rv, tir::attr::auto_vectorize_extent);
    }
    if (parsed.unroll_explicit != -1) {
      sch->Unannotate(block_rv, tir::attr::auto_unroll_explicit);
    }
    if (parsed.unroll_implicit != -1) {
      sch->Unannotate(block_rv, tir::attr::auto_unroll_implicit);
    }
  }

  static void AdjustParallelVectorize(const tir::Schedule& sch, const tir::StmtSRef& block_sref,
                                      const Array<tir::LoopRV>& loop_rvs, Parsed* parsed) {
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
      for (const tir::LoopRV& loop_rv : loop_rvs) {
        loop_srefs.push_back(sch->GetSRef(loop_rv));
        loop_types.push_back(GetLoopIterType(sch->state(), loop_srefs.back()));
      }
    }
    // check the maximal number of axes that are vectorizable (contiguous memory access)
    tir::BlockRealize realize = tir::GetBlockRealize(sch->state(), block_sref);
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

  bool Apply(const tir::Schedule& schedule) final {
    Parsed parsed;
    tir::BlockRV root_rv = schedule->GetBlock("root");
    bool find_ann = MakeAnnParser(&parsed)(schedule->Get(root_rv).get());
    if (!find_ann) {
      return true;
    }
    RemoveParsedAnn(schedule, root_rv, parsed);
    for (tir::BlockRV block_rv : schedule->GetChildBlocks(root_rv)) {
      block_rv = schedule->GetBlock(schedule->Get(block_rv)->name_hint);
      tir::StmtSRef block_sref = schedule->GetSRef(block_rv);
      Array<tir::LoopRV> loop_rvs = schedule->GetLoops(block_rv);
      int n_loops = loop_rvs.size();
      if (n_loops == 0) {
        continue;
      }
      AdjustParallelVectorize(schedule, block_sref, loop_rvs, &parsed);
      // Parallelize
      if (parsed.num_parallel_loops > 0) {
        tir::LoopRV fused =
            schedule->Fuse({loop_rvs.begin(), loop_rvs.begin() + parsed.num_parallel_loops});
        schedule->Parallel(fused);
        for (int i = 0; i < parsed.num_parallel_loops; ++i) {
          loop_rvs.Set(i, fused);
        }
      }
      // Vectorize
      if (parsed.num_vectorize_loops > 0) {
        tir::LoopRV fused =
            schedule->Fuse({loop_rvs.end() - parsed.num_vectorize_loops, loop_rvs.end()});
        schedule->Vectorize(fused);
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
        tir::LoopRV loop = loop_rvs[0];
        if (max_step > 0) {
          schedule->Annotate(loop, "pragma_auto_unroll_max_step",
                             IntImm(DataType::Int(32), max_step));
          schedule->Annotate(loop, "pragma_unroll_explicit",
                             IntImm(DataType::Int(32), unroll_explicit));
        }
      }
    }
    return true;
  }

  static constexpr const char* _type_key = "meta_schedule.RewriteParallelizeVectorizeUnroll";
  TVM_DECLARE_FINAL_OBJECT_INFO(RewriteParallelizeVectorizeUnrollNode, PostprocNode);
};

Postproc Postproc::RewriteParallelizeVectorizeUnroll() {
  ObjectPtr<RewriteParallelizeVectorizeUnrollNode> n =
      make_object<RewriteParallelizeVectorizeUnrollNode>();
  return Postproc(n);
}

/********** DisallowDynamicLoops **********/

class DisallowDynamicLoopsNode : public PostprocNode {
 public:
  void InitializeWithTuneContext(const TuneContext& context) final {}

  bool Apply(const tir::Schedule& schedule) final {
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
    tir::PreOrderVisit(FindEntryFunc(schedule->mod())->body, f_visit);
    return !has_dyn_ext;
  }

  static constexpr const char* _type_key = "meta_schedule.DisallowDynamicLoops";
  TVM_DECLARE_FINAL_OBJECT_INFO(DisallowDynamicLoopsNode, PostprocNode);
};

Postproc Postproc::DisallowDynamicLoops() {
  ObjectPtr<DisallowDynamicLoopsNode> n = make_object<DisallowDynamicLoopsNode>();
  return Postproc(n);
}

/********** VerifyGPUCode **********/

class VerifyGPUCodeNode : public PostprocNode {
 public:
  Target target_;

  static Integer Extract(const Target& target, const char* name) {
    ICHECK(target.defined());

    if (Optional<Integer> v = target->GetAttr<Integer>(name)) {
      return v.value();
    }
    LOG(FATAL) << "AttributedError: \"" << name << "\" is not defined in the target";
    throw;
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

  void InitializeWithTuneContext(const TuneContext& context) final {
    ICHECK(context->target != nullptr);
    this->target_ = context->target.value();
  }

  bool Apply(const tir::Schedule& sch) final {
    IRModule mod = sch->mod();
    try {
      mod = LowerModule(std::move(mod));
    } catch (const dmlc::Error& e) {
      return false;
    }
    for (const auto& kv : mod->functions) {
      if (const auto* func = kv.second.as<tir::PrimFuncNode>()) {
        if (!VerifyGPU(GetRef<tir::PrimFunc>(func), this->target_)) {
          return false;
        }
      }
    }
    return true;
  }

  static constexpr const char* _type_key = "meta_schedule.VerifyGPUCode";
  TVM_DECLARE_FINAL_OBJECT_INFO(VerifyGPUCodeNode, PostprocNode);
};

Postproc Postproc::VerifyGPUCode() {
  ObjectPtr<VerifyGPUCodeNode> n = make_object<VerifyGPUCodeNode>();
  return Postproc(n);
}

TVM_REGISTER_OBJECT_TYPE(PostprocNode);
TVM_REGISTER_NODE_TYPE(PyPostprocNode);
TVM_REGISTER_NODE_TYPE(VerifyGPUCodeNode);
TVM_REGISTER_NODE_TYPE(DisallowDynamicLoopsNode);
TVM_REGISTER_NODE_TYPE(RewriteParallelizeVectorizeUnrollNode);

TVM_REGISTER_GLOBAL("meta_schedule.PostprocInitializeWithTuneContext")
    .set_body_method<Postproc>(&PostprocNode::InitializeWithTuneContext);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocApply").set_body_method<Postproc>(&PostprocNode::Apply);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocPyPostproc").set_body_typed(Postproc::PyPostproc);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocVerifyGPUCode").set_body_typed(Postproc::VerifyGPUCode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocDisallowDynamicLoops")
    .set_body_typed(Postproc::DisallowDynamicLoops);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocRewriteParallelizeVectorizeUnroll")
    .set_body_typed(Postproc::RewriteParallelizeVectorizeUnroll);

}  // namespace meta_schedule
}  // namespace tvm
