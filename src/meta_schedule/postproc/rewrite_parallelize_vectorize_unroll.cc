
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
namespace tir {

/*!
 * \brief Check whether the block/loop has any annotation
 * \param sref The sref of block/loop
 * \return Whether the block/loop has any annotation
 */
inline bool HasAnnOrBinding(const ForNode* loop) {
  return loop->kind == ForKind::kThreadBinding || !loop->annotations.empty();
}

class StrideExtractor : public StmtExprVisitor {
 public:
  static int64_t Extract(const PrimExpr& expr, const Var& var) {
    StrideExtractor extractor(var);
    extractor.VisitExpr(expr);
    return extractor.strides_[expr.get()];
  }

 private:
  explicit StrideExtractor(const Var& var) : var_(var) {}

  void VisitExpr_(const MulNode* node) final {
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

  void VisitExpr_(const AddNode* node) final {
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

  void VisitExpr_(const VarNode* node) final {
    if (node == var_.get()) {
      strides_[node] = 1;
    }
  }

  const Var& var_;
  std::unordered_map<const PrimExprNode*, int64_t> strides_;
};

struct ParsedAnnotation {
  int max_parallel_extent;
  int max_vectorize_extent;
  int unroll_explicit;
  int unroll_implicit;
  int num_parallel_loops;
  int num_vectorize_loops;
};

bool ParseAnnotation(const Block& block, ParsedAnnotation* parsed) {
  bool found = false;
  *parsed = ParsedAnnotation{-1, -1, -1, -1, -1, -1};
  for (const auto& ann : block->annotations) {
    if (ann.first == tir::attr::auto_parallel_extent) {
      found = true;
      if (const auto* str_imm = ann.second.as<StringImmNode>()) {
        parsed->max_parallel_extent = std::atoi(str_imm->value.c_str());
      }
    } else if (ann.first == tir::attr::auto_vectorize_extent) {
      found = true;
      if (const auto* str_imm = ann.second.as<StringImmNode>()) {
        parsed->max_vectorize_extent = std::atoi(str_imm->value.c_str());
      }
    } else if (ann.first == tir::attr::auto_unroll_explicit) {
      found = true;
      if (const auto* str_imm = ann.second.as<StringImmNode>()) {
        parsed->unroll_explicit = std::atoi(str_imm->value.c_str());
      }
    } else if (ann.first == tir::attr::auto_unroll_implicit) {
      found = true;
      if (const auto* str_imm = ann.second.as<StringImmNode>()) {
        parsed->unroll_implicit = std::atoi(str_imm->value.c_str());
      }
    }
  }
  return found;
}

void RemoveParsedAnn(const Schedule& sch, const BlockRV& block_rv, const ParsedAnnotation& parsed) {
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

void AdjustParallelVectorize(const Schedule& sch, const BlockRV& block_rv,
                             const Array<LoopRV>& loop_rvs, ParsedAnnotation* parsed) {
  StmtSRef block_sref = sch->GetSRef(block_rv);
  if (parsed->max_parallel_extent == -1 && parsed->max_vectorize_extent == -1) {
    return;
  }
  int n_loops = loop_rvs.size();
  if (n_loops == 0) {
    parsed->max_parallel_extent = -1;
    parsed->max_vectorize_extent = -1;
    return;
  }
  // Extract loop_srefs, and calculate the iterator types
  Array<StmtSRef> loop_srefs;
  std::vector<int> loop_types;
  {
    loop_srefs.reserve(n_loops);
    loop_types.reserve(n_loops);
    for (const LoopRV& loop_rv : loop_rvs) {
      loop_srefs.push_back(sch->GetSRef(loop_rv));
      loop_types.push_back(GetLoopIterType(loop_srefs.back()));
    }
  }
  // check the maximal number of axes that are vectorizable (contiguous memory access)
  BlockRealize realize = GetBlockRealize(sch->state(), block_sref);
  Array<BufferRegion> buffer_access(realize->block->reads);
  buffer_access.insert(buffer_access.end(), realize->block->writes.begin(),
                       realize->block->writes.end());
  std::unordered_map<const VarNode*, PrimExpr> binding_map;
  for (size_t i = 0; i < realize->iter_values.size(); i++) {
    binding_map[realize->block->iter_vars[i]->var.get()] = realize->iter_values[i];
  }
  int max_fusible = INT32_MAX;
  // for each block read/write, get the strides of the loop vars and find the fusible
  // (vectorizable) axes
  for (const BufferRegion& access : buffer_access) {
    int fusible = 0;
    std::vector<int64_t> strides;
    // get strides for each loop var
    for (const StmtSRef& loop_sref : loop_srefs) {
      int64_t stride = 0, buffer_stride = 1;
      const auto* var = loop_sref->StmtAs<ForNode>();
      arith::Analyzer analyzer;
      for (int i = access->region.size() - 1; i >= 0; i--) {
        PrimExpr idx = analyzer.Simplify(Substitute(access->region[i]->min, binding_map));
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
        const auto* prev_loop = loop_srefs[prev_used_iter]->StmtAs<ForNode>();
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
    for (int i = 0; i < n_loops && loop_types[i] == IterVarType::kDataPar; ++i) {
      const StmtSRef& loop_sref = loop_srefs[i];
      const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
      if (HasAnnOrBinding(loop)) {
        break;
      }
      // Check if the loop extent is valid
      const int64_t* extent = GetLoopIntExtent(loop_sref);
      if (extent == nullptr) {
        break;
      }
      // Then we can fuse it in
      ++num_fusible;
      // Check if we need to break
      prod_extent *= *extent;
      if (prod_extent > max_extent || !IsSingleStmt(loop->body)) {
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
         i >= 0 && loop_types[i] == IterVarType::kDataPar && num_fusible < max_fusible; --i) {
      const StmtSRef& loop_sref = loop_srefs[i];
      const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
      if (HasAnnOrBinding(loop)) {
        break;
      }
      // Cannot vectorize reduce axis
      if (GetLoopIterType(loop_sref) != IterVarType::kDataPar) {
        break;
      }
      // Cannot fuse with a loop with multiple children
      if (!IsSingleStmt(loop->body)) {
        break;
      }
      // Check if the loop extent is valid
      const int64_t* extent = GetLoopIntExtent(loop_sref);
      if (extent == nullptr) {
        break;
      }
      // Check if the extent is still in a good range
      prod_extent *= *extent;
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

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

using tir::Schedule;

class RewriteParallelizeVectorizeUnrollNode : public PostprocNode {
 public:
  void InitializeWithTuneContext(const TuneContext& context) final {}

  bool Apply(const Schedule& sch) final {
    using tir::BlockRV;
    using tir::LoopRV;
    tir::ParsedAnnotation parsed;
    BlockRV root_rv = sch->GetBlock("root");  // TODO(Junru): iterator each PrimFunc
    if (!tir::ParseAnnotation(sch->Get(root_rv), &parsed)) {
      return true;
    }
    RemoveParsedAnn(sch, root_rv, parsed);
    for (BlockRV block_rv : sch->GetChildBlocks(root_rv)) {
      Array<LoopRV> loop_rvs = sch->GetLoops(block_rv);
      AdjustParallelVectorize(sch, block_rv, loop_rvs, &parsed);
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
        int n_loops = loop_rvs.size();
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
          sch->Annotate(loop, "pragma_auto_unroll_max_step", IntImm(DataType::Int(32), max_step));
          sch->Annotate(loop, "pragma_unroll_explicit", IntImm(DataType::Int(32), unroll_explicit));
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

TVM_REGISTER_NODE_TYPE(RewriteParallelizeVectorizeUnrollNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocRewriteParallelizeVectorizeUnroll")
    .set_body_typed(Postproc::RewriteParallelizeVectorizeUnroll);

}  // namespace meta_schedule
}  // namespace tvm
