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
#include <tvm/tir/analysis.h>

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "../utils.h"

namespace tvm {
namespace meta_schedule {

template <class T>
using BufferMap = std::unordered_map<const tir::BufferNode*, T>;
template <class T>
using LoopMap = std::unordered_map<const tir::LoopNode*, T>;
template <class T>
using StmtMap = std::unordered_map<const tir::StmtNode*, T>;

// A hypercube region of a buffer, using represented by [min, max) instead of [min, min + extent)
using HyperCube = std::vector<std::pair<PrimExpr, PrimExpr>>;

struct FeatureSet {
  // Group 1: Computation related features
  enum class AnnPos : int {
    kPosNone = 0,           // Does not have this kind of annotation
    kPosInnerSpatial = 1,   // The annotated iterator is the innermost spatial iterator
    kPosMiddleSpatial = 2,  // The annotated iterator is a middle spatial iterator
    kPosOuterSpatial = 3,   // The annotated iterator is the outermost spatial iterator
    kPosInnerReduce = 4,    // The annotated iterator is the innermost reduce iterator
    kPosMiddleReduce = 5,   // The annotated iterator is a middle reduce iterator
    kPosOuterReduce = 6,    // The annotated iterator is the outermost reduce iterator
    kPosMixed = 7           // The annotated iterator is a mixed space and reduce iterator
  };
  double float_mad;         // The number of float MAD (Multiply–add) ops
  double float_addsub;      // The number of float add and sub ops
  double float_mul;         // The number of float multiply ops
  double float_divmod;      // The number of float div and mod ops
  double float_cmp;         // The number of float comparison ops
  double float_math_func;   // The number of float math func calls
  double float_other_func;  // The number of other float func calls
  double int_mad;           // The number of integer MAD (Multiply–add) ops
  double int_addsub;        // The number of integer add and sub ops
  double int_mul;           // The number of integer multiply ops
  double int_divmod;        // The number of integer div and mod ops
  double int_cmp;           // The number of integer comparison ops
  double int_math_func;     // The number of integer math func calls
  double int_other_func;    // The number of other integer func calls
  double bool_op;           // The number of bool ops
  double select_op;         // The number of select ops
  double vec_num;           // The number of vectorized iterators
  double vec_prod;          // The product of the lengths of vectorized iterators
  double vec_len;           // The length of the innermost vectorized iterator
  AnnPos vec_type;          // The type of vectorization position
  double unroll_num;        // The number of unrolled iterators
  double unroll_prod;       // The product of the lengths of vectorized iterators
  double unroll_len;        // The length of the innermost unrolled iterator
  AnnPos unroll_type;       // The type of unroll position
  double parallel_num;      // The number of paralleled iterators
  double parallel_prod;     // The product of the lengths of paralleled iterators
  double parallel_len;      // The length of the innermost paralleled iterators
  AnnPos parallel_type;     // The type of parallel position
  double is_gpu;            // Whether it is a GPU task
  double blockIdx_x_len;    // The length of blockIdx.x
  double blockIdx_y_len;    // The length of blockIdx.y
  double blockIdx_z_len;    // The length of blockIdx.z
  double threadIdx_x_len;   // The length of threadIdx.x
  double threadIdx_y_len;   // The length of threadIdx.y
  double threadIdx_z_len;   // The length of threadIdx.z
  double vthread_len;       // The length of virtual thread

  // Group 2: Buffer access related features (per buffer)
  struct BufferAccessFeature {
    // Buffer access type
    enum class AccessType : int {
      kRead = 0,       //
      kWrite = 1,      //
      kReadWrite = 2,  //
      kUnknownRW = 3,  //
    };
    // Data reuse type
    enum class ReuseType : int {
      kLoopMultipleRead = 0,         // Buffer reuse because accessed on each iteration of a loop
      kSerialMultipleReadWrite = 1,  // Buffer reuse because it is serially accessed
      kNoReuse = 2,                  // No buffer reuse
    };
    String buffer_name;              // The name of the buffer
    AccessType acc_type;             // The type of the access
    double bytes;                    // The touched memory in bytes
    double unique_bytes;             // The touched unique memory in bytes
    double lines;                    // The number of touched cache lines
    double unique_lines;             // The number touched unique cache lines
    ReuseType reuse_type;            // Tye type of data reuse
    double reuse_dis_iter;           // The reuse distance in iterator number
    double reuse_dis_bytes;          // The reuse distance in total touched bytes
    double reuse_ct;                 // The reuse ratio
    double bytes_d_reuse_ct;         // bytes / reuse_ct
    double unique_bytes_d_reuse_ct;  // unique_bytes / reuse_ct
    double lines_d_reuse_ct;         // lines / reuse_ct
    double unique_lines_d_reuse_ct;  // unique_lines / reuse_ct
    double stride;                   // The stride in access
  };
  std::vector<BufferAccessFeature> accesses;

  // Group 3: Arithmetic intensity related features
  // The number of samples to extract for arithmetic intensity curves
  static const int ARITH_INTENSITY_CURVE_SAMPLE_N = 10;
  // points sampled from the arithmetic intensity curve
  double arith_intensity_curve[ARITH_INTENSITY_CURVE_SAMPLE_N];

  // Group 4: Allocation related features
  double alloc_size;        // The size of allocated buffer in bytes
  double alloc_outer_prod;  // The product of lengths of loops outside the scope of the allocation
  double alloc_inner_prod;  // The product of lengths of loops inside the score of the allocation
  double alloc_prod;        // alloc_outer_prod * alloc_inner_prod

  // Group 5: Outer scope related features
  double outer_prod;            // The product of lengths of outer loops
  double num_loops;             // The number of outer loops
  double auto_unroll_max_step;  // The value of pragma "auto_unroll_max_step"
};

class MathOpCounter : public tir::StmtExprVisitor {
 public:
#define TVM_META_SCHEDULE_FEATURE_EXTRACTION_INC_CNT(DType, FloatCounter, IntCounter) \
  if (DType.is_float()) {                                                             \
    ++result.FloatCounter;                                                            \
  } else {                                                                            \
    ++result.IntCounter;                                                              \
  }
#define TVM_META_SCHEDULE_FEATURE_EXTRACTION_SIMPLE(Type, Counter) \
  void VisitExpr_(const Type* op) final {                          \
    ++result.Counter;                                              \
    StmtExprVisitor::VisitExpr_(op);                               \
  }
#define TVM_META_SCHEDULE_FEATURE_EXTRACTION_BINARY(Type, FloatCounter, IntCounter) \
  void VisitExpr_(const Type* op) final {                                           \
    if (op->dtype.is_float()) {                                                     \
      ++result.FloatCounter;                                                        \
    } else {                                                                        \
      ++result.IntCounter;                                                          \
    }                                                                               \
    StmtExprVisitor::VisitExpr_(op);                                                \
  }

  TVM_META_SCHEDULE_FEATURE_EXTRACTION_SIMPLE(tir::AndNode, bool_op);
  TVM_META_SCHEDULE_FEATURE_EXTRACTION_SIMPLE(tir::OrNode, bool_op);
  TVM_META_SCHEDULE_FEATURE_EXTRACTION_SIMPLE(tir::NotNode, bool_op);
  TVM_META_SCHEDULE_FEATURE_EXTRACTION_SIMPLE(tir::SelectNode, select_op);
  TVM_META_SCHEDULE_FEATURE_EXTRACTION_BINARY(tir::AddNode, float_addsub, int_addsub);
  TVM_META_SCHEDULE_FEATURE_EXTRACTION_BINARY(tir::SubNode, float_addsub, int_addsub);
  TVM_META_SCHEDULE_FEATURE_EXTRACTION_BINARY(tir::MulNode, float_mul, int_mul);
  TVM_META_SCHEDULE_FEATURE_EXTRACTION_BINARY(tir::DivNode, float_divmod, int_divmod);
  TVM_META_SCHEDULE_FEATURE_EXTRACTION_BINARY(tir::ModNode, float_divmod, int_divmod);
  TVM_META_SCHEDULE_FEATURE_EXTRACTION_BINARY(tir::FloorDivNode, float_divmod, int_divmod);
  TVM_META_SCHEDULE_FEATURE_EXTRACTION_BINARY(tir::FloorModNode, float_divmod, int_divmod);
  TVM_META_SCHEDULE_FEATURE_EXTRACTION_BINARY(tir::MaxNode, float_cmp, int_cmp);
  TVM_META_SCHEDULE_FEATURE_EXTRACTION_BINARY(tir::MinNode, float_cmp, int_cmp);
  TVM_META_SCHEDULE_FEATURE_EXTRACTION_BINARY(tir::EQNode, float_cmp, int_cmp);
  TVM_META_SCHEDULE_FEATURE_EXTRACTION_BINARY(tir::NENode, float_cmp, int_cmp);
  TVM_META_SCHEDULE_FEATURE_EXTRACTION_BINARY(tir::LTNode, float_cmp, int_cmp);
  TVM_META_SCHEDULE_FEATURE_EXTRACTION_BINARY(tir::LENode, float_cmp, int_cmp);
  TVM_META_SCHEDULE_FEATURE_EXTRACTION_BINARY(tir::GTNode, float_cmp, int_cmp);
  TVM_META_SCHEDULE_FEATURE_EXTRACTION_BINARY(tir::GENode, float_cmp, int_cmp);

  void VisitExpr_(const tir::CallNode* op) final {
    static auto op_call_effect_ = Op::GetAttrMap<tir::TCallEffectKind>("TCallEffectKind");
    tir::TCallEffectKind effect_kind = op_call_effect_[Downcast<Op>(op->op)];
    bool is_pure = effect_kind == tir::CallEffectKind::kPure ||
                   effect_kind == tir::CallEffectKind::kExprAnnotation;
    if (is_pure) {
      TVM_META_SCHEDULE_FEATURE_EXTRACTION_INC_CNT(op->dtype, float_math_func, int_math_func);
    } else {
      TVM_META_SCHEDULE_FEATURE_EXTRACTION_INC_CNT(op->dtype, float_other_func, int_other_func);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

#undef TVM_META_SCHEDULE_FEATURE_EXTRACTION_BINARY
#undef TVM_META_SCHEDULE_FEATURE_EXTRACTION_SIMPLE
#undef TVM_META_SCHEDULE_FEATURE_EXTRACTION_INC_CNT

  // TODO(merrymercy,junrushao1994): Detect MAD (Multiply–add)
  struct Result {
    int float_mad{0};         // The number of float MAD (Multiply–add) ops
    int float_addsub{0};      // The number of float add and sub ops
    int float_mul{0};         // The number of float multiply ops
    int float_divmod{0};      // The number of float div and mod ops
    int float_cmp{0};         // The number of float comparison ops
    int float_math_func{0};   // The number of float math func calls
    int float_other_func{0};  // The number of other float func calls
    int int_mad{0};           // The number of integer MAD (Multiply–add) ops
    int int_addsub{0};        // The number of integer add and sub ops
    int int_mul{0};           // The number of integer multiply ops
    int int_divmod{0};        // The number of integer div and mod ops
    int int_cmp{0};           // The number of integer comparison ops
    int int_math_func{0};     // The number of integer math func calls
    int int_other_func{0};    // The number of other integer func calls
    int bool_op{0};           // The number of bool ops
    int select_op{0};         // The number of select ops
  };
  Result result;

  static Result Count(const tir::Stmt& stmt) {
    MathOpCounter counter;
    counter(stmt);
    return counter.result;
  }
};

class CoefficientExtractor : public tir::StmtExprVisitor {
 public:
  explicit CoefficientExtractor(const tir::Var& var)
      : var(var), stride(0), visited_var(false), visited_add(false), visited_mul(false) {}

  void VisitExpr_(const tir::MulNode* node) override {
    StmtExprVisitor::VisitExpr_(node);
    if (visited_var) {
      if (!visited_add) {
        if (const auto* a = node->a.as<IntImmNode>()) {
          visited_mul = true;
          stride = a->value;
        } else if (const auto* b = node->b.as<IntImmNode>()) {
          visited_mul = true;
          stride = b->value;
        }
      }
    }
  }

  void VisitExpr_(const tir::AddNode* node) override {
    StmtExprVisitor::VisitExpr_(node);
    if (visited_var) {
      if (!visited_mul) {
        visited_add = true;
        stride = 1;
      }
    }
  }

  void VisitExpr_(const tir::VarNode* node) override {
    if (node == var.get()) {
      visited_var = true;
      stride = 2;  // This is a magic default stride in case our approximation strategy fails
    }
  }

  static int64_t Extract(const PrimExpr& expr, const tir::Var& var) {
    CoefficientExtractor extractor(var);
    extractor.VisitExpr(expr);
    return extractor.visited_var ? extractor.stride : 0;
  }

  const tir::Var& var;
  int64_t stride;
  bool visited_var;
  bool visited_add;
  bool visited_mul;
};

class LoopBufferRelationExtractor : public tir::StmtExprVisitor {
 public:
  struct StrideInfo {
    int64_t min_stride;
    int64_t innermost_stride;
    int64_t prod_non_strided_loop_extent;
  };

  struct ReuseInfo {
    FeatureSet::BufferAccessFeature::ReuseType reuse_type;
    double reuse_dis_iter;
    double reuse_dis_bytes;
    int64_t reuse_ct;
  };

  void VisitStmt_(const tir::BlockRealizeNode* realize) override {
    scopes.push_back(realize);
    dfs_path.push_back(realize);
    tir::StmtExprVisitor::VisitStmt_(realize);
    dfs_path.pop_back();
    scopes.pop_back();
    // Simplify the tensor regions touched by assuming parent block vars are constants
    BufferMap<std::vector<HyperCube>> accessed_buffer_regions = ExtractBlockAccesses(realize);
    // Extract the parent loops, organized from inner to outer
    std::vector<const tir::LoopNode*> parent_loops = GetParentLoops(realize);
    int n_loops = parent_loops.size();
    // Initially, we bind all the loop variables to a constant
    for (int i = 0; i < n_loops; ++i) {
      const tir::LoopNode* loop = parent_loops[i];
      analyzer.Bind(loop->loop_var, loop->min);
    }
    // Then, we gradually bind the loops from inner to outer,
    // calculate the area the loops touch on each buffer, which is `Relation::numel`
    for (int i = 0; i < n_loops; ++i) {
      const tir::LoopNode* loop = parent_loops[i];
      analyzer.Bind(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent),
                    /*allow_override=*/true);
      int64_t touched_bytes = 0;
      for (const auto& it : accessed_buffer_regions) {
        const tir::BufferNode* buffer = it.first;
        const std::vector<HyperCube>& regions = it.second;
        int64_t numel = CalcRegionUnionSize(regions, &analyzer);
        loop_buffer_accessed_numel[loop][buffer].push_back(numel);
        touched_bytes += numel * buffer->dtype.bytes();
      }
      for_touched_bytes[loop] = touched_bytes;
    }
    // Next, for each buffer, we find the loop stride on it
    for (const auto& it : accessed_buffer_regions) {
      const tir::BufferNode* buffer = it.first;
      const std::vector<HyperCube>& regions = it.second;
      std::vector<int> buffer_shape = AsVector<PrimExpr, int>()(buffer->shape);
      // Enumerate loops from inner to outer
      int i = 0;
      StrideInfo& stride_info = buffer_accessed_stride[buffer];
      // Calculate stride_info.min_stride
      int64_t& stride = stride_info.min_stride = 0;
      for (; i < n_loops && stride == 0; ++i) {
        stride = CalcVarStrideOnRegion(regions, buffer_shape, parent_loops[i]->loop_var);
      }
      // Calculate stride_info.innermost_stride
      stride_info.innermost_stride = (i == 0) ? stride : 0;
      // Calculate stride_info.prod
      int64_t& prod = stride_info.prod_non_strided_loop_extent = 1;
      for (int j = 0; j < i; ++j) {
        prod *= GetLoopIntExtent(parent_loops[j]).value()->value;
      }
    }
    // Finally, calculate the buffer reuse
    for (const auto& it : accessed_buffer_regions) {
      const tir::BufferNode* buffer = it.first;
      const std::vector<HyperCube>& regions = it.second;
      if (regions.size() >= 2 && !parent_loops.empty()) {
        // Serial reuse
        const tir::LoopNode* loop = parent_loops[0];
        const std::vector<int64_t>& numels = loop_buffer_accessed_numel[loop][buffer];
        int64_t loop_extent = GetLoopIntExtent(loop).value()->value;
        double reuse_dis_iter = numels.empty()  //
                                    ? 1         //
                                    : *std::min_element(numels.begin(), numels.end());
        double reuse_dis_bytes = for_touched_bytes[loop];
        buffer_reuse[buffer] = {
            FeatureSet::BufferAccessFeature::ReuseType::kSerialMultipleReadWrite,
            reuse_dis_iter / loop_extent,   //
            reuse_dis_bytes / loop_extent,  //
            static_cast<int64_t>(regions.size()) - 1,
        };
        continue;
      }
      // Collect all `tir::Var`s that appears in the buffer region
      std::unordered_set<const tir::VarNode*> region_vars;
      for (const HyperCube& region : regions) {
        for (const std::pair<PrimExpr, PrimExpr>& range : region) {
          tir::PostOrderVisit(range.first, [&region_vars](const ObjectRef& obj) -> void {
            if (const auto* var = obj.as<tir::VarNode>()) {
              region_vars.insert(var);
            }
          });
        }
      }
      // Find the innermost loop that does not determine the buffer region,
      // i.e. detect reuse on each iteration of a loop
      int invariant_loop_idx = -1;
      for (int i = 0; i < n_loops; ++i) {
        if (!region_vars.count(parent_loops[i]->loop_var.get())) {
          invariant_loop_idx = i;
          break;
        }
      }
      // The region depends on all the loops, i.e. there is no reuse
      if (invariant_loop_idx == -1) {
        buffer_reuse[buffer] = {FeatureSet::BufferAccessFeature::ReuseType::kNoReuse, 0, 0, 0};
        continue;
      }
      // There is loop reuse at `invariant_loop_idx`
      const tir::LoopNode* loop = parent_loops[invariant_loop_idx];
      int64_t loop_extent = GetLoopIntExtent(loop).value()->value;
      int64_t reuse_dis_iter = 1;
      int64_t reuse_dis_bytes = 0;
      // Calculate `reuse_dis_iter` and `reuse_dis_bytes`
      if (invariant_loop_idx == 0) {
        reuse_dis_bytes = loop_buffer_accessed_numel[loop][buffer].size();
      } else {
        for (int i = 0; i < invariant_loop_idx; ++i) {
          reuse_dis_iter *= GetLoopIntExtent(parent_loops[i]).value()->value;
        }
        const std::vector<int64_t>& numels = loop_buffer_accessed_numel[loop][buffer];
        reuse_dis_bytes = std::accumulate(numels.begin(), numels.end(), int64_t(0));
      }
      reuse_dis_bytes *= buffer->dtype.bytes();
      buffer_reuse[buffer] = {
          FeatureSet::BufferAccessFeature::ReuseType::kLoopMultipleRead,
          static_cast<double>(reuse_dis_iter),   //
          static_cast<double>(reuse_dis_bytes),  //
          loop_extent,                           //
      };
    }
  }

  void VisitStmt_(const tir::LoopNode* loop) override {
    dfs_path.push_back(loop);
    tir::StmtExprVisitor::VisitStmt_(loop);
    dfs_path.pop_back();
  }

  static void Extract(const tir::Stmt& stmt) {
    LoopBufferRelationExtractor extractor;
    extractor.VisitStmt(stmt);
  }

  std::vector<const tir::BlockRealizeNode*> scopes;
  std::vector<const tir::StmtNode*> dfs_path;
  mutable arith::Analyzer analyzer;

  LoopMap<BufferMap<std::vector<int64_t>>> loop_buffer_accessed_numel;
  BufferMap<StrideInfo> buffer_accessed_stride;
  BufferMap<ReuseInfo> buffer_reuse;
  LoopMap<int64_t> for_touched_bytes;

 private:
  static int64_t CalcRegionUnionSize(const std::vector<HyperCube>& regions,
                                     arith::Analyzer* analyzer) {
    if (regions.empty()) {
      return 1;
    }
    int64_t numel = 1;
    int n_regions = regions.size();
    int ndim = regions[0].size();
    for (int i = 0; i < ndim; ++i) {
      int64_t min = arith::ConstIntBound::kPosInf;
      int64_t max = arith::ConstIntBound::kNegInf;
      for (int j = 0; j < n_regions; ++j) {
        const std::pair<PrimExpr, PrimExpr>& region = regions[j][i];
        arith::ConstIntBound l_bound = analyzer->const_int_bound(region.first);
        arith::ConstIntBound r_bound = analyzer->const_int_bound(region.second);
        min = std::min(min, l_bound->min_value);
        max = std::min(max, r_bound->max_value);
      }
      if (min <= max) {
        numel *= max - min + 1;
      }
    }
    return numel;
  }

  static int64_t CalcVarStrideOnRegion(const std::vector<HyperCube>& regions,
                                       const std::vector<int>& shape, const tir::Var& var) {
    constexpr int64_t kNotFound = std::numeric_limits<int64_t>::max();
    int ndim = shape.size();
    // Calculate the stride of buffer
    std::vector<int64_t> buffer_stride(shape.begin(), shape.end());
    for (int i = ndim - 1; i >= 1; --i) {
      buffer_stride[i - 1] *= buffer_stride[i];
    }
    // Calculate the min stride possible
    int64_t result = kNotFound;
    for (const HyperCube& region : regions) {
      CHECK_EQ(region.size(), shape.size());
      // Find the rightest dimension that contains the given variable
      for (int i = ndim - 1; i >= 0; --i) {
        const PrimExpr& idx = region[i].first;
        int64_t coef = CoefficientExtractor::Extract(idx, var);
        if (coef != 0) {
          result = std::min(result, std::abs(coef) * buffer_stride[i]);
          break;
        }
      }
    }
    return (result == kNotFound) ? 0 : result;
  }

  BufferMap<std::vector<HyperCube>> ExtractBlockAccesses(
      const tir::BlockRealizeNode* realize) const {
    arith::Analyzer analyzer;
    // Check if each region is 'update'
    int n_reads = realize->block->reads.size();
    int n_writes = realize->block->writes.size();
    std::vector<int> is_read_update(n_reads, 0);
    for (int i_r = 0; i_r < n_reads; ++i_r) {
      // Enumerate each read region
      const tir::TensorRegion& r = realize->block->reads[i_r];
      for (int i_w = 0; i_w < n_writes; ++i_w) {
        // Enumerate each write region
        const tir::TensorRegion& w = realize->block->writes[i_w];
        if (r->buffer.same_as(w->buffer)) {
          const Array<Range>& r_region = r->region;
          const Array<Range>& w_region = w->region;
          int ndim = r_region.size();
          // Check if `r_region` and `w_region` are exactly the same
          bool is_same = true;
          for (int i = 0; i < ndim; ++i) {
            if (!analyzer.CanProve(r_region[i]->min == w_region[i]->min) ||
                !analyzer.CanProve(r_region[i]->extent == w_region[i]->extent)) {
              is_same = false;
              break;
            }
          }
          // If so, mark it
          if (is_same) {
            is_read_update[i_r] = true;
          }
        }
      }
    }
    // Extract the block vars in the parent scope
    std::unordered_map<const tir::VarNode*, PrimExpr> var_substitutes;
    // Substitute block vars of the parent scope to its min
    if (!scopes.empty()) {
      const tir::BlockNode* parent_block = scopes.back()->block.operator->();
      for (const tir::IterVar& block_var : parent_block->iter_vars) {
        var_substitutes[block_var->var.get()] = block_var->dom->min;
      }
    }
    // Substitute block vars to its binding
    {
      CHECK_EQ(realize->binding_values.size(), realize->block->iter_vars.size());
      int n = realize->binding_values.size();
      for (int i = 0; i < n; ++i) {
        const tir::Var& lhs = realize->block->iter_vars[i]->var;
        const PrimExpr& rhs = realize->binding_values[i];
        var_substitutes[lhs.get()] = rhs;
      }
    }
    // Helper function to convert a TIR region into our hyper-cube and do necessary simplification
    auto f_make_hyper_cube = [&analyzer,
                              &var_substitutes](const Array<Range>& region) -> HyperCube {
      // Helper function to do the substitution
      auto f_sub = [&var_substitutes](const PrimExpr& expr) -> Optional<PrimExpr> {
        if (const auto* var = expr.as<tir::VarNode>()) {
          auto it = var_substitutes.find(var);
          if (it != var_substitutes.end()) {
            return it->second;
          }
        }
        return NullOpt;
      };
      int ndim = region.size();
      HyperCube result;
      result.reserve(ndim);
      for (int i = 0; i < ndim; ++i) {
        const Range& range = region[i];
        PrimExpr min = analyzer.Simplify(tir::Substitute(range->min, f_sub));
        PrimExpr max = analyzer.Simplify(tir::Substitute(min + range->extent, f_sub));
        result.emplace_back(min, max);
      }
      return result;
    };
    // Apply the substitution to each tensor region
    BufferMap<std::vector<HyperCube>> result;
    result.reserve(realize->block->reads.size() + realize->block->writes.size());
    for (int i = 0; i < n_reads; ++i) {
      // Skip those update regions
      if (is_read_update[i]) {
        continue;
      }
      const tir::TensorRegion& region = realize->block->reads[i];
      result[region->buffer.get()].push_back(f_make_hyper_cube(region->region));
    }
    for (int i = 0; i < n_writes; ++i) {
      const tir::TensorRegion& region = realize->block->reads[i];
      result[region->buffer.get()].push_back(f_make_hyper_cube(region->region));
    }
    return result;
  }

  std::vector<const tir::LoopNode*> GetParentLoops(const tir::BlockRealizeNode* realize) const {
    std::vector<const tir::LoopNode*> result;
    int path_depth = dfs_path.size();
    for (int i = path_depth - 1; i >= 0; --i) {
      const tir::StmtNode* stmt = dfs_path[i];
      if (stmt->IsInstance<tir::LoopNode>()) {
        result.push_back(static_cast<const tir::LoopNode*>(stmt));
      } else {
        break;
      }
    }
    return result;
  }
};

// class BufferAccessExtractor : public tir::StmtExprVisitor {
//  public:
//   struct Access {
//     FeatureSet::BufferAccessFeature::AccessType type =
//         FeatureSet::BufferAccessFeature::AccessType::kUnknownRW;
//     std::vector<MultiDimIdx> indices;
//   };

//   static BufferMap<Access> Extract(const tir::BufferStore& store) {
//     BufferAccessExtractor extractor;
//     Access& access = extractor.accesses[store->buffer];
//     access.type = FeatureSet::BufferAccessFeature::AccessType::kWrite;
//     access.indices.push_back(store->indices);
//     extractor.VisitStmt(store);
//     return extractor.accesses;
//   }

//   void VisitExpr_(const tir::BufferLoadNode* load) final {
//     const tir::Buffer& buffer = load->buffer;
//     Access& access = accesses[buffer];
//     switch (access.type) {
//       case FeatureSet::BufferAccessFeature::AccessType::kRead:
//         // do nothing
//         break;
//       case FeatureSet::BufferAccessFeature::AccessType::kWrite:
//         // from write to read-write
//         access.type = FeatureSet::BufferAccessFeature::AccessType::kReadWrite;
//         break;
//       case FeatureSet::BufferAccessFeature::AccessType::kReadWrite:
//         // do nothing
//         break;
//       case FeatureSet::BufferAccessFeature::AccessType::kUnknownRW:
//         // from unknown to read
//         access.type = FeatureSet::BufferAccessFeature::AccessType::kRead;
//         break;
//       default:
//         LOG(FATAL) << "ValueError: Cannot recognize BufferAccessFeature::AccessType: "
//                    << static_cast<int>(access.type);
//     }
//     if (access.type != FeatureSet::BufferAccessFeature::AccessType::kReadWrite) {
//       // If a buffer is both read and written, in the tvm DSL, it must be a update,
//       // so the indices should be the same. Then we can skip appending indices for it.
//       // Otherwise we do the following.
//       access.indices.push_back(load->indices);
//     }
//     StmtExprVisitor::VisitExpr_(load);
//   }

//   BufferMap<Access> accesses;
// };

class PerStoreFeatureExtractor : public tir::StmtExprVisitor {
 public:
  PerStoreFeatureExtractor() {}

  void VisitStmt_(const tir::LoopNode* loop) override {
    std::vector<const tir::LoopNode*>* ref_loops = nullptr;
    if (!loop->annotations.empty()) {
      CHECK_EQ(loop->annotations.size(), 1)
          << "ValueError: At most one annotation is allowed on a loop, but gets: "
          << GetRef<tir::Loop>(loop);
      const tir::Annotation& ann = loop->annotations[0];
      CHECK_EQ(ann->attr_key, tir::attr::loop_type)
          << "ValueError: Expects loop annotation to be 'loop_type', but gets: " << ann;
      const auto* value = ann->value.as<tir::StringImmNode>();
      CHECK(value) << "ValueError: Expevt loop annotation to be a string, but gets: " << ann;
      if (value->value == "parallel") {
        ref_loops = &parallel_;
      } else if (value->value == "vectorize") {
        ref_loops = &vectorize_;
      } else if (value->value == "unroll") {
        ref_loops = &unroll_;
      } else if (value->value == "blockIdx.x") {
        ref_loops = &blockIdx_x_;
      } else if (value->value == "blockIdx.y") {
        ref_loops = &blockIdx_y_;
      } else if (value->value == "blockIdx.z") {
        ref_loops = &blockIdx_z_;
      } else if (value->value == "threadIdx.x") {
        ref_loops = &threadIdx_x_;
      } else if (value->value == "threadIdx.y") {
        ref_loops = &threadIdx_y_;
      } else if (value->value == "threadIdx.z") {
        ref_loops = &threadIdx_z_;
      } else if (value->value == "vthread") {
        ref_loops = &vthread_;
      } else {
        LOG(FATAL) << "ValueError: Cannot recognize loop annotation: " << ann;
        throw;
      }
    }
    loops_.push_back(loop);
    if (ref_loops != nullptr) {
      ref_loops->push_back(loop);
    }
    StmtExprVisitor::VisitStmt_(loop);
    if (ref_loops != nullptr) {
      ref_loops->pop_back();
    }
    loops_.pop_back();
  }

  void VisitStmt_(const tir::BufferStoreNode* store) override {
    MathOpCounter::Result math_ops = MathOpCounter::Count(GetRef<tir::Stmt>(store));
    ExtractComputeFeature(store, math_ops);
  }

  void ExtractComputeFeature(const tir::BufferStoreNode* store,
                             const MathOpCounter::Result& math_ops) {
    FeatureSet& feature = buffer_features[store->buffer.get()];
    double loop_extent = ProdLoopExtent(loops_);
#define TVM_META_SCHEDULE_FEATURE_ASSIGN(Name) feature.Name = loop_extent * math_ops.Name;
    TVM_META_SCHEDULE_FEATURE_ASSIGN(float_mad);
    TVM_META_SCHEDULE_FEATURE_ASSIGN(float_addsub);
    TVM_META_SCHEDULE_FEATURE_ASSIGN(float_mul);
    TVM_META_SCHEDULE_FEATURE_ASSIGN(float_divmod);
    TVM_META_SCHEDULE_FEATURE_ASSIGN(float_cmp);
    TVM_META_SCHEDULE_FEATURE_ASSIGN(float_math_func);
    TVM_META_SCHEDULE_FEATURE_ASSIGN(float_other_func);
    TVM_META_SCHEDULE_FEATURE_ASSIGN(int_mad);
    TVM_META_SCHEDULE_FEATURE_ASSIGN(int_addsub);
    TVM_META_SCHEDULE_FEATURE_ASSIGN(int_mul);
    TVM_META_SCHEDULE_FEATURE_ASSIGN(int_divmod);
    TVM_META_SCHEDULE_FEATURE_ASSIGN(int_cmp);
    TVM_META_SCHEDULE_FEATURE_ASSIGN(int_math_func);
    TVM_META_SCHEDULE_FEATURE_ASSIGN(int_other_func);
    TVM_META_SCHEDULE_FEATURE_ASSIGN(bool_op);
    TVM_META_SCHEDULE_FEATURE_ASSIGN(select_op);
#undef TVM_META_SCHEDULE_FEATURE_ASSIGN
#define TVM_META_SCHEDULE_FEATURE_LOOP(Loops, Num, Len, Prod, Type) \
  if (Loops.empty()) {                                              \
    feature.Num = 0;                                                \
    feature.Len = 0;                                                \
    feature.Prod = 0;                                               \
    feature.Type = FeatureSet::AnnPos::kPosNone;                    \
  } else {                                                          \
    feature.Num = Loops.size();                                     \
    feature.Len = GetLoopIntExtent(Loops.back()).value();           \
    feature.Prod = ProdLoopExtent(Loops);                           \
    feature.Type = FeatureSet::AnnPos::kPosMixed;                   \
  }
    TVM_META_SCHEDULE_FEATURE_LOOP(parallel_, parallel_num, parallel_len, parallel_prod,
                                   parallel_type);
    TVM_META_SCHEDULE_FEATURE_LOOP(vectorize_, vec_num, vec_len, vec_prod, vec_type);
    TVM_META_SCHEDULE_FEATURE_LOOP(unroll_, unroll_num, unroll_len, unroll_prod, unroll_type);
#undef TVM_META_SCHEDULE_FEATURE_LOOP
    // TODO: feature.is_gpu
    feature.blockIdx_x_len = FirstLoopExtent(blockIdx_x_);
    feature.blockIdx_y_len = FirstLoopExtent(blockIdx_y_);
    feature.blockIdx_z_len = FirstLoopExtent(blockIdx_z_);
    feature.threadIdx_x_len = FirstLoopExtent(threadIdx_x_);
    feature.threadIdx_y_len = FirstLoopExtent(threadIdx_y_);
    feature.threadIdx_z_len = FirstLoopExtent(threadIdx_z_);
    feature.vthread_len = FirstLoopExtent(vthread_);
  }

  static double ProdLoopExtent(const std::vector<const tir::LoopNode*>& loops) {
    double prod = 1.0;
    for (const tir::LoopNode* loop : loops) {
      int64_t extent = GetLoopIntExtent(loop).value();
      prod *= extent;
    }
    return prod;
  }

  static double FirstLoopExtent(const std::vector<const tir::LoopNode*>& loops) {
    return loops.empty() ? 1.0 : GetLoopIntExtent(loops[0]).value()->value;
  }

 private:
  // The regions accessed by a specific subtree
  // i.e. maps (loop-stmt, buffer) => List[unique-bytes-accessed]
  // StmtMap<BufferMap<std::vector<int64_t>>> loop_buffer_accessed_bytes;
  // The feature vector for each buffer access
  BufferMap<FeatureSet> buffer_features;
  // The shared arithmetic analyzer
  arith::Analyzer ana_;
  // The stacks to store different kinds of for-loops
  std::vector<const tir::LoopNode*> loops_;
  std::vector<const tir::LoopNode*> parallel_;
  std::vector<const tir::LoopNode*> vectorize_;
  std::vector<const tir::LoopNode*> unroll_;
  std::vector<const tir::LoopNode*> blockIdx_x_;
  std::vector<const tir::LoopNode*> blockIdx_y_;
  std::vector<const tir::LoopNode*> blockIdx_z_;
  std::vector<const tir::LoopNode*> threadIdx_x_;
  std::vector<const tir::LoopNode*> threadIdx_y_;
  std::vector<const tir::LoopNode*> threadIdx_z_;
  std::vector<const tir::LoopNode*> vthread_;
};

}  // namespace meta_schedule
}  // namespace tvm
