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
#include <tvm/arith/int_set.h>
#include <tvm/tir/analysis.h>

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "../feature.h"
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

template <class T>
using BlockRealizeMap = std::unordered_map<const tir::BlockRealizeNode*, T>;

template <class T>
using BufferMap = std::unordered_map<const tir::BufferNode*, T>;

using NDIntSet = Array<arith::IntSet>;

struct FeatureSet {
  // Group 1: Computation related features
  struct MathOps {
    int64_t float_mad = 0;         // The number of float MAD (Multiply–add) ops
    int64_t float_addsub = 0;      // The number of float add and sub ops
    int64_t float_mul = 0;         // The number of float multiply ops
    int64_t float_divmod = 0;      // The number of float div and mod ops
    int64_t float_cmp = 0;         // The number of float comparison ops
    int64_t float_math_func = 0;   // The number of float math func calls
    int64_t float_other_func = 0;  // The number of other float func calls
    int64_t int_mad = 0;           // The number of integer MAD (Multiply–add) ops
    int64_t int_addsub = 0;        // The number of integer add and sub ops
    int64_t int_mul = 0;           // The number of integer multiply ops
    int64_t int_divmod = 0;        // The number of integer div and mod ops
    int64_t int_cmp = 0;           // The number of integer comparison ops
    int64_t int_math_func = 0;     // The number of integer math func calls
    int64_t int_other_func = 0;    // The number of other integer func calls
    int64_t bool_op = 0;           // The number of bool ops
    int64_t select_op = 0;         // The number of select ops
  };
  struct AnnIter {
    enum class Pos : int {
      kPosNone = 0,           // Does not have this kind of annotation
      kPosInnerSpatial = 1,   // The annotated iterator is the innermost spatial iterator
      kPosMiddleSpatial = 2,  // The annotated iterator is a middle spatial iterator
      kPosOuterSpatial = 3,   // The annotated iterator is the outermost spatial iterator
      kPosInnerReduce = 4,    // The annotated iterator is the innermost reduce iterator
      kPosMiddleReduce = 5,   // The annotated iterator is a middle reduce iterator
      kPosOuterReduce = 6,    // The annotated iterator is the outermost reduce iterator
      kPosMixed = 7           // The annotated iterator is a mixed space and reduce iterator
    };
    int64_t num = 0;           // The number of iterators with the annotation
    int64_t prod = 0;          // The product of the lengths of iterators with the annotation
    int64_t len = 0;           // The length of the innermost iterator with the annotation
    Pos pos = Pos::kPosMixed;  // The position of the iterators with the annotation
  };
  MathOps math_ops;   // The number of the mathematical operators
  AnnIter vectorize;  // The statistics of iterators annotated with "vectorize"
  AnnIter unroll;     // The statistics of iterators annotated with "unroll"
  AnnIter parallel;   // The statistics of iterators annotated with "parallel"
  // bool is_gpu;              // TODO(@junrushao1994): Whether it is a GPU task
  int64_t blockIdx_x_len;   // The length of blockIdx.x
  int64_t blockIdx_y_len;   // The length of blockIdx.y
  int64_t blockIdx_z_len;   // The length of blockIdx.z
  int64_t threadIdx_x_len;  // The length of threadIdx.x
  int64_t threadIdx_y_len;  // The length of threadIdx.y
  int64_t threadIdx_z_len;  // The length of threadIdx.z
  int64_t vthread_len;      // The length of virtual thread

  // Group 2: Buffer access related features (per buffer)
  struct BufferAccess {
    enum class AccessType : int {
      kRead = 0,       // The buffer is read but not written
      kWrite = 1,      // The buffer is written but not read
      kReadWrite = 2,  // The buffer is both read and written
      kUnknownRW = 3,  // Unknown type
    };
    enum class ReuseType : int {
      kLoopMultipleRead = 0,         // Buffer reuse because accessed on each iteration of a loop
      kSerialMultipleReadWrite = 1,  // Buffer reuse because it is serially accessed
      kNoReuse = 2,                  // No buffer reuse
    };
    String buffer_name;              // The name of the buffer
    AccessType access_type;          // The type of the access
    int64_t bytes;                   // The touched memory in bytes
    int64_t unique_bytes;            // The touched unique memory in bytes
    double lines;                    // The number of touched cache lines
    double unique_lines;             // The number touched unique cache lines
    ReuseType reuse_type;            // The type of data reuse
    double reuse_dis_iter;           // The reuse distance in iterator number
    double reuse_dis_bytes;          // The reuse distance in total touched bytes
    int64_t reuse_ct;                // The reuse ratio
    double bytes_d_reuse_ct;         // bytes        / reuse_ct
    double unique_bytes_d_reuse_ct;  // unique_bytes / reuse_ct
    double lines_d_reuse_ct;         // lines        / reuse_ct
    double unique_lines_d_reuse_ct;  // unique_lines / reuse_ct
    int64_t stride;                  // The stride in access
  };
  std::vector<BufferAccess> buffer_accesses;

  // Group 3: Arithmetic intensity related features
  // The number of samples to extract for arithmetic intensity curves
  static const int NUM_SAMPLE_ARITH_INTENSITY_CURVE = 10;
  // points sampled from the arithmetic intensity curve
  double arith_intensity_curve[NUM_SAMPLE_ARITH_INTENSITY_CURVE];

  // Group 4: Allocation related features
  // double alloc_size;        // The size of allocated buffer in bytes
  // double alloc_outer_prod;  // The product lengths of loops outside the scope of the allocation
  // double alloc_inner_prod;  // The product lengths of loops inside the score of the allocation
  // double alloc_prod;        // alloc_outer_prod * alloc_inner_prod

  // Group 5: Outer scope related features
  double outer_prod;            // The product of lengths of outer loops
  double num_loops;             // The number of outer loops
  double auto_unroll_max_step;  // The value of pragma "auto_unroll_max_step"
};

#define TVM_FEATURE_INC_CNT(DType, FloatCounter, IntCounter) \
  if (DType.is_float()) {                                    \
    ++result_.FloatCounter;                                  \
  } else {                                                   \
    ++result_.IntCounter;                                    \
  }
#define TVM_FEATURE_SIMPLE(Type, Counter) \
  void VisitExpr_(const Type* op) final { \
    ++result_.Counter;                    \
    StmtExprVisitor::VisitExpr_(op);      \
  }
#define TVM_FEATURE_BINARY(Type, FloatCounter, IntCounter) \
  void VisitExpr_(const Type* op) final {                  \
    if (op->dtype.is_float()) {                            \
      ++result_.FloatCounter;                              \
    } else {                                               \
      ++result_.IntCounter;                                \
    }                                                      \
    StmtExprVisitor::VisitExpr_(op);                       \
  }

class MathOpCounter : public tir::StmtExprVisitor {
 public:
  static FeatureSet::MathOps Count(const PrimExpr& expr) {
    MathOpCounter counter;
    counter(expr);
    return counter.result_;
  }

 private:
  TVM_FEATURE_SIMPLE(tir::AndNode, bool_op);
  TVM_FEATURE_SIMPLE(tir::OrNode, bool_op);
  TVM_FEATURE_SIMPLE(tir::NotNode, bool_op);
  TVM_FEATURE_SIMPLE(tir::SelectNode, select_op);
  TVM_FEATURE_BINARY(tir::AddNode, float_addsub, int_addsub);
  TVM_FEATURE_BINARY(tir::SubNode, float_addsub, int_addsub);
  TVM_FEATURE_BINARY(tir::MulNode, float_mul, int_mul);
  TVM_FEATURE_BINARY(tir::DivNode, float_divmod, int_divmod);
  TVM_FEATURE_BINARY(tir::ModNode, float_divmod, int_divmod);
  TVM_FEATURE_BINARY(tir::FloorDivNode, float_divmod, int_divmod);
  TVM_FEATURE_BINARY(tir::FloorModNode, float_divmod, int_divmod);
  TVM_FEATURE_BINARY(tir::MaxNode, float_cmp, int_cmp);
  TVM_FEATURE_BINARY(tir::MinNode, float_cmp, int_cmp);
  TVM_FEATURE_BINARY(tir::EQNode, float_cmp, int_cmp);
  TVM_FEATURE_BINARY(tir::NENode, float_cmp, int_cmp);
  TVM_FEATURE_BINARY(tir::LTNode, float_cmp, int_cmp);
  TVM_FEATURE_BINARY(tir::LENode, float_cmp, int_cmp);
  TVM_FEATURE_BINARY(tir::GTNode, float_cmp, int_cmp);
  TVM_FEATURE_BINARY(tir::GENode, float_cmp, int_cmp);
  void VisitExpr_(const tir::CallNode* op) final {
    static auto op_call_effect_ = Op::GetAttrMap<tir::TCallEffectKind>("TCallEffectKind");
    tir::TCallEffectKind effect_kind = op_call_effect_[Downcast<Op>(op->op)];
    bool is_pure = effect_kind == tir::CallEffectKind::kPure ||
                   effect_kind == tir::CallEffectKind::kExprAnnotation;
    if (is_pure) {
      TVM_FEATURE_INC_CNT(op->dtype, float_math_func, int_math_func);
    } else {
      TVM_FEATURE_INC_CNT(op->dtype, float_other_func, int_other_func);
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  FeatureSet::MathOps result_;
};
#undef TVM_FEATURE_BINARY
#undef TVM_FEATURE_SIMPLE
#undef TVM_FEATURE_INC_CNT

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

class PerBlockFeatureExtractor : public tir::StmtExprVisitor {
 public:
  BlockRealizeMap<FeatureSet> per_block_feature;

  static BlockRealizeMap<FeatureSet> Extract(const tir::PrimFunc& func) {
    PerBlockFeatureExtractor extractor;
    extractor.VisitStmt(func->body);
    return extractor.per_block_feature;
  }

 private:
  /******** Group 1: Computation related features ********/

  void CalcComputeFeature(const tir::BlockRealizeNode* realize, FeatureSet* feature) {
    if (!scopes_.empty()) {
      AddMathOpsToScope(feature->math_ops, scopes_.back());
    }
    AddLoopFeature(&feature->vectorize, this->vectorize_);
    AddLoopFeature(&feature->parallel, this->parallel_);
    AddLoopFeature(&feature->unroll, this->unroll_);
    feature->blockIdx_x_len = FirstLoopExtent(blockIdx_x_);
    feature->blockIdx_y_len = FirstLoopExtent(blockIdx_y_);
    feature->blockIdx_z_len = FirstLoopExtent(blockIdx_z_);
    feature->threadIdx_x_len = FirstLoopExtent(threadIdx_x_);
    feature->threadIdx_y_len = FirstLoopExtent(threadIdx_y_);
    feature->threadIdx_z_len = FirstLoopExtent(threadIdx_z_);
    feature->vthread_len = FirstLoopExtent(vthread_);
  }

  void AddMathOpsToScope(const FeatureSet::MathOps& math_ops, const tir::BlockRealizeNode* scope) {
    // The product of the loops up to the parent
    int64_t prod_loop_extent = 1;
    for (auto iter = dfs_path_.rbegin(); iter != dfs_path_.rend(); ++iter) {
      const tir::StmtNode* stmt = *iter;
      if (stmt == scope) {
        break;
      }
      CHECK(stmt->IsInstance<tir::LoopNode>());
      prod_loop_extent *= GetLoopIntExtent(static_cast<const tir::LoopNode*>(stmt)).value()->value;
    }
    // Add the math_ops to the parent
    FeatureSet::MathOps& parent_math_ops = per_block_feature[scope].math_ops;
#define TVM_FEATURE_MATH_OP_ADD(Name) parent_math_ops.Name = math_ops.Name * prod_loop_extent
    TVM_FEATURE_MATH_OP_ADD(float_mad);
    TVM_FEATURE_MATH_OP_ADD(float_addsub);
    TVM_FEATURE_MATH_OP_ADD(float_mul);
    TVM_FEATURE_MATH_OP_ADD(float_divmod);
    TVM_FEATURE_MATH_OP_ADD(float_cmp);
    TVM_FEATURE_MATH_OP_ADD(float_math_func);
    TVM_FEATURE_MATH_OP_ADD(float_other_func);
    TVM_FEATURE_MATH_OP_ADD(int_mad);
    TVM_FEATURE_MATH_OP_ADD(int_addsub);
    TVM_FEATURE_MATH_OP_ADD(int_mul);
    TVM_FEATURE_MATH_OP_ADD(int_divmod);
    TVM_FEATURE_MATH_OP_ADD(int_cmp);
    TVM_FEATURE_MATH_OP_ADD(int_math_func);
    TVM_FEATURE_MATH_OP_ADD(int_other_func);
    TVM_FEATURE_MATH_OP_ADD(bool_op);
    TVM_FEATURE_MATH_OP_ADD(select_op);
#undef TVM_FEATURE_MATH_OP_ADD
  }

  void AddLoopFeature(FeatureSet::AnnIter* ann,
                      const std::vector<const tir::LoopNode*>& loops) const {
    if (loops.empty()) {
      ann->num = 0;
      ann->len = 0;
      ann->prod = 0;
      ann->pos = FeatureSet::AnnIter::Pos::kPosNone;
    } else {
      ann->num = loops.size();
      ann->len = GetLoopIntExtent(loops.back()).value()->value;
      ann->prod = ProdLoopExtent(loops);
      ann->pos = FeatureSet::AnnIter::Pos::kPosNone;
    }
  }

 private:
  /******** Group 2: Buffer access related features ********/
  struct BufferInfo {
    FeatureSet::BufferAccess::AccessType access_type =
        FeatureSet::BufferAccess::AccessType::kUnknownRW;
    /*! \brief The regions that the buffer is accessed */
    Array<NDIntSet> regions = {};
    /*! \brief loop_accessed_numel[i][...] means the number of elements accessed by loops[i] */
    std::vector<std::vector<int64_t>> loop_accessed_numel = {};
    // Stride info
    int64_t min_stride = 0;
    int64_t innermost_stride = 0;
    int64_t prod_non_strided_loop_extent = 0;
    // Reuse info
    FeatureSet::BufferAccess::ReuseType reuse_type = FeatureSet::BufferAccess::ReuseType::kNoReuse;
    double reuse_dis_iter = 0.0;
    double reuse_dis_bytes = 0.0;
    int64_t reuse_ct = 0;
  };

  BufferMap<BufferInfo> CalcBufferInfo(const tir::BlockRealizeNode* realize,
                                       const std::vector<const tir::LoopNode*>& loops,
                                       std::vector<int64_t>* for_touched_bytes_) const {
    // Initialize the data structures used for the features
    int n_loops = loops.size();
    std::vector<int64_t>& for_touched_bytes = *for_touched_bytes_ =
        std::vector<int64_t>(n_loops, int64_t(0));
    BufferMap<BufferInfo> buffer_info = GatherBufferAccessInfo(realize);
    for (auto& it : buffer_info) {
      BufferInfo& info = it.second;
      info.loop_accessed_numel.resize(n_loops);
    }
    // Part 1. Area-related features
    // Step 1.1. we bind all the loop variables to a constant
    for (int i = 0; i < n_loops; ++i) {
      const tir::LoopNode* loop = loops[i];
      analyzer_.Bind(loop->loop_var, loop->min);
    }
    // Step 1.2. we gradually bind the loops from inner to outer,
    // calculate the area the loops touch on each buffer
    for (int i = 0; i < n_loops; ++i) {
      const tir::LoopNode* loop = loops[i];
      analyzer_.Bind(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent),
                     /*allow_override=*/true);
      int64_t& touched_bytes = for_touched_bytes[i] = 0;
      for (auto& it : buffer_info) {
        const tir::BufferNode* buffer = it.first;
        BufferInfo& info = it.second;
        int64_t numel = CalcRegionUnionSize(info.regions);
        info.loop_accessed_numel[i].push_back(numel);
        touched_bytes += numel * buffer->dtype.bytes();
      }
    }
    // Part 2. Stride-related features
    // For each buffer, we find the loop stride on it
    for (auto& it : buffer_info) {
      const tir::BufferNode* buffer = it.first;
      BufferInfo& info = it.second;
      std::vector<int> buffer_shape = AsVector<PrimExpr, int>()(buffer->shape);
      // Enumerate loops from inner to outer
      int i = 0;
      // Calculate info.min_stride
      int64_t& stride = info.min_stride = 0;
      for (; i < n_loops && stride == 0; ++i) {
        stride = CalcVarStrideOnRegion(info.regions, buffer_shape, loops[i]->loop_var);
      }
      // Calculate info.innermost_stride
      info.innermost_stride = (i == 0) ? stride : 0;
      // Calculate info.prod
      int64_t& prod = info.prod_non_strided_loop_extent = 1;
      for (int j = 0; j < i; ++j) {
        prod *= GetLoopIntExtent(loops[j]).value()->value;
      }
    }
    // Part 3. Reuse-related features
    for (auto& it : buffer_info) {
      const tir::BufferNode* buffer = it.first;
      BufferInfo& info = it.second;
      // Step 3.1. Check serial reuse
      int n_regions = info.regions.size();
      if (n_regions >= 2 && n_loops > 0) {
        // Serial reuse
        constexpr int i = 0;
        const tir::LoopNode* loop = loops[i];
        const std::vector<int64_t>& numels = info.loop_accessed_numel[i];
        int64_t loop_extent = GetLoopIntExtent(loop).value()->value;
        double reuse_dis_iter = numels.empty()  //
                                    ? 1         //
                                    : *std::min_element(numels.begin(), numels.end());
        double reuse_dis_bytes = for_touched_bytes[i];
        info.reuse_type = FeatureSet::BufferAccess::ReuseType::kSerialMultipleReadWrite;
        info.reuse_dis_iter = reuse_dis_iter / loop_extent;
        info.reuse_dis_bytes = reuse_dis_bytes / loop_extent;
        info.reuse_ct = n_regions - 1;
        continue;
      }
      // Step 3.2. Collect all `tir::Var`s that appears in the buffer region
      std::unordered_set<const tir::VarNode*> region_vars;
      for (const NDIntSet& region : info.regions) {
        for (const arith::IntSet& int_set : region) {
          tir::PostOrderVisit(int_set.min(), [&region_vars](const ObjectRef& obj) -> void {
            if (const auto* var = obj.as<tir::VarNode>()) {
              region_vars.insert(var);
            }
          });
        }
      }
      // Step 3.3. Find the innermost loop that does not determine the buffer region
      // i.e. detect reuse on each iteration of a loop
      int invariant_loop_idx = -1;
      for (int i = 0; i < n_loops; ++i) {
        if (!region_vars.count(loops[i]->loop_var.get())) {
          invariant_loop_idx = i;
          break;
        }
      }
      // The region depends on all the loops, i.e. there is no reuse
      if (invariant_loop_idx == -1) {
        info.reuse_type = FeatureSet::BufferAccess::ReuseType::kNoReuse;
        info.reuse_dis_iter = 0;
        info.reuse_dis_bytes = 0;
        info.reuse_ct = 0;
        continue;
      }
      // Step 3.4. There is loop reuse at `invariant_loop_idx`, i.e. reuse detected
      const tir::LoopNode* loop = loops[invariant_loop_idx];
      info.reuse_type = FeatureSet::BufferAccess::ReuseType::kLoopMultipleRead;
      info.reuse_ct = GetLoopIntExtent(loop).value()->value;
      double& reuse_dis_iter = info.reuse_dis_iter = 1;
      double& reuse_dis_bytes = info.reuse_dis_bytes = 0;
      // Calculate `reuse_dis_iter` and `reuse_dis_bytes`
      if (invariant_loop_idx == 0) {
        reuse_dis_bytes =
            buffer->dtype.bytes() * info.loop_accessed_numel[invariant_loop_idx].size();
      } else {
        for (int i = 0; i < invariant_loop_idx; ++i) {
          reuse_dis_iter *= GetLoopIntExtent(loops[i]).value()->value;
        }
        const std::vector<int64_t>& numels = info.loop_accessed_numel[invariant_loop_idx];
        reuse_dis_bytes =
            buffer->dtype.bytes() * std::accumulate(numels.begin(), numels.end(), int64_t(0));
      }
    }
    return buffer_info;
  }

  void CalcBufferAccessFeature(const tir::BlockRealizeNode* realize, FeatureSet* feature_,
                               const std::vector<const tir::LoopNode*>& loops,
                               const BufferMap<BufferInfo>& buffer_info) const {
    std::vector<FeatureSet::BufferAccess>& buffer_features = feature_->buffer_accesses;
    buffer_features.reserve(buffer_info.size());
    for (const auto& iter : buffer_info) {
      const tir::BufferNode* buffer = iter.first;
      const BufferInfo& buffer_info = iter.second;
      int64_t dtype_bytes = buffer->dtype.bytes();
      buffer_features.emplace_back();
      FeatureSet::BufferAccess& feature = buffer_features.back();
      feature.buffer_name = buffer->name;
      feature.access_type = buffer_info.access_type;
      feature.stride = buffer_info.innermost_stride;
      feature.bytes = dtype_bytes * outer_loop_prod_;
      if (loops.empty()) {
        feature.unique_bytes = 1;
        feature.lines = 1;
        feature.unique_lines = 1;
      } else {
        feature.unique_bytes = buffer_info.loop_accessed_numel[0].front() * dtype_bytes;
        double m = static_cast<double>(buffer_info.min_stride) * dtype_bytes / cache_line_bytes_;
        feature.lines =
            outer_loop_prod_ / buffer_info.prod_non_strided_loop_extent * std::min(1.0, m);
        feature.lines = std::max(1.0, feature.lines);
        feature.unique_lines = static_cast<double>(feature.lines) / cache_line_bytes_;
      }
      feature.reuse_type = buffer_info.reuse_type;
      feature.reuse_dis_iter = buffer_info.reuse_dis_iter;
      feature.reuse_dis_bytes = buffer_info.reuse_dis_bytes;
      feature.reuse_ct = buffer_info.reuse_ct;
      if (feature.reuse_ct > 0) {
        feature.bytes_d_reuse_ct =
            static_cast<double>(feature.bytes) / static_cast<double>(feature.reuse_ct);
        feature.unique_bytes_d_reuse_ct =
            static_cast<double>(feature.unique_bytes) / static_cast<double>(feature.reuse_ct);
        feature.lines_d_reuse_ct =
            static_cast<double>(feature.lines) / static_cast<double>(feature.reuse_ct);
        feature.unique_lines_d_reuse_ct =
            static_cast<double>(feature.unique_lines) / static_cast<double>(feature.reuse_ct);
      } else {
        // no reuse, multiply by a magic number '2'
        feature.bytes_d_reuse_ct = feature.bytes * 2.0;
        feature.unique_bytes_d_reuse_ct = feature.unique_bytes * 2.0;
        feature.lines_d_reuse_ct = feature.lines * 2.0;
        feature.unique_lines_d_reuse_ct = feature.unique_lines * 2.0;
      }
    }
  }

  BufferMap<BufferInfo> GatherBufferAccessInfo(const tir::BlockRealizeNode* realize) const {
    // Step 1. Check if each region is 'update'
    int n_reads = realize->block->reads.size();
    int n_writes = realize->block->writes.size();
    std::vector<int> is_read_update(n_reads, 0);
    std::vector<int> is_write_update(n_writes, 0);
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
            if (!analyzer_.CanProve(r_region[i]->min == w_region[i]->min) ||
                !analyzer_.CanProve(r_region[i]->extent == w_region[i]->extent)) {
              is_same = false;
              break;
            }
          }
          // If so, mark it
          if (is_same) {
            is_read_update[i_r] = true;
            is_write_update[i_w] = true;
          }
        }
      }
    }
    // Step 2. Extract the block vars in the parent scope
    std::unordered_map<const tir::VarNode*, PrimExpr> var_substitutes;
    // Step 2.1. Substitute block vars of the parent scope to its min
    if (!scopes_.empty()) {
      const tir::BlockNode* parent_block = scopes_.back()->block.operator->();
      for (const tir::IterVar& block_var : parent_block->iter_vars) {
        var_substitutes[block_var->var.get()] = block_var->dom->min;
      }
    }
    // Step 2.2. Substitute block vars to its binding
    {
      CHECK_EQ(realize->binding_values.size(), realize->block->iter_vars.size());
      int n = realize->binding_values.size();
      for (int i = 0; i < n; ++i) {
        const tir::Var& lhs = realize->block->iter_vars[i]->var;
        const PrimExpr& rhs = realize->binding_values[i];
        var_substitutes[lhs.get()] = rhs;
      }
    }
    // Step 2.3. Helper to convert a TIR region into our int-set and do necessary simplification
    auto f_make_int_set = [this, &var_substitutes](const Array<Range>& region) -> NDIntSet {
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
      NDIntSet result;
      result.reserve(ndim);
      for (int i = 0; i < ndim; ++i) {
        const Range& range = region[i];
        PrimExpr min = analyzer_.Simplify(tir::Substitute(range->min, f_sub));
        PrimExpr max = analyzer_.Simplify(tir::Substitute(min + range->extent - 1, f_sub));
        result.push_back(arith::IntSet::Interval(min, max));
      }
      return result;
    };
    // Step 3. Apply the substitution to each tensor region
    BufferMap<BufferInfo> result;
    result.reserve(realize->block->reads.size() + realize->block->writes.size());
    for (int i = 0; i < n_reads; ++i) {
      // Skip those update regions
      if (is_read_update[i]) {
        continue;
      }
      const tir::TensorRegion& region = realize->block->reads[i];
      BufferInfo& buffer_info = result[region->buffer.get()];
      buffer_info.access_type = FeatureSet::BufferAccess::AccessType::kRead;
      buffer_info.regions.push_back(f_make_int_set(region->region));
    }
    for (int i = 0; i < n_writes; ++i) {
      const tir::TensorRegion& region = realize->block->reads[i];
      BufferInfo& buffer_info = result[region->buffer.get()];
      if (is_write_update[i] ||
          buffer_info.access_type == FeatureSet::BufferAccess::AccessType::kRead) {
        buffer_info.access_type = FeatureSet::BufferAccess::AccessType::kReadWrite;
      } else {
        buffer_info.access_type = FeatureSet::BufferAccess::AccessType::kWrite;
      }
      buffer_info.regions.push_back(f_make_int_set(region->region));
    }
    return result;
  }

  int64_t CalcRegionUnionSize(const Array<NDIntSet>& regions) const {
    if (regions.empty()) {
      return 1;
    }
    int64_t numel = 1;
    int n_regions = regions.size();
    int ndim = regions[0].size();
    for (int i = 0; i < ndim; ++i) {
      // Calculate the union set
      Array<arith::IntSet> int_sets;
      int_sets.reserve(n_regions);
      for (int j = 0; j < n_regions; ++j) {
        int_sets.push_back(regions[j][i]);
      }
      arith::IntSet union_set = arith::Union(int_sets);
      // Update the area
      int64_t min = Downcast<IntImm>(union_set.min())->value;
      int64_t max = Downcast<IntImm>(union_set.max())->value;
      if (min <= max) {
        numel *= max - min + 1;
      }
    }
    return numel;
  }

  static int64_t CalcVarStrideOnRegion(const Array<NDIntSet>& regions,
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
    for (const NDIntSet& region : regions) {
      CHECK_EQ(region.size(), shape.size());
      // Find the rightest dimension that contains the given variable
      for (int i = ndim - 1; i >= 0; --i) {
        PrimExpr idx = region[i].min();
        int64_t coef = CoefficientExtractor::Extract(idx, var);
        if (coef != 0) {
          result = std::min(result, std::abs(coef) * buffer_stride[i]);
          break;
        }
      }
    }
    return (result == kNotFound) ? 0 : result;
  }

 private:
  /******** Group 3: Arithmetic intensity related features ********/

  void CalcAritheticIntensityFeature(const tir::BlockRealizeNode* realize, FeatureSet* feature,
                                     const std::vector<const tir::LoopNode*>& loops,
                                     const std::vector<int64_t>& for_touched_bytes,
                                     const FeatureSet::MathOps& math_ops) const {
    CHECK_EQ(loops.size(), for_touched_bytes.size());
    int n_loops = loops.size();
    // Calculate `memory_bytes`
    std::vector<double> memory_bytes;
    for (int i = 0; i < n_loops; ++i) {
      memory_bytes.push_back(std::log2(for_touched_bytes[i]));
    }
    // Calculate `compute_ops` and `cur_compute_ops`
    std::vector<double> compute_ops;
    double total_compute_ops = math_ops.float_mad + math_ops.float_addsub + math_ops.float_mul +
                               math_ops.float_divmod + math_ops.float_cmp +
                               math_ops.float_math_func + math_ops.float_other_func;
    for (int i = 0; i < n_loops; ++i) {
      int64_t extent = GetLoopIntExtent(loops[i]).value()->value;
      total_compute_ops *= extent;
      compute_ops.push_back(std::log2(total_compute_ops));
    }
    // Fill the feature set
    if (total_compute_ops <= 0 || compute_ops.empty()) {
      for (int i = 0; i < FeatureSet::NUM_SAMPLE_ARITH_INTENSITY_CURVE; ++i) {
        feature->arith_intensity_curve[i] = 0.0;
      }
      return;
    }
    int p = 0;
    for (int i = 0; i < FeatureSet::NUM_SAMPLE_ARITH_INTENSITY_CURVE; ++i) {
      double& result = feature->arith_intensity_curve[i];
      double cur_compute_ops = static_cast<double>(i + 1) /
                               FeatureSet::NUM_SAMPLE_ARITH_INTENSITY_CURVE * total_compute_ops;
      // Find the first `p` that `compute[p] >= total * (i + 1) / N`
      for (; p < n_loops; ++p) {
        if (compute_ops[p] >= cur_compute_ops - 1e-4) {
          break;
        }
      }
      CHECK_LT(p, n_loops);
      if (p == 0) {
        result = compute_ops[p] / memory_bytes[p];
      } else {
        double base = compute_ops[p - 1] / memory_bytes[p - 1];
        double slope =
            (compute_ops[p] / memory_bytes[p] - compute_ops[p - 1] / memory_bytes[p - 1]) /
            (compute_ops[p] - compute_ops[p - 1]);
        result = base + slope * (cur_compute_ops - compute_ops[p - 1]);
      }
    }
  }

 private:
  /******** Group 5: Outer scope related features ********/

  void CalcOuterScopeFeature(const tir::BlockRealizeNode* realize, FeatureSet* feature) const {
    feature->outer_prod = outer_loop_prod_;
    feature->num_loops = loops_.size();
    feature->auto_unroll_max_step = auto_unroll_.empty() ? 0 : auto_unroll_.back();
  }

 private:
  /******** Visitors ********/
  void VisitStmt_(const tir::BlockRealizeNode* realize) override {
    scopes_.push_back(realize);
    dfs_path_.push_back(realize);
    tir::StmtExprVisitor::VisitStmt_(realize);
    dfs_path_.pop_back();
    scopes_.pop_back();
    // Get the ancestor loops from inner to outer, up to the parent scope
    std::vector<const tir::LoopNode*> loops;
    for (auto iter = dfs_path_.rbegin(); iter != dfs_path_.rend(); ++iter) {
      const tir::StmtNode* stmt = *iter;
      if (stmt->IsInstance<tir::LoopNode>()) {
        loops.push_back(static_cast<const tir::LoopNode*>(stmt));
      } else {
        break;
      }
    }
    FeatureSet& feature = per_block_feature[realize];
    // Group 1: Computation related features
    CalcComputeFeature(realize, &feature);
    // Group 2: Buffer access related features
    std::vector<int64_t> for_touched_bytes;
    BufferMap<BufferInfo> buffer_info = CalcBufferInfo(realize, loops, &for_touched_bytes);
    CalcBufferAccessFeature(realize, &feature, loops, buffer_info);
    // Group 3: Arithmetic intensity related features
    CalcAritheticIntensityFeature(realize, &feature, loops, for_touched_bytes, feature.math_ops);
    // Group 5: Outer scope related features
    CalcOuterScopeFeature(realize, &feature);
  }

  void VisitStmt_(const tir::LoopNode* loop) override {
    int64_t extent = GetLoopIntExtent(loop).value()->value;
    int64_t auto_unroll = -1;
    // Handling annotated loops
    std::vector<const tir::LoopNode*>* ref_loops = nullptr;
    if (!loop->annotations.empty()) {
      std::unordered_set<std::string> annotations;
      for (const tir::Annotation& ann : loop->annotations) {
        if (ann->attr_key == tir::attr::loop_type) {
          CHECK_EQ(ann->attr_key, tir::attr::loop_type)
              << "ValueError: Expects loop annotation to be 'loop_type', but gets: " << ann;
          const auto* value = ann->value.as<tir::StringImmNode>();
          CHECK(value) << "ValueError: Expevt loop annotation to be a string, but gets: " << ann;
          annotations.insert(value->value);
        } else if (ann->attr_key == "pragma_auto_unroll_max_step") {
          auto_unroll = Downcast<Integer>(ann->value)->value;
        }
      }
      if (annotations.count("parallel")) {
        ref_loops = &parallel_;
      } else if (annotations.count("vectorize")) {
        ref_loops = &vectorize_;
      } else if (annotations.count("unroll")) {
        ref_loops = &unroll_;
      } else if (annotations.count("blockIdx.x")) {
        ref_loops = &blockIdx_x_;
      } else if (annotations.count("blockIdx.y")) {
        ref_loops = &blockIdx_y_;
      } else if (annotations.count("blockIdx.z")) {
        ref_loops = &blockIdx_z_;
      } else if (annotations.count("threadIdx.x")) {
        ref_loops = &threadIdx_x_;
      } else if (annotations.count("threadIdx.y")) {
        ref_loops = &threadIdx_y_;
      } else if (annotations.count("threadIdx.z")) {
        ref_loops = &threadIdx_z_;
      } else if (annotations.count("vthread")) {
        ref_loops = &vthread_;
      }
    }
    if (ref_loops != nullptr) {
      ref_loops->push_back(loop);
    }
    if (auto_unroll != -1) {
      auto_unroll_.push_back(auto_unroll);
    }
    outer_loop_prod_ *= extent;
    dfs_path_.push_back(loop);
    loops_.push_back(loop);
    tir::StmtExprVisitor::VisitStmt_(loop);
    loops_.pop_back();
    dfs_path_.pop_back();
    outer_loop_prod_ /= extent;
    if (auto_unroll != -1) {
      auto_unroll_.pop_back();
    }
    if (ref_loops != nullptr) {
      ref_loops->pop_back();
    }
  }

  void VisitStmt_(const tir::BufferStoreNode* store) override {
    CHECK(!scopes_.empty());
    AddMathOpsToScope(MathOpCounter::Count(store->value), scopes_.back());
  }

  void VisitStmt_(const tir::ReduceStepNode* reduce) override {
    CHECK(!scopes_.empty());
    AddMathOpsToScope(MathOpCounter::Count(reduce->rhs), scopes_.back());
  }

 private:
  static int64_t ProdLoopExtent(const std::vector<const tir::LoopNode*>& loops) {
    int64_t prod = 1.0;
    for (const tir::LoopNode* loop : loops) {
      prod *= GetLoopIntExtent(loop).value()->value;
    }
    return prod;
  }

  static int64_t FirstLoopExtent(const std::vector<const tir::LoopNode*>& loops) {
    return loops.empty() ? 1 : GetLoopIntExtent(loops[0]).value()->value;
  }

 private:
  /******** Data structure used in recursive visiting ********/
  /*! \brief The scope info used in recursive visiting */
  std::vector<const tir::BlockRealizeNode*> scopes_;
  /*! \brief The loop / block-realize visited up-down in the DFS path */
  std::vector<const tir::StmtNode*> dfs_path_;
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
  std::vector<int64_t> auto_unroll_;
  /*! \brief The persistent analyzer */
  mutable arith::Analyzer analyzer_;
  /*! \brief The product of the extents of outer loops */
  int64_t outer_loop_prod_ = 1;

  static constexpr int64_t cache_line_bytes_ = 64;
};

// shifted log to incorporate the property that slog(0) = 0
inline double slog(double x) {
  if (x < 0) {
    x = -x;
  }
  return std::log2(x + 1);
}

#define TVM_FEATURE_ADD_ANN_ITER(s)                      \
  slog(s.num), slog(s.num), slog(s.len), /**/            \
      static_cast<double>(static_cast<int>(s.pos) == 0), \
      static_cast<double>(static_cast<int>(s.pos) == 1), \
      static_cast<double>(static_cast<int>(s.pos) == 2), \
      static_cast<double>(static_cast<int>(s.pos) == 3), \
      static_cast<double>(static_cast<int>(s.pos) == 4), \
      static_cast<double>(static_cast<int>(s.pos) == 5), \
      static_cast<double>(static_cast<int>(s.pos) == 6), \
      static_cast<double>(static_cast<int>(s.pos) == 7)

std::vector<std::vector<double>> CalcPerBlockFeature(const tir::PrimFunc& func,
                                                     int max_num_buffer_access_features) {
  constexpr size_t kNumFeatureGroup1 = 8 * 2 + 11 * 3 + 7;
  constexpr size_t kNumFeatureGroup2Subgroup = 18;
  constexpr size_t kNumFeatureGroup3 = FeatureSet::NUM_SAMPLE_ARITH_INTENSITY_CURVE;
  constexpr size_t kNumFeatureGroup5 = 3;
  size_t kNumFeatureGroup2 = kNumFeatureGroup2Subgroup * max_num_buffer_access_features;

  BlockRealizeMap<FeatureSet> feature_map = PerBlockFeatureExtractor::Extract(func);
  std::vector<std::vector<double>> feature_vector;
  feature_vector.reserve(feature_map.size());
  for (const auto& iter : feature_map) {
    const FeatureSet& feature = iter.second;
    /***** Group 1: Computation related features *****/
    std::vector<double> result{
        slog(feature.math_ops.float_mad),
        slog(feature.math_ops.float_addsub),
        slog(feature.math_ops.float_mul),
        slog(feature.math_ops.float_divmod),
        slog(feature.math_ops.float_cmp),
        slog(feature.math_ops.float_math_func),
        slog(feature.math_ops.float_other_func),
        slog(feature.math_ops.int_mad),
        slog(feature.math_ops.int_addsub),
        slog(feature.math_ops.int_mul),
        slog(feature.math_ops.int_divmod),
        slog(feature.math_ops.int_cmp),
        slog(feature.math_ops.int_math_func),
        slog(feature.math_ops.int_other_func),
        slog(feature.math_ops.bool_op),
        slog(feature.math_ops.select_op),
        TVM_FEATURE_ADD_ANN_ITER(feature.vectorize),
        TVM_FEATURE_ADD_ANN_ITER(feature.unroll),
        TVM_FEATURE_ADD_ANN_ITER(feature.parallel),
        slog(feature.blockIdx_x_len),
        slog(feature.blockIdx_y_len),
        slog(feature.blockIdx_z_len),
        slog(feature.threadIdx_x_len),
        slog(feature.threadIdx_y_len),
        slog(feature.threadIdx_z_len),
        slog(feature.vthread_len),
    };
    CHECK_EQ(result.size(), kNumFeatureGroup1);
    result.reserve(kNumFeatureGroup1 + kNumFeatureGroup2 + kNumFeatureGroup3 + kNumFeatureGroup5);
    /***** Group 2: Buffer access related features *****/
    const std::vector<FeatureSet::BufferAccess>& accesses = feature.buffer_accesses;
    int n_accesses = accesses.size();
    // Sort the buffers in descending order of (line, bytes)
    std::vector<int> order(n_accesses);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&accesses](int l, int r) -> bool {
      if (accesses[l].lines != accesses[r].lines) {
        return accesses[l].lines > accesses[r].lines;
      }
      if (accesses[l].bytes != accesses[r].bytes) {
        return accesses[l].bytes > accesses[r].bytes;
      }
      return accesses[l].buffer_name < accesses[r].buffer_name;
    });
    // Make sure at most `max_num_buffer_access_features` are used
    if (n_accesses > max_num_buffer_access_features) {
      order.resize(max_num_buffer_access_features);
    }
    for (int idx : order) {
      const FeatureSet::BufferAccess& access = accesses[idx];
      std::vector<double> group2_sub{
          static_cast<double>(static_cast<int>(access.access_type) == 0),
          static_cast<double>(static_cast<int>(access.access_type) == 1),
          static_cast<double>(static_cast<int>(access.access_type) == 2),
          // FeatureSet::BufferAccess::AccessType::kUnknownRW is ignored
          slog(access.bytes),
          slog(access.unique_bytes),
          slog(access.lines),
          slog(access.unique_lines),
          static_cast<double>(static_cast<int>(access.reuse_type) == 0),
          static_cast<double>(static_cast<int>(access.reuse_type) == 1),
          static_cast<double>(static_cast<int>(access.reuse_type) == 2),
          slog(access.reuse_dis_iter),
          slog(access.reuse_dis_bytes),
          slog(access.reuse_ct),
          slog(access.bytes_d_reuse_ct),
          slog(access.unique_bytes_d_reuse_ct),
          slog(access.lines_d_reuse_ct),
          slog(access.unique_lines_d_reuse_ct),
          slog(access.stride),
      };
      CHECK_EQ(group2_sub.size(), kNumFeatureGroup2Subgroup);
    }
    // Pad to `max_num_buffer_access_features`
    if (max_num_buffer_access_features > n_accesses) {
      int n_pad = (max_num_buffer_access_features - n_accesses) * kNumFeatureGroup2Subgroup;
      result.resize(result.size() + n_pad, 0.0);
    }
    CHECK_EQ(result.size(), kNumFeatureGroup1 + kNumFeatureGroup2);
    /***** Group 3: Arithmetic intensity related features *****/
    result.insert(result.end(),  //
                  std::begin(feature.arith_intensity_curve),
                  std::end(feature.arith_intensity_curve));
    CHECK_EQ(result.size(), kNumFeatureGroup1 + kNumFeatureGroup2 + kNumFeatureGroup3);
    /***** Group 5: Outer scope related features *****/
    result.push_back(slog(feature.outer_prod));
    result.push_back(slog(feature.num_loops));
    result.push_back(slog(feature.auto_unroll_max_step));
    CHECK_EQ(result.size(),
             kNumFeatureGroup1 + kNumFeatureGroup2 + kNumFeatureGroup3 + kNumFeatureGroup5);
    // Then append it the the feature vector
    feature_vector.emplace_back(std::move(result));
  }
  return feature_vector;
}

#undef TVM_FEATURE_ADD_ANN_ITER

Array<String> PerBlockFeatureNames(const tir::PrimFunc& func, int max_num_buffer_access_features) {
  constexpr size_t kNumFeatureGroup1 = 8 * 2 + 11 * 3 + 7;
  constexpr size_t kNumFeatureGroup2Subgroup = 18;
  constexpr size_t kNumFeatureGroup3 = FeatureSet::NUM_SAMPLE_ARITH_INTENSITY_CURVE;
  constexpr size_t kNumFeatureGroup5 = 3;
  size_t kNumFeatureGroup2 = kNumFeatureGroup2Subgroup * max_num_buffer_access_features;
  size_t kTotal = kNumFeatureGroup1 + kNumFeatureGroup2 + kNumFeatureGroup3 + kNumFeatureGroup5;
  std::vector<String> result{
      "float_mad",
      "float_addsub",
      "float_mul",
      "float_divmod",
      "float_cmp",
      "float_mathfunc",
      "float_otherfunc",
      "int_mad",
      "int_addsub",
      "int_mul",
      "int_divmod",
      "int_cmp",
      "int_mathfunc",
      "int_otherfunc",
      "bool_op",
      "select_op",
      "vec_num",
      "vec_prod",
      "vec_len",
      "vec_type.kPosNone",
      "vec_type.kPosInnerSpatial",
      "vec_type.kPosMiddleSpatial",
      "vec_type.kPosOuterSpatial",
      "vec_type.kPosInnerReduce",
      "vec_type.kPosMiddleReduce",
      "vec_type.kPosOuterReduce",
      "vec_type.kPosMixed",
      "unroll_num",
      "unroll_prod",
      "unroll_len",
      "unroll_type.kPosNone",
      "unroll_type.kPosInnerSpatial",
      "unroll_type.kPosMiddleSpatial",
      "unroll_type.kPosOuterSpatial",
      "unroll_type.kPosInnerReduce",
      "unroll_type.kPosMiddleReduce",
      "unroll_type.kPosOuterReduce",
      "unroll_type.kPosMixed",
      "parallel_num",
      "parallel_prod",
      "parallel_len",
      "parallel_type.kPosNone",
      "parallel_type.kPosInnerSpatial",
      "parallel_type.kPosMiddleSpatial",
      "parallel_type.kPosOuterSpatial",
      "parallel_type.kPosInnerReduce",
      "parallel_type.kPosMiddleReduce",
      "parallel_type.kPosOuterReduce",
      "parallel_type.kPosMixed",
      "blockIdx_x_len",
      "blockIdx_y_len",
      "blockIdx_z_len",
      "threadIdx_x_len",
      "threadIdx_y_len",
      "threadIdx_z_len",
      "vthread_len",
  };
  // section total: 57
  for (int i = 0; i < max_num_buffer_access_features; ++i) {
    String prefix = "B" + std::to_string(i) + ".";
    std::vector<String> group2_sub{
        prefix + "acc_type.kRead",
        prefix + "acc_type.kWrite",
        prefix + "acc_type.kReadWrite",
        prefix + "bytes",
        prefix + "unique_bytes",
        prefix + "lines",
        prefix + "unique_lines",
        prefix + "reuse_type.kLoopMultipleRead",
        prefix + "reuse_type.kSerialMultipleReadWrite",
        prefix + "reuse_type.kNoReuse",
        prefix + "reuse_dis_iter",
        prefix + "reuse_dis_bytes",
        prefix + "reuse_ct",
        prefix + "bytes_d_reuse_ct",
        prefix + "unique_bytes_d_reuse_ct",
        prefix + "lines_d_reuse_ct",
        prefix + "unique_lines_d_reuse_ct",
        prefix + "stride",
    };
    result.insert(result.end(), group2_sub.begin(), group2_sub.end());
  }
  // section total : max_num_buffer_access_features * 18
  for (int i = 0; i < FeatureSet::NUM_SAMPLE_ARITH_INTENSITY_CURVE; ++i) {
    result.push_back("arith_intensity_curve_" + std::to_string(i));
  }
  // section total: NUM_SAMPLE_ARITH_INTENSITY_CURVE
  result.push_back(("outer_prod"));
  result.push_back(("num_loops"));
  result.push_back(("auto_unroll_max_step"));
  // section total: 3
  CHECK_EQ(result.size(), kTotal);
  return {result.begin(), result.end()};
}

}  // namespace meta_schedule
}  // namespace tvm
