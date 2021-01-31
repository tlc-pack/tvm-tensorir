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

template <class K, class V>
using ObjMap = std::unordered_map<const K*, V>;

template <class K1, class K2, class V>
using ObjPairMap = ObjMap<K1, ObjMap<K2, V>>;

using NDIntSet = Array<arith::IntSet>;

std::ostream& operator<<(std::ostream& os, const NDIntSet& nd_int_set) {
  os << '[';
  bool is_first = true;
  for (const arith::IntSet& int_set : nd_int_set) {
    if (is_first) {
      is_first = false;
    } else {
      os << ", ";
    }
    PrimExpr min = int_set.min();
    PrimExpr max = int_set.max();
    os << min << ":" << max;
  }
  os << ']';
  return os;
}

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
  const tir::BlockRealizeNode* block_realize;
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
    ++result_->FloatCounter;                                 \
  } else {                                                   \
    ++result_->IntCounter;                                   \
  }

#define TVM_FEATURE_SIMPLE(Type, Counter) \
  void VisitExpr_(const Type* op) final { \
    ++result_->Counter;                   \
    StmtExprVisitor::VisitExpr_(op);      \
  }

#define TVM_FEATURE_BINARY(Type, FloatCounter, IntCounter) \
  void VisitExpr_(const Type* op) final {                  \
    if (op->dtype.is_float()) {                            \
      ++result_->FloatCounter;                             \
    } else {                                               \
      ++result_->IntCounter;                               \
    }                                                      \
    StmtExprVisitor::VisitExpr_(op);                       \
  }

class MathOpCounter : public tir::StmtExprVisitor {
 public:
  static FeatureSet::MathOps Count(const PrimExpr& expr) {
    FeatureSet::MathOps math_ops;
    MathOpCounter counter(&math_ops);
    counter(expr);
    return math_ops;
  }

  static void Count(const PrimExpr& expr, FeatureSet::MathOps* math_ops) {
    MathOpCounter counter(math_ops);
    counter(expr);
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

  explicit MathOpCounter(FeatureSet::MathOps* result) : result_(result) {}

  FeatureSet::MathOps* result_;
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
    return (extractor.visited_var && !extractor.visited_mul && !extractor.visited_add)
               ? 1
               : (extractor.visited_var ? extractor.stride : 0);
  }

  const tir::Var& var;
  int64_t stride;
  bool visited_var;
  bool visited_add;
  bool visited_mul;
};

class PerBlockFeatureExtractor : public tir::StmtExprVisitor {
 public:
  static std::vector<FeatureSet> Extract(const tir::PrimFunc& func) {
    PerBlockFeatureExtractor extractor;
    extractor.VisitStmt(func->body);
    std::vector<FeatureSet> result;
    result.reserve(extractor.ordered_blocks_.size());
    for (const tir::BlockRealizeNode* realize : extractor.ordered_blocks_) {
      result.push_back(extractor.per_block_feature_.at(realize));
    }
    return result;
  }

 private:
  /******** Group 1: Computation related features ********/

  void CalcComputeFeature(const tir::BlockRealizeNode* realize, FeatureSet* feature) {
    if (!scopes_.empty()) {
      AddMathOpsToScope(&feature->math_ops);
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

  void AddMathOpsToScope(FeatureSet::MathOps* math_ops) {
    const tir::BlockRealizeNode* scope = scopes_.back();
    // The product of the loops up to the parent
    int64_t prod_loop_extent = 1;
    for (auto iter = dfs_path_.rbegin(); iter != dfs_path_.rend(); ++iter) {
      const tir::StmtNode* stmt = *iter;
      if (stmt == scope) {
        break;
      }
      CHECK(stmt->IsInstance<tir::LoopNode>());
      prod_loop_extent *=
          GetLoopIntExtent(static_cast<const tir::LoopNode*>(stmt)).value_or(1)->value;
    }
    // Add the math_ops to the parent
    FeatureSet::MathOps& parent_math_ops = per_block_feature_[scope].math_ops;
#define TVM_FEATURE_MATH_OP_ADD(Name)                       \
  parent_math_ops.Name = math_ops->Name * prod_loop_extent; \
  math_ops->Name *= outer_loop_prod_
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
      ann->len = GetLoopIntExtent(loops.back()).value_or(1)->value;
      ann->prod = ProdLoopExtent(loops);
      ann->pos = FeatureSet::AnnIter::Pos::kPosMixed;
    }
  }

 private:
  /******** Group 2: Buffer access related features ********/
  struct BufferInfo {
    FeatureSet::BufferAccess::AccessType access_type =
        FeatureSet::BufferAccess::AccessType::kUnknownRW;
    /*! \brief The regions that the buffer is accessed */
    Array<NDIntSet> regions = {};
    std::vector<int64_t> access_shape;
    int64_t num_continuous_bytes = 1;
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
  using BufferInfoMap = ObjMap<tir::BufferNode, BufferInfo>;

  BufferInfoMap CalcBufferInfo(const tir::BlockRealizeNode* realize,
                               const std::vector<const tir::LoopNode*>& loops,
                               std::vector<int64_t>* for_touched_bytes_) {
    // Initialize the data structures used for the features
    int n_loops = loops.size();
    std::vector<int64_t>& for_touched_bytes = *for_touched_bytes_ =
        std::vector<int64_t>(n_loops, int64_t(0));
    BufferInfoMap buffer_info = GatherBufferAccessRegion(realize);
    for (auto& it : buffer_info) {
      BufferInfo& info = it.second;
      info.loop_accessed_numel.resize(n_loops);
    }
    // Part 1. Area-related features
    // Step 1.1. we bind all the loop variables to a constant
    for (int i = 0; i < n_loops; ++i) {
      const tir::LoopNode* loop = loops[i];
      analyzer_.Bind(loop->loop_var, loop->min, /*allow_override=*/true);
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
        // Note: `info.access_shape` for `i == n_loops - 1` is the only one preserved,
        // while others are discarded
        int64_t numel = CalcRegionUnionSize(info.regions, &info.access_shape);
        info.loop_accessed_numel[i].push_back(numel);
        touched_bytes += numel * buffer->dtype.bytes();
        buffer_touched_under_loop_[loop][buffer].push_back(numel);
      }
    }
    // Part 2. Stride-related features
    // For each buffer, we find the loop stride on it
    for (auto& it : buffer_info) {
      const tir::BufferNode* buffer = it.first;
      BufferInfo& info = it.second;
      int ndim = buffer->shape.size();
      std::vector<int64_t> buffer_shape = AsVector<PrimExpr, int64_t>(buffer->shape);
      // Calculate the buffer's stride from its shape
      std::vector<int64_t> buffer_stride(ndim);
      if (ndim >= 1) {
        buffer_stride[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
          buffer_stride[i] = buffer_stride[i + 1] * buffer_shape[i + 1];
        }
      }
      // Calculate `num_continuous_bytes`
      {
        int64_t& num_continuous_bytes = info.num_continuous_bytes = 1;
        const std::vector<int64_t>& access_shape = info.access_shape;
        CHECK_EQ(access_shape.size(), buffer_shape.size());
        for (int i = ndim - 1; i >= 0; --i) {
          if (access_shape[i] == buffer_shape[i]) {
            num_continuous_bytes = buffer_shape[i] * buffer->dtype.bytes();
            break;
          }
        }
      }
      // Enumerate loops from inner to outer
      int i = 0;
      // Calculate info.min_stride
      int64_t& stride = info.min_stride = 0;
      for (; i < n_loops; ++i) {
        stride = CalcVarStrideOnRegion(info.regions, buffer_stride, loops[i]->loop_var);
        if (stride != 0) {
          break;
        }
      }
      // Calculate info.innermost_stride
      info.innermost_stride = (i == 0) ? stride : 0;
      // Calculate info.prod
      int64_t& prod = info.prod_non_strided_loop_extent = 1;
      for (int j = 0; j < i; ++j) {
        prod *= GetLoopIntExtent(loops[j]).value_or(1)->value;
      }
    }
    // Part 3. Reuse-related features
    for (auto& it : buffer_info) {
      const tir::BufferNode* buffer = it.first;
      BufferInfo& info = it.second;
      // Default case: no reuse
      FeatureSet::BufferAccess::ReuseType& reuse_type = info.reuse_type =
          FeatureSet::BufferAccess::ReuseType::kNoReuse;
      double& reuse_dis_iter = info.reuse_dis_iter = 0;
      double& reuse_dis_bytes = info.reuse_dis_bytes = 0;
      int64_t& reuse_ct = info.reuse_ct = 0;
      // Step 3.1. Collect all `tir::Var`s that appears in the buffer region
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
      // Step 3.2. Enumerate loops from inner to outer, find the first loop with reuse
      for (int i = 0; i < n_loops; ++i) {
        const tir::LoopNode* loop = loops[i];
        // Case 1. Find an invariant loop, i.e. reuse with kLoopMultipleRead
        if (!region_vars.count(loop->loop_var.get())) {
          reuse_type = FeatureSet::BufferAccess::ReuseType::kLoopMultipleRead;
          reuse_ct = GetLoopIntExtent(loop).value_or(1)->value;
          if (i == 0) {
            reuse_dis_iter = 1;
            reuse_dis_bytes = 0.0;
            for (const auto& it : buffer_info) {
              const tir::BufferNode* buffer = it.first;
              const BufferInfo& info = it.second;
              int64_t bytes = buffer->dtype.bytes();
              int64_t n_buffer = info.loop_accessed_numel[i].size();
              reuse_dis_bytes += bytes * n_buffer;
            }
          } else {
            reuse_dis_iter = 1;
            for (int j = 0; j < i; ++j) {
              reuse_dis_iter *= GetLoopIntExtent(loops[j]).value_or(1)->value;
            }
            reuse_dis_bytes = 0.0;
            for (const auto& iter : buffer_touched_under_loop_[loops[i - 1]]) {
              const tir::BufferNode* buffer = iter.first;
              const std::vector<int64_t>& numels = iter.second;
              int64_t numel = std::accumulate(numels.begin(), numels.end(), int64_t(0));
              reuse_dis_bytes += numel * buffer->dtype.bytes();
            }
          }
          break;
        }
        // Case 2. Find serial reuse, i.e. reuse with kSerialMultipleReadWrite
        const std::vector<int64_t>& touched = buffer_touched_under_loop_[loop][buffer];
        if (touched.size() >= 2) {
          int64_t extent = GetLoopIntExtent(loop).value_or(1)->value;
          reuse_type = FeatureSet::BufferAccess::ReuseType::kSerialMultipleReadWrite;
          reuse_ct = touched.size() - 1;
          reuse_dis_iter = *std::min_element(touched.begin(), touched.end());
          reuse_dis_bytes = 0.0;
          for (const auto& iter : buffer_touched_under_loop_[loop]) {
            const tir::BufferNode* buffer = iter.first;
            const std::vector<int64_t>& numels = iter.second;
            int64_t numel = std::accumulate(numels.begin(), numels.end(), int64_t(0));
            reuse_dis_bytes += numel * buffer->dtype.bytes();
          }
          reuse_dis_iter /= extent;
          reuse_dis_bytes /= extent;
          break;
        }
      }
    }
    return buffer_info;
  }

  void CalcBufferAccessFeature(const tir::BlockRealizeNode* realize, FeatureSet* feature_,
                               const std::vector<const tir::LoopNode*>& loops,
                               const BufferInfoMap& buffer_info) const {
    constexpr int64_t kCacheLineBytes = 64;
    std::vector<FeatureSet::BufferAccess>& buffer_features = feature_->buffer_accesses;
    buffer_features.reserve(buffer_info.size());
    for (const auto& iter : buffer_info) {
      const tir::BufferNode* buffer = iter.first;
      const BufferInfo& info = iter.second;
      int64_t dtype_bytes = buffer->dtype.bytes();
      buffer_features.emplace_back();
      FeatureSet::BufferAccess& feature = buffer_features.back();
      feature.buffer_name = buffer->name;
      feature.access_type = info.access_type;
      feature.stride = info.innermost_stride;
      feature.bytes = dtype_bytes * outer_loop_prod_;
      if (loops.empty()) {
        feature.unique_bytes = 1;
        feature.lines = 1;
        feature.unique_lines = 1;
      } else {
        feature.unique_bytes = info.loop_accessed_numel.back().front() * dtype_bytes;
        double m = static_cast<double>(info.min_stride) * dtype_bytes / kCacheLineBytes;
        feature.lines = outer_loop_prod_ / info.prod_non_strided_loop_extent * std::min(1.0, m);
        feature.lines = std::max(1.0, feature.lines);
        feature.unique_lines = static_cast<double>(feature.unique_bytes) /
                               std::min(kCacheLineBytes, info.num_continuous_bytes);
        feature.unique_lines = std::max(1.0, feature.unique_lines);
      }
      feature.reuse_type = info.reuse_type;
      feature.reuse_dis_iter = info.reuse_dis_iter;
      feature.reuse_dis_bytes = info.reuse_dis_bytes;
      feature.reuse_ct = info.reuse_ct;
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

  BufferInfoMap GatherBufferAccessRegion(const tir::BlockRealizeNode* realize) const {
    arith::Analyzer analyzer;
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
            if (!analyzer.CanProve(r_region[i]->min == w_region[i]->min) ||
                !analyzer.CanProve(r_region[i]->extent == w_region[i]->extent)) {
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
    auto f_substitute = [&analyzer, &var_substitutes](const PrimExpr& expr) -> PrimExpr {
      return analyzer.Simplify(
          tir::Substitute(expr, [&var_substitutes](const PrimExpr& expr) -> Optional<PrimExpr> {
            if (const auto* var = expr.as<tir::VarNode>()) {
              auto it = var_substitutes.find(var);
              if (it != var_substitutes.end()) {
                return it->second;
              }
            }
            return NullOpt;
          }));
    };
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
        var_substitutes[lhs.get()] = f_substitute(rhs);
      }
    }
    // Step 2.3. Helper to convert a TIR region into our int-set and do necessary simplification
    auto f_make_int_set = [&f_substitute](const Array<Range>& region) -> NDIntSet {
      // Helper function to do the substitution
      int ndim = region.size();
      NDIntSet result;
      result.reserve(ndim);
      for (int i = 0; i < ndim; ++i) {
        const Range& range = region[i];
        PrimExpr min = f_substitute(range->min);
        PrimExpr max = f_substitute(min + range->extent - Integer(1));
        result.push_back(arith::IntSet::Interval(min, max));
      }
      return result;
    };
    // Step 3. Apply the substitution to each tensor region
    BufferInfoMap result;
    result.reserve(realize->block->reads.size() + realize->block->writes.size());
    for (int i = 0; i < n_reads; ++i) {
      // Skip those update regions
      if (is_read_update[i]) {
        continue;
      }
      const tir::TensorRegion& region = realize->block->reads[i];
      BufferInfo& info = result[region->buffer.get()];
      info.access_type = FeatureSet::BufferAccess::AccessType::kRead;
      info.regions.push_back(f_make_int_set(region->region));
    }
    for (int i = 0; i < n_writes; ++i) {
      const tir::TensorRegion& region = realize->block->writes[i];
      BufferInfo& info = result[region->buffer.get()];
      if (is_write_update[i] || info.access_type == FeatureSet::BufferAccess::AccessType::kRead) {
        info.access_type = FeatureSet::BufferAccess::AccessType::kReadWrite;
      } else {
        info.access_type = FeatureSet::BufferAccess::AccessType::kWrite;
      }
      info.regions.push_back(f_make_int_set(region->region));
    }
    return result;
  }

  int64_t CalcRegionUnionSize(const Array<NDIntSet>& regions,
                              std::vector<int64_t>* access_shape) const {
    if (regions.empty()) {
      return 1;
    }
    access_shape->clear();
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
      int64_t min = analyzer_.const_int_bound(union_set.min())->min_value;
      int64_t max = analyzer_.const_int_bound(union_set.max())->max_value;
      if (arith::ConstIntBound::kNegInf < min && max < arith::ConstIntBound::kPosInf) {
        numel *= max - min + 1;
        access_shape->push_back(max - min + 1);
      } else {
        access_shape->push_back(1);
      }
    }
    return numel;
  }

  static int64_t CalcVarStrideOnRegion(const Array<NDIntSet>& regions,
                                       const std::vector<int64_t>& buffer_stride,
                                       const tir::Var& var) {
    constexpr int64_t kNotFound = std::numeric_limits<int64_t>::max();
    int ndim = buffer_stride.size();
    // Calculate the min stride possible
    int64_t result = kNotFound;
    for (const NDIntSet& region : regions) {
      CHECK_EQ(region.size(), buffer_stride.size());
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
    double total_compute_ops =
        static_cast<double>(math_ops.float_mad + math_ops.float_addsub + math_ops.float_mul +
                            math_ops.float_divmod + math_ops.float_cmp + math_ops.float_math_func +
                            math_ops.float_other_func) /
        outer_loop_prod_;
    for (int i = 0; i < n_loops; ++i) {
      int64_t extent = GetLoopIntExtent(loops[i]).value_or(1)->value;
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
    total_compute_ops = compute_ops.back();  // i.e. total_compute_ops = log2(total_compute_ops)
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
    if (!scopes_.empty()) {
      ordered_blocks_.push_back(realize);
    }
    scopes_.push_back(realize);
    dfs_path_.push_back(realize);
    tir::StmtExprVisitor::VisitStmt_(realize);
    dfs_path_.pop_back();
    scopes_.pop_back();
    if (scopes_.empty()) {
      return;
    }
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
    FeatureSet& feature = per_block_feature_[realize];
    feature.block_realize = realize;
    // Group 1: Computation related features
    CalcComputeFeature(realize, &feature);
    // Group 2: Buffer access related features
    std::vector<int64_t> for_touched_bytes;
    BufferInfoMap buffer_info = CalcBufferInfo(realize, loops, &for_touched_bytes);
    CalcBufferAccessFeature(realize, &feature, loops, buffer_info);
    // Group 3: Arithmetic intensity related features
    CalcAritheticIntensityFeature(realize, &feature, loops, for_touched_bytes, feature.math_ops);
    // Group 5: Outer scope related features
    CalcOuterScopeFeature(realize, &feature);
  }

  void VisitStmt_(const tir::LoopNode* loop) override {
    int64_t extent = GetLoopIntExtent(loop).value_or(1)->value;
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
    if (extent != 1 || ref_loops != nullptr) {
      dfs_path_.push_back(loop);
      loops_.push_back(loop);
      analyzer_.Bind(loop->loop_var, loop->min);
    }
    tir::StmtExprVisitor::VisitStmt_(loop);
    if (extent != 1 || ref_loops != nullptr) {
      loops_.pop_back();
      dfs_path_.pop_back();
    }
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
    FeatureSet::MathOps math_ops = MathOpCounter::Count(store->value);
    AddMathOpsToScope(&math_ops);
  }

  void VisitStmt_(const tir::ReduceStepNode* reduce) override {
    CHECK(!scopes_.empty());
    FeatureSet::MathOps math_ops = MathOpCounter::Count(reduce->rhs);
    for (const PrimExpr& expr : reduce->comm_reducer->result) {
      MathOpCounter::Count(expr, &math_ops);
    }
    AddMathOpsToScope(&math_ops);
  }

 private:
  static int64_t ProdLoopExtent(const std::vector<const tir::LoopNode*>& loops) {
    int64_t prod = 1;
    for (const tir::LoopNode* loop : loops) {
      prod *= GetLoopIntExtent(loop).value_or(1)->value;
    }
    return prod;
  }

  static int64_t FirstLoopExtent(const std::vector<const tir::LoopNode*>& loops) {
    return loops.empty() ? 1 : GetLoopIntExtent(loops[0]).value_or(1)->value;
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
  /*!
   * \brief For a specific buffer, record the regions it is acccessed under a specific loop.
   * The information is preserved across different blocks and is used for detecting serial buffer
   * reuse
   */
  ObjPairMap<tir::LoopNode, tir::BufferNode, std::vector<int64_t>> buffer_touched_under_loop_;
  /*! \brief The output: features for each BlockRealizeNode */
  ObjMap<tir::BlockRealizeNode, FeatureSet> per_block_feature_;
  /*! \brief The pre-order visit order of all the BlockRealizeNodes */
  std::vector<const tir::BlockRealizeNode*> ordered_blocks_;
};

// shifted log to incorporate the property that slog(0) = 0
inline double slog(double x) {
  if (x < 0) {
    x = -x;
  }
  return std::log2(x + 1);
}

#define TVM_FEATURE_ADD_ANN_ITER(s)                      \
  slog(s.num), slog(s.prod), slog(s.len), /**/           \
      static_cast<double>(static_cast<int>(s.pos) == 0), \
      static_cast<double>(static_cast<int>(s.pos) == 1), \
      static_cast<double>(static_cast<int>(s.pos) == 2), \
      static_cast<double>(static_cast<int>(s.pos) == 3), \
      static_cast<double>(static_cast<int>(s.pos) == 4), \
      static_cast<double>(static_cast<int>(s.pos) == 5), \
      static_cast<double>(static_cast<int>(s.pos) == 6), \
      static_cast<double>(static_cast<int>(s.pos) == 7)

void PerBlockFeature(const Schedule& sch, int max_num_buffer_access_features,
                     PrimFuncFeature* prim_func_feature) {
  constexpr size_t kNumFeatureGroup1 = 8 * 2 + 11 * 3 + 7;
  constexpr size_t kNumFeatureGroup2Subgroup = 18;
  constexpr size_t kNumFeatureGroup3 = FeatureSet::NUM_SAMPLE_ARITH_INTENSITY_CURVE;
  constexpr size_t kNumFeatureGroup5 = 3;
  size_t kNumFeature = kNumFeatureGroup1 +
                       kNumFeatureGroup2Subgroup * max_num_buffer_access_features +
                       kNumFeatureGroup3 + kNumFeatureGroup5;
  const tir::PrimFunc& func = sch->sch->func;
  std::vector<FeatureSet> feature_map = PerBlockFeatureExtractor::Extract(func);

  std::vector<double>& ret = prim_func_feature->feature;
  // Set up the shape of the returned feature
  {
    int64_t shape[] = {static_cast<int64_t>(feature_map.size()), static_cast<int64_t>(kNumFeature)};
    ret.clear();
    ret.reserve(shape[0] * shape[1]);
    prim_func_feature->shape = std::vector<int64_t>(std::begin(shape), std::end(shape));
  }

  for (const FeatureSet& feature : feature_map) {
    /***** Group 1: Computation related features *****/
    double group1[] = {
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
    CHECK_EQ(std::end(group1) - std::begin(group1), kNumFeatureGroup1);
    ret.insert(ret.end(), std::begin(group1), std::end(group1));
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
      double group2_sub[] = {
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
      CHECK_EQ(std::end(group2_sub) - std::begin(group2_sub), kNumFeatureGroup2Subgroup);
      ret.insert(ret.end(), std::begin(group2_sub), std::end(group2_sub));
    }
    // Pad to `max_num_buffer_access_features`
    if (max_num_buffer_access_features > n_accesses) {
      int n_pad = (max_num_buffer_access_features - n_accesses) * kNumFeatureGroup2Subgroup;
      ret.insert(ret.end(), n_pad, 0.0);
    }
    /***** Group 3: Arithmetic intensity related features *****/
    ret.insert(ret.end(),                                  //
               std::begin(feature.arith_intensity_curve),  //
               std::end(feature.arith_intensity_curve));
    /***** Group 5: Outer scope related features *****/
    double group5[] = {
        slog(feature.outer_prod),
        slog(feature.num_loops),
        slog(feature.auto_unroll_max_step),
    };
    ret.insert(ret.end(), std::begin(group5), std::end(group5));
  }
  // Finally check the shape of the return
  CHECK_EQ(ret.size(), prim_func_feature->shape[0] * prim_func_feature->shape[1]);
}

#undef TVM_FEATURE_ADD_ANN_ITER

Array<String> PerBlockFeatureNames(int max_num_buffer_access_features) {
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

PrimFuncFeature PerBlockFeature(const Schedule& sch, int max_num_buffer_access_features) {
  PrimFuncFeature result;
  PerBlockFeature(sch, max_num_buffer_access_features, &result);
  return result;
}

struct Internal {
  static runtime::NDArray PerBlockFeature(const Schedule& sch, int max_num_buffer_access_features) {
    static thread_local PrimFuncFeature* result =
        new PrimFuncFeature();  // persists till the program offloading
    tvm::meta_schedule::PerBlockFeature(sch, max_num_buffer_access_features, result);
    return result->AsNDArray();
  }
};

TVM_REGISTER_GLOBAL("meta_schedule.PerBlockFeature").set_body_typed(Internal::PerBlockFeature);
TVM_REGISTER_GLOBAL("meta_schedule.PerBlockFeatureNames").set_body_typed(PerBlockFeatureNames);

}  // namespace meta_schedule
}  // namespace tvm
