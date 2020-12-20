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

#include "../utils.h"

namespace tvm {
namespace meta_schedule {

template <class T>
using BufferMap = std::unordered_map<tir::Buffer, T, ObjectHash, ObjectEqual>;

using MultiDimIdx = Array<PrimExpr>;

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
      kUnknownRW = 3   //
    };
    // Data reuse type
    enum class ReuseType : int {
      kLoopMultipleRead = 0,
      kSerialMultipleReadWrite = 1,
      kNoReuse = 2
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
    TVM_META_SCHEDULE_FEATURE_EXTRACTION_INC_CNT(op->dtype,    /**/                 \
                                                 FloatCounter, /**/                 \
                                                 IntCounter);                       \
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

class BufferAccessExtractor : public tir::StmtExprVisitor {
 public:
  struct Access {
    FeatureSet::BufferAccessFeature::AccessType type =
        FeatureSet::BufferAccessFeature::AccessType::kUnknownRW;
    std::vector<MultiDimIdx> indices;
  };

  static BufferMap<Access> Extract(const tir::BufferStore& store) {
    BufferAccessExtractor extractor;
    Access& access = extractor.accesses[store->buffer];
    access.type = FeatureSet::BufferAccessFeature::AccessType::kWrite;
    access.indices.push_back(store->indices);
    extractor.VisitStmt(store);
    return extractor.accesses;
  }

  void VisitExpr_(const tir::BufferLoadNode* load) final {
    const tir::Buffer& buffer = load->buffer;
    Access& access = accesses[buffer];
    switch (access.type) {
      case FeatureSet::BufferAccessFeature::AccessType::kRead:
        // do nothing
        break;
      case FeatureSet::BufferAccessFeature::AccessType::kWrite:
        // from write to read-write
        access.type = FeatureSet::BufferAccessFeature::AccessType::kReadWrite;
        break;
      case FeatureSet::BufferAccessFeature::AccessType::kReadWrite:
        // do nothing
        break;
      case FeatureSet::BufferAccessFeature::AccessType::kUnknownRW:
        // from unknown to read
        access.type = FeatureSet::BufferAccessFeature::AccessType::kRead;
        break;
      default:
        LOG(FATAL) << "ValueError: Cannot recognize BufferAccessFeature::AccessType: "
                   << static_cast<int>(access.type);
    }
    if (access.type != FeatureSet::BufferAccessFeature::AccessType::kReadWrite) {
      // If a buffer is both read and written, in the tvm DSL, it must be a update,
      // so the indices should be the same. Then we can skip appending indices for it.
      // Otherwise we do the following.
      access.indices.push_back(load->indices);
    }
    StmtExprVisitor::VisitExpr_(load);
  }

  BufferMap<Access> accesses;
};

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
    FeatureSet& feature = buffer_features[store->buffer];
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
