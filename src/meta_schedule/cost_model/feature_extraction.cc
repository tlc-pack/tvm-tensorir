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

}  // namespace meta_schedule
}  // namespace tvm
