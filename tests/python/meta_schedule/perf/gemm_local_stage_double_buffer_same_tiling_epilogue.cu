#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#else

typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__
#define TVM_ALIGNED(x) __attribute__ ((aligned(x)))
#define TVM_HALF_OPERATOR(RTYPE, OP)                              \
  TVM_XINLINE RTYPE operator OP (half a, half b) {                \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (half a, T b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (T a, half b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }

#define TVM_HALF_ASSIGNOP(AOP, OP)                                \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const T& a) {                    \
    return *this = half(float(*this) OP float(a));                \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const volatile T& a) volatile {  \
    return *this = half(float(*this) OP float(a));                \
  }

class TVM_ALIGNED(2) half {
 public:
  uint16_t half_;

  static TVM_XINLINE half Binary(uint16_t value) {
    half res;
    res.half_ = value;
    return res;
  }

  TVM_XINLINE half() {}

  TVM_XINLINE half(const float& value) { constructor(value); }
  TVM_XINLINE explicit half(const double& value) { constructor(value); }
  TVM_XINLINE explicit half(const int8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const int32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const long long& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint64_t& value) { constructor(value); }

  TVM_XINLINE operator float() const {                          \
    return float(half2float(half_));                            \
  }                                                             \
  TVM_XINLINE operator float() const volatile {                 \
    return float(half2float(half_));                            \
  }


  TVM_HALF_ASSIGNOP(+=, +)
  TVM_HALF_ASSIGNOP(-=, -)
  TVM_HALF_ASSIGNOP(*=, *)
  TVM_HALF_ASSIGNOP(/=, /)

  TVM_XINLINE half operator+() {
    return *this;
  }

  TVM_XINLINE half operator-() {
    return half(-float(*this));
  }

  TVM_XINLINE half operator=(const half& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) {
    return *this = half(a);
  }

  TVM_XINLINE half operator=(const half& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) volatile {
    return *this = half(a);
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);   // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;   // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;   // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15;   // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;   // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;   // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;   // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;   // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  TVM_XINLINE uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  TVM_XINLINE uint16_t float2half(
    const volatile float& value) const volatile {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  TVM_XINLINE float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  TVM_XINLINE float half2float(
    const volatile uint16_t& value) const volatile {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  TVM_XINLINE void constructor(const T& value) {
    half_ = float2half(float(value));
  }
};

TVM_HALF_OPERATOR(half, +)
TVM_HALF_OPERATOR(half, -)
TVM_HALF_OPERATOR(half, *)
TVM_HALF_OPERATOR(half, /)
TVM_HALF_OPERATOR(bool, >)
TVM_HALF_OPERATOR(bool, <)
TVM_HALF_OPERATOR(bool, >=)
TVM_HALF_OPERATOR(bool, <=)

TVM_XINLINE half __float2half_rn(const float a) {
  return half(a);
}
#endif


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// fix undefined fp16 match function
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
static inline __device__ __host__ half hpow(half x, half y) {
  float tmp_x = __half2float(x);
  float tmp_y = __half2float(y);
  float result = powf(tmp_x, tmp_y);
  return __float2half(result);
}

static inline __device__ __host__ half htanh(half x) {
  float tmp_x = __half2float(x);
  float result = tanhf(tmp_x);
  return __float2half(result);
}
#endif
#include <mma.h>

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(256) main_kernel0(half* __restrict__ A, half* __restrict__ B, float* __restrict__ C) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> C_wmma_accumulator[8];
  uint4 A_shared_local[2];
  uint4 B_shared_local[2];
  __shared__ uchar shared_mem[(10240 + 8704) * 2];
  // __shared__ half A_shared[10240];
  // __shared__ half B_shared[8704];
  half* A_shared = (half*)shared_mem;
  half* B_shared = ((half*)shared_mem) + 10240;
  float* C_shared = (float*)shared_mem;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> B_shared_wmma_matrix_b[8];
  for (int i0_0_4_init = 0; i0_0_4_init < 2; ++i0_0_4_init) {
    for (int i1_0_4_init = 0; i1_0_4_init < 4; ++i1_0_4_init) {
      (void)nvcuda::wmma::fill_fragment(C_wmma_accumulator[((i0_0_4_init * 4) + i1_0_4_init)], 0.000000e+00f);
    }
  }
  for (int ax0 = 0; ax0 < 2; ++ax0) {
    ((uint4*)((half*)A_shared_local + ((ax0 * 8))))[0] = ((uint4*)(A + ((((((((int)blockIdx.x) * 131072) + (ax0 * 65536)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8)))))[0];
  }
  for (int ax01 = 0; ax01 < 2; ++ax01) {
    ((uint4*)((half*)B_shared_local + ((ax01 * 8))))[0] = ((uint4*)(B + ((((((ax01 * 16384) + (((int)threadIdx.y) * 2048)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)))))[0];
  }
  for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 2; ++ax0_ax1_fused_0) {
    ((uint4*)(A_shared + (((((ax0_ax1_fused_0 * 2560) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)))))[0] = ((uint4*)((half*)A_shared_local + ((ax0_ax1_fused_0 * 8))))[0];
  }
  for (int ax0_ax1_fused_01 = 0; ax0_ax1_fused_01 < 2; ++ax0_ax1_fused_01) {
    ((uint4*)(B_shared + (((((ax0_ax1_fused_01 * 2176) + (((int)threadIdx.y) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)))))[0] = ((uint4*)((half*)B_shared_local + ((ax0_ax1_fused_01 * 8))))[0];
  }
  __syncthreads();
  for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
    (void)nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_0], ((half *)A_shared + ((((((int)threadIdx.y) & 3) * 1280) + (ax0_0 * 640)))), 40);
  }
  for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
    (void)nvcuda::wmma::load_matrix_sync(B_shared_wmma_matrix_b[ax1_0], ((half *)B_shared + ((((((int)threadIdx.y) >> 2) * 64) + (ax1_0 * 16)))), 136);
  }
  for (int i2_0_0 = 0; i2_0_0 < 31; ++i2_0_0) {
    for (int ax02 = 0; ax02 < 2; ++ax02) {
      ((uint4*)((half*)A_shared_local + ((ax02 * 8))))[0] = ((uint4*)(A + ((((((((((int)blockIdx.x) * 131072) + (ax02 * 65536)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 2) * 1024)) + (i2_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32))))[0];
    }
    for (int ax03 = 0; ax03 < 2; ++ax03) {
      ((uint4*)((half*)B_shared_local + ((ax03 * 8))))[0] = ((uint4*)(B + ((((((((i2_0_0 * 32768) + (ax03 * 16384)) + (((int)threadIdx.y) * 2048)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 32768))))[0];
    }
    for (int ax0_01 = 0; ax0_01 < 2; ++ax0_01) {
      (void)nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[(ax0_01 + 2)], ((half *)A_shared + ((((((i2_0_0 & 1) * 5120) + ((((int)threadIdx.y) & 3) * 1280)) + (ax0_01 * 640)) + 16))), 40);
    }
    for (int ax1_01 = 0; ax1_01 < 4; ++ax1_01) {
      (void)nvcuda::wmma::load_matrix_sync(B_shared_wmma_matrix_b[(ax1_01 + 4)], ((half *)B_shared + ((((((i2_0_0 & 1) * 4352) + ((((int)threadIdx.y) >> 2) * 64)) + (ax1_01 * 16)) + 2176))), 136);
    }
    for (int i1_0_4 = 0; i1_0_4 < 4; ++i1_0_4) {
      for (int i0_0_4 = 0; i0_0_4 < 2; ++i0_0_4) {
        (void)nvcuda::wmma::mma_sync(C_wmma_accumulator[((i0_0_4 * 4) + i1_0_4)], A_shared_wmma_matrix_a[i0_0_4], B_shared_wmma_matrix_b[i1_0_4], C_wmma_accumulator[((i0_0_4 * 4) + i1_0_4)]);
      }
    }
    // __syncthreads(); // extra sync
    for (int ax0_ax1_fused_02 = 0; ax0_ax1_fused_02 < 2; ++ax0_ax1_fused_02) {
      ((uint4*)(A_shared + ((((((((i2_0_0 + 1) & 1) * 5120) + (ax0_ax1_fused_02 * 2560)) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)))))[0] = ((uint4*)((half*)A_shared_local + ((ax0_ax1_fused_02 * 8))))[0];
    }
    for (int ax0_ax1_fused_03 = 0; ax0_ax1_fused_03 < 2; ++ax0_ax1_fused_03) {
      ((uint4*)(B_shared + ((((((((i2_0_0 + 1) & 1) * 4352) + (ax0_ax1_fused_03 * 2176)) + (((int)threadIdx.y) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)))))[0] = ((uint4*)((half*)B_shared_local + ((ax0_ax1_fused_03 * 8))))[0];
    }
    __syncthreads();
    for (int ax0_02 = 0; ax0_02 < 2; ++ax0_02) {
      (void)nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_02], ((half *)A_shared + ((((((i2_0_0 + 1) & 1) * 5120) + ((((int)threadIdx.y) & 3) * 1280)) + (ax0_02 * 640)))), 40);
    }
    for (int ax1_02 = 0; ax1_02 < 4; ++ax1_02) {
      (void)nvcuda::wmma::load_matrix_sync(B_shared_wmma_matrix_b[ax1_02], ((half *)B_shared + ((((((i2_0_0 + 1) & 1) * 4352) + ((((int)threadIdx.y) >> 2) * 64)) + (ax1_02 * 16)))), 136);
    }
    for (int i1_0_41 = 0; i1_0_41 < 4; ++i1_0_41) {
      for (int i0_0_41 = 0; i0_0_41 < 2; ++i0_0_41) {
        (void)nvcuda::wmma::mma_sync(C_wmma_accumulator[((i0_0_41 * 4) + i1_0_41)], A_shared_wmma_matrix_a[(i0_0_41 + 2)], B_shared_wmma_matrix_b[(i1_0_41 + 4)], C_wmma_accumulator[((i0_0_41 * 4) + i1_0_41)]);
      }
    }
  }
  for (int ax0_03 = 0; ax0_03 < 2; ++ax0_03) {
    (void)nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[(ax0_03 + 2)], ((half *)A_shared + (((((((int)threadIdx.y) & 3) * 1280) + (ax0_03 * 640)) + 5136))), 40);
  }
  for (int ax1_03 = 0; ax1_03 < 4; ++ax1_03) {
    (void)nvcuda::wmma::load_matrix_sync(B_shared_wmma_matrix_b[(ax1_03 + 4)], ((half *)B_shared + (((((((int)threadIdx.y) >> 2) * 64) + (ax1_03 * 16)) + 6528))), 136);
  }
  for (int i1_0_42 = 0; i1_0_42 < 4; ++i1_0_42) {
    for (int i0_0_42 = 0; i0_0_42 < 2; ++i0_0_42) {
      (void)nvcuda::wmma::mma_sync(C_wmma_accumulator[((i0_0_42 * 4) + i1_0_42)], A_shared_wmma_matrix_a[i0_0_42], B_shared_wmma_matrix_b[i1_0_42], C_wmma_accumulator[((i0_0_42 * 4) + i1_0_42)]);
    }
  }
  for (int i1_0_43 = 0; i1_0_43 < 4; ++i1_0_43) {
    for (int i0_0_43 = 0; i0_0_43 < 2; ++i0_0_43) {
      (void)nvcuda::wmma::mma_sync(C_wmma_accumulator[((i0_0_43 * 4) + i1_0_43)], A_shared_wmma_matrix_a[(i0_0_43 + 2)], B_shared_wmma_matrix_b[(i1_0_43 + 4)], C_wmma_accumulator[((i0_0_43 * 4) + i1_0_43)]);
    }
  }
  // each warp 64x16, 8warps
  const int padding = 8;
  const int shared_mem_stride = 8 * 16 + padding;
  const int mul = 16;
      
  __syncthreads(); // sync because C_shared reuse A_shared/B_shared
  for (int ax0_04 = 0; ax0_04 < 2; ++ax0_04) {
    for (int ax1_04 = 0; ax1_04 < 4; ++ax1_04) {///note:why here 2,4 but 4,2 for mma_sync?
      // (void)nvcuda::wmma::store_matrix_sync(((float *)C + (((((((((int)blockIdx.x) * 131072) + ((((int)threadIdx.y) & 3) * 32768)) + (ax0_04 * 16384)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.y) >> 2) * 64)) + (ax1_04 * 16)))), C_wmma_accumulator[((ax0_04 * 4) + ax1_04)], 1024, nvcuda::wmma::mem_row_major);
      nvcuda::wmma::store_matrix_sync(C_shared + ax1_04 * mul * shared_mem_stride + threadIdx.y * 16, 
                                      C_wmma_accumulator[((ax0_04 * 4) +ax1_04)], shared_mem_stride, nvcuda::wmma::mem_row_major);
    }///what's the layout of shared memory
    for (int ax1_04 = 0; ax1_04 < 4; ++ax1_04) {
      for (int v = 0; v < 2; v++) {
        *(float4*)((float *)C + (((((((((int)blockIdx.x) * 131072) + ((((int)threadIdx.y) & 3) * 32768)) + (ax0_04 * 16384)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.y) >> 2) * 64)) + (ax1_04 * 16)) + (v * 8 + threadIdx.x / 4) * 1024 + threadIdx.x % 4 * 4)) = 
            *(float4*)(C_shared + ax1_04 * mul * shared_mem_stride + threadIdx.y * 16 + (v * 8 + threadIdx.x / 4) * shared_mem_stride + threadIdx.x % 4 * 4);
      }
    }
    __syncthreads(); // not needed, but adding it will be faster
  }
}

for 32:
    for 64:

for 2:
    for 4:
        for 16:
            for 16:
    for 8:
