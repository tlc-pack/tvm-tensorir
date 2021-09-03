# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-function-docstring,missing-module-docstring
import numpy as np
import tvm
import tvm.testing
from tvm import tir
from tvm.script import ty

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name

@tvm.script.tir
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (1024, 1024), "float32")
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.match_buffer(b, (1024, 1024), "float32")

    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "C") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = tir.float32(0)
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name
# fmt: on


################################################################################################
# Preparation and Baseline
# ------------------------
# In this tutorial, we will demo how to use TVM to optimize matrix multiplication.
# Before actually demonstrating, we first define these variables.
# Then we write a baseline implementation, the simplest way to write a matrix multiplication in TVM.


M, N, K = 1024, 1024, 1024
target = "llvm"
ctx = tvm.device(target, 0)

original_func = matmul

a_np = np.random.uniform(size=(M, K)).astype("float32")
b_np = np.random.uniform(size=(K, N)).astype("float32")
a = tvm.nd.array(a_np)
b = tvm.nd.array(b_np)
c = tvm.nd.array(np.zeros((M, N)).astype("float32"))


def build_and_test(mod):
    build_func = tvm.build(mod["main"], target=target)
    build_func(a, b, c)
    tvm.testing.assert_allclose(c.asnumpy(), np.matmul(a.asnumpy(), b.asnumpy()), rtol=1e-5)
    evaluator = build_func.time_evaluator(build_func.entry_name, ctx, number=1)
    print(tvm.script.asscript(mod))
    return evaluator(a, b, c).mean


print(tvm.script.asscript(original_func))
# print('Baseline: %f' % build_and_test(original_func))

################################################################################################
# Blocking
# --------
# A important trick to enhance the cache hit rate is blocking --- data chunk will be computed
# block by block. The memory access inside the block is a small neighborhood which is with high
# memory locality. In this tutorial, I picked up 32 as the blocking factor. So the block will
# fill 32 * 32 * sizeof(float) which is 4KB in the cache whose total size is 32KB (L1 data cache)

bn = 32

s = tir.Schedule(original_func, debug_mode=True)

update = s.get_block("C")
i, j, k = s.get_loops(update)
i_o, i_i = s.split(i, [None, bn])
j_o, j_i = s.split(j, [None, bn])
k_o, k_i = s.split(k, [None, 4])
s.reorder(i_o, j_o, k_o, k_i, i_i, j_i)
func_opt1 = s.mod

# s.decompose_reduction(update, j_o)
print("Opt1: %f" % build_and_test(s.mod))

################################################################################################
# Vectorization
# -------------
# Another important trick is vectorization. When the memory access pattern is uniform,
# the compiler can detect this pattern and pass the continuous memory to vector processor. In TVM,
# we can use `vectorize` interface to hint the compiler this pattern,
# so that we can accelerate it vastly.
#
# In this tutorial, we chose to vectorize the inner loop row data since it is cache friendly.

s = tir.Schedule(func_opt1, debug_mode=True)
update = s.get_block("C")
i_o, j_o, k_o, k_i, i_i, j_i = s.get_loops(update)

s.vectorize(j_i)
func_opt2 = s.mod

s.decompose_reduction(update, j_o)
print("Opt2: %f" % build_and_test(s.mod))

################################################################################################
# Loop Permutation
# ----------------
# If we look at the above IR, we can see the inner loop row data is vectorized and
# B is transformed into PackedB. The traversal of PackedB is sequential now.
# So we will look at the access pattern of A. In current schedule, A is accessed column by column
# which is not cache friendly. If we change the nested loop order of ki and inner axes xi,
# the access pattern for A matrix is more cache friendly.

s = tir.Schedule(func_opt2, debug_mode=True)
update = s.get_block("C")
i_o, j_o, k_o, k_i, i_i, j_i = s.get_loops(update)

s.reorder(i_o, j_o, k_o, i_i, k_i, j_i)
func_opt3 = s.mod

s.decompose_reduction(update, j_o)

print("Opt3: %f" % build_and_test(s.mod))


################################################################################################
# Array Packing
# -------------
# Another important trick is array packing. This trick is to reorder the storage dimension of the
# array to convert the continuous access pattern on certain dimension to a sequential pattern after
# flattening.
#
# .. image:: https://github.com/dmlc/web-data/raw/master/tvm/tutorial/array-packing.png
#      :align: center
#


###################################################################################################
# Just as it is shown in the figure above, after blocking the computations, we can observe the array
# access pattern of B (after flattening), which is regular but discontinuous. We expect that after
# some transformation we can get continuous access pattern. We can reorder a [16][16] array to
# a [16/4][16][4] array, so that the access pattern of B will be sequential when grabing
# the corresponding value from the packed array.
#

# We have to re-write the algorithm slightly.

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name

@tvm.script.tir
def matmul_packed(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.match_buffer(b, (1024, 1024), "float32")
    C = tir.match_buffer(c, (1024, 1024), "float32")

    packedB = tir.alloc_buffer((1024 // 32, 1024, 32))
    with tir.block([1024 // 32, 1024, 32], "packed") as [vi, vj, vk]:
        packedB[vi, vj, vk] = B[vj, vi * 32 + vk]

    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "C") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * packedB[vj // 32, vk, vj % 32]

# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name
# fmt: on


packed_func = matmul_packed

s = tir.Schedule(packed_func, debug_mode=True)
packedB = s.get_block("packed")
k = s.get_loops(packedB)[-1]
s.vectorize(k)
update = s.get_block("C")
i, j, k = s.get_loops(update)
i_o, i_i = s.split(i, [None, bn])
j_o, j_i = s.split(j, [None, bn])
k_o, k_i = s.split(k, [None, 4])
s.reorder(i_o, j_o, k_o, i_i, k_i, j_i)
s.vectorize(j_i)
func_opt3 = s.mod

s.decompose_reduction(update, j_o)
print("Opt4: %f" % build_and_test(s.mod))

################################################################################################
# Write cache for blocks
# ----------------------
# After blocking, the program will write result to C block by block, the access pattern
# is not sequential. So we can use a sequential cache array to hold the block results and
# write to C when all the block results are ready.
#

s = tir.Schedule(packed_func, debug_mode=True)
packedB = s.get_block("packed")
update = s.get_block("C")
cached_update = s.cache_write(update, 0, "global")

i, j = s.get_loops(update)
i_o, i_i = s.split(i, [None, bn])
j_o, j_i = s.split(j, [None, bn])
s.reorder(j_o, i_i)
s.compute_at(cached_update, j_o)

i, j, k = s.get_loops(cached_update)[-3:]
k_o, k_i = s.split(k, [None, 4])
s.reorder(k_o, i, k_i, j)
s.unroll(k_i)
s.vectorize(j)

x, y, z = s.get_loops(packedB)
s.vectorize(z)
s.parallel(x)
func_opt5 = s.mod

s.decompose_reduction(cached_update, j_o)
print("Opt5: %f" % build_and_test(s.mod))

###################################################################################################
# Parallel
# --------
# Futhermore, we can also utilize multi-core processors to do the thread-level parallelization.


s = tir.Schedule(func_opt5, debug_mode=True)
cached_update = s.get_block("C")
i_o, j_o, k_o, k_i, i_i, j_i = s.get_loops(cached_update)
s.parallel(i_o)

s.decompose_reduction(cached_update, j_o)
print("Opt6: %f" % build_and_test(s.mod))

###################################################################################################

##################################################################################################
# Summary
# -------
# After applying the above simple optimizations with only 18 lines of code,
# our generated code can achieve 60% of the `numpy` performance with MKL.
# Note that the outputs on the web page reflect the running times on a non-exclusive
# Docker container, thereby they are *unreliable*. It is highly encouraged to run the
# tutorial by yourself to observe the performance gain achieved by TVM.
