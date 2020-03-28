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

import numpy as np
import tvm
from tvm import tir


@tvm.tir.hybrid.script
def matmul(a, b, c):
    A = buffer_bind(a, (1024, 1024), "float32")
    B = buffer_bind(b, (1024, 1024), "float32")
    C = buffer_bind(c, (1024, 1024), "float32")

    with block({}, reads=[A[0: 1024, 0: 1024], B[0: 1024, 0: 1024]], writes=C[0: 1024, 0: 1024],
               name="root"):
        for i in range(0, 1024):
            for j in range(0, 1024):
                with block({vi(0, 1024): i, vj(0, 1024): j}, reads=[],
                           writes=C[vi: vi + 1, vj: vj + 1], name="init"):
                    C[vi, vj] = float32(0)
        for i in range(0, 1024):
            for j in range(0, 1024):
                for k in range(0, 1024):
                    with block({vi(0, 1024): i, vj(0, 1024): j, vk(0, 1024, iter_type="reduce"): k},
                               reads=[C[vi: vi + 1, vj: vj + 1], A[vi: vi + 1, vk: vk + 1],
                                      B[vj: vj + 1, vk: vk + 1]],
                               writes=[C[vi: vi + 1, vj: vj + 1]], name="update"):
                        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


################################################################################################
# Preparation and Baseline
# ------------------------
# In this tutorial, we will demo how to use TVM to optimize matrix multiplication.
# Before actually demonstrating, we first define these variables.
# Then we write a baseline implementation, the simplest way to write a matrix multiplication in TVM.


M, N, K = 1024, 1024, 1024
target = 'llvm'
ctx = tvm.context(target, 0)

mod = tir.hybrid.create_module([matmul])
original_func = mod["matmul"]
func = tvm.build(original_func, target=target)

a_np = np.random.uniform(size=(M, K)).astype("float32")
b_np = np.random.uniform(size=(K, N)).astype("float32")
a = tvm.nd.array(a_np)
b = tvm.nd.array(b_np)
c = tvm.nd.array(np.zeros((M, N)).astype("float32"))
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()), rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print('Baseline: %f' % evaluator(a, b, c).mean)

print(tvm.lower(original_func, simple_mode=True))

################################################################################################
# Blocking
# --------
# A important trick to enhance the cache hit rate is blocking --- data chunk will be computed
# block by block. The memory access inside the block is a small neighbourhood which is with high
# memory locality. In this tutorial, I picked up 32 as the blocking factor. So the block will
# fill 32 * 32 * sizeof(float) which is 4KB in the cache whose total size is 32KB (L1 data cache)

bn = 32
s = tir.create_schedule(original_func)
init = s.get_block("init")
i, j = s.get_axes(init)
i_o, i_i = s.split(i, bn)
j_o, j_i = s.split(j, bn)
update = s.get_block("update")
i, j, k = s.get_axes(update)
i_o, i_i = s.split(i, bn)
j_o, j_i = s.split(j, bn)
k_o, k_i = s.split(k, 4)
s.reorder(i_o, j_o, k_o, k_i, i_i, j_i)

func = tvm.build(s.func, target=target)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()), rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print('Opt1: %f' % evaluator(a, b, c).mean)
print(tvm.lower(s.func, simple_mode=True))

################################################################################################
# Vectorization
# -------------
# Another important trick is vectorization. When the memory access pattern is uniform,
# the compiler can detect this pattern and pass the continuous memory to vector processor. In TVM,
# we can use `vectorize` interface to hint the compiler this pattern,
# so that we can accelerate it vastly.
#
# In this tutorial, we chose to vectorize the inner loop row data since it is cache friendly.

bn = 32
s = tir.create_schedule(original_func)
init = s.get_block("init")
i, j = s.get_axes(init)
i_o, i_i = s.split(i, bn)
j_o, j_i = s.split(j, bn)
s.reorder(i_o, j_o, i_i, j_i)
s.vectorize(j_i)
update = s.get_block("update")
i, j, k = s.get_axes(update)
i_o, i_i = s.split(i, bn)
j_o, j_i = s.split(j, bn)
k_o, k_i = s.split(k, 4)
s.reorder(i_o, j_o, k_o, k_i, i_i, j_i)
s.vectorize(j_i)

func = tvm.build(s.func, target=target)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()), rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print('Opt2: %f' % evaluator(a, b, c).mean)
print(tvm.lower(s.func, simple_mode=True))

################################################################################################
# Loop Permutation
# ----------------
# If we look at the above IR, we can see the inner loop row data is vectorized and
# B is transformed into PackedB. The traversal of PackedB is sequential now.
# So we will look at the access pattern of A. In current schedule, A is accessed column by column
# which is not cache friendly. If we change the nested loop order of ki and inner axes xi,
# the access pattern for A matrix is more cache friendly.

bn = 32
s = tir.create_schedule(original_func)

init = s.get_block("init")
i, j = s.get_axes(init)
i_o, i_i = s.split(i, bn)
j_o, j_i = s.split(j, bn)
s.reorder(i_o, j_o, i_i, j_i)
s.vectorize(j_i)

update = s.get_block("update")
i, j, k = s.get_axes(update)
i_o, i_i = s.split(i, bn)
j_o, j_i = s.split(j, bn)
k_o, k_i = s.split(k, 4)
s.reorder(i_o, j_o, k_o, i_i, k_i, j_i)
s.vectorize(j_i)

func = tvm.build(s.func, target=target)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()), rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print('Opt3: %f' % evaluator(a, b, c).mean)
print(tvm.lower(s.func, simple_mode=True))


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

@tvm.tir.hybrid.script
def matmul_packed(a, b, c):
    A = buffer_bind(a, (1024, 1024), "float32")
    B = buffer_bind(b, (1024, 1024), "float32")
    C = buffer_bind(c, (1024, 1024), "float32")

    with block({}, reads=[A[0: 1024, 0: 1024], B[0: 1024, 0: 1024]], writes=C[0: 1024, 0: 1024],
               name="root"):
        packedB = buffer_allocate((1024 // 32, 1024, 32))
        for i in range(0, 1024 // 32):
            for j in range(0, 1024):
                for k in range(0, 32):
                    with block({vi(0, 1024 // 32): i, vj(0, 1024): j, vk(0, 32): k},
                               reads=B[vj: vj + 1, vi * 32 + vk: vi * 32 + vk + 1],
                               writes=packedB[vi: vi + 1, vj: vj + 1, vk: vk + 1], name="packed"):
                        packedB[vi, vj, vk] = B[vj, vi * 32 + vk]

        for i in range(0, 1024):
            for j in range(0, 1024):
                with block({vi(0, 1024): i, vj(0, 1024): j}, reads=[],
                           writes=C[vi: vi + 1, vj: vj + 1], name="init"):
                    C[vi, vj] = float32(0)

        for i in range(0, 1024):
            for j in range(0, 1024):
                for k in range(0, 1024):
                    with block({vi(0, 1024): i, vj(0, 1024): j, vk(0, 1024, iter_type="reduce"): k},
                               reads=[C[vi: vi + 1, vj: vj + 1], A[vi: vi + 1, vk: vk + 1],
                                      packedB[vj // 32: vj // 32 + 1, vk: vk + 1,
                                      vj % 32: vj % 32 + 1]],
                               writes=[C[vi: vi + 1, vj: vj + 1]], name="update"):
                        C[vi, vj] = C[vi, vj] + A[vi, vk] * packedB[vj // 32, vk, vj % 32]


mod = tir.hybrid.create_module([matmul_packed])
original_func = mod["matmul_packed"]

bn = 32
s = tir.create_schedule(original_func)
packedB = s.get_block("packed")
i, j, k = s.get_axes(packedB)
s.vectorize(k)

init = s.get_block("init")
i, j = s.get_axes(init)
i_o, i_i = s.split(i, bn)
j_o, j_i = s.split(j, bn)
s.reorder(i_o, j_o, i_i, j_i)
s.vectorize(j_i)

update = s.get_block("update")
i, j, k = s.get_axes(update)
i_o, i_i = s.split(i, bn)
j_o, j_i = s.split(j, bn)
k_o, k_i = s.split(k, 4)
s.reorder(i_o, j_o, k_o, i_i, k_i, j_i)
s.vectorize(j_i)

func = tvm.build(s.func, target=target)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()), rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print('Opt4: %f' % evaluator(a, b, c).mean)
print(tvm.lower(s.func, simple_mode=True))
