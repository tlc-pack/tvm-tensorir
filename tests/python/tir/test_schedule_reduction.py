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


def add(x, y):
    return x + y


@tvm.tir.hybrid.script
def matmul(a, b, c):
    A = buffer_bind(a, (1024, 1024), "float32")
    B = buffer_bind(b, (1024, 1024), "float32")
    C = buffer_bind(c, (1024, 1024), "float32")

    with block({}, reads=[A[0: 1024, 0: 1024], B[0: 1024, 0: 1024]], writes=C[0: 1024, 0: 1024],
               name="root"):
        for i in range(0, 1024):
            for j in range(0, 1024):
                for k in range(0, 1024):
                    with block({vi(0, 1024): i, vj(0, 1024): j, vk(0, 1024, iter_type="reduce"): k},
                               reads=[C[vi: vi + 1, vj: vj + 1], A[vi: vi + 1, vk: vk + 1],
                                      B[vj: vj + 1, vk: vk + 1]],
                               writes=[C[vi: vi + 1, vj: vj + 1]], name="C"):
                        C[vi, vj] = comm_reduce("add", A[vi, vk] * B[vk, vj], float32(0))


################################################################################################
# Preparation and Baseline
# ------------------------
# In this tutorial, we will demo how to use TVM to optimize matrix multiplication.
# Before actually demonstrating, we first define these variables.
# Then we write a baseline implementation, the simplest way to write a matrix multiplication in TVM.


M, N, K = 1024, 1024, 1024
target = 'llvm'
ctx = tvm.context(target, 0)

tir.hybrid.register(add)
mod = tir.hybrid.create_module([matmul])
original_func = mod["matmul"]

a_np = np.random.uniform(size=(M, K)).astype("float32")
b_np = np.random.uniform(size=(K, N)).astype("float32")
a = tvm.nd.array(a_np)
b = tvm.nd.array(b_np)
c = tvm.nd.array(np.zeros((M, N)).astype("float32"))
'''
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()), rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print('Baseline: %f' % evaluator(a, b, c).mean)

print(tvm.lower(original_func, simple_mode=True))
'''
################################################################################################
# Blocking
# --------
# A important trick to enhance the cache hit rate is blocking --- data chunk will be computed
# block by block. The memory access inside the block is a small neighbourhood which is with high
# memory locality. In this tutorial, I picked up 32 as the blocking factor. So the block will
# fill 32 * 32 * sizeof(float) which is 4KB in the cache whose total size is 32KB (L1 data cache)

bn = 32
s = tir.create_schedule(original_func)
update = s.get_block("C")
i, j, k = s.get_axes(update)
i_o, i_i = s.split(i, bn)
j_o, j_i = s.split(j, bn)
k_o, k_i = s.split(k, 4)
s.reorder(i_o, j_o, k_o, k_i, i_i, j_i)
init = s.split_reduction(update, j_o)
print(tir.hybrid.ashybrid(s.func))
func = tvm.build(s.func, target=target)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()), rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print('Opt1: %f' % evaluator(a, b, c).mean)
print(tvm.lower(s.func, simple_mode=True))
