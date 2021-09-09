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
"""
.. _tir_blitz:

Blitz Course to TensorIR
===========================
**Author**: `Siyuan Feng <https://github.com/Hzfengsy>`_

TensorIR is a domain specific languages for deep learning programs serving two broad purposes:

* An implement for transforming and optimizing programs on various hardware backends.
* An abstraction for automatic tensorized program optimization.
"""

################################################################################################
# IRModule
# --------
# An IRModule is the central data structure in TensorIR, which contains deep learning programs.
# It is the basic object of interest of IR transformation and model building.
#

import tvm
from tvm import tir
from tvm.script import ty

import numpy as np

################################################################################################
# Create an IRModule from TVMScript
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TVMScript is a round-trip format for TensorIR in Python style. Any TensorIR can be printed
# to TVMScript and created from it.
#
# TVMScript allows users to write a Python-styled program and convert it into TVM stack that can
# be optimized and deployed on various hardware backends. This make it possible for users to
# optimize their own tensor program
#
# For a detailed introduction to TVMScript, please see the reference.
#
# Here we write a simple IRModule with one vector add function.
#


@tvm.script.tir
class MyModule:
    # We exchange data between function by handles, which are similar to pointer.
    def main(a: ty.handle, b: ty.handle) -> None:
        # TODO: can we add it automatically?
        tir.func_attr({"global_symbol": "main", "tir.noalias": True})
        # Create high-dimensional buffer from handles.
        A = tir.match_buffer(a, (1024, 1024), dtype="float32")
        B = tir.match_buffer(b, (1024, 1024), dtype="float32")
        for i, j in tir.grid(1024, 1024):
            # A block is an abstraction for computation.
            with tir.block([1024, 1024], "B") as [vi, vj]:
                B[vi, vj] = A[vi, vj] + 2.0


################################################################################################
# We can get the module and check it is created successfully by printing it out.
#

my_module = MyModule
print(tvm.script.asscript(my_module))

################################################################################################
# Build and run an IRModule
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Also we can build and run my module on our devices using CPU. First, we random generate an
# input array and calcutate the standard result for testing.
#

a_np = np.random.rand(1024, 1024).astype("float32")
b_np = a_np + 2.0

################################################################################################
# Then we get our device and alloc the memory for TVM module.
#

ctx = tvm.cpu(0)
a = tvm.nd.array(a_np, ctx)
b = tvm.nd.array(np.zeros((1024, 1024)).astype("float32"), ctx)

################################################################################################
# Finally build our module, test the result and evaluate the performance
#

mod = tvm.build(my_module, target="llvm")
mod(a, b)
np.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)

evaluator = mod.time_evaluator(mod.entry_name, ctx, number=10)
print("Running time: %f ms" % (evaluator(a, b).mean * 1e3))

################################################################################################
# Schedule on IRModule
# --------------------
# A Schedule is a set of transformations working on an IRModule. We call each basic
# transformation a schedule primitive, including loop spliting, fusing, reordering, and more
# are comprehensively described in the API reference.
#
# We will show some simple schedule in this tutorial, please see our other tutorals for
# advanced schedule and optimization skills.
#
# Blocks and loops are two major statement of interest. They are the basic schedule unit in
# TensorIR.
#

sch = tir.Schedule(my_module)
# Get block by its name
block_b = sch.get_block("B")
# Get loops surronding the block
i, j = sch.get_loops(block_b)

################################################################################################
# Schedule on CPU
# ~~~~~~~~~~~~~~~
# In CPU schedule, we will show how to use multi-threading and vectorization to speed up our
# module. We can see the IRModule after scheduling.
#

j_0, j_1 = sch.split(j, factors=[None, 32])
sch.vectorize(j_1)
sch.parallel(i)
cpu_module = sch.mod
print(tvm.script.asscript(cpu_module))

################################################################################################
# Evaluate the performance. Here we reuse the data generated before and the allocated memory.
#

mod = tvm.build(cpu_module, target="llvm")
mod(a, b)
np.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)

evaluator = mod.time_evaluator(mod.entry_name, ctx, number=10)
print("Running time: %f ms" % (evaluator(a, b).mean * 1e3))

################################################################################################
# Schedule on GPU
# ~~~~~~~~~~~~~~~
# We create a new schedule from original IRModule.
# Different from CPU, GPU required explicit binding :code:`threadIdx` and :code:`blockIdx`.
#

sch = tir.Schedule(my_module)
block_b = sch.get_block("B")
i, j = sch.get_loops(block_b)
j_0, j_1 = sch.split(j, factors=[None, 4])
sch.bind(i, "blockIdx.x")
sch.bind(j_0, "threadIdx.x")
sch.vectorize(j_1)
gpu_module = sch.mod
print(tvm.script.asscript(gpu_module))

################################################################################################
# Allocate new memory for GPU and evaluate the result again.
#

ctx = tvm.cuda(0)
a_cuda = tvm.nd.array(a_np, ctx)
b_cuda = tvm.nd.array(np.zeros((1024, 1024)).astype("float32"), ctx)

mod = tvm.build(gpu_module, target="cuda")
mod(a_cuda, b_cuda)
np.testing.assert_allclose(b_cuda.numpy(), b_np, rtol=1e-5)

evaluator = mod.time_evaluator(mod.entry_name, ctx, number=10)
print("Running time: %f ms" % (evaluator(a_cuda, b_cuda).mean * 1e3))
