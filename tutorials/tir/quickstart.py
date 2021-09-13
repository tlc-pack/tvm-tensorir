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
# Create an IRModule
# ------------------
# IRModule can be created by writing TVMScript, which is a script syntax for TVM IR. (see the ref)
# Here is a simple module for vector add.
#


@tvm.script.tir
class MyModule:
    def main(a: ty.handle, b: ty.handle) -> None:
        # We exchange data between function by handles, which are similar to pointer.
        tir.func_attr({"global_symbol": "main", "tir.noalias": True})
        # Create high-dimensional buffer from handles.
        A = tir.match_buffer(a, (8,), dtype="float32")
        B = tir.match_buffer(b, (8,), dtype="float32")
        for i in range(8):
            # A block is an abstraction for computation.
            with tir.block([8], "B") as [vi]:
                B[vi] = A[vi] + 1.0


ir_module = MyModule
print(type(ir_module))
print(tvm.script.asscript(ir_module))

################################################################################################
# Build and Run an IRModule
# -------------------------
# We can build the IRModule into a runnable module with specific target backends.
#

mod = tvm.build(ir_module, target="llvm")  # The module for CPU backends.
print(type(mod))

################################################################################################
# Prepare the input array and output array, then run the module.
#

a = tvm.nd.array(np.arange(8).astype("float32"))
print(a)
b = tvm.nd.array(np.zeros((8,)).astype("float32"))
mod(a, b)
print(b)


################################################################################################
# Transform an IRModule
# --------------------
# The IRModule is the central data structure for program optimization, which can be transformed
# by :code:`Schedule`.
# Schedule is consist of primitives. Each primitive does a simple job on IR transformation,
# such as loop tiling or make computation parallel. (Please see ref)
#

sch = tir.Schedule(ir_module)
print(type(sch))

################################################################################################
# Tile the loops 8 into 3 loops and print the result.

# Get block by its name
block_b = sch.get_block("B")
# Get loops surronding the block
(i,) = sch.get_loops(block_b)
# Tile the loop nesting.
i_0, i_1, i_2 = sch.split(i, factors=[2, 2, 2])
print(tvm.script.asscript(sch.mod))


################################################################################################
# If you want to deploy your module on GPUs, threads binding is necessary. Fortunately, we can
# also use primitives and do incrementally transformation.
#

sch.bind(i_0, "blockIdx.x")
sch.bind(i_1, "threadIdx.x")
print(tvm.script.asscript(sch.mod))


################################################################################################
# After binding the threads, now build the IRModule with :code:`cuda` backends.
ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")
cuda_a = tvm.nd.array(np.arange(8).astype("float32"), ctx)
cuda_b = tvm.nd.array(np.zeros((8,)).astype("float32"), ctx)
cuda_mod(cuda_a, cuda_b)
print(cuda_a)
print(cuda_b)
