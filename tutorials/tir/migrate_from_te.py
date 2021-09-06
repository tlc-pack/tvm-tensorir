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
.. _migrate-from-te:

How to Migrate Current TE Schedule to TensorIR
==================================
**Author**: `Siyuan Feng <https://github.com/Hzfengsy>`_

TensorIR is the next generation schedule in TVM which enhance the schedule ability.
In this tutorial, we will demonstrate how to migrate a existing TE schedule to TensorIR.
We choose a GPU convolution schedule as an example (Please see the tutorial
with TE schedule :ref:`opt-conv-gpu`). Also this tutorial will show the
similarities and differences between TE and TensorIR.
"""

################################################################
# Preparation and Tensor Expression
# ---------------------------------
# We can still use TE to represent the computation. To be specific,
# general TE contains two major parts: the tensor expression(including
# :code:`te.placeholder`, :code:`te.compute`, etc.) and the TE schedule.
# The Tensor expression defines how to compute, which is still useful to
# TensorIR schedule. However, TE schedule is no longer needed.

import numpy as np
import tvm
import tvm.testing
from tvm import te, tir

# The sizes of inputs and filters
batch = 256
in_channel = 256
out_channel = 512
in_size = 14
kernel = 3
pad = 1
stride = 1

A = te.placeholder((in_size, in_size, in_channel, batch), name="A")
W = te.placeholder((kernel, kernel, in_channel, out_channel), name="W")
out_size = (in_size - kernel + 2 * pad) // stride + 1
# Pad input
Apad = te.compute(
    (in_size + 2 * pad, in_size + 2 * pad, in_channel, batch),
    lambda yy, xx, cc, nn: tvm.tir.if_then_else(
        tvm.tir.all(yy >= pad, yy - pad < in_size, xx >= pad, xx - pad < in_size),
        A[yy - pad, xx - pad, cc, nn],
        tvm.tir.const(0.0, "float32"),
    ),
    name="Apad",
)
# Create reduction variables
rc = te.reduce_axis((0, in_channel), name="rc")
ry = te.reduce_axis((0, kernel), name="ry")
rx = te.reduce_axis((0, kernel), name="rx")
# Compute the convolution
B = te.compute(
    (out_size, out_size, out_channel, batch),
    lambda yy, xx, ff, nn: te.sum(
        Apad[yy * stride + ry, xx * stride + rx, rc, nn] * W[ry, rx, rc, ff], axis=[ry, rx, rc]
    ),
    name="B",
)


###############################################################################
# Create Schedule
# ---------------
#
# TensorIR creates schedule from an IRModule with **schedulable PrimFunc** (i.e.,
# a PrimFunc with **block** statement), which is usually created from TVMScript or
# tensor expression by :code:`create_prim_func`.
#

# create TE schedule
s_te = te.create_schedule(B.op)
# create TensorIR(TIR) schedule
original_func = te.create_prim_func([A, W, B])
s_tir = tir.Schedule(original_func)
# We can print the scheduled IRModule/PrimFunc at any stage in TIR schedule
print(tvm.script.asscript(s_tir.mod["main"]))


###############################################################################
# Schedule Primitives
# -------------------
# We re-design all primitives in TensorIR and make them more flexible. However,
# most of the primitives keep the backward compatibility, so that we can migrate
# current TE schedule to TensorIR without much extra effort. We can see the
# similarities and differences in the following schedule. Please refer the API
# document for all primitives with detailed information.
# .. note::
#
#   *Block*
#
#   Block is a new statement introduced by TensorIR, which is the basic schedule unit
#   It is somehow corresponding to TE stage if we use tensor expression as input.
#   Please see document for detailed information.
#

block_Apad = s_tir.get_block("Apad")
block_B = s_tir.get_block("B")

###############################################################################
# First step, we need to crate cache stage to specify the memory hierarchy for buffers.
# Here is the TE schedule.
#

s_te[Apad].compute_inline()
AA = s_te.cache_read(Apad, "shared", [B])
WW = s_te.cache_read(W, "shared", [B])
AL = s_te.cache_read(AA, "local", [B])
WL = s_te.cache_read(WW, "local", [B])
BL = s_te.cache_write(B, "local")


###############################################################################
# The API for cache_read/write in TIR schedule is a little bit different from TE.
# cache_read/write operates on a buffer. TE schedule can refer a buffer by its stage
# because there is only one stage can write the buffer. However, more than one blocks
# can access the buffer in TensorIR. So we need to refer the buffer by the block and
# its read/write region index. E.g., we use block_B and the index 1 to refer Buffer A
# in cache_read since Buffer A is the second buffer the block_B reads.
#

block_AA = s_tir.cache_read(block_B, 1, "shared")
block_WW = s_tir.cache_read(block_B, 2, "shared")
block_AL = s_tir.cache_read(block_B, 1, "local")
block_WL = s_tir.cache_read(block_B, 2, "local")
block_BG = s_tir.cache_write(block_B, 0, "local")

s_tir.compute_inline(block_Apad)
print(tvm.script.asscript(s_tir.mod["main"]))


###############################################################################
# The tiling-related primitives are pretty similar between TIR and TE schedule.
# But please note that TIR schedule allows to split one loop into serveal loops
# in one step.

# tile consts
tile = 8
num_thread = 8
block_factor = tile * num_thread
step = 8
vthread = 2


# TE schedule
hi, wi, fi, ni = s_te[B].op.axis
bz = s_te[B].fuse(hi, wi)
by, fi = s_te[B].split(fi, factor=block_factor)
bx, ni = s_te[B].split(ni, factor=block_factor)
tyz, fi = s_te[B].split(fi, nparts=vthread)
txz, ni = s_te[B].split(ni, nparts=vthread)
ty, fi = s_te[B].split(fi, nparts=num_thread)
tx, ni = s_te[B].split(ni, nparts=num_thread)
s_te[B].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)
threadx_te = tx  # used later

s_te[B].bind(bz, te.thread_axis("blockIdx.z"))
s_te[B].bind(by, te.thread_axis("blockIdx.y"))
s_te[B].bind(bx, te.thread_axis("blockIdx.x"))
s_te[B].bind(tyz, te.thread_axis("vthread"))  # vthread.x is not allowed in TE
s_te[B].bind(txz, te.thread_axis("vthread"))
# s_te[B].bind(tyz, te.thread_axis("vthread.y"))
# s_te[B].bind(txz, te.thread_axis("vthread.x"))
s_te[B].bind(ty, te.thread_axis("threadIdx.y"))
s_te[B].bind(tx, te.thread_axis("threadIdx.x"))

# TIR schedule
hi, wi, fi, ni = s_tir.get_loops(block_B)[0:4]
bz = s_tir.fuse(hi, wi)
by, tyz, ty, fi = s_tir.split(fi, [None, vthread, num_thread, tile // vthread])
bx, txz, tx, ni = s_tir.split(ni, [None, vthread, num_thread, tile // vthread])
s_tir.reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)
threadx_tir = tx  # used later

s_tir.bind(bz, "blockIdx.z")
s_tir.bind(by, "blockIdx.y")
s_tir.bind(bx, "blockIdx.x")
s_tir.bind(tyz, "vthread.y")
s_tir.bind(txz, "vthread.x")
s_tir.bind(ty, "threadIdx.y")
s_tir.bind(tx, "threadIdx.x")

print(tvm.script.asscript(s_tir.mod["main"]))

###############################################################################
# Move the compution under the specific loop with compute_at.
#

# TE schedule
s_te[BL].compute_at(s_te[B], threadx_te)
yi, xi, fi, ni = s_te[BL].op.axis
ry, rx, rc = s_te[BL].op.reduce_axis
rco, rci = s_te[BL].split(rc, factor=step)
s_te[BL].reorder(rco, ry, rx, rci, fi, ni)

s_te[AL].compute_at(s_te[BL], rci)
s_te[WL].compute_at(s_te[BL], rci)
s_te[AA].compute_at(s_te[BL], rx)
s_te[WW].compute_at(s_te[BL], rx)

# TIR schedule
block_B = block_BG  # The cache_write behavior is different from upstream
s_tir.compute_at(block_B, threadx_tir)
fi, ni, ry, rx, rc = s_tir.get_loops(block_B)[-5:]
rco, rci = s_tir.split(rc, [None, step])
s_tir.reorder(rco, ry, rx, rci, fi, ni)
decompose_pos = rco  # decompose position, see details later

s_tir.compute_at(block_AL, rci)
s_tir.compute_at(block_WL, rci)
s_tir.compute_at(block_AA, rx)
s_tir.compute_at(block_WW, rx)

print(tvm.script.asscript(s_tir.mod["main"]))

###############################################################################
# Finish the rest cooperative fetching schedule. And we finally finish the schedule.
#

# TE schedule
def te_coop_fetch(stage):
    yi, xi, ci, ni = s_te[stage].op.axis
    ty, ci = s_te[stage].split(ci, nparts=num_thread)
    tx, ni = s_te[stage].split(ni, nparts=num_thread)
    _, ni = s_te[stage].split(ni, factor=4)
    s_te[stage].reorder(ty, tx, yi, xi, ci, ni)
    s_te[stage].bind(ty, te.thread_axis("threadIdx.y"))
    s_te[stage].bind(tx, te.thread_axis("threadIdx.x"))
    s_te[stage].vectorize(ni)


te_coop_fetch(AA)
te_coop_fetch(WW)

# TIR schedule
def tir_coop_fetch(block):
    ci, ni = s_tir.get_loops(block)[-2:]
    ty, ci = s_tir.split(ci, [num_thread, None])
    tx, _, ni = s_tir.split(ni, [num_thread, None, 4])
    s_tir.reorder(ty, tx, ci, ni)
    s_tir.bind(ty, "threadIdx.y")
    s_tir.bind(tx, "threadIdx.x")
    s_tir.vectorize(ni)


tir_coop_fetch(block_AA)
tir_coop_fetch(block_WW)

print(tvm.script.asscript(s_tir.mod["main"]))

###############################################################################
# There is one more step for reduction blocks in TIR schedule: decompose. In TensorIR,
# each reduction block will contain a init part by default rather than have init and
# update two blocks. Only one block makes the function much easier to schedule. However,
# we need to decompose it and return to two blocks after schedule for better performance.
# The second arg is the loop where we should put the init part.
#

s_tir.decompose_reduction(block_B, decompose_pos)
print(tvm.script.asscript(s_tir.mod["main"]))


###############################################################################
# Generate CUDA Kernel
# --------------------
#
# Finally we use TVM to generate and compile the CUDA kernel, and compare the latency
# and result between TE and TIR schedule
#

func_te = tvm.build(s_te, [A, W, B], target="cuda")
func_tir = tvm.build(s_tir.mod["main"], target="cuda")
# func_tir = tvm.build(s_tir.mod, target="cuda") # I don't know why it fails

dev = tvm.cuda(0)
a_np = np.random.uniform(size=(in_size, in_size, in_channel, batch)).astype(A.dtype)
w_np = np.random.uniform(size=(kernel, kernel, in_channel, out_channel)).astype(W.dtype)
a = tvm.nd.array(a_np, dev)
w = tvm.nd.array(w_np, dev)
b_te = tvm.nd.array(np.zeros((out_size, out_size, out_channel, batch), dtype=B.dtype), dev)
b_tir = tvm.nd.array(np.zeros((out_size, out_size, out_channel, batch), dtype=B.dtype), dev)
func_te(a, w, b_te)
func_tir(a, w, b_tir)
tvm.testing.assert_allclose(b_te.numpy(), b_tir.numpy(), rtol=1e-5)

evaluator_te = func_te.time_evaluator(func_te.entry_name, dev, number=100)
evaluator_tir = func_te.time_evaluator(func_tir.entry_name, dev, number=100)
print("Convolution with TIR schedule %f ms" % (evaluator_tir(a, w, b_tir).mean * 1e3))
print("Convolution with TE schedule %f ms" % (evaluator_te(a, w, b_te).mean * 1e3))
