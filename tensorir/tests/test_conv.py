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
"""Example code to do square matrix multiplication."""
import tvm
from tvm import tensorir
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import check_correctness

def origin_schedule():
    # The sizes of inputs and filters
    batch = 256
    in_channel = 256
    out_channel = 512
    in_size = 14
    kernel = 3
    pad = 1
    stride = 1

    # Algorithm
    A = tvm.placeholder((in_size, in_size, in_channel, batch), name='A')
    W = tvm.placeholder((kernel, kernel, in_channel, out_channel), name='W')
    out_size = (in_size - kernel + 2*pad) // stride + 1
    # Pad input
    Apad = tvm.compute(
        (in_size + 2*pad, in_size + 2*pad, in_channel, batch),
        lambda yy, xx, cc, nn: tvm.if_then_else(
            tvm.all(yy >= pad, yy - pad < in_size,
                    xx >= pad, xx - pad < in_size),
            A[yy - pad, xx - pad, cc, nn], tvm.const(0., "float32")),
        name='Apad')

    # Create reduction variables
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel), name='ry')
    rx = tvm.reduce_axis((0, kernel), name='rx')
    # Compute the convolution
    B = tvm.compute(
        (out_size, out_size, out_channel, batch),
        lambda yy, xx, ff, nn: tvm.sum(
            Apad[yy * stride + ry, xx * stride + rx, rc, nn] * W[ry, rx, rc, ff],
            axis=[ry, rx, rc]),
        name='B')

    s = tvm.create_schedule(B.op)
    s[Apad].compute_inline() # compute Apad inline
    AA = s.cache_read(Apad, 'shared', [B])
    WW = s.cache_read(W, "shared", [B])
    AL = s.cache_read(AA, "local", [B])
    WL = s.cache_read(WW, "local", [B])
    BL = s.cache_write(B, "local")

    # tile consts
    tile = 8
    num_thread = 8
    block_factor = tile * num_thread
    step = 8
    vthread = 2

    # Get the GPU thread indices
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
    thread_xz = tvm.thread_axis((0, vthread), "vthread", name="vx")
    thread_yz = tvm.thread_axis((0, vthread), "vthread", name="vy")

    # Split the workloads
    hi, wi, fi, ni = s[B].op.axis
    bz = s[B].fuse(hi, wi)
    by, fi = s[B].split(fi, factor=block_factor)
    bx, ni = s[B].split(ni, factor=block_factor)

    # Bind the iteration variables to GPU thread indices
    s[B].bind(bz, block_z)
    s[B].bind(by, block_y)
    s[B].bind(bx, block_x)

    tyz, fi = s[B].split(fi, nparts=vthread)  # virtual thread split
    txz, ni = s[B].split(ni, nparts=vthread)  # virtual thread split
    ty, fi = s[B].split(fi, nparts=num_thread)
    tx, ni = s[B].split(ni, nparts=num_thread)
    s[B].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)

    s[B].bind(tyz, thread_yz)
    s[B].bind(txz, thread_xz)
    s[B].bind(ty, thread_y)
    s[B].bind(tx, thread_x)

    # Schedule BL local write
    s[BL].compute_at(s[B], tx)
    yi, xi, fi, ni = s[BL].op.axis
    ry, rx, rc = s[BL].op.reduce_axis
    rco, rci = s[BL].split(rc, factor=step)
    s[BL].reorder(rco, ry, rx, rci, fi, ni)

    # Attach computation to iteration variables
    s[AA].compute_at(s[BL], rx)
    s[WW].compute_at(s[BL], rx)
    s[AL].compute_at(s[BL], rci)
    s[WL].compute_at(s[BL], rci)

    # Schedule for A's shared memory load
    yi, xi, ci, ni = s[AA].op.axis
    ty, ci = s[AA].split(ci, nparts=num_thread)
    tx, ni = s[AA].split(ni, nparts=num_thread)
    _, ni = s[AA].split(ni, factor=4)
    s[AA].reorder(ty, tx, yi, xi, ci, ni)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)
    s[AA].vectorize(ni)  # vectorize memory load

    # Schedule for W's shared memory load
    yi, xi, ci, fi = s[WW].op.axis
    ty, ci = s[WW].split(ci, nparts=num_thread)
    tx, fi = s[WW].split(fi, nparts=num_thread)
    _, fi = s[WW].split(fi, factor=4)
    s[WW].reorder(ty, tx, yi, xi, ci, fi)
    s[WW].bind(ty, thread_y)
    s[WW].bind(tx, thread_x)
    s[WW].vectorize(fi)  # vectorize memory load

    return s, [A, W, B]

def test_conv():

    s, args = origin_schedule()

    def _schedule_pass(stmt):
        s = tensorir.create_schedule(stmt)
        stmt = s.to_halide()
        return stmt

    check_correctness(s, args, _schedule_pass, 'cuda')

def test_conv_schedule():
    # The sizes of inputs and filters
    batch = 256
    in_channel = 256
    out_channel = 512
    in_size = 14
    kernel = 3
    pad = 1
    stride = 1

    # Algorithm
    A = tvm.placeholder((in_size, in_size, in_channel, batch), name='A')
    W = tvm.placeholder((kernel, kernel, in_channel, out_channel), name='W')
    out_size = (in_size - kernel + 2*pad) // stride + 1
    # Pad input
    Apad = tvm.compute(
        (in_size + 2*pad, in_size + 2*pad, in_channel, batch),
        lambda yy, xx, cc, nn: tvm.if_then_else(
            tvm.all(yy >= pad, yy - pad < in_size,
                    xx >= pad, xx - pad < in_size),
            A[yy - pad, xx - pad, cc, nn], tvm.const(0., "float32")),
        name='Apad')

    # Create reduction variables
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel), name='ry')
    rx = tvm.reduce_axis((0, kernel), name='rx')
    # Compute the convolution
    B = tvm.compute(
        (out_size, out_size, out_channel, batch),
        lambda yy, xx, ff, nn: tvm.sum(
            Apad[yy * stride + ry, xx * stride + rx, rc, nn] * W[ry, rx, rc, ff],
            axis=[ry, rx, rc]),
        name='B')

    s = tvm.create_schedule(B.op)
    s[Apad].compute_inline() # compute Apad inline
    AA = s.cache_read(Apad, 'shared', [B])
    WW = s.cache_read(W, "shared", [B])
    AL = s.cache_read(AA, "local", [B])
    WL = s.cache_read(WW, "local", [B])
    BL = s.cache_write(B, "local")

    def _schedule_pass(stmt):
        s = tensorir.create_schedule(stmt)

        # tile consts
        tile = 8
        num_thread = 8
        block_factor = tile * num_thread
        step = 8
        vthread = 2

        # Get blocks
        AA, AL, WW, WL, BL, BL_, B = s.blocks()
        t1, t2 = s.split(s.axis(BL)[-1], factor=1)
        BL = s.blockize(t2)

        # Get the GPU thread indices
        block_x = tvm.thread_axis("blockIdx.x")
        block_y = tvm.thread_axis("blockIdx.y")
        block_z = tvm.thread_axis("blockIdx.z")
        thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
        thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
        thread_xz = tvm.thread_axis((0, vthread), "vthread", name="vx")
        thread_yz = tvm.thread_axis((0, vthread), "vthread", name="vy")

        # Split the workloads
        hi, wi, fi, ni = s.axis(B)
        bz = s.fuse(hi, wi)
        by, fi = s.split(fi, factor=block_factor)
        bx, ni = s.split(ni, factor=block_factor)

        # Bind the iteration variables to GPU thread indices
        s.bind(bz, block_z)
        s.bind(by, block_y)
        s.bind(bx, block_x)

        tyz, fi = s.split(fi, nparts=vthread)  # virtual thread split
        txz, ni = s.split(ni, nparts=vthread)  # virtual thread split
        ty, fi = s.split(fi, nparts=num_thread)
        tx, ni = s.split(ni, nparts=num_thread)
        bz, by, bx, tyz, txz, ty, tx, fi, ni = s.reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)

        s.bind(tyz, thread_yz)
        s.bind(txz, thread_xz)
        s.bind(ty, thread_y)
        s.bind(tx, thread_x)

        s.compute_at(BL, tx)
        s.unblockize(BL)
        BL, BL_ = s.blocks()[-3:-1]
        _ = s.fuse(s.axis(BL)[-2], s.axis(BL)[-1])
        fi, ni = s.axis(BL)[-2:]
        ry, rx, rc = s.axis(BL_)[-3:]
        rco, rci = s.split(rc, factor=step)
        rco, ry, rx, rci, fi, ni = s.reorder(rco, ry, rx, rci, fi, ni)

        # Attach computation to iteration variables
        s.compute_at(AL, rci)
        s.compute_at(WL, rci)
        s.compute_at(AA, rx)
        s.compute_at(WW, rx)

        # Schedule for A's shared memory load
        ci, ni = s.axis(AA)[-2:]
        ty, ci = s.split(ci, nparts=num_thread)
        tx, ni = s.split(ni, nparts=num_thread)
        _, ni = s.split(ni, factor=4)
        ty, tx, ci, ni = s.reorder(ty, tx, ci, ni)
        s.bind(ty, thread_y)
        s.bind(tx, thread_x)
        s.annotate(ni, "vectorize")  # vectorize memory load

        # Schedule for W's shared memory load
        ci, fi = s.axis(WW)[-2:]
        ty, ci = s.split(ci, nparts=num_thread)
        tx, fi = s.split(fi, nparts=num_thread)
        _, fi = s.split(fi, factor=4)
        ty, tx, ci, fi = s.reorder(ty, tx, ci, fi)
        s.bind(ty, thread_y)
        s.bind(tx, thread_x)
        s.annotate(fi, "vectorize")  # vectorize memory load

        stmt = s.to_halide()
        return stmt

    check_correctness(s, [A, W, B], _schedule_pass, 'cuda', tvm.build(*origin_schedule(), 'cuda'))

if __name__ == "__main__":
    test_conv()
    test_conv_schedule()
