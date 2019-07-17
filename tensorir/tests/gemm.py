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
import os
import topi
from tvm.contrib import nvcc
from tvm.contrib import spirv
from tvm import ir_pass
from tvm import tensorir
from tvm import _api_internal
import numpy as np

TASK="gemm"
USE_MANUAL_CODE = False

def get_phase0(s, args, simple_mode=True):
	"""get statement after phase 0"""
	ret = []

	def fetch_pass(stmt):
		ret.append(stmt)
		return stmt

	with tvm.build_config(add_lower_pass=[(0, fetch_pass)]):
		tvm.lower(s, args, simple_mode=simple_mode)

	return ret[0]

def check_correctness(s, args, inserted_pass, target='llvm'):
	"""Check correctness by building the function with and without inserted_pass"""

	if isinstance(target, tuple) or isinstance(target, list):
		target1, target2 = target
	else:
		target1 = target2 = target

	with tvm.build_config(add_lower_pass=[(0, inserted_pass)]):
		print("============================")
		tvm.lower(s, args, target1, simple_mode=True)
		print("----------------------------")


	print("============================")
	tvm.lower(s, args, target2, simple_mode=True)
	print("----------------------------")


	with tvm.build_config(add_lower_pass=[(0, inserted_pass)]):
		func1 = tvm.build(s, args, target1)

	func2 = tvm.build(s, args, target2)

	ctx1 = tvm.context(target1)
	ctx2 = tvm.context(target2)

	bufs1 = [tvm.nd.array(np.random.randn(*topi.util.get_const_tuple(x.shape)).astype(x.dtype), ctx=ctx1)
	         for x in args]
	bufs2 = [tvm.nd.array(x, ctx=ctx2) for x in bufs1]

	func1(*bufs1)
	func2(*bufs2)

	bufs1_np = [x.asnumpy() for x in bufs1]
	bufs2_np = [x.asnumpy() for x in bufs2]

	for x, y in zip(bufs1_np, bufs2_np):
		np.testing.assert_allclose(x, y)

def test_gemm():
	# graph
	nn = 2048
	n = tvm.var('n')
	n = tvm.convert(nn)
	m, l = n, n
	A = tvm.placeholder((l, n), name='A')
	B = tvm.placeholder((l, m), name='B')
	k = tvm.reduce_axis((0, l), name='k')
	C = tvm.compute(
		(m, n),
		lambda ii, jj: tvm.sum(A[k, jj] * B[k, ii], axis=k),
		name='C')

	# schedule
	s = tvm.create_schedule(C.op)
	AA = s.cache_read(A, "shared", [C])
	BB = s.cache_read(B, "shared", [C])
	AL = s.cache_read(AA, "local", [C])
	BL = s.cache_read(BB, "local", [C])
	CC = s.cache_write(C, "local")
	#
	scale = 8
	num_thread = 8
	block_factor = scale * num_thread
	block_x = tvm.thread_axis("blockIdx.x")
	thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
	block_y = tvm.thread_axis("blockIdx.y")
	thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
	thread_xz = tvm.thread_axis((0, 2), "vthread", name="vx")
	thread_yz = tvm.thread_axis((0, 2), "vthread", name="vy")
	#
	by, yi = s[C].split(C.op.axis[0], factor=block_factor)
	bx, xi = s[C].split(C.op.axis[1], factor=block_factor)
	s[C].bind(by, block_y)
	s[C].bind(bx, block_x)
	s[C].reorder(by, bx, yi, xi)
	#
	tyz, yi = s[C].split(yi, nparts=2)
	ty, yi = s[C].split(yi, nparts=num_thread)
	txz, xi = s[C].split(xi, nparts=2)
	tx, xi = s[C].split(xi, nparts=num_thread)
	s[C].bind(tyz, thread_yz)
	s[C].bind(txz, thread_xz)
	s[C].bind(ty, thread_y)
	s[C].bind(tx, thread_x)
	s[C].reorder(tyz, txz, ty, tx, yi, xi)
	s[CC].compute_at(s[C], tx)

	yo, xo = CC.op.axis
	ko, ki = s[CC].split(k, factor=8)
	kt, ki = s[CC].split(ki, factor=1)
	s[CC].reorder(ko, kt, ki, yo, xo)
	s[AA].compute_at(s[CC], ko)
	s[BB].compute_at(s[CC], ko)
	s[CC].unroll(kt)
	s[AL].compute_at(s[CC], kt)
	s[BL].compute_at(s[CC], kt)
	# Schedule for A's shared memory load
	ty, xi = s[AA].split(s[AA].op.axis[0], nparts=num_thread)
	_, xi = s[AA].split(s[AA].op.axis[1], factor=num_thread * 4)
	tx, xi = s[AA].split(xi, nparts=num_thread)
	s[AA].bind(ty, thread_y)
	s[AA].bind(tx, thread_x)
	s[AA].vectorize(xi)
	# Schedule for B' shared memory load
	ty, xi = s[BB].split(s[BB].op.axis[0], nparts=num_thread)
	_, xi = s[BB].split(s[BB].op.axis[1], factor=num_thread * 4)
	tx, xi = s[BB].split(xi, nparts=num_thread)
	s[BB].bind(ty, thread_y)
	s[BB].bind(tx, thread_x)
	s[BB].vectorize(xi)
	s[AA].double_buffer()
	s[BB].double_buffer()

	def _schedule_pass(stmt):
		s = tensorir.create_schedule(stmt)

		stmt = s.to_halide()
		return stmt
	# print(get_phase0(s, [A, B, C], simple_mode=True))

	check_correctness(s, [A, B, C], _schedule_pass, 'cuda')

if __name__ == "__main__":
	test_gemm()
