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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
import os.path as osp
import tempfile

import sys
import pytest

import tvm
from tvm import tir
from tvm.meta_schedule import WorkloadToken
from tvm.meta_schedule.workload_registry import WorkloadRegistry
from tvm.script import ty

from tvm.meta_schedule.utils import structural_hash

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument
# fmt: off

@tvm.script.tir
class Matmul:
    def main(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
        tir.func_attr({"global_symbol": "main"})
        A = tir.match_buffer(a, (1024, 1024), "float32")
        B = tir.match_buffer(b, (1024, 1024), "float32")
        C = tir.match_buffer(c, (1024, 1024), "float32")
        with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
            with tir.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.tir
class MatmulRelu:
    def main(a: ty.handle, b: ty.handle, d: ty.handle) -> None:  # pylint: disable=no-self-argument
        tir.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = tir.match_buffer(a, (16, 16), "float32")
        B = tir.match_buffer(b, (16, 16), "float32")
        D = tir.match_buffer(d, (16, 16), "float32")
        C = tir.alloc_buffer((16, 16), "float32")
        with tir.block([16, 16, tir.reduce_axis(0, 16)], "matmul") as [vi, vj, vk]:
            with tir.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
        with tir.block([16, 16], "relu") as [vi, vj]:
            D[vi, vj] = tir.max(C[vi, vj], 0.0)


@tvm.script.tir
class BatchMatmul:
    def main(a: ty.handle, b: ty.handle, c: ty.handle) -> None:  # pylint: disable=no-self-argument
        tir.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = tir.match_buffer(a, [16, 32, 32])
        B = tir.match_buffer(b, [16, 32, 32])
        C = tir.match_buffer(c, [16, 32, 32])
        with tir.block([16, 32, 32, tir.reduce_axis(0, 32)], "update") as [vn, vi, vj, vk]:
            with tir.init():
                C[vn, vi, vj] = 0.0
            C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]


@tvm.script.tir
class Add:
    def main(a: ty.handle, b: ty.handle, c: ty.handle) -> None:  # pylint: disable=no-self-argument
        tir.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = tir.match_buffer(a, [32], "float32")
        B = tir.match_buffer(b, [32], "float32")
        C = tir.match_buffer(c, [32], "float32")
        with tir.block([32], "add") as [vi]:
            C[vi] = A[vi] + B[vi]

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def test_meta_schedule_workload_token_round_trip():
    mod = Matmul()
    shash = structural_hash(mod)
    shash = str(shash)
    token = WorkloadToken(mod, shash, token_id=0)
    json_token = token.as_json()
    new_token = WorkloadToken.from_json(json_token, token_id=1)
    assert token.shash == new_token.shash
    tvm.ir.assert_structural_equal(token.mod, new_token.mod)


def test_meta_schedule_workload_token_wrong_shash():
    mod = Matmul()
    token = WorkloadToken(mod, "wrong_hash_prefix_" + structural_hash(mod), token_id=0)
    json_token = token.as_json()
    with pytest.raises(ValueError, match="Structural hash changed."):
        WorkloadToken.from_json(json_token, token_id=1)


def test_meta_schedule_workload_registry_create():
    mod = Matmul()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = osp.join(tmpdir, "test.json")
        reg_0 = WorkloadRegistry(path, allow_missing=True)
        token_0 = reg_0.lookup_or_add(mod)
        assert token_0.same_as(reg_0.lookup_or_add(mod))
        assert len(reg_0) == 1
        reg_1 = WorkloadRegistry(path, allow_missing=False)
        assert len(reg_1) == 1
        token_1 = reg_1[0]
        assert token_0.shash == token_1.shash
        tvm.ir.assert_structural_equal(token_0.mod, mod)
        tvm.ir.assert_structural_equal(token_1.mod, mod)


def test_meta_schedule_workload_registry_multiple():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = osp.join(tmpdir, "test.json")
        mods = [Matmul(), MatmulRelu(), BatchMatmul(), Add()]
        reg = WorkloadRegistry(path, allow_missing=True)
        tokens = [reg.lookup_or_add(mod) for mod in mods]
        for i, token in enumerate(tokens):
            assert token.same_as(reg.lookup_or_add(mods[i]))
        assert len(reg) == len(mods)
        reg_1 = WorkloadRegistry(path, allow_missing=False)
        assert len(reg_1) == len(mods)
        for i, token in enumerate(tokens):
            token_1 = reg_1[i]
            assert token.shash == token_1.shash
            tvm.ir.assert_structural_equal(token.mod, mods[i])
            tvm.ir.assert_structural_equal(token_1.mod, mods[i])


def test_meta_schedule_workload_registry_missing():
    with pytest.raises(ValueError, match="File doesn't exist"):
        WorkloadRegistry(
            "./wrong-path/wrong-place/definitely-missing/incorrect.json",
            allow_missing=False,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
