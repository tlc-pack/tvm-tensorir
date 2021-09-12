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

import pytest

import tvm
from tvm import tir
from tvm.meta_schedule import WorkloadToken
from tvm.meta_schedule.workload_registry import WorkloadRegistry
from tvm.script import ty

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

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def test_meta_schedule_workload_token_round_trip():
    mod = Matmul()
    shash = tvm.ir.structural_hash(mod)
    if shash < 0:
        shash += 1 << 64
    shash = str(shash)
    token = WorkloadToken(mod, shash, token_id=0)
    json_token = token.as_json()
    new_token = WorkloadToken.from_json(json_token, token_id=1)
    assert token.shash == new_token.shash
    tvm.ir.assert_structural_equal(token.mod, new_token.mod)


def test_meta_schedule_workload_registry_create():
    mod = Matmul()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = osp.join(tmpdir, "test.json")
        reg_0 = WorkloadRegistry(path, allow_missing=True)
        token_0 = reg_0.lookup_or_add(mod)
        assert token_0.same_as(reg_0.lookup_or_add(mod))
        assert len(reg_0) == 1
        reg_1 = WorkloadRegistry(path, allow_missing=True)
        assert len(reg_1) == 1
        token_1 = reg_1[0]
        assert token_0.shash == token_1.shash
        tvm.ir.assert_structural_equal(token_0.mod, mod)
        tvm.ir.assert_structural_equal(token_1.mod, mod)


def test_meta_schedule_workload_registry_missing():
    with pytest.raises(ValueError, match="File doesn't exist"):
        WorkloadRegistry(
            "./wrong-path/wrong-place/definitely-missing/incorrect.json",
            allow_missing=False,
        )


if __name__ == "__main__":
    test_meta_schedule_workload_token_round_trip()
    test_meta_schedule_workload_registry_create()
    test_meta_schedule_workload_registry_missing()
