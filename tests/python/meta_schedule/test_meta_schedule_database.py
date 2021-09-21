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
"""Test Meta Schedule Database"""
import tempfile
import os.path as osp
import sys
import pytest

import tvm
from tvm import tir
from tvm.script import ty
from tvm.tir import Schedule
from tvm.meta_schedule import (
    JSONFileDatabase,
    TuningRecord,
    WorkloadRegistry,
    ArgInfo,
    ScheduleFn,
)

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

# Test Tuning Record


def _schedule_matmul(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block=block)
    i_tiles = sch.sample_perfect_tile(i, n=4)
    j_tiles = sch.sample_perfect_tile(j, n=4)
    k_tiles = sch.sample_perfect_tile(k, n=2)
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
    k_0, k_1 = sch.split(loop=k, factors=k_tiles)
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


def _equal_record(a: TuningRecord, b: TuningRecord):
    assert str(a.trace) == str(b.trace)
    assert str(a.run_secs) == str(b.run_secs)
    # AWAIT(@zxybazh): change to export after fixing "(bool)0"
    assert str(a.target) == str(b.target)
    assert a.workload.shash == b.workload.shash
    assert tvm.ir.structural_equal(a.workload.mod, b.workload.mod)
    for arg0, arg1 in zip(a.args_info, b.args_info):
        assert str(arg0.as_json()) == str(arg1.as_json())


def test_meta_schedule_tuning_record_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = osp.join(tmpdir, "registry.json")
        reg = WorkloadRegistry(path, allow_missing=True)
        mod = Matmul()
        workload = reg.lookup_or_add(mod)
        (example_sch,) = ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(mod)
        trace = example_sch.trace
        record = TuningRecord(
            trace,
            [1.5, 2.5, 1.8],
            workload,
            tvm.target.Target("llvm"),
            ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
        )
        new_record = TuningRecord.from_json(record.as_json(), reg)
        _equal_record(record, new_record)


# Test Database


def test_meta_schedule_database_create():
    with tempfile.TemporaryDirectory() as tmpdir:
        record_path = osp.join(tmpdir, "records.json")
        workload_path = osp.join(tmpdir, "workloads.json")
        database = JSONFileDatabase(  # pylint: disable=unused-variable
            record_path=record_path, workload_path=workload_path
        )


def test_meta_schedule_database_add_entry():
    with tempfile.TemporaryDirectory() as tmpdir:
        record_path = osp.join(tmpdir, "records.json")
        workload_path = osp.join(tmpdir, "workloads.json")
        database = JSONFileDatabase(  # pylint: disable=unused-variable
            record_path=record_path, workload_path=workload_path
        )
        mod = Matmul()
        workload = database.lookup_or_add(mod)
        (example_sch,) = ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(mod)
        trace = example_sch.trace
        record = TuningRecord(
            trace,
            [1.5, 2.5, 1.8],
            workload,
            tvm.target.Target("llvm"),
            ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
        )
        database.add(record)
        assert len(database) == 1
        ret = database.get_top_k(workload, 3)
        assert len(ret) == 1
        _equal_record(ret[0], record)


def test_meta_schedule_database_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        record_path = osp.join(tmpdir, "records.json")
        workload_path = osp.join(tmpdir, "workloads.json")
        database = JSONFileDatabase(  # pylint: disable=unused-variable
            record_path=record_path, workload_path=workload_path
        )
        mod = Matmul()
        workload = database.lookup_or_add(mod)
        (example_sch,) = ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(mod)
        trace = example_sch.trace
        record = TuningRecord(
            trace,
            [1.5, 2.5, 1.8],
            workload,
            tvm.target.Target("llvm"),
            ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
        )
        database.add(record)
        new_token = database.lookup_or_add(MatmulRelu())
        ret = database.get_top_k(new_token, 3)
        assert len(ret) == 0


def test_meta_schedule_database_sorting():
    with tempfile.TemporaryDirectory() as tmpdir:
        record_path = osp.join(tmpdir, "records.json")
        workload_path = osp.join(tmpdir, "workloads.json")
        database = JSONFileDatabase(  # pylint: disable=unused-variable
            record_path=record_path, workload_path=workload_path
        )
        mod = Matmul()
        workload = database.lookup_or_add(mod)
        (example_sch,) = ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(mod)
        trace = example_sch.trace
        records = [
            TuningRecord(
                trace,
                [7.0, 8.0, 9.0],
                workload,
                tvm.target.Target("llvm"),
                ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
            ),
            TuningRecord(
                trace,
                [1.0, 2.0, 3.0],
                workload,
                tvm.target.Target("llvm"),
                ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
            ),
            TuningRecord(
                trace,
                [4.0, 5.0, 6.0],
                workload,
                tvm.target.Target("llvm"),
                ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
            ),
            TuningRecord(
                trace,
                [1.1, 1.2, 600.0],
                workload,
                tvm.target.Target("llvm"),
                ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
            ),
            TuningRecord(
                trace,
                [1.0, 100.0, 6.0],
                workload,
                tvm.target.Target("llvm"),
                ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
            ),
            TuningRecord(
                trace,
                [4.0, 9.0, 8.0],
                workload,
                tvm.target.Target("llvm"),
                ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
            ),
        ]
        for record in records:
            database.add(record)
        ret = database.get_top_k(workload, 2)
        assert len(ret) == 2
        try:
            _equal_record(ret[0], records[2])
            _equal_record(ret[1], records[1])
        except AssertionError:
            _equal_record(ret[0], records[1])
            _equal_record(ret[1], records[2])


def test_meta_schedule_database_reload():
    with tempfile.TemporaryDirectory() as tmpdir:
        record_path = osp.join(tmpdir, "records.json")
        workload_path = osp.join(tmpdir, "workloads.json")
        database = JSONFileDatabase(  # pylint: disable=unused-variable
            record_path=record_path, workload_path=workload_path
        )
        mod = Matmul()
        workload = database.lookup_or_add(mod)
        (example_sch,) = ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(mod)
        trace = example_sch.trace
        records = [
            TuningRecord(
                trace,
                [7.0, 8.0, 9.0],
                workload,
                tvm.target.Target("llvm"),
                ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
            ),
            TuningRecord(
                trace,
                [1.0, 2.0, 3.0],
                workload,
                tvm.target.Target("llvm"),
                ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
            ),
            TuningRecord(
                trace,
                [4.0, 5.0, 6.0],
                workload,
                tvm.target.Target("llvm"),
                ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
            ),
        ]
        for record in records:
            database.add(record)
        new_database = JSONFileDatabase(  # pylint: disable=unused-variable
            record_path=record_path, workload_path=workload_path
        )
        workload = new_database.lookup_or_add(mod)
        ret = new_database.get_top_k(workload, 2)
        assert len(ret) == 2
        try:
            _equal_record(ret[0], records[2])
            _equal_record(ret[1], records[1])
        except AssertionError:
            _equal_record(ret[0], records[1])
            _equal_record(ret[1], records[2])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
