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
import tvm
from tvm import te, tir
from tvm.script import ty

def _check(original, transformed, use_native_pipeline):
    func = original
    mod = tvm.IRModule.from_expr(func)
    if use_native_pipeline:
        with tvm.transform.PassContext(
            config={"tir.InjectSoftwarePipeline": {"use_native_pipeline": True}}
        ):
            with tvm.target.Target("cuda --arch=sm_86"):
                mod = tvm.tir.transform.InjectSoftwarePipeline()(mod)
    else:
        mod = tvm.tir.transform.InjectSoftwarePipeline()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    print(tvm.script.asscript(mod))
    tvm.ir.assert_structural_equal(mod["main"], transformed)


@tvm.script.tir
def software_pipeline(a : ty.handle, c : ty.handle) -> None:
    A = tir.match_buffer(a, [100, 4], dtype="float32")
    C = tir.match_buffer(c, [100, 4], dtype="float32")
    for tx in tir.thread_binding(0, 1, 'threadIdx.x'):
        for i in range(0, 100, annotations={"pipeline_scope": 3}):
            with tir.block([], ""):
                B = tir.alloc_buffer([1, 4], dtype="float32", scope="shared")
                for j in tir.serial(0, 4):
                    with tir.block([], ""):
                        tir.reads([A[i, j]])
                        tir.writes(B[0, j])
                        B[0, j] = A[i, j]
                for j in tir.serial(0, 4):
                    with tir.block([], ""):
                        tir.reads([B[0, j]])
                        tir.writes([C[i, j]])
                        C[i, j] = B[0, j] + tir.float32(1)


@tvm.script.tir
def transformed_non_native_software_pipeline(a : ty.handle, c : ty.handle) -> None:
    C = tir.match_buffer(c, [100, 4])
    A = tir.match_buffer(a, [100, 4])
    for tx in tir.thread_binding(0, 1, thread = "threadIdx.x"):
        with tir.block([], ""):
            tir.reads([A[0:100, 0:4]])
            tir.writes([C[0:100, 0:4]])
            B = tir.alloc_buffer([3, 1, 4], scope="shared")
            for i, j in tir.grid(2, 4):
                with tir.block([], ""):
                    tir.reads([A[i, j]])
                    tir.writes([B[0:3, 0, j]])
                    B[i, 0, j] = A[i, j]
            tir.evaluate(tir.tvm_storage_sync("shared", dtype="int32"))
            for i in tir.serial(0, 98, annotations = {"pipeline_scope":1}):
                for j in tir.serial(0, 4):
                    with tir.block([], ""):
                        tir.reads([A[(i + 2), j]])
                        tir.writes([B[0:3, 0, j]])
                        B[tir.floormod((i + 2), 3), 0, j] = A[(i + 2), j]
                for j in tir.serial(0, 4):
                    with tir.block([], ""):
                        tir.reads([B[0:3, 0, j]])
                        tir.writes([C[i, j]])
                        C[i, j] = (B[tir.floormod(i, 3), 0, j] + tir.float32(1))
                tir.evaluate(tir.tvm_storage_sync("shared", dtype="int32"))
            for i, j in tir.grid(2, 4):
                with tir.block([], ""):
                    tir.reads([B[0:3, 0, j]])
                    tir.writes([C[(i + 98), j]])
                    C[(i + 98), j] = (B[tir.floormod((i + 2), 3), 0, j] + tir.float32(1))


@tvm.script.tir
def transformed_native_software_pipeline(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [100, 4])
    A = tir.match_buffer(a, [100, 4])

    for tx in tir.thread_binding(0, 1, thread = "threadIdx.x"):
        with tir.block([], ""):
            tir.reads([A[0:100, 0:4]])
            tir.writes([C[0:100, 0:4]])
            B = tir.alloc_buffer([3, 1, 4], scope="shared")
            pipeline: ty.handle = tir.tvm_create_pipeline(dtype="handle")
            for i in tir.serial(0, 2):
                tir.evaluate(tir.tvm_pipeline_producer_acquire(pipeline, dtype="handle"))
                for j in tir.serial(0, 4):
                    with tir.block([], ""):
                        tir.reads([A[i, j]])
                        tir.writes([B[0:3, 0, j]])
                        B[i, 0, j] = A[i, j]
                tir.evaluate(tir.tvm_pipeline_producer_commit(pipeline, dtype="handle"))
            for i in tir.serial(0, 98, annotations = {"pipeline_scope":1}):
                tir.evaluate(tir.tvm_pipeline_producer_acquire(pipeline, dtype="handle"))
                for j in tir.serial(0, 4):
                    with tir.block([], ""):
                        tir.reads([A[(i + 2), j]])
                        tir.writes([B[0:3, 0, j]])
                        B[tir.floormod((i + 2), 3), 0, j] = A[(i + 2), j]
                tir.evaluate(tir.tvm_pipeline_producer_commit(pipeline, dtype="handle"))
                tir.evaluate(tir.tvm_pipeline_consumer_wait(pipeline, dtype="handle"))
                tir.evaluate(tir.tvm_storage_sync("shared", dtype="int32"))
                for j in tir.serial(0, 4):
                    with tir.block([], ""):
                        tir.reads([B[0:3, 0, j]])
                        tir.writes([C[i, j]])
                        C[i, j] = (B[tir.floormod(i, 3), 0, j] + tir.float32(1))
                tir.evaluate(tir.tvm_pipeline_consumer_release(pipeline, dtype="handle"))
            for i in tir.serial(0, 2):
                tir.evaluate(tir.tvm_pipeline_consumer_wait(pipeline, dtype="handle"))
                tir.evaluate(tir.tvm_storage_sync("shared", dtype="int32"))
                for j in tir.serial(0, 4):
                    with tir.block([], ""):
                        tir.reads([B[0:3, 0, j]])
                        tir.writes([C[(i + 98), j]])
                        C[(i + 98), j] = (B[tir.floormod((i + 2), 3), 0, j] + tir.float32(1))
                tir.evaluate(tir.tvm_pipeline_consumer_release(pipeline, dtype="handle"))


def test_inject_non_native_software_pipeline():
    _check(software_pipeline, transformed_non_native_software_pipeline, False)


def test_inject_native_software_pipeline():
    _check(software_pipeline, transformed_native_software_pipeline, True)


if __name__ == "__main__":
    test_inject_non_native_software_pipeline()
    test_inject_native_software_pipeline()
