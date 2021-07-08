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


def test_inject_software_pipeline():
    n = 100
    m = 4
    num_stages = 3

    def original():
        tx = te.thread_axis("threadIdx.x")
        ib = tvm.tir.ir_builder.create()
        A = ib.pointer("float32", name="A")
        C = ib.pointer("float32", name="C")
        ib.scope_attr(tx, "thread_extent", 1)
        ib.scope_attr(None, "pipeline_scope", num_stages)
        with ib.for_range(0, n) as i:
            B = ib.allocate("float32", m, name="B", scope="shared")
            with ib.for_range(0, m) as j:
                B[j] = A[i * m + j]
            with ib.for_range(0, m) as k:
                C[k] = B[k] + 1
        stmt = ib.get()
        mod = tvm.IRModule({"main": tvm.tir.PrimFunc([A.asobject(), C.asobject()], stmt)})
        return mod

    def non_native_transformed():
        tx = te.thread_axis("threadIdx.x")
        ib = tvm.tir.ir_builder.create()
        A = ib.pointer("float32", name="A")
        C = ib.pointer("float32", name="C")
        ib.scope_attr(tx, "thread_extent", 1)
        B = ib.allocate("float32", num_stages * m, name="B", scope="shared")

        with ib.for_range(0, num_stages - 1) as i:
            with ib.for_range(0, m) as j:
                B[i * m + j] = A[i * 4 + j]

        ib.emit(tir.call_intrin("int32", "tir.tvm_storage_sync", "shared"))

        with ib.new_scope():
            ib.scope_attr(None, "pipeline_scope", 1)
            with ib.for_range(0, n - (num_stages - 1)) as i:
                with ib.for_range(0, m) as j:
                    B[tir.indexmod(i + (num_stages - 1), num_stages) * m + j] = A[
                        i * m + j + (num_stages - 1) * m
                    ]
                with ib.for_range(0, m) as k:
                    C[k] = B[tir.indexmod(i, num_stages) * m + k] + 1
                ib.emit(tir.call_intrin("int32", "tir.tvm_storage_sync", "shared"))

        with ib.for_range(0, num_stages - 1) as i:
            with ib.for_range(0, m) as k:
                C[k] = B[tir.indexmod(i + n - (num_stages - 1), num_stages) * m + k] + 1

        stmt = ib.get()
        mod = tvm.IRModule({"main": tvm.tir.PrimFunc([A.asobject(), C.asobject()], stmt)})
        return tvm.tir.transform.Simplify()(mod)

    def native_transformed():
        tx = te.thread_axis("threadIdx.x")
        ib = tvm.tir.ir_builder.create()
        A = ib.pointer("float32", name="A")
        C = ib.pointer("float32", name="C")
        ib.scope_attr(tx, "thread_extent", 1)
        B = ib.allocate("float32", num_stages * m, name="B", scope="shared")

        pipeline = tir.Var("pipeline", "handle")

        ib.emit(lambda body:tir.LetStmt(pipeline, tir.call_intrin("handle", "tir.tvm_create_pipeline"), body))
        # ib.new_scope()
        with ib.for_range(0, num_stages - 1) as i:
            ib.emit(tir.call_intrin("handle", "tir.tvm_pipeline_producer_acquire", pipeline))
            with ib.for_range(0, m) as j:
                B[i * m + j] = A[i * 4 + j]
            ib.emit(tir.call_intrin("handle", "tir.tvm_pipeline_producer_commit", pipeline))

        with ib.new_scope():
            ib.scope_attr(None, "pipeline_scope", 1)
            with ib.for_range(0, n - (num_stages - 1)) as i:
                ib.emit(tir.call_intrin("handle", "tir.tvm_pipeline_producer_acquire", pipeline))
                with ib.for_range(0, m) as j:
                    B[tir.indexmod(i + (num_stages - 1), num_stages) * m + j] = A[
                        i * m + j + (num_stages - 1) * m
                    ]
                ib.emit(tir.call_intrin("handle", "tir.tvm_pipeline_producer_commit", pipeline))
                ib.emit(tir.call_intrin("handle", "tir.tvm_pipeline_consumer_wait", pipeline))
                ib.emit(tir.call_intrin("int32", "tir.tvm_storage_sync", "shared"))
                with ib.for_range(0, m) as k:
                    C[k] = B[tir.indexmod(i, num_stages) * m + k] + 1
                ib.emit(tir.call_intrin("handle", "tir.tvm_pipeline_consumer_release", pipeline))

        with ib.for_range(0, num_stages - 1) as i:
            ib.emit(tir.call_intrin("handle", "tir.tvm_pipeline_consumer_wait", pipeline))
            ib.emit(tir.call_intrin("int32", "tir.tvm_storage_sync", "shared"))
            with ib.for_range(0, m) as k:
                C[k] = B[tir.indexmod(i + n - (num_stages - 1), num_stages) * m + k] + 1
            ib.emit(tir.call_intrin("handle", "tir.tvm_pipeline_consumer_release", pipeline))

        stmt = ib.get()
        mod = tvm.IRModule({"main": tvm.tir.PrimFunc([A.asobject(), C.asobject()], stmt)})
        return tvm.tir.transform.Simplify()(mod)

    mod = original()

    opt = tvm.transform.Sequential(
        [tvm.tir.transform.InjectSoftwarePipeline(), tvm.tir.transform.Simplify()]
    )
    with tvm.transform.PassContext(
        config={"tir.InjectSoftwarePipeline": {"use_native_pipeline": False}}
    ):
        transformed_mod = opt(mod)

    tvm.ir.assert_structural_equal(transformed_mod['main'].body,
                                   non_native_transformed()['main'].body, True)

    with tvm.transform.PassContext(
        config={"tir.InjectSoftwarePipeline": {"use_native_pipeline": True}}
    ):
        with tvm.target.Target("cuda --arch=sm_86"):
            transformed_mod = opt(mod)
    tvm.ir.assert_structural_equal(transformed_mod['main'].body, native_transformed()['main'].body,
                                   True)


if __name__ == "__main__":
    test_inject_software_pipeline()
