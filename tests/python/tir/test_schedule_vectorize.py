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
import util
from tvm import tir
from tvm.tir.ir_pass import AssertEqual


@tvm.tir.hybrid.script
def predicate_vectorize(b, c):
    C = buffer_bind(c, (16, 16), "float32")
    B = buffer_bind(b, (16, 16), "float32")
    with block({}, writes=[], reads=[], name="root"):
        for i in range(0, 16, annotation={}):
            for jo in range(0, 4, annotation={}):
                for ji in range(0, 4, annotation={"loop_type": "vectorize"}):
                    with block({vi(0, 16): i, vj(0, 16): ((jo * 4) + ji)},
                               writes=[C[vi:(vi + 1), vj:(vj + 1)]],
                               reads=[B[vi:(vi + 1), vj:(vj + 1)]],
                               predicate=(((jo * 4) + ji) < 16), name="update"):
                        C[vi, vj] = (B[vi, vj] + float32(1))


@tvm.tir.hybrid.script
def predicate_unroll(b, c):
    C = buffer_bind(c, (16, 16), "float32")
    B = buffer_bind(b, (16, 16), "float32")
    with block({}, writes=[], reads=[], name="root"):
        for i in range(0, 16, annotation={}):
            for jo in range(0, 4, annotation={}):
                for ji in range(0, 4, annotation={"loop_type": "unroll"}):
                    with block({vi(0, 16): i, vj(0, 16): ((jo * 4) + ji)},
                               writes=[C[vi:(vi + 1), vj:(vj + 1)]],
                               reads=[B[vi:(vi + 1), vj:(vj + 1)]],
                               predicate=(((jo * 4) + ji) < 16), name="update"):
                        C[vi, vj] = (B[vi, vj] + float32(1))


def test_vectorize_normal():
    func = util.predicate_stmt()

    s = tir.create_schedule(func)
    B = s.get_block("update")
    i, jo, ji = s.get_axes(B)
    s.vectorize(ji)

    mod = tir.hybrid.create_module([predicate_vectorize])
    print(tvm.lower(s.func, simple_mode=True))
    AssertEqual(s.func, mod["predicate_vectorize"])


def test_unroll_normal():
    func = util.predicate_stmt()

    s = tir.create_schedule(func)
    B = s.get_block("update")
    i, jo, ji = s.get_axes(B)
    s.unroll(ji)

    mod = tir.hybrid.create_module([predicate_unroll])
    print(tvm.lower(s.func, simple_mode=True))
    AssertEqual(s.func, mod["predicate_unroll"])


if __name__ == "__main__":
    test_vectorize_normal()
    test_unroll_normal()
