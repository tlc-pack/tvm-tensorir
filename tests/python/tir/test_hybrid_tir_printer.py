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
from tvm import tir
from tvm import ir_pass
from tvm.tir.hybrid import from_str


@tvm.tir.hybrid.script
def matmul(a, b, c):
    A = buffer_bind(a, (16, 16), "float32", name="A")
    B = buffer_bind(b, (16, 16), "float32", name="B")
    C = buffer_bind(c, (16, 16), "float32", name="C")

    with block({}, reads=[A[0: 16, 0: 16], B[0: 16, 0: 16]], writes=C[0: 16, 0: 16], name="root"):
        for i in range(0, 16):
            for j in range(0, 16):
                with block({vi(0, 16): i, vj(0, 16): j}, reads=[], writes=C[vi: vi + 1, vj: vj + 1],
                           name="init"):
                    C[vi, vj] = float32(0)
                for k in range(0, 16):
                    with block({vi(0, 16): i, vj(0, 16): j, vk(0, 16, iter_type="reduce"): k},
                               reads=[C[vi: vi + 1, vj: vj + 1], A[vi: vi + 1, vk: vk + 1],
                                      B[vj: vj + 1, vk: vk + 1]],
                               writes=[C[vi: vi + 1, vj: vj + 1]], name="update"):
                        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@tvm.tir.hybrid.script
def element_wise(a, c):
    A = buffer_bind(a, (16, 16), "float32", name="A")
    C = buffer_bind(c, (16, 16), "float32", name="C")

    with block({}, A[0: 16, 0: 16], C[0: 16, 0: 16], name="root"):
        B = buffer_allocate((16, 16), "float32", name="B")

        for i in range(0, 16):
            for j in range(0, 16):
                with block({vi(0, 16): i, vj(0, 16): j}, A[vi: vi + 1, vj: vj + 1],
                           B[vi: vi + 1, vj: vj + 1],
                           name="B"):
                    B[vi, vj] = A[vi, vj] * 2

        for i in range(0, 16):
            for j in range(0, 16):
                with block({vi(0, 16): i, vj(0, 16): j}, B[vi: vi + 1, vj: vj + 1],
                           C[vi: vi + 1, vj: vj + 1],
                           name="C"):
                    C[vi, vj] = B[vi, vj] + 1


@tvm.tir.hybrid.script
def predicate(b, c):
    B = buffer_bind(b, (16, 16), "float32", name="B")
    C = buffer_bind(c, (16, 16), "float32", name="C")

    with block({}, reads=[], writes=[], name="root"):
        for i in range(0, 16):
            for jo in range(0, 4):
                for ji in range(0, 4):
                    with block({vi(0, 16): i, vj(0, 16): jo * 3 + ji},
                               reads=B[vi: vi + 1, vj: vj + 1], writes=C[vi: vi + 1, vj: vj + 1],
                               predicate=jo * 4 + ji < 16):
                        C[vi, vj] = B[vi, vj] + 1


def test_matmul():
    a = tvm.var("a")
    b = tvm.var("b")
    c = tvm.var("c")
    func = matmul(a, b, c)
    rt_func = from_str(tvm.tir.hybrid.to_python(func, True))
    # assert tvm.ir_pass.Equal(func, rt_func)

    assert isinstance(rt_func.body, tvm.stmt.Block)
    assert isinstance(rt_func.body.body, tvm.stmt.Loop)
    assert isinstance(rt_func.body.body.body, tvm.stmt.Loop)
    assert isinstance(rt_func.body.body.body.body, tvm.stmt.SeqStmt)
    assert isinstance(rt_func.body.body.body.body[0], tvm.stmt.Block)
    assert isinstance(rt_func.body.body.body.body[1], tvm.stmt.Loop)
    assert isinstance(rt_func.body.body.body.body[1].body, tvm.stmt.Block)

    return func


def test_element_wise():
    a = tvm.var("a")
    c = tvm.var("c")
    func = element_wise(a, c)
    rt_func = from_str(tvm.tir.hybrid.to_python(func, True))
    # assert tvm.ir_pass.Equal(func, rt_func)

    assert isinstance(rt_func.body, tvm.stmt.Block)
    assert isinstance(rt_func.body.body, tvm.stmt.SeqStmt)
    assert isinstance(rt_func.body.body[0], tvm.stmt.Loop)
    assert isinstance(rt_func.body.body[0].body, tvm.stmt.Loop)
    assert isinstance(rt_func.body.body[0].body.body, tvm.stmt.Block)

    assert isinstance(rt_func.body.body[1], tvm.stmt.Loop)
    assert isinstance(rt_func.body.body[1].body, tvm.stmt.Loop)
    assert isinstance(rt_func.body.body[1].body.body, tvm.stmt.Block)

    return func


def test_predicate():
    b = tvm.var("b")
    c = tvm.var("c")
    func = predicate(b, c)
    rt_func = from_str(tvm.tir.hybrid.to_python(func, True))
    # assert tvm.ir_pass.Equal(func, rt_func)

    assert isinstance(rt_func.body, tvm.stmt.Block)
    assert isinstance(rt_func.body.body, tvm.stmt.Loop)
    assert isinstance(rt_func.body.body.body, tvm.stmt.Loop)
    assert isinstance(rt_func.body.body.body.body, tvm.stmt.Loop)
    assert isinstance(rt_func.body.body.body.body.body, tvm.stmt.Block)

    return func


def test_module():
    func1 = test_matmul()
    func2 = test_element_wise()
    func3 = test_predicate()
    mod = tvm.tir.hybrid.create_module([func1, func2, func3])
    print(tvm.tir.hybrid.to_python(mod))


@tvm.tir.hybrid.script
class MyModule:
    def matmul(a, b, c):
        C = buffer_bind(c, (16, 16), "float32", "C")
        A = buffer_bind(a, (16, 16), "float32", "A")
        B = buffer_bind(b, (16, 16), "float32", "B")
        with block({}, writes=[meta[TensorRegion][0]], reads=[meta[TensorRegion][1], meta[TensorRegion][2]], name="root"):
            for i in range(0, 16):
                for j in range(0, 16):
                    with block({vi(0, 16):i, vj(0, 16):j}, writes=[meta[TensorRegion][3]], reads=[], name="init"):
                        C[vi, vj] = float32(0)
                    for k in range(0, 16):
                        with block({vi(0, 16):i, vj(0, 16):j, vk(0, 16, iter_type="reduce"):k}, writes=[meta[TensorRegion][4]], reads=[meta[TensorRegion][5], meta[TensorRegion][6], meta[TensorRegion][7]], name="update"):
                            C[vi, vj] = (C[vi, vj] + (A[vi, vk]*B[vj, vk]))
    __tvm_meta__ = {
        "root": 1,
        "nodes": [
            {
                "type_key": ""
            },
            {
                "type_key": "StrMap",
                "keys": [
                    "TensorRegion"
                ],
                "data": [2]
            },
            {
                "type_key": "Array",
                "data": [3, 18, 33, 48, 56, 64, 70, 77]
            },
            {
                "type_key": "TensorRegion",
                "attrs": {
                    "buffer": "4",
                    "region": "11"
                }
            },
            {
                "type_key": "Buffer",
                "attrs": {
                    "buffer_type": "1",
                    "data": "5",
                    "data_alignment": "64",
                    "dtype": "float32",
                    "elem_offset": "10",
                    "name": "C",
                    "offset_factor": "1",
                    "scope": "global",
                    "shape": "6",
                    "strides": "9"
                }
            },
            {
                "type_key": "Variable",
                "attrs": {
                    "dtype": "handle",
                    "name": "C"
                }
            },
            {
                "type_key": "Array",
                "data": [7, 8]
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "16"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "16"
                }
            },
            {
                "type_key": "Array"
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "0"
                }
            },
            {
                "type_key": "Array",
                "data": [12, 15]
            },
            {
                "type_key": "Range",
                "attrs": {
                    "extent": "14",
                    "min": "13"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "0"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "16"
                }
            },
            {
                "type_key": "Range",
                "attrs": {
                    "extent": "17",
                    "min": "16"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "0"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "16"
                }
            },
            {
                "type_key": "TensorRegion",
                "attrs": {
                    "buffer": "19",
                    "region": "26"
                }
            },
            {
                "type_key": "Buffer",
                "attrs": {
                    "buffer_type": "1",
                    "data": "20",
                    "data_alignment": "64",
                    "dtype": "float32",
                    "elem_offset": "25",
                    "name": "A",
                    "offset_factor": "1",
                    "scope": "global",
                    "shape": "21",
                    "strides": "24"
                }
            },
            {
                "type_key": "Variable",
                "attrs": {
                    "dtype": "handle",
                    "name": "A"
                }
            },
            {
                "type_key": "Array",
                "data": [22, 23]
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "16"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "16"
                }
            },
            {
                "type_key": "Array"
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "0"
                }
            },
            {
                "type_key": "Array",
                "data": [27, 30]
            },
            {
                "type_key": "Range",
                "attrs": {
                    "extent": "29",
                    "min": "28"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "0"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "16"
                }
            },
            {
                "type_key": "Range",
                "attrs": {
                    "extent": "32",
                    "min": "31"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "0"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "16"
                }
            },
            {
                "type_key": "TensorRegion",
                "attrs": {
                    "buffer": "34",
                    "region": "41"
                }
            },
            {
                "type_key": "Buffer",
                "attrs": {
                    "buffer_type": "1",
                    "data": "35",
                    "data_alignment": "64",
                    "dtype": "float32",
                    "elem_offset": "40",
                    "name": "B",
                    "offset_factor": "1",
                    "scope": "global",
                    "shape": "36",
                    "strides": "39"
                }
            },
            {
                "type_key": "Variable",
                "attrs": {
                    "dtype": "handle",
                    "name": "B"
                }
            },
            {
                "type_key": "Array",
                "data": [37, 38]
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "16"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "16"
                }
            },
            {
                "type_key": "Array"
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "0"
                }
            },
            {
                "type_key": "Array",
                "data": [42, 45]
            },
            {
                "type_key": "Range",
                "attrs": {
                    "extent": "44",
                    "min": "43"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "0"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "16"
                }
            },
            {
                "type_key": "Range",
                "attrs": {
                    "extent": "47",
                    "min": "46"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "0"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "16"
                }
            },
            {
                "type_key": "TensorRegion",
                "attrs": {
                    "buffer": "4",
                    "region": "49"
                }
            },
            {
                "type_key": "Array",
                "data": [50, 53]
            },
            {
                "type_key": "Range",
                "attrs": {
                    "extent": "52",
                    "min": "51"
                }
            },
            {
                "type_key": "Variable",
                "attrs": {
                    "dtype": "int32",
                    "name": "vi"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "1"
                }
            },
            {
                "type_key": "Range",
                "attrs": {
                    "extent": "55",
                    "min": "54"
                }
            },
            {
                "type_key": "Variable",
                "attrs": {
                    "dtype": "int32",
                    "name": "vj"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "1"
                }
            },
            {
                "type_key": "TensorRegion",
                "attrs": {
                    "buffer": "4",
                    "region": "57"
                }
            },
            {
                "type_key": "Array",
                "data": [58, 61]
            },
            {
                "type_key": "Range",
                "attrs": {
                    "extent": "60",
                    "min": "59"
                }
            },
            {
                "type_key": "Variable",
                "attrs": {
                    "dtype": "int32",
                    "name": "vi"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "1"
                }
            },
            {
                "type_key": "Range",
                "attrs": {
                    "extent": "63",
                    "min": "62"
                }
            },
            {
                "type_key": "Variable",
                "attrs": {
                    "dtype": "int32",
                    "name": "vj"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "1"
                }
            },
            {
                "type_key": "TensorRegion",
                "attrs": {
                    "buffer": "4",
                    "region": "65"
                }
            },
            {
                "type_key": "Array",
                "data": [66, 68]
            },
            {
                "type_key": "Range",
                "attrs": {
                    "extent": "67",
                    "min": "59"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "1"
                }
            },
            {
                "type_key": "Range",
                "attrs": {
                    "extent": "69",
                    "min": "62"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "1"
                }
            },
            {
                "type_key": "TensorRegion",
                "attrs": {
                    "buffer": "19",
                    "region": "71"
                }
            },
            {
                "type_key": "Array",
                "data": [72, 74]
            },
            {
                "type_key": "Range",
                "attrs": {
                    "extent": "73",
                    "min": "59"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "1"
                }
            },
            {
                "type_key": "Range",
                "attrs": {
                    "extent": "76",
                    "min": "75"
                }
            },
            {
                "type_key": "Variable",
                "attrs": {
                    "dtype": "int32",
                    "name": "vk"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "1"
                }
            },
            {
                "type_key": "TensorRegion",
                "attrs": {
                    "buffer": "34",
                    "region": "78"
                }
            },
            {
                "type_key": "Array",
                "data": [79, 81]
            },
            {
                "type_key": "Range",
                "attrs": {
                    "extent": "80",
                    "min": "62"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "1"
                }
            },
            {
                "type_key": "Range",
                "attrs": {
                    "extent": "82",
                    "min": "75"
                }
            },
            {
                "type_key": "IntImm",
                "attrs": {
                    "dtype": "int32",
                    "value": "1"
                }
            }
        ],
        "b64ndarrays": [],
        "attrs": {"tvm_version": "0.7.dev0"}
    }


def test_module_class_based():
    mod = MyModule
    rt_mod = from_str(tvm.tir.hybrid.to_python(mod, True))
    # assert tvm.ir_pass.Equal(mod, rt_mod)


if __name__ == '__main__':
    test_module()
    test_module_class_based()
