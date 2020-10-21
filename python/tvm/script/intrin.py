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
"""TVM Script Parser Intrinsic Functions

IRNodes (StmtNodes without body, PrimExprNodes and more) are called intrins
"""
# pylint: disable=redefined-builtin
import tvm.tir
from .registry import register
from .utils import get_param_list


class Intrin:
    def __init__(self, intrin, stmt=False):
        self.intrin = intrin
        self.stmt = stmt

    def signature(self):
        return "tir." + self.intrin.__name__, get_param_list(self.intrin)

    def handle(self, arg_list):
        return self.intrin(*arg_list)


@register
class Bool(Intrin):
    def __init__(self):
        def bool(imm):
            return tvm.tir.const(imm, "bool")

        super().__init__(bool)


@register
class Int8(Intrin):
    def __init__(self):
        def int8(imm):
            return tvm.tir.const(imm, "int8")

        super().__init__(int8)


@register
class Int16(Intrin):
    def __init__(self):
        def int16(imm):
            return tvm.tir.const(imm, "int16")

        super().__init__(int16)


@register
class Int32(Intrin):
    def __init__(self):
        def int32(imm):
            return tvm.tir.const(imm, "int32")

        super().__init__(int32)


@register
class Int64(Intrin):
    def __init__(self):
        def int64(imm):
            return tvm.tir.const(imm, "int64")

        super().__init__(int64)


@register
class UInt8(Intrin):
    def __init__(self):
        def uint8(imm):
            return tvm.tir.const(imm, "uint8")

        super().__init__(uint8)


@register
class UInt16(Intrin):
    def __init__(self):
        def uint16(imm):
            return tvm.tir.const(imm, "uint16")

        super().__init__(uint16)


@register
class UInt32(Intrin):
    def __init__(self):
        def uint32(imm):
            return tvm.tir.const(imm, "uint32")

        super().__init__(uint32)


@register
class UInt64(Intrin):
    def __init__(self):
        def uint64(imm):
            return tvm.tir.const(imm, "uint64")

        super().__init__(uint64)


@register
class Float8(Intrin):
    def __init__(self):
        def float8(imm):
            return tvm.tir.const(imm, "float8")

        super().__init__(float8)


@register
class Float16(Intrin):
    def __init__(self):
        def float16(imm):
            return tvm.tir.const(imm, "float16")

        super().__init__(float16)


@register
class Float32(Intrin):
    def __init__(self):
        def float32(imm):
            return tvm.tir.const(imm, "float32")

        super().__init__(float32)


@register
class Float64(Intrin):
    def __init__(self):
        def float64(imm):
            return tvm.tir.const(imm, "float64")

        super().__init__(float64)


@register
class FloorDiv(Intrin):
    def __init__(self):
        def floordiv(x, y):
            return tvm.tir.floordiv(x, y)

        super().__init__(floordiv)


@register
class FloorMod(Intrin):
    def __init__(self):
        def floormod(x, y):
            return tvm.tir.floormod(x, y)

        super().__init__(floormod)


@register
class Load(Intrin):
    def __init__(self):
        def load(dtype, var, index, predicate=True):
            return tvm.tir.Load(dtype, var, index, predicate)

        super().__init__(load)


@register
class Cast(Intrin):
    def __init__(self):
        def cast(value, dtype):
            return tvm.tir.Cast(dtype, value)

        super().__init__(cast)


@register
class Ramp(Intrin):
    def __init__(self):
        def ramp(base, stride, lanes):
            return tvm.tir.Ramp(base, stride, lanes)

        super().__init__(ramp)


@register
class BroadCast(Intrin):
    def __init__(self):
        def broadcast(value, lanes):
            return tvm.tir.Broadcast(value, lanes)

        super().__init__(broadcast)


@register
class Evaluate(Intrin):
    def __init__(self):
        def evaluate(value):
            return tvm.tir.Evaluate(value)

        super().__init__(evaluate, stmt=True)


@register
class Store(Intrin):
    def __init__(self):
        def store(var, index, value, predicate=True):
            return tvm.tir.Store(var, value, index, predicate)

        super().__init__(store, stmt=True)


class StepIntrin(Intrin):
    def __init__(self, reducer):
        def intrin(lhs, rhs):
            return tvm.tir.ReduceStep(self.reducer, lhs, rhs)

        super().__init__(intrin, stmt=True)
        self.reducer = reducer

    def signature(self):
        return "TVMScriptReducer.step", get_param_list(self.intrin)


@register
class IterVar(Intrin):
    def __init__(self):
        def iter_var(var, dom, iter_type, thread_tag):
            iter_type = getattr(tvm.tir.IterVar, iter_type)
            return tvm.tir.IterVar(dom, var, iter_type, thread_tag)

        super().__init__(iter_var)


@register
class Max(Intrin):
    def __init__(self):
        def max(a, b):  # pylint: disable=redefined-builtin
            return tvm.tir.Max(a, b)

        super().__init__(max)


def get_axis(begin, end, iter_type):
    ana = tvm.arith.Analyzer()
    extent = ana.simplify(end - begin)
    block_var_dom = tvm.ir.Range.from_min_extent(begin, extent)

    iter_type_dict = {"data_par": 0, "reduce": 2, "scan": 3, "opaque": 4}
    return tvm.tir.IterVar(block_var_dom, "bv", iter_type_dict[iter_type])


@register
class Range(Intrin):
    def __init__(self):
        def range(begin, end):
            return get_axis(begin, end, "data_par")

        super().__init__(range)


@register
class ReduceAxis(Intrin):
    def __init__(self):
        def reduce_axis(begin, end):
            return get_axis(begin, end, "reduce")

        super().__init__(reduce_axis)


@register
class ScanAxis(Intrin):
    def __init__(self):
        def scan_axis(begin, end):
            return get_axis(begin, end, "scan")

        super().__init__(scan_axis)


@register
class OpaqueAxis(Intrin):
    def __init__(self):
        def opaque_axis(begin, end):
            return get_axis(begin, end, "opaque")

        super().__init__(opaque_axis)
