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
"""Stub of keywords in TIR scripts, used for type hinting."""

from typing import ContextManager, Dict, Iterable, Optional, Tuple, Union, Sequence, overload
from tvm.script import ty


"""
Redefine types
"""


class PrimExpr:
    def __init__(self: PrimExpr) -> None: ...
    @overload
    def __add__(self: PrimExpr, other: PrimExpr) -> PrimExpr: ...
    @overload
    def __sub__(self: PrimExpr, other: PrimExpr) -> PrimExpr: ...
    @overload
    def __mul__(self: PrimExpr, other: PrimExpr) -> PrimExpr: ...
    @overload
    def __div__(self: PrimExpr, other: PrimExpr) -> PrimExpr: ...
    @overload
    def __add__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    @overload
    def __radd__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    @overload
    def __sub__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    @overload
    def __rsub__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    @overload
    def __mul__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    @overload
    def __rmul__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    @overload
    def __div__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    @overload
    def __rdiv__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...


class Var(PrimExpr):
    ...


class IterVar(Var):
    ...


class Buffer(PrimExpr):
    def __getitem__(
        self: Buffer, pos: Tuple[Union[int, PrimExpr]]) -> Buffer: ...

"""
Variables and constants
"""


def var(dtype: str) -> Var: ...


def int8(imm: int) -> PrimExpr: ...


def int16(imm: int) -> PrimExpr: ...


def int32(imm: int) -> PrimExpr: ...


def int64(imm: int) -> PrimExpr: ...


def uint8(imm: int) -> PrimExpr: ...


def uint16(imm: int) -> PrimExpr: ...


def uint32(imm: int) -> PrimExpr: ...


def uint64(imm: int) -> PrimExpr: ...


def float8(imm: float) -> PrimExpr: ...


def float16(imm: float) -> PrimExpr: ...


def float32(imm: float) -> PrimExpr: ...


def float64(imm: float) -> PrimExpr: ...

"""
Binary/N-ary Operators
"""
def floormod(x: PrimExpr, y: PrimExpr) -> PrimExpr: ...


def floordiv(x: PrimExpr, y: PrimExpr) -> PrimExpr: ...


def truncmod(x: PrimExpr, y: PrimExpr) -> PrimExpr: ...


def truncdiv(x: PrimExpr, y: PrimExpr) -> PrimExpr: ...


def load(dtype: str, buffer_var: Var, index: PrimExpr,
         predicate: Optional[PrimExpr] = None) -> PrimExpr: ...


def cast(val: PrimExpr, dtype: str) -> PrimExpr: ...


def max(a: PrimExpr, b: PrimExpr) -> PrimExpr: ...


def min(a: PrimExpr, b: PrimExpr) -> PrimExpr: ...


def if_then_else(cond: PrimExpr, t: PrimExpr,
                 f: PrimExpr, dtype: str) -> PrimExpr: ...


"""
Unary operator
"""
def exp2(x: PrimExpr) -> PrimExpr: ...


def exp10(x: PrimExpr) -> PrimExpr: ...


def erf(x: PrimExpr) -> PrimExpr: ...


def tanh(x: PrimExpr) -> PrimExpr: ...


def sigmoid(x: PrimExpr) -> PrimExpr: ...


def log(x: PrimExpr) -> PrimExpr: ...


def log2(x: PrimExpr) -> PrimExpr: ...


def log10(x: PrimExpr) -> PrimExpr: ...


def log1p(x: PrimExpr) -> PrimExpr: ...


def tan(x: PrimExpr) -> PrimExpr: ...


def cos(x: PrimExpr) -> PrimExpr: ...


def cosh(x: PrimExpr) -> PrimExpr: ...


def acos(x: PrimExpr) -> PrimExpr: ...


def acosh(x: PrimExpr) -> PrimExpr: ...


def sin(x: PrimExpr) -> PrimExpr: ...


def sinh(x: PrimExpr) -> PrimExpr: ...


def asin(x: PrimExpr) -> PrimExpr: ...


def asinh(x: PrimExpr) -> PrimExpr: ...


def atan(x: PrimExpr) -> PrimExpr: ...


def atanh(x: PrimExpr) -> PrimExpr: ...


def atan2(x: PrimExpr) -> PrimExpr: ...


def sqrt(x: PrimExpr) -> PrimExpr: ...


def rsqrt(x: PrimExpr) -> PrimExpr: ...


"""
Loops
"""


def serial(begin: Union[PrimExpr, int],
           end: Union[PrimExpr, int]) -> Iterable[IterVar]: ...


def parallel(begin: Union[PrimExpr, int],
             end: Union[PrimExpr, int]) -> Iterable[IterVar]: ...


def vectorize(begin: Union[PrimExpr, int],
              end: Union[PrimExpr, int]) -> Iterable[IterVar]: ...


def unroll(begin: Union[PrimExpr, int],
           end: Union[PrimExpr, int]) -> Iterable[IterVar]: ...


def grid(*extents: Union[PrimExpr, int]) -> Iterable[Tuple[IterVar]]: ...


def range(begin: Union[PrimExpr, int],
          end: Union[PrimExpr, int]) -> Iterable[IterVar]: ...


def thread_binding(begin: Union[PrimExpr, int],
                   end: Union[PrimExpr, int], thread: str) -> Iterable[IterVar]: ...

"""
Axis
"""


def reduce_axis(begin: Union[PrimExpr, int],
                end: Union[PrimExpr, int]) -> IterVar: ...


def range(begin: Union[PrimExpr, int],
          end: Union[PrimExpr, int]) -> IterVar: ...


def scan_axis(begin: Union[PrimExpr, int],
              end: Union[PrimExpr, int]) -> IterVar: ...


def opaque_axis(begin: Union[PrimExpr, int],
                end: Union[PrimExpr, int]) -> IterVar: ...

"""
Buffers
"""


def match_buffer(param: Union[ty.handle, Buffer], shape: Sequence[Union[PrimExpr, int]], dtype: str = "float32", data=None, strides: Optional[Sequence[int]]
                 = None, elem_offset: Optional[int] = None, scope: str = "global", align: int = -1, offset_factor: int = 0, buffer_type: str = "default") -> Buffer: ...


def buffer_decl(shape: Sequence[Union[PrimExpr, int]], dtype: str = "float32", data=None, strides: Optional[Sequence[int]] = None,
                elem_offset: Optional[int] = None, scope: str = "global", align: int = -1, offset_factor: int = 0, buffer_type: str = "default") -> Buffer: ...


def alloc_buffer(shape: Sequence[Union[PrimExpr, int]], dtype: str = "float32", data=None, strides: Optional[Sequence[int]] = None,
                 elem_offset: Optional[int] = None, scope: str = "global", align: int = -1, offset_factor: int = 0, buffer_type: str = "default") -> Buffer: ...

"""
Reads/Writes
"""
def reads(*args: Buffer) -> None: ...


def writes(*args: Buffer) -> None: ...

"""
Scope handler
"""


class block(ContextManager):
    def __init__(
        self, axes: Sequence[Union[int, PrimExpr]], name: str = "") -> None: ...

    def __enter__(self) -> Sequence[IterVar]: ...


class init(ContextManager):
    def __init__(self) -> None: ...

"""
Threads and Bindings
"""
def env_thread(thread: str) -> IterVar: ...


def bind(iter_var: IterVar, expr: PrimExpr) -> None: ...


def where(cond: PrimExpr) -> None: ...


"""
Annotations
"""
def func_attr(attrs: Dict) -> None: ...


def block_attr(attrs: Dict) -> None: ...
