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
"""Hybrid Script Parser Intrinsic Functions

This module provides the functions registered into parser under intrin category.
Typically, a intrin function has return value.
Meanwhile, user can register intrin functions into parser.

Example
-------

.. code-block:: python

    def add(a, b):
        return a + b
    def mul(a, b=1):
        return a * b

"""

import tvm.tir


def bool(imm):
    return tvm.tir.const(imm.value, "bool")


def int16(imm):
    return tvm.tir.const(imm.value, "int16")


def int32(imm):
    return tvm.tir.const(imm.value, "int32")


def int64(imm):
    return tvm.tir.const(imm.value, "int64")


def uint8(imm):
    return tvm.tir.const(imm.value, "uint8")


def uint16(imm):
    return tvm.tir.const(imm.value, "uint16")


def uint32(imm):
    return tvm.tir.const(imm.value, "uint32")


def uint64(imm):
    return tvm.tir.const(imm.value, "uint64")


def float16(imm):
    return tvm.tir.const(imm.value, "float16")


def float32(imm):
    return tvm.tir.const(imm.value, "float32")


def float64(imm):
    return tvm.tir.const(imm.value, "float64")


def floordiv(x, y):
    return tvm.tir.floordiv(x, y)


def floormod(x, y):
    return tvm.tir.floormod(x, y)


def load(dtype, var, index, predicate):
    return tvm.tir.Load(dtype, var, index, predicate)


def cast(dtype, value):
    return tvm.tir.Cast(dtype, value)


def evaluate(value):
    return tvm.tir.Evaluate(value)


def store(var, index, value, predicate):
    return tvm.tir.Store(var, value, index, predicate)


def iter_var(var, dom, iter_type, thread_tag):
    iter_type = getattr(tvm.tir.IterVar, iter_type)
    return tvm.tir.IterVar(dom, var, iter_type, thread_tag)
