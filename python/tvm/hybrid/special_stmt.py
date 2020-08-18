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
"""Hybrid Script Parser Special Stmt Functions
This module provides the functions registered into parser under special_stmt category.
special_stmt functions don't correspond to an IRNode in the AST directly. It is usually
used for some information that is not suitable to be printed directly.
special_stmt can appear as 2 formats
.. code-block:: python

    target = tir.name():
    tir.name()

When registering a special stmt, the first two arguments must be parser, node
"""
# pylint: disable=unused-argument, no-self-argument, inconsistent-return-statements
from typed_ast import ast3 as ast

import tvm.tir
from tvm import te
from .registry import register_special_stmt


@register_special_stmt()
def buffer_bind(parser, node, param, shape, dtype="float32", data=None, strides=None,
                elem_offset=None, scope="global", align=-1, offset_factor=0, buffer_type="default"):
    """ Special function buffer_bind(var, shape, dtype, data, strides, elem_offset, scope, align,
                                     offset_factor, buffer_type)
    Example
    -------
    .. code-block:: python

        A = tir.buffer_bind(a, (128, 128), dtype="float32")
    """

    if param not in parser.params:
        parser.report_error("Can not bind non-input param to buffer")
    if strides is None:
        strides = []
    align = align.value if not isinstance(align, int) else align
    offset_factor = offset_factor.value if not isinstance(offset_factor, int) else offset_factor
    buffer = tvm.tir.decl_buffer(shape, dtype, parser._assign_target, data, strides, elem_offset,
                                 scope, align, offset_factor, buffer_type)
    parser.buffer_map[param] = buffer
    return buffer


@register_special_stmt()
def buffer_allocate(parser, node, shape, dtype="float32", data=None, strides=None, elem_offset=None,
                    scope="global", align=-1, offset_factor=0, buffer_type="default"):
    """ Special function buffer_allocate(shape, dtype, data, strides, elem_offset, scope, align,
                                         offset_factor, buffer_type)

    Example
    -------
    .. code-block:: python

        A = tir.buffer_allocate((128, 128), dtype="float32")

    """

    if strides is None:
        strides = []
    align = align.value if not isinstance(align, int) else align
    offset_factor = offset_factor.value if not isinstance(offset_factor, int) else offset_factor
    buffer = tvm.tir.decl_buffer(shape, dtype, parser._assign_target, data, strides, elem_offset,
                                 scope, align, offset_factor, buffer_type)
    parser.scope_emitter.alloc(tvm.tir.BufferAllocate(buffer, scope))
    return buffer


@register_special_stmt()
def buffer_decl(parser, node, shape, dtype="float32", data=None, strides=None, elem_offset=None,
                scope="global", align=-1, offset_factor=0, buffer_type="default"):
    """ Special function buffer_decl(shape, dtype, data, strides, elem_offset, scope, align,
                                         offset_factor, buffer_type)
    Example
    -------
    .. code-block:: python

        A = tir.buffer_decl((128, 128), dtype="float32")
    """

    if strides is None:
        strides = []
    align = align.value if not isinstance(align, int) else align
    offset_factor = offset_factor.value if not isinstance(offset_factor, int) else offset_factor
    buffer = tvm.tir.decl_buffer(shape, dtype, parser._assign_target, data, strides, elem_offset,
                                 scope, align, offset_factor, buffer_type)
    return buffer



@register_special_stmt()
def var(parser, node, dtype):
    """ Special function for defining a Var"""
    return te.var(parser._assign_target, dtype)


@register_special_stmt()
def block_vars(parser, node, begin, end, iter_type="data_par"):
    """ Special function for defining a block var

    Example
    -------
    .. code-block:: python

        vi(0, 128, iter_type="reduce")

    """

    ana = tvm.arith.Analyzer()
    extent = end if begin == 0 else ana.simplify(end - begin)
    block_var_dom = tvm.ir.Range.from_min_extent(begin, extent)

    iter_type_dict = {"data_par": 0, "reduce": 2, "scan": 3, "opaque": 4}
    if iter_type not in iter_type_dict:
        parser.report_error("Unknown iter_type")

    if not isinstance(node.func, ast.Name):
        parser.report_errot("Invalid block var name")
    return tvm.tir.IterVar(block_var_dom, node.func.id, iter_type_dict[iter_type])


class HybridLambda:
    """Hybrid Lambda, used in lambda expression"""
    def __init__(self, args, body):
        self.args = args
        self.body = body


class HybridReducer:
    """Hybrid Reducer, used in reducer declaration"""
    def __init__(self, combiner, identity):
        self.combiner = combiner
        self.identity = identity
        self.reducer = tvm.tir.CommReducer([self.combiner.args[0]], [self.combiner.args[1]],
                                           [self.combiner.body], [self.identity])

    @register_special_stmt()
    def step(parser, node, lhs, rhs):
        obj = parser.visit(node.func.value)
        return tvm.tir.ReduceStep(obj.reducer, lhs, rhs)


@register_special_stmt()
def comm_reducer(parser, node, combiner, identity):
    """ Special function for defining a comm_reducer

    Example
    -------
    .. code-block:: python

        reducer = tir.comm_reducer(lambda x, y: x + y, float32(0))

    """

    if isinstance(combiner, HybridLambda) and len(combiner.args) == 2:
        return HybridReducer(combiner, identity)
    parser.report_error("comm_reducer expect a 2-argument lambda function as first argument")


@register_special_stmt()
def func_attr(parser, node, dict_attr):
    """ Special function for declaring the DictAttr of PrimFunc

    Example
    -------
    .. code-block:: python

         tir.func_attr({"tir.noalias": True, "global_symbol"})
    """

    parser.dict_attr = dict_attr