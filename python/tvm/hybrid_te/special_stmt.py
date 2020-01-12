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
Special Stmt functions are used to provide some primitive functions for specific use.
Typically, a special stmt function has return value and accepts parser and
node as its first 2 arguments.
"""
# pylint: disable=unused-argument

from .. import api as _api
from .. import ir_pass as _pass
from .. import make as _make


def buffer_bind(parser, node, var, shape, dtype="float32", name="buf"):
    """ Special function buffer_bind(var, shape, dtype, name)

    Example
    -------
    .. code-block:: python

        A = buffer_bind(a, (128, 128), dtype="float32", name="A")

    """
    if var not in parser.params:
        parser.report_error("Can not bind non-input args to buffer")
    return _api.decl_buffer(shape, dtype=dtype, name=name)


def buffer_allocate(parser, node, shape, dtype="float32", name="buf", scope=""):
    """ Special function buffer_allocate(var, shape, dtype, name)

    Example
    -------
    .. code-block:: python

        A = buffer_allocate((128, 128), dtype="float32", name="A")

    """
    _buffer = _api.decl_buffer(shape, dtype=dtype, name=name)
    parser.scope_emitter.allocate_stack[-1].append(_make.BufferAllocate(_buffer, scope))
    return _buffer


def block_vars(parser, node, begin, end, name="bv", iter_type="data_par"):
    """ Special function buffer_bind(var, shape, dtype, name)

    Example
    -------
    .. code-block:: python

        vi(0, 128, iter_type="reduce")

    """
    extent = end if begin == 0 else _pass.Simplify(end - begin)
    block_var_dom = _make.range_by_min_extent(begin, extent)

    if iter_type == "data_par":
        iter_type_id = 0
    elif iter_type == "reduce":
        iter_type_id = 2
    elif iter_type == "scan":
        iter_type_id = 3
    elif iter_type == "opaque":
        iter_type_id = 4
    else:
        raise ValueError("Unknown iter_type")

    block_var = _api._IterVar(block_var_dom, name, iter_type_id)
    parser.update_symbol(block_var.var.name, parser.Symbol.IterVar, block_var.var)
    return block_var
