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
"""Hybrid Script Parser Scope Handler Functions

This module provides the functions registered into parser under with_scope or for_scope category.
Scope handler functions are used to handle such scenarios.

.. code-block:: python

    for x in name():
    with name():

Typically, a scope handler function has no return value and accepts parser and node as its first
2 arguments.
"""
# pylint: disable=redefined-builtin

import tvm.tir
from .registry import register_scope_handler

# With scope handler


@register_scope_handler("with_scope", concise=False)
def block(parser, node, block_vars_info, reads, writes, predicate=True, annotations=None, name=""):
    """ With scope handler function block(block_vars， reads, writes, predicate, annotations, name)

    Example
    -------
    .. code-block:: python

        with block({vi(0, 128): i, vj(0, 128): j}, reads=[], writes=C[vi : vi + 1, vj : vj + 1], \
        name="init"):

    """

    block_vars = [info[0] for info in block_vars_info]
    values = [info[1] for info in block_vars_info]
    if not isinstance(reads, list):
        reads = [reads]
    if not isinstance(writes, list):
        writes = [writes]
    if annotations is None:
        annotations = []

    body = get_body(parser, node)
    allocations = parser.scope_emitter.pop_scope(is_block=True)

    inner = tvm.tir.Block(block_vars, reads, writes, body, allocations, annotations, name)
    return tvm.tir.BlockRealize(values, predicate, inner)


@register_scope_handler("with_scope", concise=True)
def Assert(parser, node, condition, message, body):
    """ With scope handler function assert(condition, message, body) """

    return tvm.tir.AssertStmt(condition, tvm.runtime.convert(message), body)


@register_scope_handler("with_scope", concise=True)
def realize(parser, node, buffer_bounds, body, condition=True):
    """ With scope handler function realize(buffer_bounds, condition, body) """

    buffer = buffer_bounds.buffer
    bounds = buffer_bounds.region
    return tvm.tir.BufferRealize(buffer, bounds, condition, body)


@register_scope_handler("with_scope", concise=True)
def attr(parser, node, attr_node, attr_key, value, body):
    """ With scope handler function attr(attr_node, attr_key, value, bdoy) """

    return tvm.tir.AttrStmt(attr_node, attr_key, tvm.runtime.convert(value), body)


@register_scope_handler("with_scope", concise=True)
def allocate(parser, node, buffer_var, dtype, extents, condition, body):
    """ With scope handler function allocate(buffer_var, dtype, extents, condition, body) """

    return tvm.tir.Allocate(buffer_var, dtype, extents, condition, body)


# For scope handler


@register_scope_handler("for_scope")
def range(parser, node, begin, end, for_type="serial"):
    """ For scope handler function range(begin, end, annotation)"""
    ana = tvm.arith.Analyzer()
    extent = end if begin == 0 else ana.simplify(end - begin)
    loop_var_name = node.target.id
    loop_var = tvm.te.var(loop_var_name, dtype="int32")

    parser.scope_emitter.new_scope()
    parser.scope_emitter.update_symbol(loop_var_name, loop_var)
    body = get_body(parser, node)
    parser.scope_emitter.pop_scope()

    for_type_dict = {"serial": 0, "parallel": 1, "vectorized": 2, "Unrolled": 3, }
    return tvm.tir.For(loop_var, begin, extent, for_type_dict[for_type], 0, body)


@register_scope_handler("for_scope")
def grid(parser, node, begin, end, annotation=None):
    """ For scope handler function grid(begin, end, annotation)"""
    ana = tvm.arith.Analyzer()
    extent = end if begin == 0 else ana.simplify(end - begin)
    loop_var_name = node.target.id
    loop_var = tvm.te.var(loop_var_name, dtype="int32")

    parser.scope_emitter.new_scope()
    parser.scope_emitter.update_symbol(loop_var_name, loop_var)
    body = get_body(parser, node)
    parser.scope_emitter.pop_scope()

    if annotation is None:
        annotation = []
    else:
        annotation = [
            tvm.tir.Annotation(key, tvm.runtime.convert(val) if isinstance(val, str) else val)
            for key, val in annotation.items()
        ]

    return tvm.tir.Loop(loop_var, begin, extent, annotation, body)


def get_body(parser, node):
    parser.scope_emitter.node_stack[-1].extend(reversed(node.body))
    body = []
    while len(parser.scope_emitter.node_stack[-1]):
        res = parser.visit(parser.scope_emitter.node_stack[-1].pop())
        if res is not None:
            body.append(res)
    body = tvm.tir.SeqStmt(body) if len(body) > 1 else body[0]
    return body
