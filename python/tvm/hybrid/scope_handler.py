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
Scope handler nodes are StmtNodes with body, which are used to handle such scenarios.

.. code-block:: python

    for x in tir.name():
    with tir.name():
    tir.name() # with scope handlers + concise scoping

When registering a with scope handler, the first three arguments must be parser, node, body
When registering a for scope handler, the first four arguments must be parser, node, body, loop_vars
These parameters will handled by Hybrid Script parser automatically
"""
# pylint: disable=redefined-builtin, unused-argument, invalid-name

import tvm.tir
from .registry import register_with_scope, register_for_scope


# With scope handler
@register_with_scope(concise=False)
def block(parser, node, body, block_vars_info, reads, writes,
          predicate=True, annotations=None, name=""):
    """ With scope handler function block(block_vars， reads, writes, predicate, annotations, name)

    Example
    -------
    .. code-block:: python

        with tir.block({vi(0, 128): i, vj(0, 128): j}, \
                       reads=[], writes=C[vi : vi + 1, vj : vj + 1], name="init"):

    """
    block_vars = [info[0] for info in block_vars_info]
    values = [info[1] for info in block_vars_info]
    reads_in = [reads] if not isinstance(reads, list) else reads
    writes_in = [writes] if not isinstance(writes, list) else writes
    if annotations is None:
        annotations = []

    reads = []
    for read in reads_in:
        if isinstance(read, tvm.tir.BufferLoad):
            doms = []
            for index in read.indices:
                doms.append(tvm.ir.Range.from_min_extent(index, 1))
            reads.append(tvm.tir.TensorRegion(read.buffer, doms))
        else:
            reads.append(read)

    writes = []
    for write in writes_in:
        if isinstance(write, tvm.tir.BufferLoad):
            doms = []
            for index in write.indices:
                doms.append(tvm.ir.Range.from_min_extent(index, 1))
            writes.append(tvm.tir.TensorRegion(write.buffer, doms))
        else:
            writes.append(write)

    allocations = parser.scope_emitter.pop_scope(is_block=True)
    inner = tvm.tir.Block(block_vars, reads, writes, body, allocations, annotations, name)
    return tvm.tir.BlockRealize(values, predicate, inner)


@register_with_scope(concise=False)
def Assert(parser, node, body, condition, message):
    """ With scope handler function tir.Assert(condition, message) """
    return tvm.tir.AssertStmt(condition, tvm.runtime.convert(message), body)


@register_with_scope(concise=False)
def let(parser, node, body, var, value):
    """ With scope handler function tir.let(var, value) """
    return tvm.tir.LetStmt(var, value, body)


@register_with_scope(concise=True)
def realize(parser, node, body, buffer_bounds, condition=True):
    """ With scope handler function tir.realize(buffer_bounds, condition) """
    buffer, bounds = buffer_bounds.buffer, buffer_bounds.region
    return tvm.tir.BufferRealize(buffer, bounds, condition, body)


@register_with_scope(concise=True)
def attr(parser, node, body, attr_node, attr_key, value):
    """ With scope handler function tir.attr(attr_node, attr_key, value) """
    attr_node = tvm.runtime.convert(attr_node)
    value = tvm.runtime.convert(value)
    return tvm.tir.AttrStmt(attr_node, attr_key, value, body)


@register_with_scope(concise=True)
def allocate(parser, node, body, buffer_var, dtype, extents, condition=True):
    """ With scope handler function tir.allocate(buffer_var, dtype, extents, condition) """
    return tvm.tir.Allocate(buffer_var, dtype, extents, tvm.runtime.convert(condition), body)


# For scope handler
@register_for_scope()
def serial(parser, node, body, loop_vars, begin, end):
    """ For scope handler function tir.serial(begin, end)"""
    if len(loop_vars) != 1:
        parser.report_error("Expect exact 1 loop var")
    ana = tvm.arith.Analyzer()
    extent = end if begin == 0 else ana.simplify(end - begin)
    return tvm.tir.For(loop_vars[0], begin, extent, 0, 0, body)


@register_for_scope()
def parallel(parser, node, body, loop_vars, begin, end):
    """ For scope handler function tir.parallel(begin, end)"""
    if len(loop_vars) != 1:
        parser.report_error("Expect exact 1 loop var")
    ana = tvm.arith.Analyzer()
    extent = end if begin == 0 else ana.simplify(end - begin)
    return tvm.tir.For(loop_vars[0], begin, extent, 1, 0, body)


@register_for_scope()
def vectorized(parser, node, body, loop_vars, begin, end):
    """ For scope handler function tir.vectorized(begin, end)"""
    if len(loop_vars) != 1:
        parser.report_error("Expect exact 1 loop var")
    ana = tvm.arith.Analyzer()
    extent = end if begin == 0 else ana.simplify(end - begin)
    return tvm.tir.For(loop_vars[0], begin, extent, 2, 0, body)


@register_for_scope()
def unroll(parser, node, body, loop_vars, begin, end):
    """ For scope handler function tir.unroll(begin, end)"""
    if len(loop_vars) != 1:
        parser.report_error("Expect exact 1 loop var")
    ana = tvm.arith.Analyzer()
    extent = end if begin == 0 else ana.simplify(end - begin)
    return tvm.tir.For(loop_vars[0], begin, extent, 3, 0, body)


@register_for_scope(name="range")
def range(parser, node, body, loop_vars, begin, end, annotation=None):
    """ For scope handler function range(begin, end, annotation)"""
    if len(loop_vars) != 1:
        parser.report_error("Expect exact 1 loop var")
    ana = tvm.arith.Analyzer()
    extent = end if begin == 0 else ana.simplify(end - begin)
    if annotation is None:
        annotation = []
    else:
        annotation = [
            tvm.tir.Annotation(key, tvm.runtime.convert(val) if isinstance(val, str) else val)
            for key, val in annotation.items()
        ]
    return tvm.tir.Loop(loop_vars[0], begin, extent, annotation, body)


@register_for_scope()
def grid(parser, node, body, loop_vars, *extents):
    """ For scope handler function tir.grid(*extents) """
    if len(loop_vars) != len(extents):
        parser.report_error("Inconsitent number of loop vars and extents")
    for loop_var, extent in zip(reversed(loop_vars), reversed(extents)):
        body = tvm.tir.Loop(loop_var, 0, extent, [], body)
    return body
