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

from .scope_emitter import ScopeEmitter


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

    for stmt in node.body:
        parser.visit(stmt)

    allocations, body = parser.scope_emitter.pop_scope(is_block=True)
    inner = tvm.tir.Block(block_vars, reads, writes, body, allocations, annotations, name)
    parser.scope_emitter.emit(tvm.tir.BlockRealize(values, predicate, inner, False))


def range(parser, node, begin, end, annotation=None):
    """ For scope handler function range(begin, end, annotation)"""
    extent = end if begin == 0 else tvm.tir.ir_pass.Simplify(end - begin)
    loop_var_name = node.target.id
    loop_var = tvm.te.var(loop_var_name, dtype="int32")

    parser.scope_emitter.new_scope()
    parser.scope_emitter.update_symbol(loop_var_name, ScopeEmitter.Symbol.LoopVar, loop_var)

    for stmt in node.body:
        parser.visit(stmt)

    if annotation is None:
        annotation = []
    else:
        annotation = [tvm.tir.Annotation(arg[0],
                                         tvm.runtime.convert(arg[1]) if isinstance(arg[1], str) else
                                         arg[1]) for arg in annotation]

    parser.scope_emitter.emit(
        tvm.tir.Loop(loop_var, begin, extent, annotation, parser.scope_emitter.pop_scope()))
