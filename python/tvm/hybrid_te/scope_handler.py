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

from .. import api as _api
from .. import ir_pass as _pass
from .. import make as _make


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
    parser.scope_emitter.new_scope(is_block=True)
    for stmt in node.body:
        parser.visit(stmt)
    for block_var in block_vars:
        parser.remove_symbol(block_var.var.name)
    parser.scope_emitter.emit(
        _make.TeBlock(block_vars, values, reads, writes, parser.scope_emitter.pop_seq(), predicate,
                      parser.scope_emitter.allocate_stack.pop(), annotations, name))


def range(parser, node, begin, end):
    """ For scope handler function range(begin, end)"""
    extent = end if begin == 0 else _pass.Simplify(end - begin)
    loop_var_name = node.target.id
    loop_var = _api.var(loop_var_name, dtype="int32")
    parser.update_symbol(loop_var_name, parser.Symbol.LoopVar, loop_var)
    parser.scope_emitter.new_scope()
    for stmt in node.body:
        parser.visit(stmt)
    parser.scope_emitter.emit(
        _make.Loop(loop_var, begin, extent, [], parser.scope_emitter.pop_seq()))
    parser.remove_symbol(loop_var_name)
