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

from . import registry
from .. import api as _api
from .. import ir_pass as _pass
from .. import make as _make


def register_scope_handler(origin_func, scope_name):
    registry.register_func(WithScope if scope_name == "with" else ForScope, origin_func, need_parser_and_node=True,
                           need_return=False)


class WithScope:
    pass


def block(parser, node, block_vars, values, reads, writes, predicate=True, annotations=[], name=""):
    """ With scope handler function block(block_vars, values, reads, writes, predicate, annotations, name)

    e.g.
        with block([vi(0, 128), vj(0, 128)], [i, j], reads=[], writes=C[vi : vi + 1, vj : vj + 1], name="init"):
    <=> with ib.block([vi, vj], [i, j], reads=[], writes=C[vi : vi + 1, vj : vj + 1], name="init"):
        (Note that block_vars has been processed ahead)
    """
    if not isinstance(reads, list):
        reads = [reads]
    if not isinstance(writes, list):
        writes = [writes]
    parser.scope_emitter.new_scope(is_block=True)
    for stmt in node.body:
        parser.visit(stmt)
    for block_var in block_vars:
        parser.remove_symbol(block_var.var.name)
    parser.scope_emitter.emit(
        _make.TeBlock(block_vars, values, reads, writes, parser.scope_emitter.pop_seq(), predicate,
                      parser.scope_emitter.allocate_stack.pop(), annotations, name))


class ForScope:
    pass


def range(parser, node, begin, end):
    """ For scope handler function range(begin, end)"""
    extent = end if begin == 0 else _pass.Simplify(end - begin)
    loop_var_name = node.target.id
    loop_var = _api.var(loop_var_name, dtype="int32")
    parser.add_symbol(loop_var_name, parser.Symbol.LoopVar, loop_var)
    parser.scope_emitter.new_scope()
    for stmt in node.body:
        parser.visit(stmt)
    parser.scope_emitter.emit(_make.Loop(loop_var, begin, extent, [], parser.scope_emitter.pop_seq()))
    parser.remove_symbol(loop_var_name)
