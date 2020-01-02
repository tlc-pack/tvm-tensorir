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
"""Intrinsic Function Calls in Hybrid Script Parser For TE IR"""

from .. import api as _api
from .. import ir_pass as _pass
from .. import make as _make
from ..ir_builder import Buffer


class Symbol:
    """Enumerates types in the symbol table"""
    Var = 0
    Buffer = 1
    IterVar = 2
    LoopVar = 3
    ListOfTensorRegions = 4


class CallArgumentReader:
    """A helper class which read argument and do type check if needed"""

    def __init__(self, func_name, args, kwargs, parser):
        self.func_name = func_name
        self.args = args
        self.kwargs = kwargs
        self.parser = parser

    def get_func_compulsory_arg(self, pos, name):
        """Get corresponding function argument from argument list which is compulsory"""

        if len(self.args) >= pos:
            arg, arg_node = self.args[pos - 1]
        elif name not in self.kwargs.keys():
            self.parser.report_error(self.func_name + " misses argument " + name)
            return
        else:
            arg, arg_node = self.kwargs[name]

        return arg

    def get_func_optional_arg(self, pos, name, default):
        """Get corresponding function argument from argument list which is optional.
        If user doesn't provide the argument, set it to default value
        """

        if len(self.args) >= pos:
            arg, arg_node = self.args[pos - 1]
        elif name in self.kwargs.keys():
            arg, arg_node = self.kwargs[name]
        else:
            return default

        return arg


class GlobalScope:
    pass


class WithScope:
    pass


class ForScope:
    pass


def register_func(scope, func_name, func_to_register, arg_list, need_parser_and_node, need_return):
    """Helper function to register a function to the scope """

    def wrap_func(parser, node, args, kwargs):
        reader = CallArgumentReader(func_name, args, kwargs, parser)
        internal_args = list()

        if need_parser_and_node:
            internal_args.append(parser)
            internal_args.append(node)

        for i, arg_info in enumerate(arg_list):
            if len(arg_info) == 1:
                arg_name, = arg_info
                internal_args.append(reader.get_func_compulsory_arg(i + 1, arg_name))
            else:
                arg_name, default = arg_info
                internal_args.append(reader.get_func_optional_arg(i + 1, arg_name, default=default))

        if need_return:
            return func_to_register(*internal_args)
        else:
            func_to_register(*internal_args)

    setattr(scope, func_name, wrap_func)


def buffer_bind(parser, node, var, shape, dtype="float32", name="buf"):
    """ Intrin function buffer_bind(var, shape, dtype, name)

    e.g.
        A = buffer_bind(a, (128, 128), dtype="float32", name="A")
    <=> A = ib.declare_buffer((128, 128), dtype="float32", name="A")
        buffer_map[a] = A
    """
    if var not in parser.params:
        parser.report_error("Can not bind non-input args to buffer")
    return parser.ir_builder.declare_buffer(shape=shape, dtype=dtype, name=name)


def buffer_allocate(parser, node, shape, dtype="float32", name="buf", scope=""):
    """ Intrin function buffer_allocate(var, shape, dtype, name)

    e.g.
        A = buffer_allocate((128, 128), dtype="float32", name="A")
    <=> A = ib.allocate_buffer((128, 128), dtype="float32", name="A")
    """
    _buffer = _api.decl_buffer(shape, dtype=dtype, name=name)
    parser.scope_emitter.allocate_stack[-1].append(_make.BufferAllocate(_buffer, scope))
    return Buffer(parser.ir_builder, _buffer, dtype)


def block_vars(parser, node, begin, end, name="bv", iter_type="data_par"):
    """ Intrin function buffer_bind(var, shape, dtype, name)

    e.g.
        vi(0, 128, iter_type="reduce")
    <=> ib.IterVar(tvm.make_range_by_min_text(0, 128), name="vi", iter_type="reduce")
    """
    extent = end if begin == 0 else _pass.Simplify(end - begin)
    block_var_dom = _make.range_by_min_extent(begin, extent)
    block_var = parser.ir_builder.iter_var(block_var_dom, name=name, iter_type=iter_type)
    parser.add_symbol(block_var.var.name, Symbol.IterVar, block_var.var)
    return block_var


def block(parser, node, block_vars, values, reads, writes, predicate=True, annotations=[], name="", ):
    """ Intrin function block(block_vars, values, reads, writes, predicate, annotations, name)

    e.g.
        with block([vi(0, 128), vj(0, 128)], [i, j], reads=[], writes=C[vi : vi + 1, vj : vj + 1], name="init"):
    <=> with ib.block([vi, vj], [i, j], reads=[], writes=C[vi : vi + 1, vj : vj + 1], name="init"):
        (Note that block_vars has been processed ahead)
    """
    if not isinstance(reads, list):
        reads = [reads]
    if not isinstance(writes, list):
        writes = [writes]
    parser.scope_emitter.new_block_scope()
    for stmt in node.body:
        parser.visit(stmt)
    for block_var in block_vars:
        parser.remove_symbol(block_var.var.name)
    parser.scope_emitter.emit(
        _make.TeBlock(block_vars, values, reads, writes, parser.scope_emitter.pop_seq(), predicate,
                      parser.scope_emitter.allocate_stack.pop(), annotations, name))


def range(parser, node, begin, end, ):
    """ Intrin function range(begin, end)"""
    extent = end if begin == 0 else _pass.Simplify(end - begin)
    loop_var_name = node.target.id
    loop_var = _api.var(loop_var_name, dtype="int32")
    parser.add_symbol(loop_var_name, Symbol.LoopVar, loop_var)
    parser.scope_emitter.new_loop_scope()
    for stmt in node.body:
        parser.visit(stmt)
    parser.scope_emitter.emit(_make.Loop(loop_var, begin, extent, [], parser.scope_emitter.pop_seq()))
    parser.remove_symbol(loop_var_name)
