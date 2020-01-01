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

from typing import List, Tuple, Union

from typeguard import check_type

from .. import api as _api
from .. import expr as _expr
from .. import ir_pass as _pass
from .. import make as _make
from .. import schedule as _schedule
from ..ir_builder import TensorRegion, Buffer


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

    def get_func_compulsory_arg(self, pos, name, type_expected=None):
        """Get corresponding function argument from argument list which is compulsory"""

        if len(self.args) >= pos:
            arg, arg_node = self.args[pos - 1]
        elif name not in self.kwargs.keys():
            self.parser.report_error(self.func_name + " misses argument " + name)
            return
        else:
            arg, arg_node = self.kwargs[name]

        if type_expected is not None:
            self._type_check(name, arg, arg_node, type_expected)

        return arg

    def get_func_optional_arg(self, pos, name, default, type_expected=None):
        """Get corresponding function argument from argument list which is optional.
        If user doesn't provide the argument, set it to default value
        """

        if len(self.args) >= pos:
            arg, arg_node = self.args[pos - 1]
        elif name in self.kwargs.keys():
            arg, arg_node = self.kwargs[name]
        else:
            return default

        if type_expected is not None:
            self._type_check(name, arg, arg_node, type_expected)

        return arg

    def _type_check(self, name, arg, arg_node, type_expected):
        try:
            check_type(name, arg, type_expected)
        except TypeError as e:
            self.parser.report_error(str(e), self.parser.current_lineno, arg_node.col_offset)


class GlobalScope:
    pass


class WithScope:
    pass


class ForScope:
    pass


def register_func(class_name, func_name, func_to_register, arg_list, need_parser_and_node, need_return):
    def wrap_func(parser, node, args, kwargs):
        reader = CallArgumentReader(func_name, args, kwargs, parser)
        internal_args = list()
        for i, arg_info in enumerate(arg_list):
            if len(arg_info) == 2:
                arg_name, type_expcted = arg_info
                internal_args.append(reader.get_func_compulsory_arg(i + 1, arg_name, type_expected=type_expcted))
            else:
                arg_name, type_expcted, default = arg_info
                internal_args.append(
                    reader.get_func_optional_arg(i + 1, arg_name, type_expected=type_expcted, default=default))
        if need_parser_and_node:
            internal_args.append(parser)
            internal_args.append(node)

        if need_return:
            return func_to_register(*internal_args)
        else:
            func_to_register(*internal_args)

    setattr(class_name, func_name, wrap_func)


def register_buffer_bind(scope):
    def buffer_bind(var, shape, dtype, name, parser, node):
        """ Intrin function buffer_bind(var, shape, dtype, name)

        e.g.
            A = buffer_bind(a, (128, 128), dtype="float32", name="A")
        <=> A = ib.declare_buffer((128, 128), dtype="float32", name="A")
            buffer_map[a] = A
        """

        if var not in parser.params:
            parser.report_error("Can not bind non-input args to buffer")
        return parser.ir_builder.declare_buffer(shape=shape, dtype=dtype, name=name)

    arg_list = [("var", _expr.Var), ("shape", Tuple[_expr.Expr, ...]), ("dtype", str, "float32"), ("name", str, "buf")]
    register_func(scope, "buffer_bind", buffer_bind, arg_list, need_parser_and_node=True, need_return=True)


def register_buffer_allocate(scope):
    def buffer_allocate(shape, dtype, name, scope, parser, node):
        """ Intrin function buffer_allocate(var, shape, dtype, name)

        e.g.
            A = buffer_allocate((128, 128), dtype="float32", name="A")
        <=> A = ib.allocate_buffer((128, 128), dtype="float32", name="A")
        """

        _buffer = _api.decl_buffer(shape, dtype=dtype, name=name)
        parser.scope_emitter.allocate_stack[-1].append(_make.BufferAllocate(_buffer, scope))
        return Buffer(parser.ir_builder, _buffer, dtype)

    arg_list = [("shape", Tuple[_expr.Expr, ...]), ("dtype", str, "float32"), ("name", str, "buf"), ("scope", str, "")]
    register_func(scope, "buffer_allocate", buffer_allocate, arg_list, need_parser_and_node=True, need_return=True)


def register_block_vars(scope):
    def block_vars(begin, end, name, iter_type, parser, node):
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

    arg_list = [("begin", _expr.Expr), ("end", _expr.Expr), ("name", str, "bv"), ("iter_type", str, "data_par")]
    register_func(scope, "block_vars", block_vars, arg_list, need_parser_and_node=True, need_return=True)


def register_block(scope):
    def block(block_vars, values, reads, writes, predicate, annotations, name, parser, node):
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

    arg_list = [("block_vars", List[_schedule.IterVar]), ("values", List[_expr.Expr]),
                ("reads", Union[TensorRegion, List[TensorRegion]]), ("writes", Union[TensorRegion, List[TensorRegion]]),
                ("predicate", _expr.Expr, True), ("annotations", None, []), ("name", str, "")]
    register_func(scope, "block", block, arg_list, need_parser_and_node=True, need_return=False)


def register_range(scope):
    def range(begin, end, parser, node):
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

    arg_list = [("begin", _expr.Expr), ("end", _expr.Expr)]
    register_func(scope, "range", range, arg_list, need_parser_and_node=True, need_return=False)
