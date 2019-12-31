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

    def get_func_compulsory_arg(self, pos, name, wrap_list=False, type_expected=None, elem_type_expected=None):
        """Get corresponding function argument from argument list which is compulsory"""

        if len(self.args) >= pos:
            arg, arg_node = self.args[pos - 1]
        elif name not in self.kwargs.keys():
            self.parser.report_error(self.func_name + " misses argument " + name)
            return
        else:
            arg, arg_node = self.kwargs[name]

        if wrap_list:
            arg = self._wrap_list(arg)

        if type_expected is not None:
            self._type_check(arg, arg_node, type_expected)

        if elem_type_expected is not None:
            self._type_check(arg, arg_node, (list, tuple))
            self._type_check_list(arg, arg_node, elem_type_expected)

        return arg

    def get_func_optional_arg(self, pos, name, default, wrap_list=False, type_expected=None, elem_type_expected=None):
        """Get corresponding function argument from argument list which is optional.
        If user doesn't provide the argument, set it to default value
        """
        if len(self.args) >= pos:
            arg, arg_node = self.args[pos - 1]
        elif name in self.kwargs.keys():
            arg, arg_node = self.kwargs[name]
        else:
            return default

        if wrap_list:
            arg = self._wrap_list(arg)

        if type_expected is not None:
            self._type_check(arg, arg_node, type_expected)

        if elem_type_expected is not None:
            self._type_check(arg, arg_node, (list, tuple))
            self._type_check_list(arg, arg_node, elem_type_expected)

        return arg

    @staticmethod
    def _wrap_list(sth):
        if not isinstance(sth, list):
            return [sth]
        return sth

    def _type_check(self, arg, arg_node, type_expected):
        if not isinstance(arg, type_expected):
            self.parser.report_error(str(type_expected) + " expected while " + str(type(arg[0])) + " found",
                                     self.parser.current_lineno, arg_node.col_offset)

    def _type_check_list(self, args, arg_node, type_expected):
        for arg in args:
            self._type_check(arg, arg_node, type_expected)


def buffer_bind(parser, node, args, kwargs):
    """ Intrin function buffer_bind(var, shape, dtype, name)

    e.g.
        A = buffer_bind(a, (128, 128), dtype="float32", name="A")
    <=> A = ib.declare_buffer((128, 128), dtype="float32", name="A")
        buffer_map[a] = A
    """

    reader = CallArgumentReader("buffer_bind", args, kwargs, parser)
    var = reader.get_func_compulsory_arg(1, "var", type_expected=_expr.Var)
    shape = reader.get_func_compulsory_arg(2, "shape", type_expected=tuple, elem_type_expected=_expr.Expr)
    dtype = reader.get_func_optional_arg(3, "dtype", "float32", type_expected=str)
    name = reader.get_func_optional_arg(4, "name", "buf", type_expected=str)

    if var not in parser.params:
        parser.report_error("Can not bind non-input args to buffer")

    return parser.ir_builder.declare_buffer(shape=shape, dtype=dtype, name=name)


def buffer_allocate(parser, node, args, kwargs):
    """ Intrin function buffer_allocate(var, shape, dtype, name)

    e.g.
        A = buffer_allocate((128, 128), dtype="float32", name="A")
    <=> A = ib.allocate_buffer((128, 128), dtype="float32", name="A")
    """

    reader = CallArgumentReader("buffer_allocate", args, kwargs, parser)
    shape = reader.get_func_compulsory_arg(1, "shape", type_expected=tuple, elem_type_expected=_expr.Expr)
    dtype = reader.get_func_optional_arg(2, "dtype", "float32", type_expected=str)
    name = reader.get_func_optional_arg(3, "name", "buf", type_expected=str)
    scope = reader.get_func_optional_arg(4, "scope", "", type_expected=str)

    _buffer = _api.decl_buffer(shape, dtype=dtype, name=name)
    parser.scope_emitter.allocate_stack[-1].append(_make.BufferAllocate(_buffer, scope))
    return Buffer(parser.ir_builder, _buffer, dtype)


def block_vars(parser, node, args, kwargs):
    """ Intrin function buffer_bind(var, shape, dtype, name)

    e.g.
        vi(0, 128, iter_type="reduce")
    <=> ib.IterVar(tvm.make_range_by_min_text(0, 128), name="vi", iter_type="reduce")
    """

    reader = CallArgumentReader("block_var", args, kwargs, parser)
    begin = reader.get_func_compulsory_arg(1, "begin", type_expected=_expr.Expr)
    end = reader.get_func_compulsory_arg(2, "end", type_expected=_expr.Expr)
    name = reader.get_func_optional_arg(3, "name", "bv", type_expected=str)
    iter_type = reader.get_func_optional_arg(4, "iter_type", "data_par", type_expected=str)

    extent = end if begin == 0 else _pass.Simplify(end - begin)
    block_var_dom = _make.range_by_min_extent(begin, extent)
    block_var = parser.ir_builder.iter_var(block_var_dom, name=name, iter_type=iter_type)

    parser.add_symbol(block_var.var.name, Symbol.IterVar, block_var.var)

    return block_var


class With:
    """All the functions supported in With stmt are registered here"""

    @staticmethod
    def block(parser, node, args, kwargs):
        """ Intrin function block(block_vars, values, reads, writes, predicate, annotations, name)

        e.g.
            with block([vi(0, 128), vj(0, 128)], [i, j], reads=[], writes=C[vi : vi + 1, vj : vj + 1], name="init"):
        <=> with ib.block([vi, vj], [i, j], reads=[], writes=C[vi : vi + 1, vj : vj + 1], name="init"):
            (Note that block_vars has been processed ahead)
        """

        reader = CallArgumentReader("block", args, kwargs, parser)
        block_vars = reader.get_func_compulsory_arg(1, "block_vars", wrap_list=True,
                                                    elem_type_expected=_schedule.IterVar)
        values = reader.get_func_compulsory_arg(2, "values", wrap_list=True, elem_type_expected=_expr.Expr)
        reads = reader.get_func_compulsory_arg(3, "reads", wrap_list=True, elem_type_expected=TensorRegion)
        writes = reader.get_func_compulsory_arg(4, "writes", wrap_list=True, elem_type_expected=TensorRegion)
        predicate = reader.get_func_optional_arg(5, "predicate", True, type_expected=_expr.Expr)
        annotations = reader.get_func_optional_arg(6, "annotations", [], wrap_list=True)
        name = reader.get_func_optional_arg(7, "name", "", type_expected=str)

        parser.scope_emitter.new_block_scope()

        for stmt in node.body:
            parser.visit(stmt)

        for block_var in block_vars:
            parser.remove_symbol(block_var.var.name)

        parser.scope_emitter.emit(
            _make.TeBlock(block_vars, values, reads, writes, parser.scope_emitter.pop_seq(), predicate,
                          parser.scope_emitter.allocate_stack.pop(), annotations, name))


class For:
    """All the functions supported in For stmt are registered here"""

    @staticmethod
    def range(parser, node, args, kwargs):
        """ Intrin function range(begin, end)"""

        reader = CallArgumentReader("range", args, kwargs, parser)
        begin = reader.get_func_compulsory_arg(1, "begin", type_expected=_expr.Expr)
        end = reader.get_func_compulsory_arg(2, "end", type_expected=_expr.Expr)

        extent = end if begin == 0 else _pass.Simplify(end - begin)

        loop_var_name = node.target.id
        loop_var = _api.var(loop_var_name, dtype="int32")
        parser.add_symbol(loop_var_name, Symbol.LoopVar, loop_var)

        parser.scope_emitter.new_loop_scope()

        for stmt in node.body:
            parser.visit(stmt)

        parser.scope_emitter.emit(_make.Loop(loop_var, begin, extent, [], parser.scope_emitter.pop_seq()))
        parser.remove_symbol(loop_var_name)
