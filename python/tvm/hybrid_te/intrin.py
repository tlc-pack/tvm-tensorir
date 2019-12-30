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
from ..ir_builder import Buffer


class Symbol:
    """Enumerates types in the symbol table"""
    Var = 0
    Buffer = 1
    IterVar = 2
    LoopVar = 3
    ListOfTensorRegions = 4


def _get_func_compulsory_arg(args, kwargs, pos, func_name, name, parser):
    """Get corresponding function argument from argument list which is compulsory"""
    if len(args) >= pos:
        return args[pos - 1]
    else:
        if name not in kwargs.keys():
            parser.report_error(func_name + " misses argument " + name)
        return kwargs[name]


def _get_func_optional_arg(args, kwargs, pos, func_name, name, parser, default):
    """Get corresponding function argument from argument list which is optional.
    If user doesn't provide the argument, set it to default value
    """
    if len(args) >= pos:
        return args[pos - 1]
    else:
        if name in kwargs.keys():
            return kwargs[name]
        else:
            return default


def _wrap_list(sth):
    if not isinstance(sth, list):
        return [sth]
    return sth


def _type_check(arg, type_expected, parser):
    if not isinstance(arg, type_expected):
        parser.report_error(str(type_expected) + " expected while " + str(type(arg)) + " found")


def _type_check_list(args, type_expected, parser):
    for arg in args:
        _type_check(arg, type_expected, parser)


def _buffer_bind(parser, node, args, kwargs):
    """ Intrin function buffer_bind(var, shape, dtype, name)

    e.g.
        A = buffer_bind(a, (128, 128), dtype="float32", name="A")
    <=> A = ib.declare_buffer((128, 128), dtype="float32", name="A")
        buffer_map[a] = A
    """

    # var
    var = _get_func_compulsory_arg(args, kwargs, 1, "buffer_bind", "var", parser)
    _type_check(var, _expr.Var, parser)
    if var not in parser.params:
        parser.report_error("Can not bind non-input args to buffer")
    # shape
    shape = _get_func_compulsory_arg(args, kwargs, 2, "buffer_bind", "shape", parser)
    _type_check(shape, tuple, parser)
    _type_check_list(shape, _expr.Expr, parser)
    # dtype
    dtype = _get_func_optional_arg(args, kwargs, 3, "buffer_bind", "dtype", parser, "float32")
    _type_check(dtype, str, parser)
    # name
    name = _get_func_optional_arg(args, kwargs, 4, "buffer_bind", "name", parser, "buf")
    _type_check(name, str, parser)

    return parser.ir_builder.declare_buffer(shape=shape, dtype=dtype, name=name)


buffer_bind = _buffer_bind


def _buffer_allocate(parser, node, args, kwargs):
    """ Intrin function buffer_allocate(var, shape, dtype, name)

    e.g.
        A = buffer_allocate((128, 128), dtype="float32", name="A")
    <=> A = ib.allocate_buffer((128, 128), dtype="float32", name="A")
    """

    # shape
    shape = _get_func_compulsory_arg(args, kwargs, 1, "buffer_allocate", "shape", parser)
    _type_check(shape, tuple, parser)
    _type_check_list(shape, _expr.Expr, parser)
    # dtype
    dtype = _get_func_optional_arg(args, kwargs, 2, "buffer_allocate", "dtype", parser, "float32")
    _type_check(dtype, str, parser)
    # name
    name = _get_func_optional_arg(args, kwargs, 3, "buffer_allocate", "name", parser, "buf")
    _type_check(name, str, parser)
    # scope
    scope = _get_func_optional_arg(args, kwargs, 4, "buffer_allocate", "scope", parser, "")
    _type_check(scope, str, parser)

    _buffer = _api.decl_buffer(shape, dtype=dtype, name=name)
    parser.allocate_stack[-1].append(_make.BufferAllocate(_buffer, scope))
    return Buffer(parser.ir_builder, _buffer, dtype)


buffer_allocate = _buffer_allocate


def _block_vars(parser, node, args, kwargs):
    """ Intrin function buffer_bind(var, shape, dtype, name)

    e.g.
        vi(0, 128, iter_type="reduce")
    <=> ib.IterVar(tvm.make_range_by_min_text(0, 128), name="vi", iter_type="reduce")
    """

    # begin
    begin = _get_func_compulsory_arg(args, kwargs, 1, "block_vars", "begin", parser)
    _type_check(begin, _expr.Expr, parser)
    # end
    end = _get_func_compulsory_arg(args, kwargs, 2, "block_vars", "end", parser)
    _type_check(end, _expr.Expr, parser)
    # name
    name = _get_func_optional_arg(args, kwargs, 3, "block_vars", "name", parser, "bv")
    _type_check(name, str, parser)
    # iter_type
    iter_type = _get_func_optional_arg(args, kwargs, 4, "block_vars", "iter_type", parser, "data_par")

    extent = end if begin == 0 else _pass.Simplify(end - begin)

    block_var_dom = _make.range_by_min_extent(begin, extent)
    block_var = parser.ir_builder.iter_var(block_var_dom, name=name, iter_type=iter_type)

    parser.add_symbol(block_var.var.name, Symbol.IterVar, block_var.var)

    return block_var


block_vars = _block_vars


class With:
    """All the functions supported in With stmt are registered here"""

    @staticmethod
    def _block(parser, node, args, kwargs):
        """ Intrin function block(block_vars, values, reads, writes, predicate, annotations, name)

        e.g.
            with block([vi(0, 128), vj(0, 128)], [i, j], reads=[], writes=C[vi : vi + 1, vj : vj + 1], name="init"):
        <=> with ib.block([vi, vj], [i, j], reads=[], writes=C[vi : vi + 1, vj : vj + 1], name="init"):
            (Note that block_vars has been processed ahead)
        """

        # block_vars
        block_vars = _get_func_compulsory_arg(args, kwargs, 1, "block", "block_vars", parser)
        block_vars = _wrap_list(block_vars)
        # values
        values = _get_func_compulsory_arg(args, kwargs, 2, "block", "values", parser)
        values = _wrap_list(values)
        _type_check_list(values, _expr.Expr, parser)
        # reads
        reads = _get_func_compulsory_arg(args, kwargs, 3, "block", "reads", parser)
        reads = _wrap_list(reads)
        # writes
        writes = _get_func_compulsory_arg(args, kwargs, 4, "block", "writes", parser)
        writes = _wrap_list(writes)
        # predicate
        predicate = _get_func_optional_arg(args, kwargs, 5, "block", "predicate", parser, True)
        # annotations
        annotations = _get_func_optional_arg(args, kwargs, 6, "block", "annotations", parser, [])
        annotations = _wrap_list(annotations)
        # name
        name = _get_func_optional_arg(args, kwargs, 7, "block", "name", parser, "")

        parser.seq_stack.append([])
        parser.allocate_stack.append([])

        for stmt in node.body:
            parser.visit(stmt)

        for block_var in block_vars:
            parser.remove_symbol(block_var.var.name)

        parser.emit(
            _make.TeBlock(block_vars, values, reads, writes, parser.pop_seq(), predicate, parser.allocate_stack.pop(),
                          annotations, name))

    block = _block


class For:
    """All the functions supported in For stmt are registered here"""

    @staticmethod
    def _range(parser, node, args, kwargs):
        """ Intrin function range(begin, end)"""

        # begin
        begin = _get_func_compulsory_arg(args, kwargs, 1, "range", "begin", parser)
        _type_check(begin, _expr.Expr, parser)
        # end
        end = _get_func_compulsory_arg(args, kwargs, 2, "range", "end", parser)
        _type_check(end, _expr.Expr, parser)

        extent = end if begin == 0 else _pass.Simplify(end - begin)

        loop_var_name = node.target.id
        loop_var = _api.var(loop_var_name, dtype="int32")
        parser.add_symbol(loop_var_name, Symbol.LoopVar, loop_var)
        parser.seq_stack.append([])

        for stmt in node.body:
            parser.visit(stmt)

        parser.emit(_make.Loop(loop_var, begin, extent, [], parser.pop_seq()))
        parser.remove_symbol(loop_var_name)

    range = _range
