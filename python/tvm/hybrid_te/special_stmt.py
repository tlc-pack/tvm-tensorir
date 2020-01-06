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
from ..ir_builder import Buffer


class SpecialStmt:
    pass


def register_special_stmt(origin_func):
    registry.register_func(SpecialStmt, origin_func, need_parser_and_node=True, need_return=True)


def buffer_bind(parser, node, var, shape, dtype="float32", name="buf"):
    """ Special function buffer_bind(var, shape, dtype, name)

    e.g.
        A = buffer_bind(a, (128, 128), dtype="float32", name="A")
    <=> A = ib.declare_buffer((128, 128), dtype="float32", name="A")
        buffer_map[a] = A
    """
    if var not in parser.params:
        parser.report_error("Can not bind non-input args to buffer")
    return parser.ir_builder.declare_buffer(shape=shape, dtype=dtype, name=name)


def buffer_allocate(parser, node, shape, dtype="float32", name="buf", scope=""):
    """ Special function buffer_allocate(var, shape, dtype, name)

    e.g.
        A = buffer_allocate((128, 128), dtype="float32", name="A")
    <=> A = ib.allocate_buffer((128, 128), dtype="float32", name="A")
    """
    _buffer = _api.decl_buffer(shape, dtype=dtype, name=name)
    parser.scope_emitter.allocate_stack[-1].append(_make.BufferAllocate(_buffer, scope))
    return Buffer(parser.ir_builder, _buffer, dtype)


def block_vars(parser, node, begin, end, name="bv", iter_type="data_par"):
    """ Special function buffer_bind(var, shape, dtype, name)

    e.g.
        vi(0, 128, iter_type="reduce")
    <=> ib.IterVar(tvm.make_range_by_min_text(0, 128), name="vi", iter_type="reduce")
    """
    extent = end if begin == 0 else _pass.Simplify(end - begin)
    block_var_dom = _make.range_by_min_extent(begin, extent)
    block_var = parser.ir_builder.iter_var(block_var_dom, name=name, iter_type=iter_type)
    parser.add_symbol(block_var.var.name, parser.Symbol.IterVar, block_var.var)
    return block_var
