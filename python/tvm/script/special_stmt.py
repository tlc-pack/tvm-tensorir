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
"""TVM Script Parser Special Stmt Classes"""
# pylint: disable=unused-argument, no-self-argument, inconsistent-return-statements
# pylint: disable=relative-beyond-top-level
from synr import ast

import tvm.tir
from tvm import te
from .utils import get_param_list, from_synr_span
from .registry import register


class SpecialStmt:
    """Base class for all Special Stmts"""

    def __init__(self, func, def_symbol):
        self.func = func
        self.def_symbol = def_symbol
        self.node = None
        self.context = None

    def signature(self):
        return "tir." + self.func.__name__, get_param_list(self.func)

    def handle(self, node, context, arg_list, span):
        self.node = node
        self.context = context
        return self.func(*arg_list, span=from_synr_span(span))


@register
class MatchBuffer(SpecialStmt):
    """Special Stmt match_buffer(var, shape, dtype, data, strides, elem_offset, scope, align,
                                 offset_factor, buffer_type)
    Example
    -------
    .. code-block:: python
        A = tir.match_buffer(a, (128, 128), dtype="float32")
    """

    def __init__(self):
        def match_buffer(
            param,
            shape,
            dtype="float32",
            data=None,
            strides=None,
            elem_offset=None,
            scope="global",
            align=-1,
            offset_factor=0,
            buffer_type="default",
            span=None,
        ):
            assert isinstance(self.node, ast.Assign)

            if param not in self.context.func_params:
                self.context.report_error(
                    "Can not bind non-input param to buffer", self.node.rhs.params[0].span
                )
            if strides is None:
                strides = []
            align = align.value if not isinstance(align, int) else align
            offset_factor = (
                offset_factor.value if not isinstance(offset_factor, int) else offset_factor
            )
            buffer = tvm.tir.decl_buffer(
                shape,
                dtype,
                self.node.lhs.id.name,
                data,
                strides,
                elem_offset,
                scope,
                align,
                offset_factor,
                buffer_type,
                span=span,
            )
            self.context.func_buffer_map[param] = buffer
            self.context.update_symbol(self.node.lhs.id.name, buffer)

        super().__init__(match_buffer, def_symbol=True)


@register
class BufferDeclare(SpecialStmt):
    """Special Stmt buffer_decl(shape, dtype, data, strides, elem_offset, scope, align,
                                offset_factor, buffer_type)
    Example
    -------
    .. code-block:: python
        A = tir.buffer_decl((128, 128), dtype="float32")
    """

    def __init__(self):
        def buffer_decl(
            shape,
            dtype="float32",
            data=None,
            strides=None,
            elem_offset=None,
            scope="global",
            align=-1,
            offset_factor=0,
            buffer_type="default",
            span=None,
        ):
            assert isinstance(self.node, ast.Assign)

            if strides is None:
                strides = []
            align = align.value if not isinstance(align, int) else align
            offset_factor = (
                offset_factor.value if not isinstance(offset_factor, int) else offset_factor
            )
            buffer = tvm.tir.decl_buffer(
                shape,
                dtype,
                self.node.lhs.id.name,
                data,
                strides,
                elem_offset,
                scope,
                align,
                offset_factor,
                buffer_type,
                span=span,
            )
            self.context.update_symbol(self.node.lhs.id.name, buffer)
            return buffer

        super().__init__(buffer_decl, def_symbol=True)


@register
class AllocBuffer(SpecialStmt):
    """Special function alloc_buffer(shape, dtype, data, strides, elem_offset, scope, align,
                                     offset_factor, buffer_type)

    Example
    -------
    .. code-block:: python

        A = tir.alloc_buffer((128, 128), dtype="float32")

    """

    def __init__(self):
        def alloc_buffer(
            shape,
            dtype="float32",
            data=None,
            strides=None,
            elem_offset=None,
            scope="",
            align=-1,
            offset_factor=0,
            buffer_type="default",
            span=None,
        ):
            assert isinstance(self.node, ast.Assign)

            if strides is None:
                strides = []
            align = align.value if not isinstance(align, int) else align
            offset_factor = (
                offset_factor.value if not isinstance(offset_factor, int) else offset_factor
            )
            buffer = tvm.tir.decl_buffer(
                shape,
                dtype,
                self.node.lhs.id.name,
                data,
                strides,
                elem_offset,
                scope,
                align,
                offset_factor,
                buffer_type,
                span=span,
            )
            self.context.block_scope().alloc_buffers.append(buffer)
            self.context.update_symbol(self.node.lhs.id.name, buffer)

        super().__init__(alloc_buffer, def_symbol=True)


@register
class BlockVarBind(SpecialStmt):
    """Special function bind(block_iter, binding_value)

    Example
    -------
    .. code-block:: python

        tir.bind(vx, i)

    """

    def __init__(self):
        def bind(block_var, binding, span=None):
            self.context.block_scope().binding[block_var] = binding

        super().__init__(bind, def_symbol=False)


@register
class BlockReads(SpecialStmt):
    """Special function reads([read_buffer_regions])

    Example
    -------
    .. code-block:: python

        tir.reads([A[vi: vi + 4, vk: vk + 4], B[vk: vk + 4, vj]])

    """

    def __init__(self):
        def reads(reads, span=None):
            self.context.block_scope().reads = [reads] if not isinstance(reads, list) else reads

        super().__init__(reads, def_symbol=False)


@register
class BlockWrites(SpecialStmt):
    """Special function writes([write_buffer_regions])

    Example
    -------
    .. code-block:: python

        tir.writes([C[vi: vi + 4, vj])

    """

    def __init__(self):
        def writes(writes, span=None):
            self.context.block_scope().writes = [writes] if not isinstance(writes, list) else writes

        super().__init__(writes, def_symbol=False)


@register
class BlockAttr(SpecialStmt):
    """Special function block_attr({attr_key: attr_value})

    Example
    -------
    .. code-block:: python

        tir.block_attr({"double_buffer_scope": 1})

    """

    def __init__(self):
        def block_attr(attrs, span=None):
            attrs = {
                key: tvm.tir.StringImm(val) if isinstance(val, str) else val
                for key, val in attrs.items()
            }
            self.context.block_scope().annotations = attrs

        super().__init__(block_attr, def_symbol=False)


@register
class BlockPredicate(SpecialStmt):
    """Special function where(predicate)

    Example
    -------
    .. code-block:: python

        tir.where(i < 4)

    """

    def __init__(self):
        def where(predicate, span=None):
            self.context.block_scope().predicate = predicate

        super().__init__(where, def_symbol=False)


@register
class BlockMatchBufferRegion(SpecialStmt):
    """Special function match_buffer_region(source, strides, elem_offset, align, offset_factor)

    Example
    -------
    .. code-block:: python

        B = tir.match_buffer_region(A[0: 4])

    """

    def __init__(self):
        def match_buffer_region(
            source,
            strides=None,
            elem_offset=None,
            align=-1,
            offset_factor=0,
            span=None,
        ):
            assert isinstance(self.node, ast.Assign)

            if strides is None:
                strides = []
            align = align.value if not isinstance(align, int) else align
            offset_factor = (
                offset_factor.value if not isinstance(offset_factor, int) else offset_factor
            )

            shape = []
            if (
                not isinstance(source, tuple)
                or len(source) != 2
                or not isinstance(source[0], tvm.tir.Buffer)
            ):
                self.context.report_error(
                    "match_buffer_region needs a buffer region as source", span=self.node.span
                )
            source_buffer, ranges = source
            region = []
            for index in ranges:
                if isinstance(index, tvm.ir.Range):
                    shape.append(index.extent)
                    region.append(index)
                elif isinstance(index, tvm.tir.PrimExpr):
                    shape.append(1)
                    region.append(tvm.ir.Range.from_min_extent(index, 1))
                else:
                    self.context.report_error(
                        "error during handling source buffer region", span=self.node.span
                    )
            buffer_region = tvm.tir.BufferRegion(source_buffer, region)
            buffer = tvm.tir.decl_buffer(
                shape,
                source_buffer.dtype,
                self.node.lhs.id.name,
                data=None,
                strides=strides,
                elem_offset=elem_offset,
                scope=source_buffer.scope,
                data_alignment=align,
                offset_factor=offset_factor,
                span=span,
            )
            self.context.block_scope().match_buffers.append(
                tvm.tir.MatchBufferRegion(buffer, buffer_region)
            )
            self.context.update_symbol(self.node.lhs.id.name, buffer)

        super().__init__(match_buffer_region, def_symbol=True)


@register
class ExecScope(SpecialStmt):
    """Special function match_buffer_region(source, strides, elem_offset, align, offset_factor)

    Example
    -------
    .. code-block:: python

        B = tir.match_buffer_region(A[0: 4])

    """

    def __init__(self):
        def exec_scope(scope, span=None):
            if not isinstance(scope, str):
                self.context.report_error(
                    "exec_scope expects a str as parameter, but get " + str(type(scope)),
                    span=self.node.span,
                )
            if self.context.block_scope().exec_scope != "":
                self.context.report_error(
                    "get duplicate exec_scope, previous definition is "
                    + self.context.block_scope().exec_scope,
                    span=self.node.span,
                )
            self.context.block_scope().exec_scope = scope

        super().__init__(exec_scope, def_symbol=False)


@register
class VarDef(SpecialStmt):
    """ Special function for defining a Var"""

    def __init__(self):
        def var(dtype, span):
            assert isinstance(self.node, ast.Assign)
            v = te.var(self.node.lhs.id.name, dtype, span=span)
            self.context.update_symbol(v.name, v)

        super().__init__(var, def_symbol=True)


@register
class EnvThread(SpecialStmt):
    """ Bind a var to thread env """

    def __init__(self):
        def env_thread(env_name, span):
            assert isinstance(self.node, ast.Assign)
            v = te.var(self.node.lhs.id.name, span=span)
            self.context.func_var_env_dict[v] = env_name
            self.context.update_symbol(v.name, v)

        super().__init__(env_thread, def_symbol=True)


@register
class FuncAttr(SpecialStmt):
    """Special Stmt for declaring the DictAttr of PrimFunc
    Example
    -------
    .. code-block:: python
         tir.func_attr({"tir.noalias": True, "global_symbol"})
    """

    def __init__(self):
        def func_attr(dict_attr, span):
            self.context.func_dict_attr = dict_attr

        super().__init__(func_attr, def_symbol=False)
