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
"""TVM Script Parser Special Stmt Functions
This module provides the functions registered into parser under special_stmt category.
special_stmt functions don't correspond to an IRNode in the AST directly. It is usually
used for some information that is not suitable to be printed directly.
special_stmt can appear as 2 formats
.. code-block:: python

    target = tir.name():
    tir.name()

When registering a special stmt, the first two arguments must be parser, node
"""
# pylint: disable=unused-argument, no-self-argument, inconsistent-return-statements
from typed_ast import ast3 as ast

import tvm.tir
from .utils import get_param_list
from .intrin import StepIntrin
from .registry import register


class SpecialStmt:
    def __init__(self, func, def_symbol):
        self.func = func
        self.def_symbol = def_symbol
        self.node = None
        self.context = None

    def signature(self):
        return "tir." + self.func.__name__, get_param_list(self.func)

    def handle(self, node, context, arg_list):
        self.node = node
        self.context = context
        return self.func(*arg_list)


@register
class MatchBuffer(SpecialStmt):
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
        ):
            assert isinstance(self.node, ast.Assign)

            if param not in self.context.parser.params:
                self.context.report_error("Can not bind non-input param to buffer")
            if strides is None:
                strides = []
            align = align.value if not isinstance(align, int) else align
            offset_factor = (
                offset_factor.value if not isinstance(offset_factor, int) else offset_factor
            )
            buffer = tvm.tir.decl_buffer(
                shape,
                dtype,
                self.node.targets[0].id,
                data,
                strides,
                elem_offset,
                scope,
                align,
                offset_factor,
                buffer_type,
            )
            self.context.parser.buffer_map[param] = buffer
            self.context.update_symbol(self.node.targets[0].id, buffer)

        super().__init__(match_buffer, def_symbol=True)


@register
class BufferAllocate(SpecialStmt):
    def __init__(self):
        def buffer_allocate(
            shape,
            dtype="float32",
            data=None,
            strides=None,
            elem_offset=None,
            scope="global",
            align=-1,
            offset_factor=0,
            buffer_type="default",
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
                self.node.targets[0].id,
                data,
                strides,
                elem_offset,
                scope,
                align,
                offset_factor,
                buffer_type,
            )
            self.context.block_scope().allocates.append(tvm.tir.BufferAllocate(buffer, scope))
            self.context.update_symbol(self.node.targets[0].id, buffer)

        super().__init__(buffer_allocate, def_symbol=True)


@register
class BlockVarBind(SpecialStmt):
    def __init__(self):
        def bind(block_var, binding):
            self.context.block_scope().binding[block_var] = binding

        super().__init__(bind, def_symbol=False)


@register
class BlockReads(SpecialStmt):
    def __init__(self):
        def reads(reads):
            self.context.block_scope().reads = [reads] if not isinstance(reads, list) else reads

        super().__init__(reads, def_symbol=False)


@register
class BlockWrites(SpecialStmt):
    def __init__(self):
        def writes(writes):
            self.context.block_scope().writes = [writes] if not isinstance(writes, list) else writes

        super().__init__(writes, def_symbol=False)


@register
class BlockAttr(SpecialStmt):
    def __init__(self):
        def block_attr(attrs):
            self.context.block_scope().annotations = attrs

        super().__init__(block_attr, def_symbol=False)


@register
class BlockPredicate(SpecialStmt):
    def __init__(self):
        def where(predicate):
            self.context.block_scope().predicate = predicate

        super().__init__(where, def_symbol=False)


@register
class BufferDeclare(SpecialStmt):
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
        ):
            pass

        super().__init__(buffer_decl, def_symbol=True)


@register
class VarDef(SpecialStmt):
    def __init__(self):
        def var(dtype):
            pass

        super().__init__(var, def_symbol=True)


@register
class EnvThread(SpecialStmt):
    def __init__(self):
        def env_thread(env_name):
            pass

        super().__init__(env_thread, def_symbol=True)


class TVMScriptLambda:
    """TVM Script Lambda, used in lambda expression"""

    def __init__(self, args, body):
        self.args = args
        self.body = body


class TVMScriptReducer:
    """TVM Script Reducer, used in reducer declaration"""

    def __init__(self, combiner, identity):
        self.combiner = combiner
        self.identity = identity
        self.reducer = tvm.tir.CommReducer(
            [self.combiner.args[0]], [self.combiner.args[1]], [self.combiner.body], [self.identity]
        )
        self.step = StepIntrin(self)


@register
class CommReducer(SpecialStmt):
    def __init__(self):
        def comm_reducer(combiner, identity):
            if isinstance(combiner, TVMScriptLambda) and len(combiner.args) == 2:
                assert isinstance(self.node, ast.Assign)
                assert isinstance(self.node.targets[0], ast.Name)
                self.context.update_symbol(
                    self.node.targets[0].id, TVMScriptReducer(combiner, identity)
                )
            else:
                self.context.report_error(
                    "comm_reducer expect a 2-argument lambda function as first argument"
                )

        super().__init__(comm_reducer, def_symbol=True)


@register
class FuncAttr(SpecialStmt):
    def __init__(self):
        def func_attr(dict_attr):
            pass

        super().__init__(func_attr, def_symbol=False)
