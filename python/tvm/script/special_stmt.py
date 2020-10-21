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
from .intrin import Intrin
from .registry import register


class SpecialStmt:
    def __init__(self, def_symbol):
        self.def_symbol = def_symbol

    @staticmethod
    def signature():
        pass

    def handle(self, node, context, arg_list):
        pass


@register
class MatchBuffer(SpecialStmt):
    def __init__(self):
        super().__init__(def_symbol=True)

    @staticmethod
    def signature():
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
            pass

        return "match_buffer", get_param_list(match_buffer)

    def handle(self, node, context, arg_list):
        pass


@register
class BufferAllocate(SpecialStmt):
    def __init__(self):
        super().__init__(def_symbol=True)

    @staticmethod
    def signature():
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
            pass

        return "buffer_allocate", get_param_list(buffer_allocate)

    def handle(self, node, context, arg_list):
        pass


@register
class BlockVarBind(SpecialStmt):
    def __init__(self):
        super().__init__(def_symbol=False)

    @staticmethod
    def signature():
        def bind(block_var, binding):
            pass

        return "bind", get_param_list(bind)

    def handle(self, node, context, arg_list):
        pass


@register
class BlockReads(SpecialStmt):
    def __init__(self):
        super().__init__(def_symbol=False)

    @staticmethod
    def signature():
        def reads(reads):
            pass

        return "reads", get_param_list(reads)

    def handle(self, node, context, arg_list):
        pass


@register
class BlockWrites(SpecialStmt):
    def __init__(self):
        super().__init__(def_symbol=False)

    @staticmethod
    def signature():
        def writes(writes):
            pass

        return "writes", get_param_list(writes)

    def handle(self, node, context, arg_list):
        pass


@register
class BlockAttr(SpecialStmt):
    def __init__(self):
        super().__init__(def_symbol=False)

    @staticmethod
    def signature():
        def block_attr(attrs):
            pass

        return "block_attr", get_param_list(block_attr)

    def handle(self, node, context, arg_list):
        pass


@register
class BlockPredicate(SpecialStmt):
    def __init__(self):
        super().__init__(def_symbol=False)

    @staticmethod
    def signature():
        def where(predicate):
            pass

        return "where", get_param_list(where)

    def handle(self, node, context, arg_list):
        pass


@register
class BufferDeclare(SpecialStmt):
    def __init__(self):
        super().__init__(def_symbol=True)

    @staticmethod
    def signature():
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

        return "buffer_decl", get_param_list(buffer_decl)

    def handle(self, node, context, arg_list):
        pass


@register
class VarDef(SpecialStmt):
    def __init__(self):
        super().__init__(def_symbol=True)

    @staticmethod
    def signature():
        def var(dtype):
            pass

        return "var", get_param_list(var)

    def handle(self, node, context, arg_list):
        pass


@register
class EnvThread(SpecialStmt):
    def __init__(self):
        super().__init__(def_symbol=True)

    @staticmethod
    def signature():
        def env_thread(env_name):
            pass

        return "env_thread", get_param_list(env_thread)

    def handle(self, node, context, arg_list):
        pass


class TVMScriptLambda:
    """TVM Script Lambda, used in lambda expression"""

    def __init__(self, args, body):
        self.args = args
        self.body = body


class TVMScriptReducer:
    """TVM Script Reducer, used in reducer declaration"""

    class StepIntrin(Intrin):
        def __init__(self, reducer):
            def intrin(lhs, rhs):
                return tvm.tir.ReduceStep(self.reducer, lhs, rhs)

            super().__init__(intrin)
            self.reducer = reducer

        def signature(self):
            return "TVMScriptReducer.step", get_param_list(self.intrin)

    def __init__(self, combiner, identity):
        self.combiner = combiner
        self.identity = identity
        self.reducer = tvm.tir.CommReducer(
            [self.combiner.args[0]], [self.combiner.args[1]], [self.combiner.body], [self.identity]
        )
        self.step = TVMScriptReducer.StepIntrin(self)


@register
class CommReducer(SpecialStmt):
    def __init__(self):
        super().__init__(def_symbol=True)

    @staticmethod
    def signature():
        def comm_reducer(combiner, identity):
            pass

        return "comm_reducer", get_param_list(comm_reducer)

    def handle(self, node, context, arg_list):
        pass


@register
class FuncAttr(SpecialStmt):
    def __init__(self):
        super().__init__(def_symbol=False)

    @staticmethod
    def signature():
        def func_attr(dict_attr):
            pass

        return "func_attr", get_param_list(func_attr)

    def handle(self, node, context, arg_list):
        pass
