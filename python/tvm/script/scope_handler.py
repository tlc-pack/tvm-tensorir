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
"""TVM Script Parser Scope Handler Classes"""
# pylint: disable=redefined-builtin, unused-argument, invalid-name

import inspect
from typed_ast import ast3 as ast
import tvm.tir
from .utils import get_param_list
from .registry import register


class ScopeHandler:
    def __init__(self, func):
        self.func = func
        self.body = None

    def signature(self):
        return "tir." + self.func.__name__, get_param_list(self.func)

    def enter_scope(self, node, context):
        pass

    def exit_scope(self, node, context, arg_list):
        pass


class WithScopeHandler(ScopeHandler):
    def __init__(self, func, concise_scope, def_symbol):
        super().__init__(func)
        self.concise_scope = concise_scope
        self.def_symbol = def_symbol


@register
class Block(WithScopeHandler):
    def __init__(self):
        def block(axes=None, name="", exec_scope=""):
            pass

        super().__init__(func=block, concise_scope=False, def_symbol=True)

    def enter_scope(self, node, context):
        # define block vars
        pass

    def exit_scope(self, node, context, arg_list):
        # construct a BlockRealize(Block)
        pass


@register
class Allocate(WithScopeHandler):
    def __init__(self):
        def allocate(extents, dtype, scope, condition=True):
            pass

        super().__init__(allocate, concise_scope=True, def_symbol=True)

    def enter_scope(self, node, context):
        # define buffer vars in symbol table
        pass

    def exit_scope(self, node, context, arg_list):
        # construct an Attr(Allocate())
        pass


@register
class LaunchThread(WithScopeHandler):
    def __init__(self):
        def launch_thread(env_var, extent):
            pass

        super().__init__(launch_thread, concise_scope=True, def_symbol=False)

    def enter_scope(self, node, context):
        # do nothing
        pass

    def exit_scope(self, node, context, arg_list):
        # construct Attr(IterVar)
        pass


@register
class Realize(WithScopeHandler):
    def __init__(self):
        def realize(buffer_bounds, scope, condition=True):
            pass

        super().__init__(realize, concise_scope=True, def_symbol=False)

    def enter_scope(self, node, context):
        # do nothing
        pass

    def exit_scope(self, node, context, arg_list):
        # construct Attr(BufferRealize)
        pass


@register
class Attr(WithScopeHandler):
    def __init__(self):
        def attr(attr_key, value):
            pass

        super().__init__(attr, concise_scope=True, def_symbol=False)

    def enter_scope(self, node, context):
        # do nothing
        pass

    def exit_scope(self, node, context, arg_list):
        # construct Attr
        pass


@register
class AssertHandler(WithScopeHandler):
    def __init__(self):
        def Assert(condition, message):
            pass

        super().__init__(Assert, concise_scope=True, def_symbol=False)

    def enter_scope(self, node, context):
        # do nothing
        pass

    def exit_scope(self, node, context, arg_list):
        # construct Assert
        pass


@register
class Let(WithScopeHandler):
    def __init__(self):
        def let(var, value):
            pass

        super().__init__(let, concise_scope=False, def_symbol=False)

    def enter_scope(self, node, context):
        # do nothing
        pass

    def exit_scope(self, node, context, arg_list):
        # construct Let
        pass


class ForScopeHandler(ScopeHandler):
    def __init__(self, func):
        super().__init__(func)


@register
class Serial(ForScopeHandler):
    def __init__(self):
        def serial(begin, end):
            pass

        super().__init__(serial)

    def enter_scope(self, node, context):
        super().enter_scope(node, context)

    def exit_scope(self, node, context, arg_list):
        super().exit_scope(node, context, arg_list)


@register
class Parallel(ForScopeHandler):
    def __init__(self):
        def parallel(begin, end):
            pass

        super().__init__(parallel)

    def enter_scope(self, node, context):
        super().enter_scope(node, context)

    def exit_scope(self, node, context, arg_list):
        super().exit_scope(node, context, arg_list)


@register
class Vectorized(ForScopeHandler):
    def __init__(self):
        def vectorized(begin, end):
            pass

        super().__init__(vectorized)

    def enter_scope(self, node, context):
        super().enter_scope(node, context)

    def exit_scope(self, node, context, arg_list):
        super().exit_scope(node, context, arg_list)


@register
class Unroll(ForScopeHandler):
    def __init__(self):
        def unroll(begin, end):
            pass

        super().__init__(unroll)

    def enter_scope(self, node, context):
        super().enter_scope(node, context)

    def exit_scope(self, node, context, arg_list):
        super().exit_scope(node, context, arg_list)


@register
class RangeHandler(ForScopeHandler):
    def __init__(self):
        def Range(begin, end, annotation=None):
            pass

        super().__init__(Range)

    def enter_scope(self, node, context):
        super().enter_scope(node, context)

    def exit_scope(self, node, context, arg_list):
        super().exit_scope(node, context, arg_list)


@register
class Grid(ForScopeHandler):
    def __init__(self):
        def grid(*extents):
            pass

        super().__init__(grid)

    def enter_scope(self, node, context):
        super().enter_scope(node, context)

    def exit_scope(self, node, context, arg_list):
        super().exit_scope(node, context, arg_list)
