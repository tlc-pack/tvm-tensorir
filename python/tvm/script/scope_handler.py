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
    def __init__(self):
        self.body = None

    @staticmethod
    def signature():
        pass

    def enter_scope(self, node, context):
        pass

    def exit_scope(self, node, context, arg_list):
        pass


class WithScopeHandler(ScopeHandler):
    def __init__(self, concise_scope, def_symbol):
        super().__init__()
        self.concise_scope = concise_scope
        self.def_symbol = def_symbol


@register
class Block(WithScopeHandler):
    def __init__(self):
        super().__init__(concise_scope=False, def_symbol=True)

    @staticmethod
    def signature():
        def block(axes=None, name="", exec_scope=""):
            pass

        return "block", get_param_list(block)

    def enter_scope(self, node, context):
        # define block vars
        pass

    def exit_scope(self, node, context, arg_list):
        # construct a BlockRealize(Block)
        pass


@register
class Allocate(WithScopeHandler):
    def __init__(self):
        super().__init__(concise_scope=True, def_symbol=True)

    @staticmethod
    def signature():
        def allocate(extents, dtype, scope, condition=True):
            pass

        return "allocate", get_param_list(allocate)

    def enter_scope(self, node, context):
        # define buffer vars in symbol table
        pass

    def exit_scope(self, node, context, arg_list):
        # construct an Attr(Allocate())
        pass


@register
class LaunchThread(WithScopeHandler):
    def __init__(self):
        super().__init__(concise_scope=True, def_symbol=False)

    @staticmethod
    def signature():
        def launch_thread(env_var, extent):
            pass

        return "launch_thread", get_param_list(launch_thread)

    def enter_scope(self, node, context):
        # do nothing
        pass

    def exit_scope(self, node, context, arg_list):
        # construct Attr(IterVar)
        pass


@register
class Realize(WithScopeHandler):
    def __init__(self):
        super().__init__(concise_scope=True, def_symbol=False)

    @staticmethod
    def signature():
        def realize(buffer_bounds, scope, condition=True):
            pass

        return "realize", get_param_list(realize)

    def enter_scope(self, node, context):
        # do nothing
        pass

    def exit_scope(self, node, context, arg_list):
        # construct Attr(BufferRealize)
        pass


@register
class Attr(WithScopeHandler):
    def __init__(self):
        super().__init__(concise_scope=True, def_symbol=False)

    @staticmethod
    def signature():
        def attr(attr_key, value):
            pass

        return "attr", get_param_list(attr)

    def enter_scope(self, node, context):
        # do nothing
        pass

    def exit_scope(self, node, context, arg_list):
        # construct Attr
        pass


@register
class Assert(WithScopeHandler):
    def __init__(self):
        super().__init__(concise_scope=True, def_symbol=False)

    @staticmethod
    def signature():
        def assert_sig(condition, message):
            pass

        return "Assert", get_param_list(assert_sig)

    def enter_scope(self, node, context):
        # do nothing
        pass

    def exit_scope(self, node, context, arg_list):
        # construct Assert
        pass


@register
class Let(WithScopeHandler):
    def __init__(self):
        super().__init__(concise_scope=False, def_symbol=False)

    @staticmethod
    def signature():
        def let(var, value):
            pass

        return "let", get_param_list(let)

    def enter_scope(self, node, context):
        # do nothing
        pass

    def exit_scope(self, node, context, arg_list):
        # construct Let
        pass


class ForScopeHandler(ScopeHandler):
    def __init__(self):
        super().__init__()


@register
class Serial(ForScopeHandler):
    def __init__(self):
        super().__init__()

    @staticmethod
    def signature():
        def serial(begin, end):
            pass

        return "serial", get_param_list(serial)

    def enter_scope(self, node, context):
        super().enter_scope(node, context)

    def exit_scope(self, node, context, arg_list):
        super().exit_scope(node, context, arg_list)


@register
class Parallel(ForScopeHandler):
    def __init__(self):
        super().__init__()

    @staticmethod
    def signature():
        def parallel(begin, end):
            pass

        return "parallel", get_param_list(parallel)

    def enter_scope(self, node, context):
        super().enter_scope(node, context)

    def exit_scope(self, node, context, arg_list):
        super().exit_scope(node, context, arg_list)


@register
class Vectorized(ForScopeHandler):
    def __init__(self):
        super().__init__()

    @staticmethod
    def signature():
        def vectorized(begin, end):
            pass

        return "vectorized", get_param_list(vectorized)

    def enter_scope(self, node, context):
        super().enter_scope(node, context)

    def exit_scope(self, node, context, arg_list):
        super().exit_scope(node, context, arg_list)


@register
class Unroll(ForScopeHandler):
    def __init__(self):
        super().__init__()

    @staticmethod
    def signature():
        def unroll(begin, end):
            pass

        return "unroll", get_param_list(unroll)

    def enter_scope(self, node, context):
        super().enter_scope(node, context)

    def exit_scope(self, node, context, arg_list):
        super().exit_scope(node, context, arg_list)


@register
class Range(ForScopeHandler):
    def __init__(self):
        super().__init__()

    @staticmethod
    def signature():
        def loop_range(begin, end, annotation=None):
            pass

        return "Range", get_param_list(loop_range)

    def enter_scope(self, node, context):
        super().enter_scope(node, context)

    def exit_scope(self, node, context, arg_list):
        super().exit_scope(node, context, arg_list)


@register
class Grid(ForScopeHandler):
    def __init__(self):
        super().__init__()

    @staticmethod
    def signature():
        def grid(*extents):
            pass

        return "grid", get_param_list(grid)

    def enter_scope(self, node, context):
        super().enter_scope(node, context)

    def exit_scope(self, node, context, arg_list):
        super().exit_scope(node, context, arg_list)
