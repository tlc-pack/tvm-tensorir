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
        self.node = None
        self.context = None

    def signature(self):
        return "tir." + self.func.__name__, get_param_list(self.func)

    def enter_scope(self, node, context):
        pass

    def exit_scope(self, node, context, arg_list):
        self.node = node
        self.context = context
        return self.func(*arg_list)


class WithScopeHandler(ScopeHandler):
    def __init__(self, func, concise_scope, def_symbol):
        super().__init__(func)
        self.concise_scope = concise_scope
        self.def_symbol = def_symbol


@register
class Block(WithScopeHandler):
    def __init__(self):
        def block(axes=None, name="", exec_scope=""):
            block_info = self.context.block_scope()
            if axes is None:
                axes = []
            if len(axes) != len(self.block_vars):
                self.context.report_error("Inconsistent number of block vars")
            block_iters = []
            for i in range(len(axes)):
                axis = tvm.runtime.convert(axes[i])
                if isinstance(axis, tvm.tir.PrimExpr):
                    block_var_dom = tvm.ir.Range.from_min_extent(0, axis)
                    block_iters.append(tvm.tir.IterVar(block_var_dom, self.block_vars[i], 0))
                elif isinstance(axis, tvm.ir.Range):
                    block_iters.append(tvm.tir.IterVar(axis, self.block_vars[i], 0))
                elif isinstance(axis, tvm.tir.IterVar):
                    block_iters.append(
                        tvm.tir.IterVar(axis.dom, self.block_vars[i], axis.iter_type)
                    )
                else:
                    self.context.report_error("Invalid argument of tir.block()")
            # create block IO info
            if block_info.reads is None:
                reads = None
            else:
                reads = []
                for read in block_info.reads:
                    if isinstance(read, tvm.tir.BufferLoad):
                        doms = []
                        for index in read.indices:
                            doms.append(tvm.ir.Range.from_min_extent(index, 1))
                        reads.append(tvm.tir.TensorRegion(read.buffer, doms))
                    else:
                        reads.append(read)
            if block_info.writes is None:
                writes = None
            else:
                writes = []
                for write in block_info.writes:
                    if isinstance(write, tvm.tir.BufferLoad):
                        doms = []
                        for index in write.indices:
                            doms.append(tvm.ir.Range.from_min_extent(index, 1))
                        writes.append(tvm.tir.TensorRegion(write.buffer, doms))
                    else:
                        writes.append(write)
            inner = tvm.tir.Block(
                block_iters,
                reads,
                writes,
                self.body,
                block_info.allocates,
                block_info.annotations,
                name,
            )
            # create block var binding
            if not block_info.binding:
                values = self.context.loop_stack[-1].copy()
                if len(values) == 0:
                    values = [None] * len(block_iters)
                elif len(values) != len(block_iters):
                    self.context.report_error(
                        "Autocomplete block var binding expect larger number of loops"
                    )
            else:
                for block_var in self.block_vars:
                    if block_var not in block_info.binding:
                        self.context.report_error("Missing block var binding for " + block_var.name)
                values = [block_info.binding[block_var] for block_var in self.block_vars]

            body = tvm.tir.BlockRealize(values, block_info.predicate, inner, exec_scope)
            return body

        super().__init__(func=block, concise_scope=False, def_symbol=True)
        self.block_vars = None

    def enter_scope(self, node, context):
        # define block vars
        assert isinstance(node, ast.With)

        var_names = None
        if isinstance(node.items[0].optional_vars, ast.Name):
            var_names = [node.items[0].optional_vars.id]
        elif isinstance(node.items[0].optional_vars, (ast.List, ast.Tuple)):
            for var in node.items[0].optional_vars.elts:
                if not isinstance(var, ast.Name):
                    context.parser.report_error("Invalid optional var definition")
            var_names = [var.id for var in node.items[0].optional_vars.elts]
        else:
            context.parser.report_error("Invalid optional var definition")

        self.block_vars = [tvm.te.var(name) for name in var_names]
        for block_var in self.block_vars:
            context.update_symbol(block_var.name, block_var)


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
