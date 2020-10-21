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

    @staticmethod
    def get_optional_var_names(node, context):
        assert isinstance(node, ast.With)

        var_names = None
        if isinstance(node.items[0].optional_vars, ast.Name):
            var_names = [node.items[0].optional_vars.id]
        elif isinstance(node.items[0].optional_vars, (ast.List, ast.Tuple)):
            for var in node.items[0].optional_vars.elts:
                if not isinstance(var, ast.Name):
                    context.report_error("Invalid optional var definition")
            var_names = [var.id for var in node.items[0].optional_vars.elts]
        else:
            context.report_error("Invalid optional var definition")
        return var_names


@register
class Block(WithScopeHandler):
    def __init__(self):
        def block(axes=None, name="", exec_scope=""):
            block_info = self.context.block_info_stack[-1]
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
                values = self.context.loop_stack[-2].copy()
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

        var_names = WithScopeHandler.get_optional_var_names(node, context)
        self.block_vars = [tvm.te.var(name) for name in var_names]
        for block_var in self.block_vars:
            context.update_symbol(block_var.name, block_var)


@register
class Allocate(WithScopeHandler):
    def __init__(self):
        def allocate(extents, dtype, scope, condition=True):
            condition = tvm.runtime.convert(condition)
            scope = tvm.runtime.convert(scope)
            body = tvm.tir.Allocate(self.buffer_var, dtype, extents, condition, self.body)
            return tvm.tir.AttrStmt(self.buffer_var, "storage_scope", scope, body)

        super().__init__(allocate, concise_scope=True, def_symbol=True)
        self.buffer_var = None

    def enter_scope(self, node, context):
        # define buffer vars in symbol table
        if isinstance(node, ast.With):
            names = WithScopeHandler.get_optional_var_names(node, context)
            if len(names) != 1:
                context.report_error("Unexpected number of vars")
            name = names[0]
        elif isinstance(node, ast.Assign):
            name = node.targets[0].id
        else:
            raise Exception("Internal Bug")

        self.buffer_var = tvm.te.var(name, "handle")
        context.update_symbol(name, self.buffer_var)


@register
class LaunchThread(WithScopeHandler):
    def __init__(self):
        def launch_thread(env_var, extent):
            extent = tvm.runtime.convert(extent)
            return tvm.tir.AttrStmt(
                tvm.tir.IterVar(
                    None,
                    env_var,
                    getattr(tvm.tir.IterVar, "ThreadIndex"),
                    self.context.func_var_env_dict[env_var],
                ),
                "thread_extent",
                extent,
                self.body,
            )

        super().__init__(launch_thread, concise_scope=True, def_symbol=False)


@register
class Realize(WithScopeHandler):
    def __init__(self):
        def realize(buffer_bounds, scope, condition=True):
            buffer, bounds = buffer_bounds.buffer, buffer_bounds.region
            scope = tvm.runtime.convert(scope)
            return tvm.tir.AttrStmt(
                buffer,
                "realize_scope",
                scope,
                tvm.tir.BufferRealize(buffer, bounds, condition, self.body),
            )

        super().__init__(realize, concise_scope=True, def_symbol=False)


@register
class Attr(WithScopeHandler):
    def __init__(self):
        def attr(attr_node, attr_key, value):
            attr_node = tvm.runtime.convert(attr_node)
            value = tvm.runtime.convert(value)
            return tvm.tir.AttrStmt(attr_node, attr_key, value, self.body)

        super().__init__(attr, concise_scope=True, def_symbol=False)


@register
class AssertHandler(WithScopeHandler):
    def __init__(self):
        def Assert(condition, message):
            return tvm.tir.AssertStmt(condition, tvm.runtime.convert(message), self.body)

        super().__init__(Assert, concise_scope=True, def_symbol=False)


@register
class Let(WithScopeHandler):
    def __init__(self):
        def let(var, value):
            return tvm.tir.LetStmt(var, value, self.body)

        super().__init__(let, concise_scope=False, def_symbol=False)


class ForScopeHandler(ScopeHandler):
    def __init__(self, func):
        super().__init__(func)
        self.loop_vars = None

    def enter_scope(self, node, context):
        assert isinstance(node, ast.For)

        loop_var_names = list()
        if isinstance(node.target, ast.Name):
            loop_var_names.append(node.target.id)
        elif isinstance(node.target, ast.Tuple):
            for elt in node.target.elts:
                if not isinstance(elt, ast.Name):
                    context.report_error("Invalid loop var")
                loop_var_names.append(elt.id)
        else:
            context.report_error("Invalid loop var")

        self.loop_vars = [tvm.te.var(name, dtype="int32") for name in loop_var_names]
        for loop_var in self.loop_vars:
            context.update_symbol(loop_var.name, loop_var)
            context.loop_stack[-1].append(loop_var)

    def exit_scope(self, node, context, arg_list):
        for loop_var in self.loop_vars:
            context.loop_stack[-1].pop()
        return super().exit_scope(node, context, arg_list)


@register
class Serial(ForScopeHandler):
    def __init__(self):
        def serial(begin, end):
            if len(self.loop_vars) != 1:
                self.context.report_error("Expect exact 1 loop var")
            ana = tvm.arith.Analyzer()
            extent = end if begin == 0 else ana.simplify(end - begin)
            return tvm.tir.For(self.loop_vars[0], begin, extent, 0, 0, self.body)

        super().__init__(serial)


@register
class Parallel(ForScopeHandler):
    def __init__(self):
        def parallel(begin, end):
            if len(self.loop_vars) != 1:
                self.context.report_error("Expect exact 1 loop var")
            ana = tvm.arith.Analyzer()
            extent = end if begin == 0 else ana.simplify(end - begin)
            return tvm.tir.For(self.loop_vars[0], begin, extent, 1, 0, self.body)

        super().__init__(parallel)


@register
class Vectorized(ForScopeHandler):
    def __init__(self):
        def vectorized(begin, end):
            if len(self.loop_vars) != 1:
                self.context.report_error("Expect exact 1 loop var")
            ana = tvm.arith.Analyzer()
            extent = end if begin == 0 else ana.simplify(end - begin)
            return tvm.tir.For(self.loop_vars[0], begin, extent, 2, 0, self.body)

        super().__init__(vectorized)


@register
class Unroll(ForScopeHandler):
    def __init__(self):
        def unroll(begin, end):
            if len(self.loop_vars) != 1:
                self.context.report_error("Expect exact 1 loop var")
            ana = tvm.arith.Analyzer()
            extent = end if begin == 0 else ana.simplify(end - begin)
            return tvm.tir.For(self.loop_vars[0], begin, extent, 3, 0, self.body)

        super().__init__(unroll)


@register
class RangeHandler(ForScopeHandler):
    def __init__(self):
        def Range(begin, end, annotation=None):
            if len(self.loop_vars) != 1:
                self.context.report_error("Expect exact 1 loop var")
            ana = tvm.arith.Analyzer()
            extent = end if begin == 0 else ana.simplify(end - begin)
            if annotation is None:
                annotation = []
            else:
                annotation = [
                    tvm.tir.Annotation(
                        key, tvm.runtime.convert(val) if isinstance(val, str) else val
                    )
                    for key, val in annotation.items()
                ]
            return tvm.tir.Loop(self.loop_vars[0], begin, extent, annotation, self.body)

        super().__init__(Range)

    def signature(self):
        return "range", get_param_list(self.func)


@register
class Grid(ForScopeHandler):
    def __init__(self):
        def grid(*extents):
            if len(self.loop_vars) != len(extents):
                self.context.report_error("Inconsistent number of loop vars and extents")
            body = self.body
            for loop_var, extent in zip(reversed(self.loop_vars), reversed(extents)):
                body = tvm.tir.Loop(loop_var, 0, extent, [], body)
            return body

        super().__init__(grid)
