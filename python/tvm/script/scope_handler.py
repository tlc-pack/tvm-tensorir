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
# pylint: disable=redefined-builtin, unused-argument, invalid-name, relative-beyond-top-level
import synr as synr
from synr import ast
import tvm.tir
from tvm.runtime import Object
from tvm.ir import Span, Range
from tvm.tir import Stmt, PrimExpr, IterVar, Var
from tvm.tir import Buffer, BufferLoad, BufferRegion

from .context_maintainer import ContextMaintainer
from .utils import get_param_list, from_synr_span
from .registry import register
from typing import Tuple, Any, Callable, Optional, List, Union, Mapping


class ScopeHandler:
    """Base class for all scope handlers"""

    def __init__(self, func: Callable):
        self.func: Callable = func
        self.body: Optional[Stmt] = None
        self.node: Optional[synr.ast.Node] = None
        self.context: Optional[ContextMaintainer] = None

    def signature(self) -> Tuple[str, Tuple[list, list, Any]]:
        return "tir." + self.func.__name__, get_param_list(self.func)

    def enter_scope(
        self,
        node: synr.ast.Node,
        context: ContextMaintainer,
        arg_list: List[Any],
        span: synr.ast.Span,
    ):
        pass

    def exit_scope(
        self,
        node: synr.ast.Node,
        context: ContextMaintainer,
        arg_list: List[Any],
        span: synr.ast.Span,
    ):
        self.node = node
        self.context = context
        return self.func(*arg_list, span=from_synr_span(span))


class WithScopeHandler(ScopeHandler):
    """Base class for all with scope handlers"""

    def __init__(self, func, concise_scope, def_symbol):
        super().__init__(func)
        self.concise_scope = concise_scope
        self.def_symbol = def_symbol

    @staticmethod
    def get_optional_var_names(node, context):
        """Get list of names from ast.With's optional_vars"""
        assert isinstance(node, ast.With)

        if isinstance(node.lhs, list):
            for var in node.lhs:
                if not isinstance(var, ast.Var):
                    context.report_error("Invalid optional var definition", node.span)
            var_names = [var.id.name for var in node.lhs]
        else:
            context.report_error("Invalid optional var definition", node.span)
        return var_names


@register
class Allocate(WithScopeHandler):
    """ With scope handler tir.alloc_with_scope(var, extents, dtype, scope, condition) """

    def __init__(self):
        def allocate(extents, dtype, scope, condition=True, span=None):
            condition = tvm.runtime.convert(condition)
            scope = tvm.runtime.convert(scope)
            body = tvm.tir.Allocate(
                self.buffer_var, dtype, extents, condition, self.body, span=span
            )
            return tvm.tir.AttrStmt(self.buffer_var, "storage_scope", scope, body, span=span)

        super().__init__(allocate, concise_scope=True, def_symbol=True)
        self.buffer_var = None

    def enter_scope(
        self,
        node: synr.ast.Node,
        context: ContextMaintainer,
        arg_list: List[Any],
        span: synr.ast.Span,
    ):
        # define buffer vars in symbol table
        if isinstance(node, ast.With):
            names = WithScopeHandler.get_optional_var_names(node, context)
            if len(names) != 1:
                context.report_error("Unexpected number of vars", node.span)
            name = names[0]
        elif isinstance(node, ast.Assign):
            name = node.lhs.id.name
        else:
            raise Exception("Internal Bug")

        def setup_buffer_var(extents, dtype, scope, condition=True, span: Span = None):
            """Setup buffer var for a given type."""
            buffer_ptr_type = tvm.ir.PointerType(tvm.ir.PrimType(dtype))
            self.buffer_var = tvm.tir.Var(name, buffer_ptr_type, span)

        setup_buffer_var(*arg_list, span=from_synr_span(node.lhs.id.span))
        context.update_symbol(name, self.buffer_var, node)


@register
class LaunchThread(WithScopeHandler):
    """ With scope handler tir.launch_thread(env_var, extent) """

    def __init__(self):
        def launch_thread(env_var, extent, span):
            extent = tvm.runtime.convert(extent, span=span)
            return tvm.tir.AttrStmt(
                IterVar(
                    None,
                    env_var,
                    getattr(IterVar, "ThreadIndex"),
                    self.context.func_var_env_dict[env_var],
                    span=span,
                ),
                "thread_extent",
                extent,
                self.body,
                span=span,
            )

        super().__init__(launch_thread, concise_scope=True, def_symbol=False)


@register
class Realize(WithScopeHandler):
    """ With scope handler tir.realize(buffer_bounds, scope, condition) """

    def __init__(self):
        def realize(buffer_bounds, scope, condition=True, span=None):
            buffer, bounds = buffer_bounds
            scope = tvm.runtime.convert(scope, span=span)
            return tvm.tir.AttrStmt(
                buffer,
                "realize_scope",
                scope,
                tvm.tir.BufferRealize(buffer, bounds, condition, self.body, span=span),
                span=span,
            )

        super().__init__(realize, concise_scope=True, def_symbol=False)


@register
class Attr(WithScopeHandler):
    """ With scope handler tir.attr(attr_node, attr_key, value) """

    def __init__(self):
        def attr(attr_node, attr_key, value, span):
            attr_node = tvm.runtime.convert(attr_node, span=span)
            value = tvm.runtime.convert(value, span=span)
            return tvm.tir.AttrStmt(attr_node, attr_key, value, self.body, span=span)

        super().__init__(attr, concise_scope=True, def_symbol=False)


@register
class AssertHandler(WithScopeHandler):
    """ With scope handler tir.Assert(condition, message) """

    def __init__(self):
        def Assert(condition, message, span):
            return tvm.tir.AssertStmt(condition, tvm.runtime.convert(message), self.body, span=span)

        super().__init__(Assert, concise_scope=True, def_symbol=False)


@register
class Let(WithScopeHandler):
    """ With scope handler tir.let(var, value) """

    def __init__(self):
        def let(var, value, span):
            return tvm.tir.LetStmt(var, value, self.body, span=span)

        super().__init__(let, concise_scope=False, def_symbol=False)


@register
class Block(WithScopeHandler):
    """ With scope handler tir.block(extents, name) as iter_vars"""

    def __init__(self):
        def block(axes=None, name_hint: str = "", span: Optional[Span] = None):
            assert self.node
            assert self.context
            assert self.body
            block_info = self.context.block_info_stack[-1]
            if axes is None:
                axes = []
            if len(axes) != len(self.block_vars):
                self.context.report_error("Inconsistent number of block vars", self.node.span)
            block_iters: List[IterVar] = []
            reads: List[BufferRegion] = []
            writes: List[BufferRegion] = []
            for i in range(len(axes)):
                axis = tvm.runtime.convert(axes[i])
                if isinstance(axis, tvm.tir.PrimExpr):
                    block_var_dom = Range.from_min_extent(0, axis)
                    block_iters.append(IterVar(block_var_dom, self.block_vars[i], 0))
                elif isinstance(axis, Range):
                    block_iters.append(IterVar(axis, self.block_vars[i], 0))
                elif isinstance(axis, IterVar):
                    block_iters.append(IterVar(axis.dom, self.block_vars[i], axis.iter_type))
                else:
                    self.context.report_error("Invalid argument of tir.block()", self.node.span)

            # create block read/write regions

            def create_buffer_region(
                inputs: Union[BufferLoad, Tuple[Buffer, List[Union[Range, PrimExpr]]]]
            ) -> BufferRegion:
                region: List[Range] = []
                if isinstance(inputs, tvm.tir.BufferLoad):
                    buffer = inputs.buffer
                    for index in inputs.indices:
                        region.append(Range.from_min_extent(index, 1))
                else:
                    buffer, indices = inputs
                    for index in indices:
                        if isinstance(index, Range):
                            region.append(index)
                        else:
                            region.append(Range.from_min_extent(index, 1))
                return BufferRegion(buffer, region)

            if block_info.reads is None:
                reads = []
            else:
                reads = [create_buffer_region(read) for read in block_info.reads]

            if block_info.writes is None:
                writes = []
            else:
                writes = [create_buffer_region(write) for write in block_info.writes]

            inner = tvm.tir.Block(
                block_iters,
                reads,
                writes,
                name_hint,
                self.body,
                block_info.init,
                block_info.alloc_buffers,
                block_info.match_buffers,
                block_info.annotations,
                span,
            )
            # create block var binding
            values: List[PrimExpr]
            if not block_info.binding:
                values = self.context.loop_stack[-2].copy()
                if len(values) == 0:
                    values = [tvm.tir.const(float("nan"), dtype="float32")] * len(block_iters)
                elif len(values) != len(block_iters):
                    self.context.report_error(
                        "Autocomplete block var binding expect larger number of loops",
                        self.node.span,
                    )
            else:
                for block_var in self.block_vars:
                    if block_var not in block_info.binding:
                        self.context.report_error(
                            "Missing block var binding for " + block_var.name,
                            self.node.span,
                        )
                values = [block_info.binding[block_var] for block_var in self.block_vars]
            body = tvm.tir.BlockRealize(values, block_info.predicate, inner, span)
            return body

        super().__init__(func=block, concise_scope=False, def_symbol=True)
        self.block_vars = None

    def enter_scope(
        self,
        node: synr.ast.Node,
        context: ContextMaintainer,
        arg_list: List[Any],
        span: synr.ast.Span,
    ):
        # define block vars
        assert isinstance(node, ast.With)

        var_names = WithScopeHandler.get_optional_var_names(node, context)
        self.block_vars = [tvm.te.var(name) for name in var_names]
        for block_var in self.block_vars:
            context.update_symbol(block_var.name, block_var, node)


@register
class InitBlock(WithScopeHandler):
    """ With scope handler tir.init()"""

    def __init__(self):
        def init(span: Span = None):
            assert self.context
            self.context.block_info_stack[-2].init = self.body

        super().__init__(func=init, concise_scope=False, def_symbol=True)


class ForScopeHandler(ScopeHandler):
    """Base class for all for scope handlers"""

    def __init__(self, func):
        super().__init__(func)
        self.loop_vars: Optional[List[Var]] = None

    def enter_scope(
        self,
        node: synr.ast.Node,
        context: ContextMaintainer,
        arg_list: List[Any],
        span: synr.ast.Span,
    ):
        assert isinstance(node, ast.For)

        loop_var_names = list()
        spans = list()
        if isinstance(node.lhs, ast.Var):
            loop_var_names.append(node.lhs.id.name)
            spans.append(from_synr_span(node.lhs.id.span))
        elif isinstance(node.lhs, list):
            for elt in node.lhs:
                if not isinstance(elt, ast.Var):
                    context.report_error("Invalid loop var", elt.span)
                loop_var_names.append(elt.id.name)
                spans.append(from_synr_span(elt.id.span))
        else:
            context.report_error("Invalid loop var in loop", span)

        self.loop_vars = [
            tvm.te.var(name, dtype="int32", span=span) for name, span in zip(loop_var_names, spans)
        ]
        for loop_var in self.loop_vars:
            context.update_symbol(loop_var.name, loop_var, node)
            context.loop_stack[-1].append(loop_var)

    def exit_scope(
        self,
        node: synr.ast.Node,
        context: ContextMaintainer,
        arg_list: List[Any],
        span: synr.ast.Span,
    ):
        assert self.loop_vars
        for loop_var in self.loop_vars:
            context.loop_stack[-1].pop()
        return super().exit_scope(node, context, arg_list, span)

    def create_loop(
        self,
        begin: PrimExpr,
        end: PrimExpr,
        kind: int,
        thread_binding: Optional[str] = None,
        annotations: Optional[Mapping[str, Object]] = None,
        span: Optional[Span] = None,
    ):
        assert self.node
        assert self.context
        assert self.loop_vars
        if len(self.loop_vars) != 1:
            self.context.report_error("Expect exact 1 loop var", self.node.span)
        extent = end if begin == 0 else self.context.analyzer.simplify(end - begin)
        annos: Mapping[str, Object]
        if annotations is None:
            annos = {}
        else:
            annos = {
                key: tvm.tir.StringImm(val) if isinstance(val, str) else val
                for key, val in annotations.items()
            }
        return tvm.tir.For(
            self.loop_vars[0],
            begin,
            extent,
            kind,
            self.body,
            thread_binding=thread_binding,
            annotations=annos,
            span=span,
        )


@register
class Serial(ForScopeHandler):
    """ For scope handler tir.serial(begin, end, annotations)"""

    def __init__(self):
        def serial(
            begin: PrimExpr,
            end: PrimExpr,
            annotations: Optional[Mapping[str, Object]] = None,
            span: Optional[Span] = None,
        ):
            return self.create_loop(begin, end, 0, annotations=annotations, span=span)

        super().__init__(serial)


@register
class Parallel(ForScopeHandler):
    """ For scope handler tir.parallel(begin, end, annotations)"""

    def __init__(self):
        def parallel(
            begin: PrimExpr,
            end: PrimExpr,
            annotations: Optional[Mapping[str, Object]] = None,
            span: Optional[Span] = None,
        ):
            return self.create_loop(begin, end, 1, annotations=annotations, span=span)

        super().__init__(parallel)


@register
class Vectorized(ForScopeHandler):
    """ For scope handler tir.vectorized(begin, end, annotations)"""

    def __init__(self):
        def vectorized(
            begin: PrimExpr,
            end: PrimExpr,
            annotations: Optional[Mapping[str, Object]] = None,
            span: Optional[Span] = None,
        ):
            return self.create_loop(begin, end, 2, annotations=annotations, span=span)

        super().__init__(vectorized)


@register
class Unroll(ForScopeHandler):
    """ For scope handler tir.unroll(begin, end, annotations)"""

    def __init__(self):
        def unroll(
            begin: PrimExpr,
            end: PrimExpr,
            annotations: Optional[Mapping[str, Object]] = None,
            span: Optional[Span] = None,
        ):
            return self.create_loop(begin, end, 3, annotations=annotations, span=span)

        super().__init__(unroll)


@register
class ThreadBinding(ForScopeHandler):
    """ For scope handler tir.thread_binding(begin, end, thread, annotations)"""

    def __init__(self):
        def thread_binding(
            begin: PrimExpr,
            end: PrimExpr,
            thread: str,
            annotations: Optional[Mapping[str, Object]] = None,
            span: Optional[Span] = None,
        ):
            thread_iter_var = IterVar(None, None, 1, thread, span=span)
            return self.create_loop(
                begin,
                end,
                4,
                thread_binding=thread_iter_var,
                annotations=annotations,
                span=span,
            )

        super().__init__(thread_binding)


@register
class RangeHandler(ForScopeHandler):
    """For scope handler tir.range(begin, end, annotations)
    Note that tir.range is totally the same as tir.serial
    """

    def __init__(self):
        def Range(
            begin: PrimExpr,
            end: PrimExpr,
            annotations: Optional[Mapping[str, Object]] = None,
            span: Optional[Span] = None,
        ):
            return self.create_loop(begin, end, 0, annotations=annotations, span=span)

        super().__init__(Range)

    def signature(self):
        return "range", get_param_list(self.func)


@register
class Grid(ForScopeHandler):
    """ For scope handler tir.grid(extents)"""

    def __init__(self):
        def grid(*extents: List[PrimExpr], span: Span):
            assert self.node
            assert self.context
            assert self.loop_vars
            if len(self.loop_vars) != len(extents):
                self.context.report_error(
                    "Inconsistent number of loop vars and extents", self.node.span
                )
            body = self.body
            for loop_var, extent in zip(reversed(self.loop_vars), reversed(extents)):
                body = tvm.tir.For(loop_var, 0, extent, 0, body, span=span)
            return body

        super().__init__(grid)
