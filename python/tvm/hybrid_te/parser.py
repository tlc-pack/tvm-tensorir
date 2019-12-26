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
"""Hybrid Script Parser For TE IR"""

import numbers
import operator

from typed_ast import ast3 as ast

from . import intrin
from .intrin import Symbol
from .utils import _internal_assert
from .. import api as _api
from .. import expr as _expr
from .. import ir_builder as _ib
from .. import ir_pass as _pass
from .. import make as _make
from .. import stmt as _stmt
from ..api import all as _all
from ..api import any as _any


def _floordiv(x, y):
    if isinstance(x, _expr.ExprOp) or isinstance(y, _expr.ExprOp):
        return _api.floordiv(x, y)
    return operator.floordiv(x, y)


def _floormod(x, y):
    if isinstance(x, _expr.ExprOp) or isinstance(y, _expr.ExprOp):
        return _api.floormod(x, y)
    return operator.mod(x, y)


class HybridParser(ast.NodeVisitor):
    """Python AST visitor pass which finally lowers it to TE IR"""

    _symbol_type = {
        list: Symbol.ListOfTensorRegions,
        _ib.Buffer: Symbol.Buffer
    }

    _binop_maker = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: _floordiv,
        ast.Mod: _floormod,
        ast.BitOr: operator.or_,
        ast.BitAnd: operator.and_,
        ast.BitXor: operator.xor,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.And: _all,
        ast.Or: _any,
    }

    _unaryop_maker = {
        ast.USub: operator.neg,
        ast.Invert: operator.invert,
        ast.Not: operator.not_
    }

    def __init__(self, args):
        self.args = list(args)
        self.symbols = {} # Symbol table
        self.te_function = None, None, None
        self.te_function_name = None
        self.ir_builder = _ib.create()

        self._is_block_vars = False
        self._in_with_func_arg = False
        self.buffer_map = {}
        self.seq_stack = [[]] # IR stmts of scopes
        self.allocate_stack = [[]] # Buffer allocations of scopes

    def add_symbol(self, name, symbol_type, symbol, lineno):
        """Add value to the symbol table context"""
        if name in self.symbols.keys():
            old = str(self.symbols[name])
            new = str((symbol_type, symbol))
            _internal_assert(False, "Name conflict in symbol table! [%s] %s -> %s" % (name, old, new), lineno)

        self.symbols[name] = (symbol_type, symbol)

    def update_symbol(self, name, symbol_type, symbol, lineno):
        """Update value to the symbol table context"""
        if name in self.symbols.keys():
            self.symbols[name] = (symbol_type, symbol)
        else:
            self.add_symbol(name, symbol_type, symbol, lineno)

    def remove_symbol(self, name):
        """Remove value to the symbol table context"""
        self.symbols.pop(name)

    def emit(self, stmt):
        """Emit a stmt into current scope"""
        if isinstance(stmt, _expr.Call):
            stmt = _make.Evaluate(stmt)
        self.seq_stack[-1].append(stmt)

    def pop_seq(self):
        """Pop the inner most scope"""
        seq = self.seq_stack.pop()
        if not seq or callable(seq[-1]):
            seq.append(_make.Evaluate(0))
        stmt = seq[-1]
        for s in reversed(seq[:-1]):
            if callable(s):
                stmt = s(stmt)
            else:
                assert isinstance(s, _stmt.Stmt)
                stmt = _make.Block(s, stmt)
        return stmt

    def generic_visit(self, node):
        """To directly filter out invalidate type of stmt"""
        _internal_assert(False, type(node).__name__ + " stmt is not supported now", node.lineno)

    def visit_Module(self, node):
        _internal_assert(len(node.body) == 1, "Only one-function source code is allowed")
        return self.visit(node.body[0])

    def visit_FunctionDef(self, node):
        _internal_assert(len(node.args.args) == len(self.args), "The number of arguments passed", node.lineno)
        if self.te_function_name is None:
            self.te_function_name = node.name

        params = []
        for idx, arg in enumerate(node.args.args):
            self.add_symbol(arg.arg, Symbol.Input, self.args[idx], node.lineno)
            params.append(self.args[idx])

        for body_element in node.body:
            self.visit(body_element)

        body = self.pop_seq()
        _internal_assert(len(self.seq_stack) == 0, "Runtime Error", node.lineno)
        return self.ir_builder.function(params, self.buffer_map, body, name=self.te_function_name)

    def visit_Assign(self, node):
        _internal_assert(len(node.targets) == 1, "Only one-valued assignment is supported now", node.lineno)
        target = node.targets[0]
        if isinstance(target, ast.Name):
            # Name = List of TensorRegion, Buffer(buffer_bind, buffer_allocate)
            rhs = self.visit(node.value)
            _internal_assert(isinstance(rhs, (_ib.Buffer, list)),
                             "The value of assign ought to be list of TensorRegions or Buffer typed", node.lineno)
            self.update_symbol(target.id, HybridParser._symbol_type[type(rhs)], rhs, node.lineno)
            if isinstance(node.value, ast.Call) and node.value.func.id == "buffer_bind":
                self.buffer_map[self.symbols[node.value.args[0].id][1]] = rhs
        elif isinstance(target, ast.Subscript):
            # Buffer[expr, expr, .. expr] = Expr
            buffer, buffer_indexes = self.visit(target)
            rhs = self.visit(node.value)
            _internal_assert(isinstance(rhs, _expr.Expr), "The rhs of Assign stmt ought to be Expr typed", node.lineno)
            value = _api.convert(rhs)
            if value.dtype != buffer._content_type:
                raise ValueError("data type does not match content type %s vs %s" % (value.dtype, buffer._content_type))
            self.emit(_make.BufferStore(buffer._buffer, value, buffer_indexes))

    def visit_For(self, node):
        _internal_assert(isinstance(node.target, ast.Name), "The loop variable should be a name variable", node.lineno)
        _internal_assert(isinstance(node.iter, ast.Call), "The loop iter should be a Call", node.lineno)

        func_name = node.iter.func.id
        args = [self.visit(arg) for arg in node.iter.args]
        kw_args = [self.visit(keyword) for keyword in node.iter.keywords]
        kw_args = {kw_arg[0]: kw_arg[1] for kw_arg in kw_args}
        _internal_assert(hasattr(intrin.For, func_name),
                         "Function " + func_name + " used in For stmt is not supported now", node.lineno)
        getattr(intrin.For, func_name)(self, node, args, kw_args)

    def visit_With(self, node):
        _internal_assert(len(node.items) == 1, "Only one with element is supported now", node.lineno)
        _internal_assert(isinstance(node.items[0].context_expr, ast.Call),
                         "The context expression of with should be a Call", node.lineno)

        func_name = node.items[0].context_expr.func.id

        # preprocess block_var definitions
        block_vars_arg = None
        if len(node.items[0].context_expr.args) >= 1:
            block_vars_arg = node.items[0].context_expr.args[0]
        else:
            for keyword in node.items[0].context_expr.keywords:
                if keyword.arg == 'block_vars':
                    block_vars_arg = keyword.value

        if block_vars_arg is None:
            _internal_assert(False, "block() misses argument block_vars", node.lineno)

        self._is_block_vars = True
        block_vars = self.visit(block_vars_arg)
        self._is_block_vars = False

        args = [block_vars] + [self.visit(arg) for arg in node.items[0].context_expr.args[1:]]
        kw_args = [self.visit(keyword) for keyword in node.items[0].context_expr.keywords if
                   not keyword.arg == "block_vars"]
        kw_args = {kw_arg[0]: kw_arg[1] for kw_arg in kw_args}
        _internal_assert(hasattr(intrin.With, func_name),
                         "Function " + func_name + " used in With stmt is not supported now", node.lineno)
        getattr(intrin.With, func_name)(self, node, args, kw_args)

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        return HybridParser._binop_maker[type(node.op)](lhs, rhs)

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        return HybridParser._unaryop_maker[type(node.op)](operand)

    def visit_Call(self, node):
        _internal_assert(isinstance(node.func, ast.Name), "Only id function call is supported now", node.lineno)

        func_name = node.func.id
        args = [self.visit(arg) for arg in node.args]
        kw_args = [self.visit(keyword) for keyword in node.keywords]
        kw_args = {kw_arg[0]: kw_arg[1] for kw_arg in kw_args}

        if self._is_block_vars:
            kw_args["name"] = func_name
            func_name = "block_vars"
        else:
            _internal_assert(hasattr(intrin, func_name), "Function " + func_name + " is not supported now", node.lineno)
        return getattr(intrin, func_name)(self, node, args, kw_args)

    def visit_Subscript(self, node):
        _internal_assert(isinstance(node.value, ast.Name), "Only buffer variable can be subscriptable", node.lineno)
        _internal_assert(node.value.id in self.symbols, node.value.id + " is not defined", node.lineno)
        symbol_type, symbol = self.symbols[node.value.id]

        if isinstance(node.slice, ast.Index):
            # BufferLoad & BufferStore
            indexes = []
            if isinstance(node.slice.value, ast.Tuple):
                indexes = [self.visit(element) for element in node.slice.value.elts]
            else:
                indexes = [self.visit(node.slice.value)]

            for index in indexes:
                _internal_assert(isinstance(index, _expr.Expr), "Expression expected", node.lineno)

            if isinstance(node.ctx, ast.Load):
                return _make.BufferLoad(symbol._content_type, symbol._buffer, indexes)
            else:
                return symbol, indexes
        else:
            # TensorRegion
            slices = []
            if isinstance(node.slice, ast.Slice):
                _internal_assert(node.slice.step is None, "step is not allowed in TensorRegion", node.lineno)
                slices = [(self.visit(node.slice.lower), self.visit(node.slice.upper))]
            elif isinstance(node.slice, ast.ExtSlice):
                for dim in node.slice.dims:
                    _internal_assert(dim.step is None, "step is not allowed in TensorRegion", node.lineno)
                    slices.append((self.visit(dim.lower), self.visit(dim.upper)))

            doms = []
            for dom in slices:
                _internal_assert(isinstance(dom[0], _expr.Expr), "Expression expected", node.lineno)
                _internal_assert(isinstance(dom[1], _expr.Expr), "Expression expected", node.lineno)
                extent = dom[1] - dom[0]
                if isinstance(extent, _expr.Expr):
                    extent = _pass.Simplify(dom[1] - dom[0])
                doms.append(_make.range_by_min_extent(dom[0], extent))

            return _make.TensorRegion(symbol._buffer, doms)

    def visit_Name(self, node):
        name = node.id
        _internal_assert(name in self.symbols, "Unknown symbol %s" % name, node.lineno)
        symbol_type, symbol = self.symbols[name]
        return symbol

    def visit_Tuple(self, node):
        return tuple(self.visit(element) for element in node.elts)

    def visit_List(self, node):
        return [self.visit(element) for element in node.elts]

    def visit_keyword(self, node):
        return node.arg, self.visit(node.value)

    def visit_Constant(self, node):
        return _api.convert(node.value)

    def visit_Num(self, node):
        if isinstance(node.n, numbers.Integral):
            dtype = "int32"
        elif isinstance(node.n, float):
            dtype = "float32"
        else:
            _internal_assert(False, "The data type should be one of (int, float)", node.lineno)
        return _api.const(node.n, dtype)

    def visit_Str(self, node):
        return node.s


def source_to_op(src, args):
    """ Another level of wrapper

    Parameters
    ----------
    src : str
        Pruned source of original function

    args : list of Vars
        input of original function

    Returns
    -------
    function : TeFunction
        The TeFunction in IR.

    tensors : list of Placeholders
        List of tensors for buffers in function

    tensor_maps: dict of TeBuffer to Tensor
        Map between buffers in function and tensors
    """
    root = ast.parse(src)
    parser = HybridParser(args)
    return parser.visit(root)
