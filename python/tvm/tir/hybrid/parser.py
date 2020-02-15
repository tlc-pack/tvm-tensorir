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
"""Hybrid Script Parser For TIR"""
# pylint: disable=invalid-name, missing-docstring, inconsistent-return-statements, no-else-return
# pylint: disable=unnecessary-comprehension, unused-argument

import json
import numbers
import operator
from typed_ast import ast3 as ast

from tvm import api as _api
from tvm import expr as _expr
from tvm import ir_pass as _pass
from tvm import make as _make
from tvm import schedule as _schedule
from tvm._ffi.base import TVMError
from tvm.api import all as _all
from tvm.api import any as _any
from tvm.api import _init_api

from .. import module
from . import scope_emitter
from .scope_emitter import ScopeEmitter
from .meta_unparser import MetaUnparser
from .registry import Registry


def _floordiv(x, y):
    """Helper function to make operator floordiv"""
    if isinstance(x, _expr.ExprOp) or isinstance(y, _expr.ExprOp):
        return _api.floordiv(x, y)
    return operator.floordiv(x, y)


def _floormod(x, y):
    """Helper function to make operator floormod"""
    if isinstance(x, _expr.ExprOp) or isinstance(y, _expr.ExprOp):
        return _api.floormod(x, y)
    return operator.mod(x, y)


class HybridParserError(RuntimeError):
    """Hybrid Parser Runtime Error"""


class HybridParser(ast.NodeVisitor):
    """Python AST visitor pass which finally lowers it to TIR
    Notes for extension:
    1. To support new types of AST nodes. Add a function visit_xxx().
    2. To support new functions
        We divide allowed function calls in hybrid script into 3 categories,
        which is intrin, scope_handler and special_stmt.
        1) intrin functions ought to have return value.
        User can also register intrin category function into parser.
        2) scope_handler functions have no return value and accepts parser and AST node
        as its arguments, which is used in for scope and with scope.
        3) special_stmt functions have return value and accepts parser and AST node as its arguments
        When visiting Call node, we check special_stmt registry at first. If no registered function
        is found, we then check intrin.
        When visiting With node, we check with_scope registry.
        When visiting For node, we check for_scope registry.
    """

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

    def __init__(self, src, base_lienno):
        self.params = None
        self.buffer_map = None
        self.scope_emitter = None

        self.src = src.split('\n')
        self.base_lineno = base_lienno
        self.current_lineno = 0
        self.current_col_offset = 0
        self.meta = None

        self._is_block_vars = False
        self._in_with_func_arg = False
        self._assign_target = None

    def init_function_parsing_env(self):
        """Initialize function parsing environment"""
        self.params = []  # parameter list
        self.buffer_map = {}  # buffer map
        self.scope_emitter = scope_emitter.ScopeEmitter(self)  # scope emitter

    # TODO : if meta related functions grow, consider moving them to a new file
    @staticmethod
    def is_meta(node):
        """Judge whether an AST node is META"""
        return isinstance(node, ast.Assign) \
               and len(node.targets) == 1 \
               and isinstance(node.targets[0], ast.Name) \
               and node.targets[0].id == "__tvm_meta__"

    def init_meta(self, meta_dict):
        if meta_dict is not None:
            self.meta = _api.load_json(json.dumps(meta_dict))

    def mutate_meta(self, meta_node):
        Mutate_Meta(self.scope_emitter.symbols, meta_node)
        return meta_node

    def visit(self, node):
        """Override method in ast.NodeVisitor"""
        old_lineno, old_col_offset = self.current_lineno, self.current_col_offset

        if hasattr(node, "lineno"):
            self.current_lineno = self.base_lineno + node.lineno - 1
        if hasattr(node, "col_offset"):
            self.current_col_offset = node.col_offset

        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        visit_res = visitor(node)

        self.current_lineno, self.current_col_offset = old_lineno, old_col_offset

        return visit_res

    def wrap_line_col(self, message, lineno, col_offset):
        """Wrap the message with line number and column offset"""
        src_line = self.src[lineno - self.base_lineno]
        leading_space = len(src_line) - len(src_line.lstrip(' '))
        col_offset = col_offset - leading_space
        src_line = src_line[leading_space:]
        return "\n  " + src_line + "\n  " + " " * col_offset + "^\n" + "ParserError in line " \
               + str(lineno) + " : " + message

    def report_error(self, message, lineno=None, col_offset=None):
        """ Report an error occur in line lineno and column col_offset
        Parameters
        ----------
        message : str
            Error message
        lineno : int
            Line number of error line
        col_offset : int
            Column offset of error line
        """

        if lineno is None:
            lineno = self.current_lineno
        if col_offset is None:
            col_offset = self.current_col_offset
        raise HybridParserError(self.wrap_line_col(message, lineno, col_offset))

    def generic_visit(self, node):
        """ Override method in ast.NodeVisitor.
        To directly filter out invalidate type of stmt.
        """

        self.report_error(type(node).__name__ + " stmt is not supported now")

    def visit_Module(self, node):
        """ Module visitor
        AST abstract grammar:
            Module(stmt* body, type_ignore* type_ignore)
        By now we support two format of hybrid script shown below.

        Example
        -------
        1. Generate a Function(If the code is printed, then it may bring meta)
        .. code-block:: python

            import tvm

            @tvm.tir.hybrid.script
            def A(...):
                ...

            # call hybrid parser when call this function, get a Function
            func = A

        2. Generate a Module
        .. code-block:: python

            import tvm

            @tvm.tir.hybrid.script
            class MyMod():
               def A(...):
                  ...

               def B(...):
                   ...

                __tvm_meta__ = ...

            # call hybrid parser during construction, get a Module
            mod = MyMod
        """

        if len(node.body) == 1 and isinstance(node.body[0], (ast.ClassDef, ast.FunctionDef)):
            # class or single function
            return self.visit(node.body[0])
        elif len(node.body) == 2:
            if isinstance(node.body[0], ast.Assign):
                node.body[0], node.body[1] = node.body[1], node.body[0]
            if isinstance(node.body[0], ast.FunctionDef) and HybridParser.is_meta(node.body[1]):
                # function with meta
                self.init_meta(MetaUnparser().visit(node.body[1].value))
                return self.visit(node.body[0])
        self.report_error(
            "Only one-function, one-class or function-with-meta source code is allowed")

    def visit_ClassDef(self, node):
        """ ClassDef visitor
        AST abstract grammar:
            ClassDef(identifier name, expr* bases, keyword* keywords, stmt* body,
                     expr* decorator_list)
        """

        # parse meta
        count = False
        for body_element in node.body:
            if isinstance(body_element, ast.FunctionDef):
                pass
            elif HybridParser.is_meta(body_element) and not count:
                count = True
                self.init_meta(MetaUnparser().visit(body_element.value))
            else:
                self.report_error("invalid class member")

        # parse member functions
        funcs = []
        for body_element in node.body:
            if isinstance(body_element, ast.FunctionDef):
                funcs.append(self.visit(body_element))
        return module.create_module(funcs)

    def visit_FunctionDef(self, node):
        """ FunctionDef visitor
        AST abstract grammar:
            FunctionDef(identifier name, arguments args, stmt* body, expr* decorator_list,
                        expr? returns, string? type_comment)
            arguments = (arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs,
                         expr* kw_defaults, arg? kwarg, expr* defaults)
            arg = (identifier arg, expr? annotation, string? type_comment)
        """

        self.init_function_parsing_env()
        # add parameters of function
        for arg in node.args.args:
            arg_var = _api.var(arg.arg)
            self.scope_emitter.update_symbol(arg.arg, ScopeEmitter.Symbol.Var, arg_var)
            self.params.append(arg_var)
        # visit the body of function
        for body_element in node.body:
            self.visit(body_element)
        # fetch the body and return a tir.Function
        body = self.scope_emitter.pop_scope()
        return _make.Function(self.params, self.buffer_map, node.name, body)

    def visit_Assign(self, node):
        """ Assign visitor
        AST abstract grammar:
            Assign(expr* targets, expr value, string? type_comment)
        By now only 2 types of Assign is supported:
            1. Target = List, Buffer(buffer_bind, buffer_allocate)
            2. Buffer[expr, expr, .. expr] = Expr
        """

        if not len(node.targets) == 1:
            self.report_error("Only one-valued assignment is supported now")

        target = node.targets[0]
        if isinstance(target, ast.Name):
            # Target = List, Buffer(buffer_bind, buffer_allocate)
            self._assign_target = target.id
            rhs = self.visit(node.value)
            if not isinstance(rhs, (_schedule.Buffer, list)):
                self.report_error(
                    "The value of assign ought to be list of TensorRegions or Buffer typed")
            self.scope_emitter.update_symbol(target.id, ScopeEmitter._symbol_type[type(rhs)], rhs)
            # special judge buffer_bind
            if isinstance(node.value, ast.Call) and node.value.func.id == "buffer_bind":
                self.buffer_map[self.scope_emitter.lookup_symbol(node.value.args[0].id)] = rhs
        elif isinstance(target, ast.Subscript):
            # Buffer[expr, expr, .. expr] = Expr
            buffer, buffer_indexes = self.visit(target)
            rhs = self.visit(node.value)
            value = _api.convert(rhs)
            self.scope_emitter.emit(_make.BufferStore(buffer, value, buffer_indexes))
        else:
            self.report_error(
                "The target of Assign ought to be a name variable or a Buffer element")

    def visit_For(self, node):
        """ For visitor
        AST abstract grammar:
            For(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
        By now only 1 type of For is supported:
            1. for name in range(begin, end)
        """

        if not isinstance(node.target, ast.Name):
            self.report_error("The loop variable should be a name variable")
        # check node.iter, which is a Call
        if not isinstance(node.iter, ast.Call):
            self.report_error("The loop iter should be a Call")
        func_name = node.iter.func.id
        # collect arguments
        args = [self.visit(arg) for arg in node.iter.args]
        kw_args = [self.visit(keyword) for keyword in node.iter.keywords]
        kw_args = {kw_arg[0]: kw_arg[1] for kw_arg in kw_args}
        # All the functions supported in For stmt are registered in scope_handler.ForScope
        if func_name not in Registry.for_scope.keys():
            self.report_error("Function " + func_name + " used in For stmt is not supported now",
                              self.current_lineno,
                              node.iter.col_offset)

        old_lineno, old_col_offset = self.current_lineno, self.current_col_offset
        self.current_lineno, self.current_col_offset = \
            self.base_lineno + node.iter.lineno - 1, node.iter.col_offset
        Registry.for_scope.get(func_name)(self, node, args, kw_args)
        self.current_lineno, self.current_col_offset = old_lineno, old_col_offset

    def visit_With(self, node):
        """ With visitor
        AST abstract grammar:
            With(withitem* items, stmt* body, string? type_comment)
            withitem = (expr context_expr, expr? optional_vars)
        By now only 1 type of With is supported:
            1. with block(block_vars, values, reads, writes, predicate, annotations, name):
                Note that block_vars is a list of Calls, e.g. vi(0, 128, "reduce")
                It's a syntax sugar, which is equivalent with defining a IterVar named vi and
                used in the following block definition
        Example
        -------
        If we want to define a block and we use the primitive APIs in IRBuilder, we should write
        .. code-block:: python
            bv_i = ib.iter_var(tvm.make.range_by_min_extent(0, 128), name="vi")
            bv_j = ib.iter_var(tvm.make.range_by_min_extent(0, 128), name="vj")
            vi = bv_i.var
            vj = bv_j.var
            with ib.block([bv_i, bv_j], [i, j], reads = A[vi:vi+1, vj:vj+1], \
            write = B[vi:vi+1, vj:vj+1])
        The IterVar variable bv_i and bv_j are only used once, so I planned to give a sugar here and
        the user can simply write in one line code like
        .. code-block:: python
            with block({vi(0, 128): i, vj(0, 128): j}, reads = A[vi:vi+1, vj:vj+1], \
            write = B[vi:vi+1, vj:vj+1])
        The problem it brings is that vi, vj will be parsed as Call here, so when parsing the Call
        which is actually to defining a block var, we leave it to a intrinsic function block_vars()
        to handle them.
        """

        if not len(node.items) == 1:
            self.report_error("Only one with element is supported now")
        if not isinstance(node.items[0].context_expr, ast.Call):
            self.report_error("The context expression of with should be a Call")

        func_call = node.items[0].context_expr
        func_name = func_call.func.id

        if func_name == 'block':
            # preprocess block_var definitions
            block_vars_arg = None
            if len(func_call.args) >= 1:
                block_vars_arg = func_call.args[0]
            else:
                for keyword in func_call.keywords:
                    if keyword.arg == 'block_vars_info':
                        block_vars_arg = keyword.value

            if block_vars_arg is None:
                self.report_error("block() misses argument block_vars_info",
                                  lineno=self.current_lineno,
                                  col_offset=block_vars_arg.col_offset)

            self._is_block_vars = True
            block_vars = self.visit(block_vars_arg)
            self._is_block_vars = False

            # update block vars into symbol table
            self.scope_emitter.new_scope(is_block=True)
            for block_var, _ in block_vars:
                self.scope_emitter.update_symbol(block_var.var.name, ScopeEmitter.Symbol.IterVar,
                                                 block_var.var)
            # collect arguments
            args = [block_vars] + [self.visit(arg) for arg in func_call.args[1:]]
            kw_args = [self.visit(keyword) for keyword in func_call.keywords if
                       not keyword.arg == "block_vars_info"]
            kw_args = {kw_arg[0]: kw_arg[1] for kw_arg in kw_args}
        elif func_name in Registry.with_scope.keys():
            # reserved for future use
            # collect arguments
            args = [self.visit(arg) for arg in func_call.args]
            kw_args = [self.visit(keyword) for keyword in func_call.keywords]
            kw_args = {kw_arg[0]: kw_arg[1] for kw_arg in kw_args}
        else:
            self.report_error("Function " + func_name + " used in With stmt is not supported now")

        # All the functions supported in With stmt are registered in scope_handler.WithScope
        old_lineno, old_col_offset = self.current_lineno, self.current_col_offset
        self.current_lineno, self.current_col_offset = \
            self.base_lineno + func_call.lineno - 1, func_call.col_offset
        Registry.with_scope.get(func_name)(self, node, args, kw_args)
        self.current_lineno, self.current_col_offset = old_lineno, old_col_offset

    def visit_Call(self, node):
        """ Call visitor
        AST abstract grammar:
            Call(expr func, expr* args, keyword* keywords)
            keyword = (identifier? arg, expr value)
        All the functions used outside With and For are registered in special_stmt or intrin
        """

        if not isinstance(node.func, ast.Name):
            self.report_error("Only id function call is supported now")

        func_name = node.func.id

        # collect arguments
        args = [self.visit(arg) for arg in node.args]
        kw_args = [self.visit(keyword) for keyword in node.keywords]
        kw_args = {kw_arg[0]: kw_arg[1] for kw_arg in kw_args}

        if self._is_block_vars:
            # special judge block_var sugar
            kw_args["name"] = func_name
            func_name = "block_vars"

        if func_name in Registry.special_stmt.keys():
            return Registry.special_stmt.get(func_name)(self, node, args, kw_args)
        if func_name in Registry.intrin.keys():
            return Registry.intrin.get(func_name)(self, node, args, kw_args)
        self.report_error("Function " + func_name + " is not supported now")

    def visit_BinOp(self, node):
        """ BinOp visitor
        AST abstract grammar:
            BinOp(expr left, operator op, expr right)
        """

        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if not isinstance(node.op, tuple(HybridParser._binop_maker.keys())):
            self.report_error("BinOp " + str(type(node.op)) + " is not supported now")
        return HybridParser._binop_maker[type(node.op)](lhs, rhs)

    def visit_Compare(self, node):
        """ Compare visitor
        AST abstract grammar:
            Compare(expr left, expr right, ops=)
        """

        ops = [self.visit(node.left)]
        ops += [self.visit(comparator) for comparator in node.comparators]
        res = []
        for i in range(len(node.ops)):
            lhs = ops[i]
            rhs = ops[i + 1]
            res.append(HybridParser._binop_maker[type(node.ops[i])](lhs, rhs))
        return _all(*res)

    def visit_BoolOp(self, node):
        """ BoolOp visitor
        AST abstract grammar:
            BoolOp(boolop op, expr* values)
        """

        values = [self.visit(value) for value in node.values]
        return HybridParser._binop_maker[type(node.op)](*values)

    def visit_UnaryOp(self, node):
        """ UnaryOp visitor
        AST abstract grammar:
            UnaryOp(unaryop op, expr operand)
        """

        operand = self.visit(node.operand)
        if not isinstance(node.op, tuple(HybridParser._unaryop_maker.keys())):
            self.report_error("UnaryOp " + str(type(node.op)) + " is not supported now")
        return HybridParser._unaryop_maker[type(node.op)](operand)

    def visit_Subscript(self, node):
        """ Subscript visitor
        AST abstract grammar:
            Subscript(expr value, slice slice, expr_context ctx)
            slice = Slice(expr? lower, expr? upper, expr? step)
                    | ExtSlice(slice* dims)
                    | Index(expr value)
        By now only 3 types of Subscript are supported:
            1. Buffer[index, index, ...], Buffer element access(BufferLoad & BufferStore)
            2. Buffer[slice, slice, ...], TensorRegion
            3. meta[type_key][index], Meta info access
            TODO(long term): TensorRegion can be Buffer[index, index, ...]?
        """

        if isinstance(node.value, ast.Name):
            symbol = self.scope_emitter.lookup_symbol(node.value.id)
            if symbol is None :
                    self.report_error(node.value.id + " is not defined")

            if isinstance(node.slice, ast.Index):
                # BufferLoad & BufferStore
                if isinstance(node.slice.value, ast.Tuple):
                    # Buffer[index, index, ...]
                    indexes = [self.visit(element) for element in node.slice.value.elts]
                else:
                    # Buffer[index]
                    indexes = [self.visit(node.slice.value)]

                if isinstance(node.ctx, ast.Load):
                    return _make.BufferLoad(symbol.dtype, symbol, indexes)
                return symbol, indexes
            else:
                # TensorRegion
                slices = []
                if isinstance(node.slice, ast.Slice):
                    # Buffer[begin:end]
                    if node.slice.step is not None:
                        self.report_error("step is not allowed in TensorRegion")
                    slices = [(self.visit(node.slice.lower), self.visit(node.slice.upper))]
                elif isinstance(node.slice, ast.ExtSlice):
                    # Buffer[begin:end, begin:end]
                    for dim in node.slice.dims:
                        if dim.step is not None:
                            self.report_error("step is not allowed in TensorRegion")
                        slices.append((self.visit(dim.lower), self.visit(dim.upper)))

                doms = []
                for dom in slices:
                    extent = dom[1] - dom[0]
                    if isinstance(extent, _expr.PrimExpr):
                        extent = _pass.Simplify(dom[1] - dom[0])
                    doms.append(_make.range_by_min_extent(dom[0], extent))

                return _make.TensorRegion(symbol, doms)

        elif isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name) \
                and node.value.value.id == 'meta':
            # meta[type_key][index]
            if not (isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Num)) \
                    or not (isinstance(node.value.slice, ast.Index) \
                            and isinstance(node.value.slice.value, ast.Name)):
                self.report_error("The meta access format ought to be meta[type_key][index]")

            type_key = node.value.slice.value.id
            index = node.slice.value.n
            node_list = self.meta[type_key]
            if node_list is None:
                self.report_error("type_key " + type_key + " in meta not found")
            if len(node_list) <= index:
                self.report_error("index " + index + " out of range " + len(node_list))
            return self.mutate_meta(node_list[index])
        else:
            self.report_error("Only buffer variable and meta can be subscriptable")

    def visit_Name(self, node):
        """ Name visitor
        AST abstract grammar:
            Name(identifier id, expr_context ctx)
        """

        name = node.id
        symbol = self.scope_emitter.lookup_symbol(name)
        if symbol is None:
            self.report_error("Unknown symbol %s" % name)
        return symbol

    def visit_Dict(self, node):
        """ Dict visitor
        AST abstract grammar:
            Dict(expr* keys, expr* values)
        """

        keys = [self.visit(key) for key in node.keys]
        values = [self.visit(value) for value in node.values]
        if self._is_block_vars:
            return list((key, value) for key, value in zip(keys, values))
        return {key: value for key, value in zip(keys, values)}

    def visit_Tuple(self, node):
        """ Tuple visitor
        AST abstract grammar:
            Tuple(expr* elts, expr_context ctx)
        """

        return tuple(self.visit(element) for element in node.elts)

    def visit_List(self, node):
        """ List visitor
        AST abstract grammar:
            List(expr* elts, expr_context ctx)
        """

        return [self.visit(element) for element in node.elts]

    def visit_keyword(self, node):
        """ Keyword visitor
        AST abstract grammar:
            keyword = (identifier? arg, expr value)
        """

        return node.arg, self.visit(node.value)

    def visit_NameConstant(self, node):
        return _api.convert(node.value)

    def visit_Constant(self, node):
        return _api.convert(node.value)

    def visit_Num(self, node):
        if isinstance(node.n, numbers.Integral):
            dtype = "int32"
        elif isinstance(node.n, float):
            dtype = "float32"
        else:
            self.report_error("The data type should be one of (int, float)")
        return _api.const(node.n, dtype)

    def visit_Str(self, node):
        return node.s


def source_to_op(src, func_lineno=0):
    """ Another level of wrapper
    Parameters
    ----------
    src : str
        Pruned source of original script
    func_lineno : Optional[int]
        The line number of the first line of the script to be parsed
    Returns
    -------
    functions : Function or Module
        The Function or Module in IR.
    """

    root = ast.parse(src)
    parser = HybridParser(src, func_lineno)

    try:
        return parser.visit(root)
    except TVMError as e:
        # TVM internal c++ error, we have to process the error message and inject line info
        inject_e = str(e).split('\n')
        msg = inject_e[-1].split(':', maxsplit=1)[1].strip()
        inject_e = inject_e[:-1]
        inject_e.extend(
            parser.wrap_line_col(msg, parser.current_lineno, parser.current_col_offset).split('\n'))
        inject_e[-1] = "TVM" + inject_e[-1][6:]
        raise TVMError('\n'.join(inject_e))


_init_api("tvm.tir.hybrid.parser")
