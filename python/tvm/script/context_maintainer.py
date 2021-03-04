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
"""TVM Script Context Maintainer for TIR"""

from typing import List, Mapping, Union, Optional, Dict, Callable

import tvm
from tvm.tir import Var, Buffer, PrimExpr, Stmt, MatchBufferRegion
from tvm.runtime import Object
from .node import BufferSlice

import synr


class BlockInfo:
    """Information for block and block_realize signature"""

    alloc_buffers: List[Buffer]
    match_buffers: List[MatchBufferRegion]
    binding: Mapping[Var, PrimExpr]
    reads: List[BufferSlice]
    writes: List[BufferSlice]
    annotations: Mapping[str, Object]
    predicate: PrimExpr
    init: Optional[Stmt]

    def __init__(self):
        self.alloc_buffers = []
        self.match_buffers = []
        self.binding = {}
        self.reads = []
        self.writes = []
        self.annotations = {}
        self.predicate = tvm.tir.const(True, "bool")
        self.init = None


class ContextMaintainer:
    """Maintain all the necessary context info"""

    # scope context
    node_stack: List[List[synr.ast.Node]]
    block_info_stack: List[BlockInfo]
    loop_stack: List[List[Var]]
    symbols: List[Dict[str, Union[Var, Buffer]]]
    # function context
    func_params: List[Var]
    func_buffer_map: Mapping[Var, Buffer]
    func_dict_attr: Mapping[str, Object]
    func_var_env_dict: Mapping[Var, str]
    # parser and analyzer
    _report_error: Callable[[str, synr.ast.Span], None]
    analyzer: tvm.arith.Analyzer

    def __init__(self, report_error: Callable[[str, synr.ast.Span], None]):
        # scope context
        self.node_stack = []  # AST nodes of scopes
        self.block_info_stack = []  # Block info of scopes
        self.loop_stack = []  # stack of loop vars
        self.symbols = []  # symbols of scopes
        # function context
        self.func_params = []  # parameter list of function
        self.func_buffer_map = {}  # buffer_map of function
        self.func_dict_attr = {}  # func_attr of function
        self.func_var_env_dict = {}  # map from var to env_name
        # parser and analyzer
        self._report_error = report_error
        self.analyzer = tvm.arith.Analyzer()

    def enter_scope(self, nodes: Optional[List[synr.ast.Node]] = None):
        """Creating a new scope

        Parameters
        ----------
        nodes : Optional[List[synr.ast.Node]]
            The synr AST nodes in new scope
        """
        if nodes is None:
            nodes = []
        self.node_stack.append(list(reversed(nodes)))
        self.symbols.append(dict())

    def enter_block_scope(self, nodes: Optional[List[synr.ast.Node]] = None):
        """Creating a new block scope, the function will call `enter_scope` implicitly

        Parameters
        ----------
        nodes : Optional[List[synr.ast.Node]]
            The synr AST nodes in new scope
        """
        self.enter_scope(nodes)
        # Create a new loop stack for the new block
        self.loop_stack.append([])
        # Create a new BlockInfo for the new block
        self.block_info_stack.append(BlockInfo())

    def exit_scope(self):
        """Pop the inner most scope"""
        self.symbols.pop()
        self.node_stack.pop()

    def exit_block_scope(self):
        """Pop the inner most block scope, the function will call `exit_scope` implicitly"""
        self.exit_scope()
        # Pop loop stack
        self.loop_stack.pop()
        # Pop block_info
        self.block_info_stack.pop()

    def update_symbol(self, name: str, symbol: Union[Buffer, Var], node: synr.ast.Node):
        """Append a symbol into current scope"""
        if isinstance(symbol, Buffer):
            if name in self.symbols[0]:
                raise self.report_error("Duplicate Buffer name: " + symbol.name, node.span)
            self.symbols[0][name] = symbol
        else:
            self.symbols[-1][name] = symbol

    def remove_symbol(self, name: str):
        """Remove a symbol"""
        for symbols in reversed(self.symbols):
            if name in symbols:
                symbols.pop(name)
                return
        raise RuntimeError("Internal error of tvm script parser: no symbol named " + name)

    def lookup_symbol(self, name: str) -> Optional[Union[Buffer, Var]]:
        """Look up symbol by name"""
        for symbols in reversed(self.symbols):
            if name in symbols:
                return symbols[name]
        return None

    def report_error(self, message: str, span: synr.ast.Span):
        self._report_error(message, span)

    def current_block_scope(self) -> BlockInfo:
        return self.block_info_stack[-1]
