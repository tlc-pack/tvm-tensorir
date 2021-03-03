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
from tvm.tir import Var, Buffer, PrimExpr, Stmt, MatchBufferRegion, BufferSlice
from tvm.runtime import Object

import synr


class BlockInfo:
    """Information for block and block_realize signature"""

    def __init__(self):
        self.alloc_buffers: List[Buffer] = []
        self.match_buffers: List[MatchBufferRegion] = []
        self.binding = {}
        self.reads: List[BufferSlice] = []
        self.writes: List[BufferSlice] = []
        self.annotations: Mapping[str, Object] = {}
        self.predicate: PrimExpr = tvm.tir.const(True, "bool")
        self.init: Optional[Stmt] = None


class ContextMaintainer:
    """Maintain all the necessary context info"""

    def __init__(self, report_error: Callable[[str, synr.ast.Span], None]):
        # scope context
        self.node_stack: List[List[synr.ast.Node]] = []  # AST nodes of scopes
        self.block_info_stack: List[BlockInfo] = []  # Block info of scopes
        self.loop_stack: List[List[Var]] = []  # stack of loop vars
        self.symbols: List[Dict[str, Union[Var, Buffer]]] = []  # symbols of scopes
        # function context
        self.func_params: List[Var] = []  # parameter list of function
        self.func_buffer_map: Mapping[Var, Buffer] = {}  # buffer_map of function
        self.func_dict_attr: Mapping[str, Object] = {}  # func_attr of function
        self.func_var_env_dict: Mapping[Var, str] = {}  # map from var to env_name
        # parser
        self._report_error: Callable[[str, synr.ast.Span], None] = report_error
        # analyzer
        self.analyzer: tvm.arith.Analyzer = tvm.arith.Analyzer()

    def enter_scope(self, nodes: Optional[List[synr.ast.Node]] = None):
        """Creating a new scope"""
        if nodes is None:
            nodes = []
        self.node_stack.append(list(reversed(nodes)))
        self.symbols.append(dict())

    def enter_block_scope(self, nodes: Optional[List[synr.ast.Node]] = None):
        """Creating a new block scope"""
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
        """Pop the inner most block scope"""
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

    def block_scope(self) -> BlockInfo:
        return self.block_info_stack[-1]
