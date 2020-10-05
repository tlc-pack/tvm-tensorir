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
"""Hybrid Script Scope Emitter for TIR"""

import tvm
from tvm.te import schedule


class ScopeEmitter:
    """Maintain the nodes, symbols of scopes and information of blocks"""

    class BlockInfo:
        def __init__(self):
            self.allocates = []
            self.binding = dict()
            self.reads = None
            self.writes = None
            self.annotations = []
            self.predicate = tvm.runtime.convert(True)

    def __init__(self, parser):
        self.node_stack = []  # AST nodes of scopes
        self.block_info_stack = []  # Block info of scopes
        self.loop_stack = []  # stack of loop vars
        self.symbols = []  # Symbols of scopes
        self.parser = parser

    def pop_scope(self, is_block=False):
        """Pop the inner most scope"""
        self.symbols.pop()
        self.node_stack.pop()
        if is_block:
            self.loop_stack.pop()
            return self.block_info_stack.pop()

    def new_scope(self, is_block=False):
        """ Creating a new scope """
        self.node_stack.append([])
        self.symbols.append(dict())
        if is_block:
            self.loop_stack.append([])
            self.block_info_stack.append(ScopeEmitter.BlockInfo())

    def update_symbol(self, name, symbol):
        """Append a symbol into current scope"""
        if isinstance(symbol, schedule.Buffer):
            if name in self.symbols[0]:
                self.parser.report_error("Duplicate Buffer name")
            self.symbols[0][name] = symbol
        else:
            self.symbols[-1][name] = symbol

    def remove_symbol(self, name):
        """Remove a symbol"""
        for symbols in reversed(self.symbols):
            if name in symbols:
                symbols.pop(name)
                return
        raise RuntimeError("Internal error of hybrid parser: no symbol named" + name)

    def lookup_symbol(self, name):
        """Look up symbol by name"""
        for symbols in reversed(self.symbols):
            if name in symbols:
                return symbols[name]
        return None

    def block_scope(self):
        return self.block_info_stack[-1]