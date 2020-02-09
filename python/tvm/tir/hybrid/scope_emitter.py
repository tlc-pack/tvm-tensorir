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

from enum import Enum

from tvm import schedule as _schedule
from tvm import expr as _expr
from tvm import make as _make
from tvm import stmt as _stmt


class ScopeEmitter:
    """Maintain the stmts, allocations, and symbols of scopes"""

    class Symbol(Enum):
        """Enumerates types in the symbol table"""
        Var = 0      # params
        Buffer = 1   # Buffer
        IterVar = 2  # block_var
        LoopVar = 3  # loop_var
        List = 4     # list
        Dict = 5     # dict

    _symbol_type = {
        list: Symbol.List,
        dict: Symbol.Dict,
        _schedule.Buffer: Symbol.Buffer
    }

    def __init__(self, parser):
        self.seq_stack = [[]]  # IR stmts of scopes
        self.allocate_stack = [[]]  # Buffer allocations of scopes
        self.symbols = [dict()]  # Symbols of scopes
        self.parser = parser

    def pop_scope(self, is_block=False):
        """Pop the inner most scope"""
        self.symbols.pop()
        seq = self.seq_stack.pop()
        if len(seq) == 1:
            seq = seq[0]
        else:
            seq = _stmt.SeqStmt(seq)

        if is_block:
            return self.allocate_stack.pop(), seq
        return seq

    def new_scope(self, is_block=False):
        """Creating a new scope"""
        self.seq_stack.append([])
        self.symbols.append(dict())
        if is_block:
            self.allocate_stack.append([])

    def emit(self, stmt):
        """Emit a stmt into current scope"""
        if isinstance(stmt, _expr.Call):
            stmt = _make.Evaluate(stmt)
        self.seq_stack[-1].append(stmt)

    def alloc(self, allocation):
        """Append an allocation into current scope"""
        self.allocate_stack[-1].append(allocation)

    def update_symbol(self, name, symbol_type, symbol):
        """Append a symbol into current scope"""
        if symbol_type == ScopeEmitter.Symbol.Buffer:
            if name in self.symbols[0]:
                self.parser.report_error("Duplicate buffer name")
            self.symbols[0][name] = symbol
        else:
            self.symbols[-1][name] = symbol

    def lookup_symbol(self, name):
        """Look up symbol by name"""
        for symbols in reversed(self.symbols):
            if name in symbols:
                return symbols[name]
        return None
