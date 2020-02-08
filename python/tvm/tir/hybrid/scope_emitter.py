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

from tvm import expr as _expr
from tvm import make as _make
from tvm import stmt as _stmt


class ScopeEmitter:
    """Maintain the stmt of scopes"""

    def __init__(self, parser):
        self.seq_stack = [[]]  # IR stmts of scopes
        self.allocate_stack = [[]]  # Buffer allocations of scopes
        self.parser = parser

    def emit(self, stmt):
        """Emit a stmt into current scope"""
        if isinstance(stmt, _expr.Call):
            stmt = _make.Evaluate(stmt)
        self.seq_stack[-1].append(stmt)

    def pop_seq(self):
        """Pop the inner most scope"""
        seq = self.seq_stack.pop()
        if not seq:
            seq.append(_make.Evaluate(0))
        if len(seq) == 1:
            return seq[0]
        return _stmt.SeqStmt(seq)

    def new_scope(self, is_block=False):
        self.seq_stack.append([])
        if is_block:
            self.allocate_stack.append([])
