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
"""TVM Script Parser for Sparse Dialect Classes"""
# pylint: disable=unused-argument, no-self-argument, inconsistent-return-statements
# pylint: disable=relative-beyond-top-level
import synr
from synr import ast

import tvm.tir

from .special_stmt import SpecialStmt
from .registry import register

@register
class MatchSparseBuffer(SpecialStmt):
    def __init__(self):
        def match_buffer(
            param,
            shape,
            fmt,
            idtype="int32",
            dtype="float32",
            span=None,
        ):
            pass

