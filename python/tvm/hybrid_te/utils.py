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
"""Helper functions in Hybrid Script Parser"""

import inspect

from .registry import register_func


def _pruned_source(func):
    """Prune source code's extra leading spaces"""
    lines = inspect.getsource(func).split('\n')
    leading_space = len(lines[0]) - len(lines[0].lstrip(' '))
    lines = [line[leading_space:] for line in lines]
    return '\n'.join(lines)


def register_intrin(origin_func):
    """Register function under category intrin"""
    register_func("intrin", origin_func, need_parser_and_node=False, need_return=True)


def register_scope_handler(origin_func, scope_name):
    """Register function under category with_scope or for_scope"""
    register_func(scope_name, origin_func, need_parser_and_node=True, need_return=False)


def register_special_stmt(origin_func):
    """Register function under category special_stmt"""
    register_func("special_stmt", origin_func, need_parser_and_node=True, need_return=True)
