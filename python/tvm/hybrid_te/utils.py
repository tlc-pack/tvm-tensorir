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
"""Internal utilities for parsing Python subset to TE IR"""

import inspect


def _internal_assert(cond, err, lineno=1):
    """Simplify the code segment like if not cond then raise an error"""
    if not cond:
        raise ValueError("TVM Hybrid Script Error in line " + str(lineno) + " : " + err)


def _pruned_source(func):
    """Prune source code's extra leading spaces"""
    lines = inspect.getsource(func).split('\n')
    leading_space = len(lines[0]) - len(lines[0].lstrip(' '))
    lines = [line[leading_space:] for line in lines]
    return '\n'.join(lines)
