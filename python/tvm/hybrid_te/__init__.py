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
"""Hybrid Programming APIs of TVM Python Package, aimed to support TE IR"""

from __future__ import absolute_import as _abs

from . import utils, registry, intrin, special_stmt, scope_handler
from .parser import source_to_op
from .utils import _pruned_source
from .._ffi.base import decorate
from ..api import _init_api


def to_python(func):
    """Transform a TeFunction to python syntax script

    Parameters
    ----------
    func : TeFunction
        The TeFunction to be dumped

    Returns
    -------
    script : str
        The Python script
    """

    return Print(func)


def register(origin_func):
    """Register an external function to parser under intrin

    The registered function ought to have return value.

    Parameters
    ----------
    origin_func : python function
        The function to be registered.
        Default value in parameter list is supported.
    """
    registry.register_intrin(origin_func)


def _init_scope():
    """Register primitive functions"""
    registry.register_intrin(intrin.int16)
    registry.register_intrin(intrin.int32)
    registry.register_intrin(intrin.int64)
    registry.register_intrin(intrin.float16)
    registry.register_intrin(intrin.float32)
    registry.register_intrin(intrin.float64)
    registry.register_special_stmt(special_stmt.buffer_bind)
    registry.register_special_stmt(special_stmt.buffer_allocate)
    registry.register_special_stmt(special_stmt.block_vars)
    registry.register_scope_handler(scope_handler.block, scope_name="with_scope")
    registry.register_scope_handler(scope_handler.range, scope_name="for_scope")


def script(origin_func):
    """Decorate a python function function as hybrid script.

    The hybrid function support parsing to the internal TE IR.

    Returns
    -------
    function : TeFunction
        The TeFunction in IR.
    """

    def wrapped_func(func, *args, **kwargs):
        _init_scope()
        src = _pruned_source(func)
        return source_to_op(func.__code__.co_firstlineno, src, *args, **kwargs)

    return decorate(origin_func, wrapped_func)


_init_api("tvm.hybrid_te")
