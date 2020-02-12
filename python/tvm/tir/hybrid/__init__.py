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
"""Hybrid Programming APIs of TVM Python Package, aimed to support TIR"""
# pylint: disable=no-else-return, unused-argument

from __future__ import absolute_import as _abs

import inspect

from tvm._ffi.base import decorate
from tvm.api import _init_api

from . import module
from . import utils, registry, intrin, special_stmt, scope_handler
from .parser import source_to_op
from .utils import _pruned_source


def from_str(src, lineno=0):
    """Parsing the hybrid script source string

    Parameters
    ----------
    src : string
        The source code of hybrid script
    lineno : int
        The line number of the first line of src

    Returns
    -------
    funcs : Function or Module
        The built Function or Module
    """

    return source_to_op(lineno, src)


def create_module(funcs=None):
    """Construct a module from list of functions.

    Parameters
    -----------
    funcs : Optional[list]
        list of functions

    Returns
    -------
    mod : Module
        A module containing the passed definitions
    """

    return module.create_module(functions=funcs)


def to_python(funcs, show_meta=False):
    """Transform a Function or Module to python syntax script

    Parameters
    ----------
    funcs : Union[Function, Module]
        The Function or Module to be dumped

    show_meta : bool
        Whether show meta

    Returns
    -------
    script : str
        The Python script
    """

    return AsHybrid(funcs, show_meta)


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


def script(origin_script):
    """Decorate a python function function as hybrid script.

    The hybrid function support parsing to the internal TIR.

    Returns
    -------
    function : Function
        The Function in IR.
    """

    if inspect.isfunction(origin_script):
        def wrapped_func(func, *args, **kwargs):
            _init_scope()
            src = _pruned_source(func)
            return source_to_op(inspect.getsourcelines(func)[1], src)

        return decorate(origin_script, wrapped_func)

    elif inspect.isclass(origin_script):
        _init_scope()
        return from_str(inspect.getsource(origin_script), inspect.getsourcelines(origin_script)[1])

    else:
        raise TypeError("Only function and class are supported")


_init_api("tvm.tir.hybrid")
