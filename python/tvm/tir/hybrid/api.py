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
"""Hybrid Script APIs"""
# pylint: disable=invalid-name

import inspect

from tvm.api import _init_api

from .utils import HybridClass, HybridFunction, _parse
from .. import module


def create_module(funcs=None):
    """Construct a module from list of functions.

    Parameters
    -----------
    funcs : Optional[List[Union[Function, HybridFunction]]]
        list of functions

    Returns
    -------
    mod : Module
        A module containing the passed definitions
    """

    funcs = [_parse(func) if isinstance(func, HybridFunction) else func for func in funcs]
    return module.create_module(funcs=funcs)


def ashybrid(ir, show_meta=False):
    """Transform a Function or Module to python syntax script

    Parameters
    ----------
    ir : Union[Function, Module, HybridFunction]
        The Function or Module to be dumped

    show_meta : bool
        Whether show meta

    Returns
    -------
    script : str
        The Python script
    """

    if isinstance(ir, HybridFunction):
        # transform HybridFunction to Function
        ir = _parse(ir)
    elif isinstance(ir, module.Module):
        ir = ir.module  # get the inner IRModule of Module
    return AsHybrid(ir, show_meta)


def script(origin_script):
    """Decorate a python function or class as hybrid script.

    The hybrid function or parsing support parsing to the internal TIR.

    Returns
    -------
    output : Union[Function, Module]
        The Function or Module in IR.
    """

    if inspect.isfunction(origin_script):
        return HybridFunction(origin_script)

    if inspect.isclass(origin_script):
        return HybridClass(origin_script)

    raise TypeError("Only function and class are supported")


_init_api("tvm.tir.hybrid.api")
