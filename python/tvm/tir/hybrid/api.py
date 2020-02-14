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

from . import registry, utils
from .parser import source_to_op
from .. import module


def from_source(src, lineno=0):
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
    funcs : Optional[List[Union[Function, HybridFunction]]]
        list of functions

    Returns
    -------
    mod : Module
        A module containing the passed definitions
    """

    funcs = [utils.parse(func) if isinstance(func, HybridFunction) else func for func in funcs]
    return module.create_module(funcs=funcs)


def to_python(ir, show_meta=False):
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
        ir = utils.parse(ir)  # transform HybridFunction to Function
    elif isinstance(ir, module.Module):
        ir = ir.module  # get the inner IRModule of Module
    return AsHybrid(ir, show_meta)


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


class HybridClass:
    """Helper class for decorating a class"""

    def __init__(self, origin_script):
        self.origin_script = origin_script

    def __call__(self, *args, **kwargs):
        # call the parser to transform hybrid script into TIR
        return utils.parse(self)


class HybridFunction:
    """Helper class for decorating a function"""

    def __init__(self, origin_script):
        self.origin_script = origin_script


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
