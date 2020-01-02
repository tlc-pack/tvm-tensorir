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

import inspect

from . import intrin
from .parser import source_to_op
from .utils import _pruned_source
from .._ffi.base import decorate


def register(origin_func, scope_name="global", need_parser_node=False):
    """Register an external function to parser under scope

    Parameters
    ----------
    origin_func: python function
        The function to be registered
        You can write annotations of the arguments to ask the parser do runtime type check.
        Default value is also supported.

    scope_name: str
        The scope name where the function is to be registered， could be "global", "for", "with"

    need_parser_node: Bool
        Whether the function to be registered need parser and ast node as its argument.
        If True, the first and the second argument of origin_func ought to be parser and node
        Check intrin.buffer_bind, intrin.buffer_allocate for an example
    """

    scope_dict = {"global": intrin.GlobalScope, "for": intrin.ForScope, "with": intrin.WithScope}
    if scope_name not in scope_dict.keys():
        raise RuntimeError("TVM Hybrid Script register error : scope should be \"global\", \"for\" or \"with\"")
    else:
        scope = scope_dict[scope_name]

    full_arg_spec = inspect.getfullargspec(origin_func)
    type_hints, args, defaults = full_arg_spec.annotations, full_arg_spec.args, full_arg_spec.defaults
    if defaults is None:
        defaults = tuple()
    if need_parser_node:
        args = args[2:]

    if full_arg_spec.varargs is not None:
        raise RuntimeError("TVM Hybrid Script register error : variable argument is not supported now")
    if full_arg_spec.varkw is not None:
        raise RuntimeError("TVM Hybrid Script register error : variable keyword argument is not supported now")
    if not len(full_arg_spec.kwonlyargs) == 0:
        raise RuntimeError("TVM Hybrid Script register error : keyword only argument is not supported now")

    arg_list = list()
    for arg in args[: len(args) - len(defaults)]:
        arg_list.append((arg, type_hints.get(arg, None)))
    for default, arg in zip(defaults, args[len(args) - len(defaults):]):
        arg_list.append((arg, type_hints.get(arg, None), default))

    intrin.register_func(scope, origin_func.__name__, origin_func, arg_list, need_parser_and_node=need_parser_node,
                         need_return="return" in type_hints.keys())


def _init_scope():
    """Register primitive intrinsic functions"""
    register(intrin.buffer_bind, need_parser_node=True)
    register(intrin.buffer_allocate, need_parser_node=True)
    register(intrin.block_vars, need_parser_node=True)
    register(intrin.block, scope_name="with", need_parser_node=True)
    register(intrin.range, scope_name="for", need_parser_node=True)


def script(origin_func):
    """Decorate a python function function as hybrid script.

    The hybrid function support parsing to the internal TE IR.

    Returns
    -------
    function : TeFunction
        The TeFunction in IR.

    tensors : list of Placeholders
        List of tensors for buffers in function

    tensor_maps : dict of Buffer to Tensor
        Map between buffers in function and tensors
    """

    def wrapped_func(func, *args, **kwargs):
        _init_scope()
        src = _pruned_source(func)
        return source_to_op(func.__code__.co_firstlineno, src, *args, **kwargs)

    return decorate(origin_func, wrapped_func)
