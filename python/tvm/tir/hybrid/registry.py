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
"""Hybrid Script Parser Function Registry """
# pylint: disable=inconsistent-return-statements

import inspect


class Registry(object):
    """Registration map
    All these maps are static
    """
    intrin = dict()
    with_scope = dict()
    for_scope = dict()
    special_stmt = dict()

    host_dict = {
        "intrin": intrin,
        "with_scope": with_scope,
        "for_scope": for_scope,
        "special_stmt": special_stmt
    }


class CallArgumentReader(object):
    """A helper class which read required argument from passed arguments"""

    def __init__(self, func_name, args, kwargs, parser):
        self.func_name = func_name
        self.args = args
        self.kwargs = kwargs
        self.parser = parser

    def get_func_compulsory_arg(self, pos, name):
        """Get corresponding function argument from argument list which is compulsory"""

        if len(self.args) >= pos:
            arg = self.args[pos - 1]
        elif name not in self.kwargs.keys():
            self.parser.report_error(self.func_name + " misses argument " + name)
        else:
            arg = self.kwargs[name]

        return arg

    def get_func_optional_arg(self, pos, name, default):
        """Get corresponding function argument from argument list which is optional.
        If user doesn't provide the argument, set it to default value
        """

        if len(self.args) >= pos:
            arg = self.args[pos - 1]
        elif name in self.kwargs.keys():
            arg = self.kwargs[name]
        else:
            return default

        return arg


def func_wrapper(func_name, func_to_register, arg_list, need_parser_and_node, need_return):
    """Helper function to wrap a function to be registered """

    def wrap_func(parser, node, args, kwargs):
        reader = CallArgumentReader(func_name, args, kwargs, parser)
        internal_args = list()

        if need_parser_and_node:
            internal_args.append(parser)
            internal_args.append(node)

        for i, arg_info in enumerate(arg_list):
            if len(arg_info) == 1:
                arg_name, = arg_info
                internal_args.append(reader.get_func_compulsory_arg(i + 1, arg_name))
            else:
                arg_name, default = arg_info
                internal_args.append(reader.get_func_optional_arg(i + 1, arg_name, default=default))

        if need_return:
            return func_to_register(*internal_args)
        func_to_register(*internal_args)

    return wrap_func


def register_func(category, origin_func, need_parser_and_node, need_return):
    """Helper function to register a function under category

    Parameters
    ----------
    category: str
        The category of function to be registered, ought to be "intrin", "with_scope", "for_scope",
        "special_stmt"

    origin_func: function
        The function to be registered

    need_parser_and_node: bool
        Whether the function need parser and node in its arguments

    need_return: bool
        Whether the function has return value
    """

    full_arg_spec = inspect.getfullargspec(origin_func)

    args, defaults = full_arg_spec.args, full_arg_spec.defaults

    if defaults is None:
        defaults = tuple()
    if need_parser_and_node:
        args = args[2:]

    if full_arg_spec.varargs is not None:
        raise RuntimeError(
            "TVM Hybrid Script register error : variable argument is not supported now")
    if full_arg_spec.varkw is not None:
        raise RuntimeError(
            "TVM Hybrid Script register error : variable keyword argument is not supported now")
    if not len(full_arg_spec.kwonlyargs) == 0:
        raise RuntimeError(
            "TVM Hybrid Script register error : keyword only argument is not supported now")

    arg_list = list()
    for arg in args[: len(args) - len(defaults)]:
        arg_list.append((arg,))
    for default, arg in zip(defaults, args[len(args) - len(defaults):]):
        arg_list.append((arg, default))

    Registry.host_dict[category][origin_func.__name__] = \
        func_wrapper(origin_func.__name__,
                     origin_func, arg_list,
                     need_parser_and_node=need_parser_and_node,
                     need_return=need_return)


def register_intrin(origin_func):
    """Register function under category intrin"""
    register_func("intrin", origin_func, need_parser_and_node=False, need_return=True)


def register_scope_handler(origin_func, scope_name):
    """Register function under category with_scope or for_scope"""
    register_func(scope_name, origin_func, need_parser_and_node=True, need_return=False)


def register_special_stmt(origin_func):
    """Register function under category special_stmt"""
    register_func("special_stmt", origin_func, need_parser_and_node=True, need_return=True)
