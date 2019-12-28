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

from .parser import source_to_op
from .utils import _pruned_source
from .._ffi.base import decorate


def script(origin_func):
    """Decorate a python function function as hybrid script.

    The hybrid function support emulation mode and parsing to
    the internal language IR.

    Returns
    -------
    function : TeFunction
        The TeFunction in IR.

    tensors : list of Placeholders
        List of tensors for buffers in function

    tensor_maps : dict of TeBuffer to Tensor
        Map between buffers in function and tensors
    """

    def wrapped_func(func, *args, **kwargs):
        utils.set_lineno(func.__code__.co_firstlineno)
        src = _pruned_source(func)
        return source_to_op(src, *args, **kwargs)

    return decorate(origin_func, wrapped_func)
