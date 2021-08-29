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
"""Abstraction for sparse data structures."""
from numbers import Integral
import tvm._ffi

from tvm._ffi.base import string_types
from tvm.runtime import Object, convert
from tvm.ir import PrimExpr, PointerType, PrimType
from . import _ffi_api

@tvm._ffi.register_object("tir.Buffer")
class Buffer(Object):
    """Symbolic sparse data buffer in TVM.

    Buffer provide a way to represent sparse data layout
    specialization of data structure in TVM.

    See Also
    --------
    decl_buffer : Declare a buffer
    """
    pass

