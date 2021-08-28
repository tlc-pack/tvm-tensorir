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
"""ArgInfo"""
from typing import Any, List, Tuple, Union

from tvm._ffi import register_object
from tvm.runtime import DataType, Device, NDArray, Object, ndarray
from tvm.runtime.container import ShapeTuple

from . import _ffi_api

Arg = Union[
    int,
    float,
    NDArray,
]
Args = List[Arg]
PyArgInfo = List[Any]
PyArgsInfo = List[PyArgInfo]


@register_object("meta_schedule.ArgInfo")
class ArgInfo(Object):
    def as_python(self) -> PyArgInfo:
        raise NotImplementedError

    @staticmethod
    def alloc(arg_info: PyArgInfo, device: Device) -> Arg:
        subtype = _TYPE_DICT.get(arg_info[0], None)
        if subtype is None:
            raise ValueError(f"Unable to recognize argument information: {arg_info}")
        return subtype.alloc(arg_info, device)


@register_object("meta_schedule.TensorArgInfo")
class TensorArgInfo(Object):
    TYPE_STR = "TENSOR"

    dtype: DataType
    shape: ShapeTuple

    def __init__(
        self,
        dtype: DataType,
        shape: Union[ShapeTuple, List[int]],
    ) -> None:
        if isinstance(shape, ShapeTuple):
            shape_tuple = shape
        else:
            shape_tuple = ShapeTuple(shape)
        self.__init_handle_by_constructor__(
            _ffi_api.TensorArgInfo,  # type: ignore # pylint: disable=no-member
            dtype,
            shape_tuple,
        )

    def as_python(self) -> Tuple[str, str, List[int]]:
        return (
            TensorArgInfo.TYPE_STR,
            str(self.dtype),
            list(self.shape[:]),
        )

    @staticmethod
    def alloc(arg_info: PyArgInfo, device: Device) -> NDArray:
        _, dtype, shape = arg_info
        return ndarray.empty(shape=shape, dtype=dtype, device=device)


_TYPE_DICT = {
    TensorArgInfo.TYPE_STR: TensorArgInfo,
}
