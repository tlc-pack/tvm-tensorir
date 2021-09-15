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
"""Workload Registry"""

from typing import Any, Tuple

from tvm._ffi import register_object
from tvm.ir import IRModule
from tvm.runtime import Object

from . import _ffi_api


@register_object("meta_schedule.WorkloadToken")
class WorkloadToken(Object):
    """The class of workload tokens.

    Parameters
    ----------
    mod : tvm.ir.IRModule
        The IRModule of the workload.
    shash : int
        The structural hash of the workload.
    """

    mod: IRModule
    shash: int

    def __init__(self, mod: IRModule, shash: int, token_id: int) -> None:
        """Constructor."""
        self.__init_handle_by_constructor__(
            _ffi_api.WorkloadToken,  # type: ignore # pylint: disable=no-member
            mod,
            shash,
            token_id,
        )

    @staticmethod
    def from_json(json_obj: Any, token_id: int) -> "WorkloadToken":
        """Create a workload token from json object.

        Parameters
        ----------
        json_obj : Any
            The json object.
        token_id : int
            The token id.

        Returns
        -------
        workload_token : WorkloadToken
            The workload token.
        """
        return _ffi_api.WorkloadTokenFromJSON(json_obj, token_id)  # type: ignore # pylint: disable=no-member

    def as_json(self) -> Tuple[str, str]:
        """Export the workload token to a JSON string.

        Returns
        -------
        json_str_tuple : Tuple[str, str]
            An array containing the structural hash and the base64 json string.
        """
        shash_str, mod = _ffi_api.WorkloadTokenAsJSON(self)  # type: ignore # pylint: disable=no-member
        return (shash_str, mod)


@register_object("meta_schedule.WorkloadRegistry")
class WorkloadRegistry(Object):
    """The class for workload registry.

    Parameters
    ----------
    path : str
        The path of the workload registry.
    """

    path: str

    def __init__(self, path: str, allow_missing: bool) -> None:
        """Constructor."""
        self.__init_handle_by_constructor__(
            _ffi_api.WorkloadRegistry,  # type: ignore # pylint: disable=no-member
            path,
            allow_missing,
        )

    def lookup_or_add(self, mod: IRModule) -> WorkloadToken:
        """Lookup or add a workload.

        Parameters
        ----------
        mod : tvm.ir.IRModule
            The IRModule of the workload.

        Returns
        -------
        workload_token : WorkloadToken
            The workload token of the IRModule.
        """
        return _ffi_api.WorkloadRegistryLookupOrAdd(self, mod)  # type: ignore # pylint: disable=no-member

    def __len__(self) -> int:
        """Get size of the workload registry.

        Returns
        -------
        size : int
            The size of the workload registry.
        """
        return _ffi_api.WorkloadRegistrySize(self)  # type: ignore # pylint: disable=no-member

    def __getitem__(self, index: int) -> WorkloadToken:
        """Get a workload token by index.

        Parameters
        ----------
        index : int
            The index of the workload token.

        Returns
        -------
        workload_token : WorkloadToken
            The workload token with given index.
        """
        return _ffi_api.WorkloadRegistryAt(self, index)  # type: ignore # pylint: disable=no-member
