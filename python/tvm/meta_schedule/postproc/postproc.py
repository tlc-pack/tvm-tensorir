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
"""Meta Schedule Postproc."""

from typing import TYPE_CHECKING

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.tir.schedule import Schedule

from .. import _ffi_api
from ..utils import _get_hex_address, check_override

if TYPE_CHECKING:
    from ..tune_context import TuneContext


@register_object("meta_schedule.Postproc")
class Postproc(Object):
    """Rules to apply a post processing to a schedule.

    Note
    ----
    Post processing is designed to deal with the problem of undertermined schedule validity after
    applying some schedule primitves at runtime. E.g., Fuse the first X loops to reach the maximum
    number below 1024, X is only decided at runtime.
    """

    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        """Initialize the post processing with a tune context.

        Parameters
        ----------
        tune_context : TuneContext
            The tuning context for initializing the post processing.
        """
        _ffi_api.PostprocInitializeWithTuneContext(  # type: ignore # pylint: disable=no-member
            self, tune_context
        )

    def apply(self, sch: Schedule) -> bool:
        """Apply a post processing to the given schedule.

        Parameters
        ----------
        sch : Schedule
            The schedule to be post processed.

        Returns
        -------
        result : bool
            Whether the post processing was successfully applied.
        """
        return _ffi_api.PostprocApply(self, sch)


@register_object("meta_schedule.PyPostproc")
class PyPostproc(Postproc):
    """An abstract Postproc with customized methods on the python-side."""

    def __init__(self):
        """Constructor."""

        @check_override(self.__class__, Postproc)
        def f_initialize_with_tune_context(tune_context: "TuneContext") -> None:
            self.initialize_with_tune_context(tune_context)

        @check_override(self.__class__, Postproc)
        def f_apply(sch: Schedule) -> bool:
            return self.apply(sch)

        def f_as_string() -> str:
            return str(self)

        self.__init_handle_by_constructor__(
            _ffi_api.PostprocPyPostproc,  # type: ignore # pylint: disable=no-member
            f_initialize_with_tune_context,
            f_apply,
            f_as_string,
        )

    def __str__(self) -> str:
        return f"PyPostproc({_get_hex_address(self.handle)})"


@register_object("meta_schedule.VerifyGPUCode")
class VerifyGPUCode(Postproc):

    def __init__(self) -> None:
        self.__init_handle_by_constructor__(_ffi_api.PostprocVerifyGPUCode)


@register_object("meta_schedule.DisallowDynamicLoops")
class DisallowDynamicLoops(Postproc):

    def __init__(self) -> None:
        self.__init_handle_by_constructor__(_ffi_api.PostprocDisallowDynamicLoops)
