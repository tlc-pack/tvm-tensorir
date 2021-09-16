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
"""Tuning record database"""
from typing import Any, List, TYPE_CHECKING

from tvm._ffi import register_object
from tvm.ir.module import IRModule
from tvm.runtime import Object
from tvm.target import Target
from tvm.tir.schedule import Trace

from .. import _ffi_api
from ..arg_info import ArgInfo
from ..workload_registry import WorkloadRegistry, WorkloadToken

if TYPE_CHECKING:
    from ..tune_context import TuneContext


@register_object("meta_schedule.TuningRecord")
class TuningRecord(Object):
    """
    The class of tuning records.

    Parameters
    ----------
    trace : tvm.ir.Trace
        The trace of the tuning record.
    run_secs : List[float]
        The run time of the tuning record.
    workload : WorkloadToken
        The workload token of the tuning record.
    target : Target
        The target of the tuning record.
    args_info : List[ArgInfo]
        The argument information of the tuning record.
    """

    trace: Trace
    run_secs: List[float]
    workload: WorkloadToken
    target: Target
    args_info: List[ArgInfo]

    def __init__(
        self,
        trace: Trace,
        run_secs: List[float],
        workload: WorkloadToken,
        target: Target,
        args_info: List[ArgInfo],
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.TuningRecord,  # type: ignore # pylint: disable=no-member
            trace,
            run_secs,
            workload,
            target,
            args_info,
        )

    def as_json(self) -> Any:
        """Export the tuning record to a JSON string.

        Returns
        -------
        json_str : str
            The JSON string exported.
        """
        return _ffi_api.TuningRecordAsJSON(self)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def from_json(json_obj: Any, reg: WorkloadRegistry) -> "TuningRecord":
        """Create a tuning record from a json object.

        Parameters
        ----------
        json_obj : Any
            The json object to parse.
        reg : WorkloadRegistry
            The workload registry.

        Returns
        -------
        tuning_record : TuningRecord
            The parsed tuning record.
        """
        return _ffi_api.TuningRecordFromJSON(json_obj, reg)  # type: ignore # pylint: disable=no-member


@register_object("meta_schedule.Database")
class Database(Object):
    """The abstract database interface."""

    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        """Initialize the database with tuning context.

        Parameters
        ----------
        tune_context : TuneContext
            The tuning context.
        """
        _ffi_api.DatabaseInitializeWithTuneContext(self, tune_context)  # type: ignore # pylint: disable=no-member

    def add(self, record: TuningRecord) -> None:
        """Add a tuning record to the database.

        Parameters
        ----------
        record : TuningRecord
            The tuning record to add.
        """
        _ffi_api.DatabaseAdd(self, record)  # type: ignore # pylint: disable=no-member

    def get_top_k(self, workload: WorkloadToken, top_k: int) -> List[TuningRecord]:
        """Get the top K tuning records of given workload from the database.

        Parameters
        ----------
        workload : WorkloadToken
            The workload to be searched for.
        top_k : int
            The number of top records to get.

        Returns
        -------
        top_k_records : List[TuningRecord]
            The top K records.
        """
        return _ffi_api.DatabaseGetTopK(self, workload, top_k)  # type: ignore # pylint: disable=no-member

    def lookup_or_add(self, mod: IRModule) -> WorkloadToken:
        """Look up or add workload to the database if missing.

        Parameters
        ----------
        mod : IRModule
            The IRModule to be searched for or added.

        Returns
        -------
        workload : WorkloadToken
            The workload token of the given IRModule.
        """
        return _ffi_api.DatabaseLookupOrAdd(self, mod)  # type: ignore # pylint: disable=no-member

    def __len__(self) -> int:
        """Get the number of records in the database.

        Returns
        -------
        num_records : int
        """
        return _ffi_api.DatabaseSize(self)  # type: ignore # pylint: disable=no-member


@register_object("meta_schedule.PyDatabase")
class PyDatabase(Database):
    """An abstract Database with customized methods on the python-side."""

    def __init__(self):
        """Constructor."""

        def f_initialize_with_tune_context(tune_context: "TuneContext") -> None:
            self.initialize_with_tune_context(tune_context)

        def f_add(record: TuningRecord) -> None:
            self.add(record)

        def f_get_top_k(workload: WorkloadToken, top_k: int) -> List[TuningRecord]:
            return self.get_top_k(workload, top_k)

        def f_lookup_or_add(mod: IRModule) -> WorkloadToken:
            return self.lookup_or_add(mod)

        def f_size() -> int:
            return self.__len__()

        self.__init_handle_by_constructor__(
            _ffi_api.DatabasePyDatabase,  # pylint: disable=no-member
            f_initialize_with_tune_context,
            f_add,
            f_get_top_k,
            f_lookup_or_add,
            f_size,
        )

    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        raise NotImplementedError

    def add(self, record: TuningRecord) -> None:
        raise NotImplementedError

    def get_top_k(self, workload: WorkloadToken, top_k: int) -> List[TuningRecord]:
        raise NotImplementedError

    def lookup_or_add(self, mod: IRModule) -> WorkloadToken:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
