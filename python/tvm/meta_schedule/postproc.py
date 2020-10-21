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
"""Postprocessors are applied at the final stage before a schedule is sent to be measured"""

from typing import Optional

from tvm._ffi import register_object
from tvm.runtime import Object

from . import _ffi_api_postproc
from .schedule import Schedule


@register_object("meta_schedule.Postproc")
class Postproc(Object):
    """A post processor, used for the search strategy for postprocess the schedule it gets"""

    name: str

    def apply(
        self,
        sch: Schedule,
        seed: Optional[int] = None,
    ) -> bool:
        """Postprocess the schedule.

        Parameters
        ----------
        sch : Schedule
            The schedule to be mutated
        seed : Optional[int]
            The random seed

        Returns
        -------
        result : bool
            If the post-processing succeeds
        """
        return _ffi_api_postproc.Apply(self, sch, seed)  # pylint: disable=no-member
