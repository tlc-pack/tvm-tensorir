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
"""Rule that mark parallelize, vectorize and unroll to each block correspondingly"""
from typing import List, Optional

from tvm._ffi import register_object

from .. import _ffi_api
from .schedule_rule import ScheduleRule


@register_object("meta_schedule.ParallelizeVectorizeUnroll")
class ParallelizeVectorizeUnroll(ScheduleRule):
    """Rule that mark parallelize, vectorize and unroll to each block correspondingly

    Parameters
    ----------
    max_jobs_per_core: int
        The maximum number of jobs to be launched per CPU core. It sets the uplimit of CPU
        parallelism, i.e. `num_cores * max_jobs_per_core`.
        Use -1 to disable parallelism.
    max_vectorize_extent: int
        The maximum extent to be vectorized. It sets the uplimit of the CPU vectorization.
        Use -1 to disable vectorization.
    unroll_max_steps: Optional[List[int]]
        The maximum number of unroll steps to be done.
        Use None to disable unroll
    unroll_explicit: bool
        Whether to explicitly unroll the loop, or just add a unroll pragma
    """

    def __init__(
        self,
        max_jobs_per_core: int = 16,
        max_vectorize_extent: int = 32,
        unroll_max_steps: Optional[List[int]] = None,
        unroll_explicit: bool = True,
    ) -> None:
        if unroll_max_steps is None:
            unroll_max_steps = []
        self.__init_handle_by_constructor__(
            _ffi_api.ScheduleRuleParallelizeVectorizeUnroll,  # type: ignore # pylint: disable=no-member
            max_jobs_per_core,
            max_vectorize_extent,
            unroll_max_steps,
            unroll_explicit,
        )
