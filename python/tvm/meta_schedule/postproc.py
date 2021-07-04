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

from typing import List, Optional

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.target import Target
from tvm.tir import TensorIntrin

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


def rewrite_parallel() -> Postproc:
    """Creates a postprocessor that fuses the loops which are marked as "lazy_parallel",
    and then parallelize the fused loop

    Returns
    ----------
    postproc: Postproc
        The postprocessor created
    """
    return _ffi_api_postproc.RewriteParallel()  # pylint: disable=no-member


def rewrite_vectorize() -> Postproc:
    """Creates a postprocessor that fuses the loops which are marked as "lazy_vectorize",
    and then apply vectorization on the fused loop

    Returns
    ----------
    postproc: Postproc
        The postprocessor created
    """
    return _ffi_api_postproc.RewriteVectorize()  # pylint: disable=no-member


def rewrite_tensorize(tensor_intrins: List[TensorIntrin]) -> Postproc:
    """Creates a postprocessor that matches the region that is marked as "lazy_tensorize"

    Returns
    ----------
    postproc: Postproc
        The postprocessor created
    """
    return _ffi_api_postproc.RewriteTensorize(tensor_intrins)  # pylint: disable=no-member


def rewrite_cuda_thread_bind(warp_size: int) -> Postproc:
    """Creates a postprocessor that do block/vthread/thread binding for cuda

    Returns
    ----------
    postproc: Postproc
        The postprocessor created
    """
    return _ffi_api_postproc.RewriteCudaThreadBind(warp_size)  # pylint: disable=no-member


def verify_gpu_code(target: Target.TYPE) -> Postproc:
    """Creates a postprocessor that do block/vthread/thread binding for cuda

    Returns
    ----------
    postproc: Postproc
        The postprocessor created
    """
    return _ffi_api_postproc.VerifyGPUCode(Target(target))  # pylint: disable=no-member
