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
from tvm.tir import Schedule, TensorIntrin

from . import _ffi_api_postproc
from .search import SearchTask


@register_object("meta_schedule.Postproc")
class Postproc(Object):
    """A post processor, used for the search strategy for postprocess the schedule it gets"""

    name: str

    def apply(
        self,
        task: SearchTask,
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
        return _ffi_api_postproc.Apply(self, task, sch, seed)  # pylint: disable=no-member


def rewrite_tensorize(tensor_intrins: List[TensorIntrin]) -> Postproc:
    """Creates a postprocessor that matches the region that is marked as auto tensorized

    Returns
    ----------
    postproc: Postproc
        The postprocessor created
    """
    return _ffi_api_postproc.RewriteTensorize(tensor_intrins)  # pylint: disable=no-member


def rewrite_tensorize_tensor_core(
    compute_intrin: str,
    load_intrin_A: str,
    load_intrin_B: str,
    store_intrin: str,
    init_intrin: str,
) -> Postproc:
    """Creates a postprocessor that matches the region that is marked as auto tensorized

    Parameters
    ----------
    compute_intrin: str
        The tensor intrinsinc for doing computation
    load_intrin_A: str
        The corresponding data load intrinsic for compute_intrin
    load_intrin_B: str
        The corresponding data load intrinsic for compute_intrin
    store_intrin: str
        The corresponing data store instrinsic for compute_intrin
    init_intrin: str
        The intrinsic for tensor core computation

    Returns
    ----------
    postproc: Postproc
        The postprocessor created
    """
    return _ffi_api_postproc.RewriteTensorizeTensorCore(
        compute_intrin, load_intrin_A, load_intrin_B, store_intrin, init_intrin
    )  # pylint: disable=no-member


def rewrite_cooperative_fetch() -> Postproc:
    """Creates a postprocessor rewrites "lazy_cooperative_fetch" with the actual threadIdx

    Returns
    ----------
    postproc: Postproc
        The postprocessor created
    """
    return _ffi_api_postproc.RewriteCooperativeFetch()  # pylint: disable=no-member


def rewrite_cooperative_fetch_tensor_core() -> Postproc:
    """Creates a postprocessor rewrites "lazy_cooperative_fetch" with the threadIdx==32

    Returns
    ----------
    postproc: Postproc
        The postprocessor created
    """
    return _ffi_api_postproc.RewriteCooperativeFetchTensorCore()  # pylint: disable=no-member


def rewrite_unbound_blocks() -> Postproc:
    """Creates a postprocessor that finds each block that is not bound to thread axes,
    and bind them to `blockIdx.x` and `threadIdx.x`

    Returns
    ----------
    postproc: Postproc
        The postprocessor created
    """
    return _ffi_api_postproc.RewriteUnboundBlocks()  # pylint: disable=no-member


def rewrite_parallel_vectorize_unroll() -> Postproc:
    """Creates a postprocessor that applies parallelization, vectorization and auto unrolling,
    according to the annotation of each block

    Returns
    ----------
    postproc: Postproc
        The postprocessor created
    """
    return _ffi_api_postproc.RewriteParallelizeVectorizeUnroll()  # pylint: disable=no-member


def disallow_dynamic_loops() -> Postproc:
    """Create a postprocessor that checks if all loops are static

    Returns
    ----------
    postproc: Postproc
        The postprocessor created
    """
    return _ffi_api_postproc.DisallowDynamicLoops()  # pylint: disable=no-member


def verify_gpu_code() -> Postproc:
    """Creates a postprocessor that do block/vthread/thread binding for cuda

    Returns
    ----------
    postproc: Postproc
        The postprocessor created
    """
    return _ffi_api_postproc.VerifyGPUCode()  # pylint: disable=no-member


def rewrite_reduction_block() -> Postproc:
    """Creates a postprocessor that decomposes ReduceStep

    Returns
    ----------
    postproc: Postproc
        The postprocessor created
    """
    return _ffi_api_postproc.RewriteReductionBlock()  # pylint: disable=no-member


def rewrite_reduction_block_tensor_core() -> Postproc:
    """Creates a postprocessor that decomposes ReduceStep

    Returns
    ----------
    postproc: Postproc
        The postprocessor created
    """
    return _ffi_api_postproc.RewriteReductionBlockTensorCore()  # pylint: disable=no-member


def rewrite_layout() -> Postproc:
    """Creates a postprocessor that rewrites layout

    Returns
    ----------
    postproc: Postproc
        The postprocessor created
    """
    return _ffi_api_postproc.RewriteLayout()  # pylint: disable=no-member
