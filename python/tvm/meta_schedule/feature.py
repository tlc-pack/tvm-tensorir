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
"""Feature extraction interface"""
from typing import List

import numpy as np

from . import _ffi_api
from ..tir.schedule import Schedule


def per_block_feature(sch: Schedule, max_num_buffer_access_features: int = 5) -> np.ndarray:
    """Calculate the per-block feature

    Parameters
    ----------
    sch : Schedule
        The meta schedule class
    max_num_buffer_access_features : int
        The maximum number of buffer accesses

    Returns
    -------
    features: np.ndarray
        A 2d matrix, the feature vectors for each block
    """
    return _ffi_api.PerBlockFeature(  # pylint: disable=no-member
        sch, max_num_buffer_access_features
    ).asnumpy()


def per_block_feature_batched(
    schs: List[Schedule],
    max_num_buffer_access_features: int = 5,
) -> List[np.ndarray]:
    """Calculate the per-block feature in a batch

    Parameters
    ----------
    sch : Schedule
        The meta schedule class
    max_num_buffer_access_features : int
        The maximum number of buffer accesses

    Returns
    -------
    features: np.ndarray
        A 2d matrix, the feature vectors for each block
    """
    result = _ffi_api.PerBlockFeatureBatched(  # pylint: disable=no-member
        schs, max_num_buffer_access_features
    )
    return [x.asnumpy() for x in result]


def per_bloc_feature_names(max_num_buffer_access_features: int = 5) -> List[str]:
    return _ffi_api.PerBlockFeatureNames(  # pylint: disable=no-member
        max_num_buffer_access_features
    )
