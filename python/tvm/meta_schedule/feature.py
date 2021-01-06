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

from .schedule import Schedule
from . import _ffi_api
from ..runtime.ndarray import NDArray


def calc_per_block_feature(sch: Schedule, max_num_buffer_access_features: int = 5) -> NDArray:
    return _ffi_api.CalcPerBlockFeature(  # pylint: disable=no-member
        sch, max_num_buffer_access_features
    )


def per_bloc_feature_names(max_num_buffer_access_features: int = 5) -> NDArray:
    return _ffi_api.PerBlockFeatureNames(  # pylint: disable=no-member
        max_num_buffer_access_features
    )
