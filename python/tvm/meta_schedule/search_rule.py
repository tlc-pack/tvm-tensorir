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
"""Search rules in meta schedule"""

from . import _ffi_api_rule
from .search import SearchRule


def multi_level_tiling(tiling_structure: str) -> SearchRule:
    """Create a rule that does multi-level tiling if there is sufficient amout of data reuse

    Parameters
    ----------
    tiling_structure: str
        Structure of tiling. On CPU, recommended to use 'SSRSRS';
        On GPU, recommended to use 'SSSRRSRS'

    Returns
    ----------
    rule: SearchRule
        A search rule that does multi-level tiling
    """

    return _ffi_api_rule.MultiLevelTiling(tiling_structure)  # pylint: disable=no-member


def multi_level_tiling_with_fusion(tiling_structure: str) -> SearchRule:
    """Create a rule that does multi-level tiling and fusion together
    if there is sufficient amout of data reuse

    Parameters
    ----------
    tiling_structure: str
        Structure of tiling. On CPU, recommended to use 'SSRSRS';
        On GPU, recommended to use 'SSSRRSRS'

    Returns
    ----------
    rule: SearchRule
        A search rule that does multi-level tiling with fusion
    """

    return _ffi_api_rule.MultiLevelTilingWithFusion(  # pylint: disable=no-member
        tiling_structure
    )
