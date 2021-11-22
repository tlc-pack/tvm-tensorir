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
"""Cost model metrics for meta schedule"""
from typing import List
import numpy as np


def max_curve(trial_scores: np.ndarray) -> List[float]:
    """f(n) = max([s[i] fo i < n])

    Parameters
    ----------
    trial_scores : List[float]
        the score of i-th trial

    Returns
    -------
    curve : List[float]
        function values
    """
    ret = np.empty(len(trial_scores))
    keep = -1e9
    for i, score in enumerate(trial_scores):
        keep = max(keep, score)
        ret[i] = keep
    return ret
