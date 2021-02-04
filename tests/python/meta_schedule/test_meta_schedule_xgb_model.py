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
"""Test XGB cost model"""
import numpy as np

# pylint: disable=missing-function-docstring
import pytest
from tvm.auto_scheduler.cost_model import xgb_model as ansor_xgb_model
from tvm.meta_schedule import xgb_model as ms_xgb_model


def _get_dmatrix(batch):
    xs = [np.random.rand(b, 159) for b in batch]  # pylint: disable=invalid-name
    ys = np.random.rand(len(batch))  # pylint: disable=invalid-name
    ys_pred = np.random.rand(sum(batch))
    ansor_dmatrix = ansor_xgb_model.pack_sum_xgbmatrix(xs=xs, ys=ys, weights=ys)
    ms_dmatrix = ms_xgb_model.PackSum(xs=xs, ys=ys)
    return ansor_dmatrix, ms_dmatrix, ys_pred


@pytest.mark.parametrize(
    "batch",
    [
        [2, 3, 5],
        [2, 3, 5, 1],
        [1],
        [5],
    ],
)
def test_meta_schedule_xgb_model_obj_square_error(batch):
    ansor_dmatrix, ms_dmatrix, ys_pred = _get_dmatrix(batch)
    ansor_result_0, ansor_result_1 = ansor_xgb_model.pack_sum_square_error(ys_pred, ansor_dmatrix)
    ms_result_0, ms_result_1 = ms_dmatrix.obj_square_error(ys_pred)
    np.testing.assert_allclose(actual=ms_result_0, desired=ansor_result_0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual=ms_result_1, desired=ansor_result_1, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "batch",
    [
        [2, 3, 5],
        [2, 3, 5, 1],
        [1],
        [5],
    ],
)
def test_meta_schedule_xgb_model_rmse(batch):
    ansor_dmatrix, ms_dmatrix, ys_pred = _get_dmatrix(batch)
    _, ms_result = ms_dmatrix.rmse(ys_pred)
    _, ansor_result = ansor_xgb_model.pack_sum_rmse(ys_pred, ansor_dmatrix)
    np.testing.assert_allclose(actual=ms_result, desired=ansor_result, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "batch",
    [
        [2, 3, 5],
        [2, 3, 5, 1],
        [1],
        [5],
    ],
)
@pytest.mark.parametrize(
    "n",
    [1, 2, 4, 6, 10, 32],
)
def test_meta_schedule_xgb_model_average_peak_score(batch, n):
    ansor_dmatrix, ms_dmatrix, ys_pred = _get_dmatrix(batch)
    _, ms_result = ms_dmatrix.average_peak_score(ys_pred, n=n)
    _, ansor_result = ansor_xgb_model.pack_sum_average_peak_score(N=n)(ys_pred, ansor_dmatrix)
    np.testing.assert_allclose(actual=ms_result, desired=ansor_result, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_meta_schedule_xgb_model_obj_square_error([2, 3, 5])
    test_meta_schedule_xgb_model_rmse([2, 3, 5])
    test_meta_schedule_xgb_model_average_peak_score([2, 3, 5], 32)
