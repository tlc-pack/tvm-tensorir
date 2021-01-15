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
"""XGBoost-based cost model"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, List, Optional, Dict, Tuple

import numpy as np

from .cost_model import PyCostModel
from .feature import per_block_feature
from .measure_record import MeasureInput, MeasureResult
from .search import SearchTask
from .schedule import Schedule

if TYPE_CHECKING:
    import xgboost as xgb


class XGBDMatrixContext:
    """A global context to hold additional attributes of xgb.DMatrix"""

    context_dict: defaultdict

    def __init__(self):
        self.context_dict = defaultdict(dict)

    def get(
        self,
        key: str,
        matrix: xgb.DMatrix,
        default: Optional[Any] = None,
    ) -> Any:
        """
        Get an attribute of a xgb.DMatrix
        Parameters
        ----------
        key: str
            The name of the attribute
        matrix: xgb.DMatrix
            The matrix
        default: Optional[Any]
            The default value if the item does not exist
        """
        return self.context_dict[key].get(matrix.handle.value, default)

    def set(
        self,
        key: str,
        matrix: xgb.DMatrix,
        value: Any,
    ):
        """
        Set an attribute for a xgb.DMatrix
        Parameters
        ----------
        key: str
            The name of the attribute
        matrix: xgb.DMatrix
            The matrix
        value: Optional[Any]
            The new value
        """
        self.context_dict[key][matrix.handle.value] = value


xgb_dmatrix_context = XGBDMatrixContext()


class PackSum:
    """The pack-sum format

    'dmatrix' is a float64 array of shape [n, m],
    where `n` is the packed number of blocks,
    and `m` is the length of feature vector on each block

    'ids' is an int64 array of shape [n] containing nonnegative integers,
    indicating which the index of a sample that a block belongs to
    """

    dmatrix: xgb.DMatrix  # pylint: disable=invalid-name
    ids: np.ndarray

    def __init__(
        self,
        xs: List[np.ndarray],
        ys: Optional[List[np.ndarray]],
    ):
        repeats = [x.shape[0] for x in xs]
        xs = np.concatenate(xs, axis=0)
        self.ids = np.concatenate([[i] * repeat for i, repeat in enumerate(repeats)], axis=0)
        if ys is None:
            self.dmatrix = xgb.DMatrix(data=xs, label=None)
            return
        ys = np.concatenate([[y] * repeat for y, repeat in zip(ys, repeats)], axis=0)
        self.dmatrix = xgb.DMatrix(data=xs, label=ys)
        self.dmatrix.set_weight(ys)
        xgb_dmatrix_context.set("pack-sum", self.dmatrix, self)

    def predict_with_booster(self, booster: xgb.Booster) -> np.ndarray:
        pred = booster.predict(self.dmatrix)
        return np.bincount(self.ids, weights=pred)

    def predict_with_score(self, pred) -> np.ndarray:
        return np.bincount(self.ids, weights=pred)

    def square_error_loss(self, xs_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Implement square error loss on pack-sum format as
        a custom objective function for xgboost.

        Parameters
        ----------
        xs_pred: np.ndarray
            The predictions

        Returns
        -------
        gradient: np.ndarray
            The gradient according to the xgboost format
        hessian: np.ndarray
            The hessian according to the xgboost format
        """
        ys = self.dmatrix.get_label()  # pylint: disable=invalid-name
        # Making prediction
        xs_pred = self.predict_with_score(xs_pred)
        # Propagate prediction to each block
        xs_pred = xs_pred[self.ids]
        # The gradient and hessian
        gradient = xs_pred - ys
        hessian = np.ones_like(gradient)
        return gradient * ys, hessian * ys


class XGBModel(PyCostModel):
    """XGBoost model"""

    booster: Optional[xgb.Booster]
    xgb_params: Dict[str, Any]
    plan_size: int
    num_warmup_sample: int
    verbose_eval: int
    model_file: Optional[str]

    cached_features: List[np.ndarray]
    cached_mean_costs: np.ndarray

    def __init__(
        self,
        verbose_eval: int = 25,
        num_warmup_sample: int = 100,
        model_file: Optional[str] = None,
        seed: int = 43,
    ):
        super().__init__()
        self.xgb_params = {
            "max_depth": 10,
            "gamma": 0.001,
            "min_child_weight": 0,
            "eta": 0.2,
            "n_gpus": 0,
            # "nthread": multiprocessing.cpu_count() // 2,
            "verbosity": 0,
            "seed": seed,
            "disable_default_eval_metric": 1,
        }
        self.booster = None
        self.plan_size = 32
        self.num_warmup_sample = num_warmup_sample
        self.verbose_eval = verbose_eval
        self.model_file = model_file
        self.cached_features = []
        self.cached_mean_costs = np.empty((0,), dtype="float64")

    def update(self, inputs: List[MeasureInput], results: List[MeasureResult]):
        """Update the cost model according to new measurement results (training data).
        XGBoost does not support incremental training, so we re-train a new model every time.
        Parameters
        ----------
        inputs : List[MeasureInput]
            The measurement inputs
        results : List[MeasureResult]
            The measurement results
        """
        assert len(inputs) == len(results)
        if len(inputs) == 0:
            return

        # extract feature
        self.cached_features.extend(per_block_feature(x.sch) for x in inputs)
        self.cached_mean_costs = np.append(self.cached_mean_costs, [x.mean_cost for x in results])
        features = self.cached_features
        mean_costs = self.cached_throughputs.min() / self.cached_throughputs
        # train xgb model
        d_train = PackSum(xs=features, ys=mean_costs)
        self.booster = xgb.train(
            self.xgb_params,
            d_train.dmatrix,
            num_boost_round=10000,
            obj=pack_sum_square_error,
            callbacks=[
                # custom_callback(
                #     stopping_rounds=50,
                #     metric="tr-p-rmse",
                #     fevals=[
                #         pack_sum_rmse,
                #         pack_sum_average_peak_score(self.plan_size),
                #     ],
                #     evals=[(d_train, "tr")],
                #     maximize=False,
                #     verbose_eval=self.verbose_eval,
                # )
            ],
        )
        # Update the model file if it has been set
        # if self.model_file:
        #     self.save(self.model_file)

    def predict(self, task: SearchTask, schedules: List[Schedule]) -> np.ndarray:
        """Predict the scores of states

        Parameters
        ----------
        search_task : SearchTask
            The search task of states
        schedules : List[Schedule]
            The input states

        Returns
        -------
        scores: np.ndarray
            The predicted scores for all states
        """
        if self.booster is not None and len(self.inputs) > self.num_warmup_sample:
            features = [per_block_feature(x) for x in schedules]
            d_test = PackSum(xs=features, ys=None)
            ret = d_test.predict_with_booster(self.booster)
        else:
            n = len(schedules)
            ret = np.random.uniform(0, 1, (n,))
        ret = ret.astype("float64")
        return ret


def pack_sum_square_error(xs_pred, d_train):
    """Implement square error loss on pack-sum format as
     a custom objective function for xgboost.

    Parameters
    ----------
    xs_pred: np.ndarray
        The predictions
    d_train: xgb.DMatrix
        The training set

    Returns
    -------
    gradient: np.ndarray
        The gradient according to the xgboost format
    hessian: np.ndarray
        The hessian according to the xgboost format
    """
    d_train: PackSum = xgb_dmatrix_context.get("pack-sum", d_train)
    ys = d_train.dmatrix.get_label()  # pylint: disable=invalid-name
    # TODO(@junrushao1994): isn't the following two statements useless?
    xs_pred = d_train.predict_with_score(xs_pred)
    xs_pred = xs_pred[d_train.ids]
    gradient = xs_pred - ys
    hessian = np.ones_like(gradient)
    if len(ys) == 0:
        return gradient, hessian
    return gradient * ys, hessian * ys
