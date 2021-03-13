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

import logging
from collections import defaultdict
from itertools import chain as itertools_chain
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..autotvm.tuner.metric import max_curve
from .cost_model import PyCostModel
from .feature import per_block_feature_batched
from .measure_record import MeasureInput, MeasureResult
from .schedule import Schedule
from .search import SearchTask
from .utils import cpu_count

if TYPE_CHECKING:
    import xgboost as xgb


logger = logging.getLogger("meta_schedule")


def _make_metric_sorter(focused_metric):
    def metric_name_for_sort(name):
        if focused_metric == name:
            return "!" + name
        return name

    def sort_key(key):
        key, _ = key
        return metric_name_for_sort(key)

    return sort_key


class XGBDMatrixContext:
    """A global context to hold additional attributes of xgb.DMatrix"""

    context_dict: defaultdict

    def __init__(self):
        self.context_dict = defaultdict(dict)

    def clear(self) -> None:
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

    Parameters
    ----------
    dmatrix : xgb.DMatrix
        A float64 array of shape [n, m],
        where `n` is the packed number of blocks,
        and `m` is the length of feature vector on each block
    ids : np.ndarray
        An int64 array of shape [n] containing nonnegative integers,
        indicating which the index of a sample that a block belongs to
    """

    dmatrix: xgb.DMatrix  # pylint: disable=invalid-name
    ids: np.ndarray

    def __init__(
        self,
        xs: List[np.ndarray],
        ys: Optional[List[float]],
    ):
        """Create PackSum format given a batch of samples

        Parameters
        ----------
        xs : List[np.ndarray]
            A batch of input samples
        ys : Optional[List[float]]
            A batch of labels. None means no lables available.
        """
        import xgboost as xgb  # pylint: disable=import-outside-toplevel

        repeats = [x.shape[0] for x in xs]
        xs = np.concatenate(xs, axis=0)
        self.ids = np.concatenate([[i] * repeat for i, repeat in enumerate(repeats)], axis=0)
        if ys is None:
            self.dmatrix = xgb.DMatrix(data=xs, label=None)
        else:
            ys = np.concatenate([[y] * repeat for y, repeat in zip(ys, repeats)], axis=0)
            self.dmatrix = xgb.DMatrix(data=xs, label=ys)
            self.dmatrix.set_weight(ys)

    def predict_with_score(self, pred: np.ndarray) -> np.ndarray:
        return np.bincount(self.ids, weights=pred)

    def obj_square_error(self, ys_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Implement square error loss on pack-sum format as
        a custom objective function for xgboost.

        Parameters
        ----------
        ys_pred: np.ndarray
            The predictions

        Returns
        -------
        gradient: np.ndarray
            The gradient according to the xgboost format
        hessian: np.ndarray
            The hessian according to the xgboost format
        """
        # Making prediction
        ys_pred = self.predict_with_score(ys_pred)
        # Propagate prediction to each block
        ys_pred = ys_pred[self.ids]
        # The gradient and hessian
        ys = self.dmatrix.get_label()  # pylint: disable=invalid-name
        gradient = ys_pred - ys
        hessian = np.ones_like(gradient)
        return gradient * ys, hessian * ys

    def rmse(self, ys_pred: np.ndarray) -> Tuple[str, float]:
        """Evaluate RMSE (rooted mean square error) in the pack-sum format

        Parameters
        ----------
        ys_pred: np.ndarray
            The raw predictions

        Returns
        -------
        name: str
            The name of the metric
        score: float
            The score of the metric
        """
        # Making prediction
        ys_pred = self.predict_with_score(ys_pred)
        # Propagate prediction to each block
        ys_pred = ys_pred[self.ids]
        # The RMSE
        ys = self.dmatrix.get_label()  # pylint: disable=invalid-name
        square_error = np.square(ys_pred - ys)
        rmse = np.sqrt(square_error.mean())
        return "p-rmse", rmse

    def average_peak_score(
        self,
        ys_pred: np.ndarray,
        n: int,
    ) -> Tuple[str, float]:  # pylint: disable=invalid-name
        """Evaluate average-peak-score@N in the pack-sum format

        Parameters
        ----------
        ys_pred: np.ndarray
            The raw prediction
        n : int
            The N in average-peak-score@N

        Returns
        -------
        name: str
            The name of the metric
        score: float
            The score of the metric
        """
        ys = self.dmatrix.get_label()  # pylint: disable=invalid-name
        ys = self.predict_with_score(ys)  # pylint: disable=invalid-name
        ys = ys / np.unique(self.ids, return_counts=True)[1]  # pylint: disable=invalid-name
        ys_pred = self.predict_with_score(ys_pred)
        trials = np.argsort(ys_pred)[::-1][:n]
        trial_scores = ys[trials]
        curve = max_curve(trial_scores) / np.max(ys)
        score = np.mean(curve)
        return "a-peak@%d" % n, score


class XGBModel(PyCostModel):
    """XGBoost model"""

    # model-related
    xgb_max_depth: int
    xgb_gamma: float
    xgb_min_child_weight: float
    xgb_eta: float
    xgb_seed: int
    # serialization-related
    path: Optional[str]
    # behavior of randomness
    num_warmup_samples: int
    # evaluation
    early_stopping_rounds: int
    verbose_eval: int
    average_peak_n: int
    # states
    cached_features: List[np.ndarray]
    cached_mean_costs: np.ndarray
    cached_normalizer: Optional[float]
    booster: Optional[xgb.Booster]

    def __init__(
            self,
            *,
            # model-related
            xgb_max_depth: int = 10,
            xgb_gamma: float = 0.001,
            xgb_min_child_weight: float = 0,
            xgb_eta: float = 0.1,
            xgb_seed: int = 43,
            # serialization-related
            path: Optional[str] = None,
            # behavior of randomness
            num_warmup_samples: int = 100,
            # evaluation
            early_stopping_rounds: int = 50,
            verbose_eval: int = 25,
            average_peak_n: int = 32,
    ):
        super().__init__()
        # model-related
        self.xgb_max_depth = xgb_max_depth
        self.xgb_gamma = xgb_gamma
        self.xgb_min_child_weight = xgb_min_child_weight
        self.xgb_eta = xgb_eta
        self.xgb_seed = xgb_seed
        # serialization-related
        self.path = path
        # behavior of randomness
        self.num_warmup_samples = num_warmup_samples
        # evaluation
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.average_peak_n = average_peak_n
        # states
        self.cached_features = []
        self.cached_mean_costs = np.empty((0,), dtype="float64")
        self.cached_normalizer = None
        self.booster = None

    def update(self, inputs: List[MeasureInput], results: List[MeasureResult]) -> None:
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
        # extract feature and do validation
        new_features = per_block_feature_batched([x.sch for x in inputs])
        new_mean_costs = [x.mean_cost() for x in results]
        if self.booster is not None and self.cached_normalizer is not None:
            logger.debug(
                "XGB validation: %s",
                "\t".join(
                    "%s: %.6f" % (key, score)
                    for key, score in self._validate(
                        xs=new_features,
                        ys=new_mean_costs,
                    )
                ),
            )
        # use together with previous features
        self.cached_features.extend(new_features)
        self.cached_mean_costs = np.append(self.cached_mean_costs, new_mean_costs)
        self.cached_normalizer = np.min(self.cached_mean_costs)
        # train xgb model
        self._train(
            xs=self.cached_features,
            ys=self.cached_mean_costs,
        )
        # Update the model file if it has been set
        if self.path:
            self.save(self.path)

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
        n_measured = len(self.cached_features)
        if self.booster is not None and n_measured >= self.num_warmup_samples:
            ret = self._predict(xs=per_block_feature_batched(schedules))
        else:
            ret = np.random.uniform(  # TODO(@junrushao1994): leaked source of randomness (?)
                low=0,
                high=1,
                size=(len(schedules),),
            ).astype("float64")
        return ret

    def _xgb_params(self) -> Dict[str, Any]:
        return {
            "max_depth": self.xgb_max_depth,
            "gamma": self.xgb_gamma,
            "min_child_weight": self.xgb_min_child_weight,
            "eta": self.xgb_eta,
            "seed": self.xgb_seed,
            "nthread": cpu_count(False),
            "n_gpus": 0,
            "verbosity": 0,
            "disable_default_eval_metric": 1,
        }

    def _train(  # pylint: disable=invalid-name
        self,
        xs: List[np.ndarray],
        ys: List[float],
    ) -> None:
        import xgboost as xgb  # pylint: disable=import-outside-toplevel

        d_train = PackSum(
            xs=xs,
            ys=self.cached_normalizer / ys,
        )

        xgb_dmatrix_context.set("pack-sum", d_train.dmatrix, d_train)

        def _get(d_train: xgb.DMatrix) -> PackSum:
            return xgb_dmatrix_context.get("pack-sum", d_train)

        def obj(ys_pred: np.ndarray, d_train: xgb.DMatrix):
            return _get(d_train).obj_square_error(ys_pred)

        def rmse(ys_pred: np.ndarray, d_train: xgb.DMatrix):
            return _get(d_train).rmse(ys_pred)

        def average_peak_score(ys_pred: np.ndarray, d_train: xgb.DMatrix):
            return _get(d_train).average_peak_score(ys_pred, self.average_peak_n)

        self.booster = xgb.train(
            self._xgb_params(),
            d_train.dmatrix,
            num_boost_round=10000,
            obj=obj,
            callbacks=[
                custom_callback(
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose_eval=self.verbose_eval,
                    fevals=[
                        rmse,
                        average_peak_score,
                    ],
                    evals=[(d_train.dmatrix, "tr")],
                )
            ],
        )
        xgb_dmatrix_context.clear()

    def _predict(  # pylint: disable=invalid-name
        self,
        xs: List[np.ndarray],
    ) -> np.ndarray:
        d_test = PackSum(xs=xs, ys=None)
        pred = self.booster.predict(d_test.dmatrix)
        ret = d_test.predict_with_score(pred)
        return ret.astype("float64")

    def _validate(  # pylint: disable=invalid-name
        self,
        xs: List[np.ndarray],
        ys: List[float],
    ) -> List[Tuple[str, float]]:
        """Evaluate the score of states

        Parameters
        ----------
        xs : List[np.ndarray]
            A batch of input samples
        ys : List[float]
            A batch of labels

        Returns
        -------
        scores: np.ndarray
            The predicted scores for all states
        """
        if self.booster is None or self.cached_normalizer is None:
            return []

        d_valid = PackSum(
            xs=xs,
            ys=self.cached_normalizer / ys,
        )

        def average_peak_score(ys_pred: np.ndarray):
            return d_valid.average_peak_score(ys_pred, n=self.average_peak_n)

        ys_pred = self.booster.predict(d_valid.dmatrix)
        eval_result: List[Tuple[str, float]] = [
            feval(ys_pred)
            for feval in (
                average_peak_score,
                d_valid.rmse,
            )
        ]
        eval_result.sort(key=_make_metric_sorter("p-rmse"))
        return eval_result

    def load(self, path: str) -> XGBModel:
        """Load the model from a file

        Parameters
        ----------
        path: str
            The filename
        """
        if self.booster is None:
            self.bst = xgb.Booster(self._xgb_params())
        self.booster.load_model(path)
        self.num_warmup_sample = 0
        return self

    def save(self, path: str) -> XGBModel:
        """Save the model to a file

        Parameters
        ----------
        path: str
            The filename
        """
        self.booster.save_model(path)
        return self


def custom_callback(
    early_stopping_rounds: int,
    verbose_eval: int,
    fevals: List[Callable],
    evals: List[Tuple[xgb.DMatrix, str]],
    focused_metric: str = "tr-p-rmse",
):
    """Callback function for xgboost to support multiple custom evaluation functions"""
    # pylint: disable=import-outside-toplevel
    import xgboost as xgb
    from xgboost.callback import _fmt_metric
    from xgboost.core import EarlyStopException

    try:
        from xgboost.training import aggcv
    except ImportError:
        from xgboost.callback import _aggcv as aggcv
    # pylint: enable=import-outside-toplevel

    sort_key = _make_metric_sorter(focused_metric=focused_metric)

    state = {}

    def init(env: xgb.core.CallbackEnv):
        """Internal function"""
        booster: xgb.Booster = env.model

        state["best_iteration"] = 0
        state["best_score"] = float("inf")
        if booster is None:
            assert env.cvfolds is not None
            return
        if booster.attr("best_score") is not None:
            state["best_score"] = float(booster.attr("best_score"))
            state["best_iteration"] = int(booster.attr("best_iteration"))
            state["best_msg"] = booster.attr("best_msg")
        else:
            booster.set_attr(best_iteration=str(state["best_iteration"]))
            booster.set_attr(best_score=str(state["best_score"]))

    def callback(env: xgb.core.CallbackEnv):
        if not state:
            init(env)
        booster: xgb.Booster = env.model
        iteration: int = env.iteration
        cvfolds: List[xgb.training.CVPack] = env.cvfolds
        ##### Evaluation #####
        # `eval_result` is a list of (key, score)
        eval_result: List[Tuple[str, float]] = []
        if cvfolds is None:
            eval_result = itertools_chain.from_iterable(
                [
                    (key, float(value))
                    for key, value in map(
                        lambda x: x.split(":"),
                        booster.eval_set(
                            evals=evals,
                            iteration=iteration,
                            feval=feval,
                        ).split()[1:],
                    )
                ]
                for feval in fevals
            )
        else:
            eval_result = itertools_chain.from_iterable(
                [
                    (key, score)
                    for key, score, _std in aggcv(
                        fold.eval(
                            iteration=iteration,
                            feval=feval,
                        )
                        for fold in cvfolds
                    )
                ]
                for feval in fevals
            )
        eval_result = list(eval_result)
        eval_result.sort(key=sort_key)

        ##### Print eval result #####
        if verbose_eval and iteration % verbose_eval == 0:
            info = []
            for key, score in eval_result:
                if "null" in key:
                    continue
                info.append("%s: %.6f" % (key, score))
            logger.debug("XGB iter %3d: %s", iteration, "\t".join(info))

        ##### Choose score and do early stopping #####
        score = None
        for key, _score in eval_result:
            if key == focused_metric:
                score = _score
                break
        assert score is not None

        best_score = state["best_score"]
        best_iteration = state["best_iteration"]
        if score < best_score:
            msg = "[%d] %s" % (env.iteration, "\t".join([_fmt_metric(x) for x in eval_result]))
            state["best_msg"] = msg
            state["best_score"] = score
            state["best_iteration"] = env.iteration
            # save the property to attributes, so they will occur in checkpoint.
            if env.model is not None:
                env.model.set_attr(
                    best_score=str(state["best_score"]),
                    best_iteration=str(state["best_iteration"]),
                    best_msg=state["best_msg"],
                )
        elif env.iteration - best_iteration >= early_stopping_rounds:
            best_msg = state["best_msg"]
            if verbose_eval and env.rank == 0:
                logger.debug("XGB stopped. Best iteration: %s ", best_msg)
            raise EarlyStopException(best_iteration)

    return callback
