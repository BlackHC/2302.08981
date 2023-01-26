import functools
import sys

import catboost
import torch
from sklearn import ensemble
import numpy as np
from torch import nn

from bmdal_reg.splittable_module import SplittableModule


# Abstract base class for all sklearn models
class SklearnModel(SplittableModule):
    def __init__(self):
        super().__init__(10)
        self.model = None

    def eval(self):
        pass

    def fit(self, x: np.ndarray, y: np.ndarray, eval_x: np.ndarray, eval_y: np.ndarray):
        # raise ValueError(f"{x.shape} {y.shape} {eval_x.shape} {eval_y.shape}")
        return self._fit(x, y.squeeze(-1), eval_x, eval_y.squeeze(-1))

    def _fit(self, x, y, eval_x, eval_y):
        raise NotImplementedError

    def __call__(self, x):
        # Assert that we have [1, n_points, n_features] shape
        if len(x.shape) != 3 or x.shape[0] != 1:
            raise ValueError(f"{x.shape}")
        x = x.squeeze(0)
        result = torch.as_tensor(self.predict(x.cpu().numpy()), device=x.device)
        return result[None, :, None]

    def predict(self, x: np.ndarray):
        result = self._predict(x)
        # print(f"x.shape: {x.shape} result.shape: {result.shape}", file=sys.stderr)
        return result

    def _predict(self, x):
        raise NotImplementedError

    def sample_all(self, x):
        return self._sample_all(x)

    def _sample_all(self, x):
        raise NotImplementedError

    def get_single_model(self, i: int) -> nn.Module:
        raise ValueError('Sklearn models cannot be split')


class HistGradientBoostingRegressor(SklearnModel):
    def _fit(self, x, y, eval_x, eval_y):
        self.model = ensemble.BaggingRegressor(
            estimator=ensemble.HistGradientBoostingRegressor(), n_estimators=self.n_models
        )
        self.model.fit(x, y)

    def _predict(self, x):
        return self.model.predict(x)

    def _sample_all(self, x):
        x = np.stack([estimator.predict(x) for estimator in self.model.estimators_])
        return x


class RandomForestRegressor(SklearnModel):
    def _fit(self, x, y, eval_x, eval_y):
        self.model = ensemble.RandomForestRegressor()
        self.model.fit(x, y)

    def _predict(self, x, num_samples=None, transform=None):
        return self.model.predict(x)

    def _sample_all(self, x):
        x = np.stack([estimator.predict(x) for estimator in self.model.estimators_])
        return x


class CatBoostRegressor(SklearnModel):
    def _fit(self, x, y, eval_x, eval_y):
        eval_set = (eval_x, eval_y)
        base_model = catboost.CatBoostRegressor(verbose=False, task_type="GPU", devices="0:1", gpu_ram_part=1/12,
                                                used_ram_limit=10*2**30)
        self.model = [base_model.copy().fit(x, y, eval_set=eval_set, early_stopping_rounds=10) for _ in
                      range(self.n_models)]

    def _predict(self, x):
        return np.mean([model.predict(x) for model in self.model], axis=0)

    def _sample_all(self, x):
        x = np.asarray([estimator.predict(x) for estimator in self.model])
        return x
