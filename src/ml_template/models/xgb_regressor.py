"""Single-target XGBoost regression with native persistence."""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


class XGBRegressionAdapter:
    def __init__(self, params, seed: int):
        self.params = params
        self.seed = seed
        self.max_rounds = params.n_estimators
        self.model = None
        self._loaded_booster = None

    @staticmethod
    def _target(y):
        if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
            raise ValueError("Regression targets must be numeric, not boolean")
        values = np.asarray(y, dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Regression targets must be a nonempty finite vector")
        return values

    def _build(self, rounds, stopping):
        params = self.params.model_dump()
        params.update(
            n_estimators=int(rounds),
            early_stopping_rounds=stopping,
            tree_method="hist",
            enable_categorical=True,
            random_state=self.seed,
            verbosity=0,
        )
        return xgb.XGBRegressor(**params)

    def _booster(self):
        if self.model is not None:
            return self.model.get_booster()
        if self._loaded_booster is not None:
            return self._loaded_booster
        raise RuntimeError("Model is not fitted")

    def fit_with_validation(self, X_train, y_train, X_val, y_val):
        model = self._build(self.max_rounds, self.params.early_stopping_rounds)
        model.fit(
            X_train, self._target(y_train), eval_set=[(X_val, self._target(y_val))], verbose=False
        )
        self.model, self._loaded_booster = model, None
        return int(model.best_iteration) + 1

    def refit(self, X, y, n_rounds):
        if int(n_rounds) != n_rounds or n_rounds < 1:
            raise ValueError("n_rounds must be a positive integer")
        model = self._build(n_rounds, None)
        model.fit(X, self._target(y), verbose=False)
        self.model, self._loaded_booster = model, None

    def predict(self, X):
        booster = self._booster()
        names = booster.feature_names
        if not isinstance(X, pd.DataFrame) or list(X.columns) != names:
            raise ValueError("Prediction columns must match fitted feature order")
        if len(X) == 0:
            return np.empty(0, dtype=float)
        best = booster.attr("best_iteration")
        rounds = (0, int(best) + 1) if best is not None else (0, 0)
        values = np.asarray(
            booster.predict(xgb.DMatrix(X, enable_categorical=True), iteration_range=rounds),
            dtype=float,
        )
        if values.shape != (len(X),) or not np.isfinite(values).all():
            raise ValueError("Expected finite one-dimensional regression predictions")
        return values

    def save(self, path: Path):
        self._booster().save_model(str(path))

    @classmethod
    def load(cls, params, seed, path: Path):
        import json

        adapter = cls(params, seed)
        booster = xgb.Booster()
        booster.load_model(str(path))
        objective = json.loads(booster.save_config())["learner"]["objective"]["name"]
        if objective != params.objective:
            raise ValueError("Saved model objective disagrees with regression config")
        booster.set_param({"device": params.device})
        adapter._loaded_booster = booster
        return adapter

    def feature_importances(self):
        booster = self._booster()
        scores = booster.get_score(importance_type="gain")
        return {name: float(scores.get(name, 0.0)) for name in booster.feature_names}
