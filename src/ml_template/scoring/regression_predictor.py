"""Registry-free, sealed regression inference with canonical keyed input."""

from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.evaluation.evaluation_config import PolicyConfig
from ml_template.models.regression_predictor import continuous
from ml_template.scoring.input_contract import coerce_frame, validate_batch
from ml_template.scoring.pre_gold_schema import get_preprocessor_schema
from ml_template.scoring.release_artifacts import read_json, sha256, verify, verify_runtime


class RegressionReleasedPredictor:
    def __init__(self, directory, *, expected_release_id=None):
        self.directory = Path(directory).resolve()
        self.manifest = verify(self.directory, expected_release_id)
        verify_runtime(read_json(self.directory / "runtime.json"))
        self.release_id = self.manifest["release_id"]
        self.info = read_json(self.directory / "release.json")
        if (
            self.info.get("schema_version") != 2
            or self.info.get("task") != "regression"
            or self.info.get("prediction_kind") != "continuous"
        ):
            raise ValueError("Unsupported regression release semantics")
        if self.info["calibration_id"] is not None or self.info["release_variant"] != "raw":
            raise ValueError("Regression release must be uncalibrated")
        self.policy = PolicyConfig.model_validate(self.info["policy"])
        if self.policy.kind != "none":
            raise ValueError("Classification policy is invalid for regression")
        self.calibrator = None
        self.contract = read_json(self.directory / "contract.json")
        self.recipe = read_json(self.directory / "feature_recipe.json")
        self.preprocessor_input_schema = get_preprocessor_schema(self.contract)
        names = [item["name"] for item in self.contract["source_schema"]]
        if self.recipe != {
            "schema_version": 1,
            "kind": "identity",
            "raw_columns": names,
            "columns": names,
        }:
            raise ValueError("Only raw identity regression features are supported")
        if names != [item["name"] for item in self.preprocessor_input_schema]:
            raise ValueError("Regression input schema order differs")
        from ml_template.models.predictor_selection import load_selected_predictor
        from ml_template.training.run_config import RunConfig

        cfg = RunConfig.model_validate(read_json(self.directory / "config.json"))
        if cfg.task != "regression" or cfg.fitted_features or cfg.data.feature_set_id:
            raise ValueError("Unsupported regression feature recipe")
        descriptor = read_json(self.directory / "predictor.json")
        evidence = read_json(self.directory / "evidence.json")
        provenance, spec = evidence["provenance"], evidence["evaluation_spec"]
        if (
            descriptor != provenance["predictor"]
            or descriptor["kind"] != self.info["predictor_kind"]
        ):
            raise ValueError("Packaged/evaluated predictor differs")
        if (
            spec.get("task") != "regression"
            or spec["parent_run_id"] != self.info["parent_run_id"]
            or spec["predictor"]["kind"] != descriptor["kind"]
        ):
            raise ValueError("Release evaluation binding differs")
        summary = evidence["evaluation_summary"]
        if (
            summary["evaluation_id"] != self.info["evaluation_id"]
            or summary["parent_run_id"] != self.info["parent_run_id"]
        ):
            raise ValueError("Release summary identity differs")
        from ml_template.evaluation.evaluation_inputs import digest

        expected_identity = digest({"spec": spec, "provenance": provenance})
        if expected_identity != summary["identity_hash"]:
            raise ValueError("Evaluation evidence digest differs")
        if sha256(self.directory / "config.json") != provenance["input_hashes"]["config"]:
            raise ValueError("Packaged run config differs from evaluated config")
        outputs = {"raw": "y_pred_raw", "release": "y_pred_release"}
        if self.info.get("prediction_columns") != outputs:
            raise ValueError("Unexpected regression output schema")
        self.model = load_selected_predictor(cfg, self.directory, descriptor)

    def _prepare_source_features(self, source):
        return coerce_frame(source, self.preprocessor_input_schema)

    def prepare_features(self, frame):
        return self._prepare_source_features(validate_batch(frame, self.contract))

    def predict(self, frame, *, chunk_size=None, include_policy=True):
        verify(self.directory, self.release_id)
        verify_runtime(read_json(self.directory / "runtime.json"))
        raw = validate_batch(frame, self.contract)
        chunk_size = len(raw) if chunk_size is None else chunk_size
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        pieces = []
        for start in range(0, len(raw), chunk_size):
            features = self._prepare_source_features(raw.iloc[start : start + chunk_size])
            pieces.append(continuous(self.model.predict(features), len(features)))
        values = continuous(np.concatenate(pieces), len(raw))
        return pd.DataFrame(
            {
                self.contract["key_col"]: raw.index.to_numpy(),
                "y_pred_raw": values,
                "y_pred_release": values.copy(),
                "release_id": self.release_id,
            }
        )
