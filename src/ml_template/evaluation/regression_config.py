"""Frozen point-estimate regression evaluation specification."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ml_template.evaluation.predictor_config import PredictorConfig


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DisabledUncertainty(Strict):
    enabled: Literal[False] = False


class RegressionCriterion(Strict):
    metric: Literal["rmse", "mae", "r2"]
    operator: Literal["<=", ">="]
    value: float
    basis: Literal["point"] = "point"


class RegressionEvaluationConfig(Strict):
    schema_version: Literal[2] = 2
    task: Literal["regression"] = "regression"
    parent_run_id: str = Field(min_length=1)
    predictor: PredictorConfig = Field(default_factory=PredictorConfig)
    primary_metric: Literal["rmse", "mae"] = "rmse"
    expected_silver_dataset_version_id: str | None = None
    expected_feature_set_id: str | None = None
    expected_split_id: str | None = None
    intended_use: str = "Research; no automatic deployment approval"
    limitations: tuple[str, ...] = ()
    criteria: tuple[RegressionCriterion, ...] = ()
    uncertainty: DisabledUncertainty = Field(default_factory=DisabledUncertainty)
    calibration_id: None = None
    release_variant: Literal["raw"] = "raw"
    acknowledge_prior_exposure: bool = False
    prior_exposure_note: str = ""
    supersedes_evaluation_id: str | None = None

    @model_validator(mode="after")
    def _check(self):
        if not self.intended_use.strip():
            raise ValueError("Document intended use")
        if self.acknowledge_prior_exposure and not self.prior_exposure_note.strip():
            raise ValueError("Prior-exposure acknowledgement requires a note")
        return self
