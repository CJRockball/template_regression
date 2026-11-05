"""Package frozen regression evidence; never train or read target columns."""

import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.scoring.release_artifacts import (
    inside,
    read_json,
    runtime_info,
    seal,
    sha256,
    verify,
    write_json,
)


def regression_evidence(engine, request):
    from ml_template.db.evaluation_registry import get_evaluation
    from ml_template.evaluation.evaluation_inputs import digest
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig
    from ml_template.evaluation.regression_evaluate import _verify_cache
    from ml_template.evaluation.regression_inputs import prepare_regression
    from ml_template.tracking.run_artifacts import resolve_stored_path

    record = get_evaluation(engine, request.evaluation_id)
    if not record or record["status"] != "succeeded":
        raise ValueError("An explicit successful regression evaluation is required")
    _verify_cache(record, record["identity_hash"])
    directory = resolve_stored_path(record["artifact_dir"])
    spec = RegressionEvaluationConfig.model_validate(read_json(directory / "spec.json"))
    registered = RegressionEvaluationConfig.model_validate_json(record["spec_json"])
    if spec != registered or spec.parent_run_id != request.parent_run_id:
        raise ValueError("Release request/evaluation registry/spec mismatch")
    summary = read_json(directory / "summary.json")
    provenance = read_json(directory / "provenance.json")
    if (
        summary["evaluation_id"] != request.evaluation_id
        or summary["parent_run_id"] != request.parent_run_id
        or summary.get("task") != "regression"
    ):
        raise ValueError("Evaluation summary binding differs")
    if (
        summary["identity_hash"] != record["identity_hash"]
        or digest({"spec": spec.model_dump(mode="json"), "provenance": provenance})
        != record["identity_hash"]
    ):
        raise ValueError("Evaluation evidence digest differs")
    p = prepare_regression(engine, spec)
    if (
        p.spec != spec
        or p.provenance["predictor"] != provenance["predictor"]
        or p.provenance["input_hashes"] != provenance["input_hashes"]
    ):
        raise ValueError("Evaluated predictor/data artifacts changed")
    current_code = runtime_info()["code_sha256"]
    for name, expected in provenance["code_hashes"].items():
        current_code = runtime_info()["code_sha256"]

        for name, expected in provenance["code_hashes"].items():
            if (
                name.startswith(("models/", "features/", "preprocessing/"))
                or name
                in {
                    "training/run_config.py",
                    "evaluation/regression_inputs.py",
                }
            ) and current_code.get(name) != expected:
                raise ValueError(f"Evaluated prediction implementation changed: {name}")
                raise ValueError(f"Evaluated prediction implementation changed: {name}")
    return p, directory, spec, summary, provenance


def prepare_regression_release(engine, request):
    from ml_template.scoring.input_contract import validate_batch
    from ml_template.scoring.pre_gold_schema import build_pre_gold_contract
    from ml_template.scoring.source_spec import freeze_source_spec

    p, directory, spec, summary, provenance = regression_evidence(engine, request)
    # Project only the key and model-neutral feature columns. Never request y_true/target.
    source = pd.read_parquet(p.parquet_path, columns=[p.key_col, *p.feature_names])
    if (
        source[p.key_col].isna().any()
        or source[p.key_col].duplicated().any()
        or set(source[p.key_col]) != set(p.assignments[p.key_col])
    ):
        raise ValueError("Source key coverage differs")
    source = source.set_index(p.key_col).sort_index()
    model_features = source.copy()
    categories = p.feature_info.get("categorical_schema", {})
    for name, item in categories.items():
        model_features[name] = model_features[name].astype(
            pd.CategoricalDtype(item["categories"], ordered=item["ordered"])
        )
    if list(model_features.columns) != p.feature_names:
        raise ValueError("Feature order differs")
    for name, dtype in p.feature_info["dtypes"].items():
        if str(model_features[name].dtype) != dtype:
            raise ValueError(f"Source dtype differs from training: {name}")
    if sha256(p.parquet_path) != provenance["input_hashes"]["silver_parquet"]:
        raise ValueError("Silver changed while reading source features")
    input_spec, source_policy, model_policy = freeze_source_spec(p, source)
    keys = source.index.to_series()
    is_int = keys.map(
        lambda v: isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
    ).all()
    is_str = keys.map(lambda v: isinstance(v, str)).all()
    if not is_int and not is_str:
        raise ValueError("Entity keys must be homogeneous int or string")
    contract = build_pre_gold_contract(
        source,
        model_features,
        key_col=p.key_col,
        key_kind="int" if is_int else "str",
        input_spec=input_spec,
        source_policy=source_policy,
        model_policy=model_policy,
    )
    validate_batch(source.reset_index(), contract, enforce_rates=True)
    recipe = {
        "schema_version": 1,
        "kind": "identity",
        "raw_columns": p.feature_names,
        "columns": p.feature_names,
    }
    return {
        "prepared": p,
        "evaluation_dir": directory,
        "spec": spec,
        "summary": summary,
        "provenance": provenance,
        "source": source,
        "contract": contract,
        "recipe": recipe,
        "reference_sha256": read_json(directory / "artifact_manifest.json")["files"][
            "holdout_predictions.parquet"
        ]["sha256"],
    }


def check_regression_parity(predictor, material, atol):
    p = material["prepared"]
    path = material["evaluation_dir"] / "holdout_predictions.parquet"
    expected_hash = material["reference_sha256"]
    if sha256(path) != expected_hash:
        raise ValueError("Evaluation predictions changed")
    reference = pd.read_parquet(path, columns=[p.key_col, "y_pred"])
    if (
        reference[p.key_col].isna().any()
        or reference[p.key_col].duplicated().any()
        or set(reference[p.key_col]) != set(p.hold_keys)
    ):
        raise ValueError("Evaluation prediction keys differ")
    reference = reference.set_index(p.key_col).sort_index()
    values = reference.y_pred.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite evaluation predictions")
    batch = material["source"].loc[reference.index].reset_index()
    whole = predictor.predict(batch).set_index(p.key_col).sort_index()
    chunked = (
        predictor.predict(
            batch.iloc[::-1].reset_index(drop=True), chunk_size=max(1, min(137, len(batch) // 3))
        )
        .set_index(p.key_col)
        .sort_index()
    )
    for column in ["y_pred_raw", "y_pred_release"]:
        if not np.allclose(values, whole[column], rtol=0, atol=atol) or not np.allclose(
            whole[column], chunked[column], rtol=0, atol=atol
        ):
            raise ValueError(f"Regression prediction parity failed: {column}")
    singleton = predictor.predict(batch.iloc[:1]).y_pred_release.iloc[0]
    if abs(singleton - whole.y_pred_release.iloc[0]) > atol:
        raise ValueError("Regression singleton parity failed")
    if sha256(path) != expected_hash:
        raise ValueError("Evaluation predictions changed during parity")
    return {
        "n_rows": len(reference),
        "atol": atol,
        "max_abs_diff": float(np.max(np.abs(values - whole.y_pred_release.to_numpy()))),
        "keyed": True,
        "labels_loaded": False,
        "metrics_recomputed": False,
        "shuffled_chunk_parity": True,
        "singleton_parity": True,
    }


def build_regression_release(engine, request, *, check_only=False, output_root=None):
    material = prepare_regression_release(engine, request)
    p, spec = material["prepared"], material["spec"]
    if check_only:
        return {
            "ready": True,
            "task": "regression",
            "parent_run_id": request.parent_run_id,
            "evaluation_id": request.evaluation_id,
            "predictor_kind": spec.predictor.kind,
            "holdout_labels_loaded": False,
            "parity_executed": False,
            "files_written": False,
        }
    from ml_template.db.scoring_registry import register_release
    from ml_template.paths import ARTIFACTS
    from ml_template.scoring.predictor import ReleasedPredictor

    root = Path(output_root or ARTIFACTS / "releases").resolve()
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".regression-release-", dir=root))
    try:
        policy = {
            "kind": "none",
            "value": None,
            "source": "Regression has no classification policy",
            "batch_definition": "entire scoring input",
            "selection": None,
        }
        info = {
            "schema_version": 2,
            "task": "regression",
            "prediction_kind": "continuous",
            "name": request.name,
            "created_at": datetime.now(UTC).isoformat(),
            "parent_run_id": request.parent_run_id,
            "evaluation_id": request.evaluation_id,
            "calibration_id": None,
            "release_variant": "raw",
            "policy": policy,
            "intended_use": request.intended_use,
            "research_only": True,
            "automatic_promotion": False,
            "predictor_kind": spec.predictor.kind,
            "prediction_columns": {"raw": "y_pred_raw", "release": "y_pred_release"},
            "evaluation_summary": material["summary"],
            "limitations": list(spec.limitations),
        }
        write_json(stage / "release.json", info)
        write_json(stage / "contract.json", material["contract"])
        write_json(stage / "feature_recipe.json", material["recipe"])
        write_json(stage / "runtime.json", runtime_info())
        descriptor = material["provenance"]["predictor"]
        write_json(stage / "predictor.json", descriptor)
        write_json(
            stage / "evidence.json",
            {
                "provenance": material["provenance"],
                "evaluation_spec": spec.model_dump(mode="json"),
                "evaluation_summary": material["summary"],
            },
        )
        for name in ["config.json", *descriptor["files_sha256"]]:
            dest = inside(stage, name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p.run_dir / name, dest)
            expected = (
                material["provenance"]["input_hashes"]["config"]
                if name == "config.json"
                else descriptor["files_sha256"][name]
            )
            if sha256(dest) != expected:
                raise ValueError(f"Copied predictor artifact changed: {name}")
        seal(stage)
        parity = check_regression_parity(ReleasedPredictor(stage), material, request.parity_atol)
        write_json(stage / "parity.json", parity)
        manifest = seal(stage)
        verify(stage)
        destination = root / manifest["release_id"]
        if destination.exists():
            raise FileExistsError("Immutable release destination exists")
        stage.rename(destination)
        register_release(engine, manifest, info, destination)
        return {
            "release_id": manifest["release_id"],
            "artifact_dir": str(destination),
            "task": "regression",
            "status": "succeeded",
            "parity": parity,
            "holdout_labels_loaded": False,
        }
    finally:
        if stage.exists():
            shutil.rmtree(stage)
