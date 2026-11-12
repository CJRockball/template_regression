# NN regression lifecycle — Package 01

Base: main at 063519f55522852957273dd6c0d691420411e8aa.
Delivery: candidate implementation with 42 isolated pure cases executed and 16 REAL CPU NN
integration cases supplied for local acceptance. No native NN integration is claimed here.

## Scope
Verified NN continuous predictors (final/cv_mean) and shared METADATA preparation only.
Existing evaluate/calibrate/release entry points stay XGB-only. No holdout evaluation, correction,
interval fitting, release, scoring pipeline, monitoring, SHAP or permutation support is enabled.
Do NOT run the old NN lifecycle YAMLs through those entry points yet.

- New schema-3 NN descriptors record task/model/serialization/embedded preprocessing explicitly.
- Schema-1 binary and schema-2 XGB paths are retained.
- Existing .ubj checkpoint filenames and all parent artifact manifests remain unchanged.
- Descriptor binds original model files, config.json, features.json and CV inventory hashes.
- Expected preprocessing fit-row counts derive from verified stored DEVELOPMENT assignments:
  final = n_dev; fold k = n_dev - n_validation_k. These are necessary invariants, not a mathematical
  proof of which rows were used; native tests also verify actual train-partition numeric statistics.
- Checkpoint hashes are verified before adapter/Torch loading. Existing adapter retains
  torch.load(map_location='cpu', weights_only=True), task/config/seed/dimension/weight checks.
- Loaded Gold config/roles must match saved estimator-input metadata and NN parameters.
- Every prediction checks that Gold state remains unchanged. No fitting, filling outside Gold,
  feature-set-cache lookup, or category vocabulary relearning occurs inside the predictor.
- Direct predict(X) consumes a uniquely keyed POST-RECIPE, PRE-GOLD estimator-input DataFrame.
  It is NOT a raw CSV batch-scoring API. Frozen stateless replay belongs upstream.
- NN external fitted_features remain rejected; stateless FE + NN-owned preprocessing is allowed.
- NN preparation reads the existing registry using SELECT, not get_run/ensure_tables, and the CLI
  uses SQLite mode=ro. No schema migrations, registry rows or outputs are written.
- Metadata preparation hashes source bytes and reads assignments; it never decodes Silver
  feature/target values or model checkpoints. Explicit checkpoint verification loads weights,
  but still reads no historical feature/target values and computes no predictions.
- Runtime provenance records current Torch distribution version without importing Torch during
  metadata-only checking. The version can say not-installed; weights are NOT semantically
  verified until --verify-checkpoints. CUDA is not auto-selected; native acceptance targets CPU.

## Files
Modified:
- src/ml_template/models/predictor_selection.py
- src/ml_template/evaluation/regression_inputs.py
New:
- src/ml_template/models/nn_regression_predictor.py
- src/ml_template/evaluation/nn_predictor_preflight.py
- tests/models/test_nn_lifecycle_01.py
- tests/models/test_nn_lifecycle_01_native.py
- configs/predictor/concrete_nn_package_01.yaml

Training, NN preprocessor/adapter and dashboard files are not modified. In particular the
scalar fix, residual-analysis code and report exporter are left alone.

## Install
Preserve original checkouts/environments for existing frozen releases. Changes to source
runtime hashes can invalidate old evaluation/release reuse in the NEW checkout. Never edit
sealed evidence, reseal it, or relax a runtime guard to resolve this.

Extract outside the repository. Set BUNDLE to the extracted nn_lifecycle_package_01 folder.
From repository root:

```bash
uv run python "$BUNDLE/install_nn_lifecycle_01.py" --repo . --check
uv run python "$BUNDLE/install_nn_lifecycle_01.py" --repo . --check \
  --emit-patch nn_lifecycle_01_exact.patch
uv run python "$BUNDLE/install_nn_lifecycle_01.py" --repo . --apply
```

The installer uses EXACT source/dependency Git byte hashes, unique anchors and payload SHA256.
There are NO ast.dump fingerprint comparisons. AST is used only for syntax checking with the
3.11 grammar, then compile. A mismatch prints expected/actual and must be investigated, not
forced. No project imports, data reads, model execution, network operations or lockfile changes
occur in installation. --check writes nothing unless --emit-patch was explicitly requested.

Backups and exact applied.patch live in .nn_lifecycle_01_backup. Do not commit that directory,
the delivery directory or generated exact patch. Use the installer OR patch, not both.
The supplied static patch has authentic edited-section context for predictor_selection and the
full original regression_inputs source verified against its Git blob. Guarded installation is
preferred; the installer emits a full-file exact patch from YOUR matching sources.

Alternative after successful guarded --check/--emit-patch (do not also use --apply):

```bash
git apply --check nn_lifecycle_01_exact.patch
git apply nn_lifecycle_01_exact.patch
```

Git application does not create installer rollback backups. Installer rollback, before later
edits or Ruff formatting:

```bash
uv run python "$BUNDLE/install_nn_lifecycle_01.py" --repo . --rollback
```

Rollback refuses modified installed/backup files. It retains backup history. After formatting
or further edits use your reviewed Git diff, not forced rollback. Reinstallation after rollback
requires deliberately archiving the old backup directory. A partial write attempts restoration.

## Local acceptance — BEFORE your real saved NN
Your existing working NN environment is appropriate. The current pyproject extra is pytorch,
not nn. If needed, install the EXISTING locked extra; no lockfile is supplied/rewritten:

```bash
uv sync --locked --extra pytorch --group dev
uv run pytest -v -rs -x --tb=long \
  tests/models/test_nn_lifecycle_01.py \
  tests/models/test_nn_lifecycle_01_native.py
```

Expect 42 pure cases + 16 native cases. Native cases MUST execute, not skip. They create real
synthetic CPU NN runs through experiment_runner with nullable numeric/categorical data and
raw/stateless variants, verify final/CV-mean checkpoints, numeric fit-partition statistics,
first-inference missingness, unknown-category fallback, singleton/chunk/shuffle parity,
wrong payload/config/fit-count rejection, pre-Torch tamper rejection, and continued NN holdout
and evaluation blocking. Training is done ONLY inside these synthetic tests, not by preflight.

Run formatting/lint (review fixes; rollback hashes no longer match after formatting):

```bash
uv run ruff check --fix \
  src/ml_template/models/nn_regression_predictor.py \
  src/ml_template/models/predictor_selection.py \
  src/ml_template/evaluation/regression_inputs.py \
  src/ml_template/evaluation/nn_predictor_preflight.py \
  tests/models/test_nn_lifecycle_01.py \
  tests/models/test_nn_lifecycle_01_native.py
uv run ruff format \
  src/ml_template/models/nn_regression_predictor.py \
  src/ml_template/models/predictor_selection.py \
  src/ml_template/evaluation/regression_inputs.py \
  src/ml_template/evaluation/nn_predictor_preflight.py \
  tests/models/test_nn_lifecycle_01.py \
  tests/models/test_nn_lifecycle_01_native.py
uv run ruff check
uv run pytest -q tests/models tests/evaluation tests/scoring
uv run pytest -q
```

Formatting should be completed before future evaluation/release provenance is frozen.

## Existing NN run: no retraining or target reads
Installer places configs/predictor/concrete_nn_package_01.yaml:

```yaml
parent_run_id: run_20261111_135508_ccf0e8
predictor_kind: final
database: data/project.db
expected_silver_dataset_version_id: silver_concrete_v1_6fe40821
expected_feature_set_id: null
expected_split_id: split_841e3a09cb9d
```

Confirm references still describe your intended stored run. Expected fields are optional extra
constraints; null does not itself mean 'assert raw feature set'. Archived config/feature/registry
identities are verified regardless.

```bash
uv run python -m ml_template.evaluation.nn_predictor_preflight \
  --config configs/predictor/concrete_nn_package_01.yaml --check
```

Expected: ready:true, purpose:predictor_only, descriptor_schema:3, models_loaded:false,
predictions_computed:false, labels_loaded:false, files_written:false; lifecycle flags all false.

Then explicitly deserialize and verify the selected checkpoint, with no prediction/target access:

```bash
uv run python -m ml_template.evaluation.nn_predictor_preflight \
  --config configs/predictor/concrete_nn_package_01.yaml --verify-checkpoints
```

Expected: checkpoints_verified:1 for final, models_loaded:true, predictions_computed:false,
labels_loaded:false. To verify all saved fold checkpoints, change predictor_kind to cv_mean
and repeat BOTH commands. For the five-fold run expect checkpoints_verified:5.
There is deliberately no --execute flag or automatic final/CV-mean comparison.

The supported Python API for later packages is prepare_nn_regression_predictor_inputs(engine,
RegressionEvaluationConfig(...)) followed by load_selected_predictor(cfg, root, descriptor).
The returned PreparedRegression has purpose='predictor_only'; load_holdout rejects it.
Preparing a NN through existing prepare_regression still raises 'Expected XGB regression parent'.
Descriptor is generated in memory, not appended to the sealed parent. The original parent is
never repaired/resealed. No evidence of deployment approval is created.

## Authoring validation / remaining gates
Executed here: 42 pure tests against the actual new descriptor/config/pre-deserialization logic
using an isolated verifier source-fragment harness. This is NOT full installed-package testing.
The harness contains no Torch models and makes no native replay claims. New runtime/test files
and changed source contexts parse with Python 3.11 grammar and compile in Python 3.12.8.
Full original regression_inputs context matched the inspected e9202... Git blob exactly.
Installer fixture check/apply/idempotence/rollback and changed-dependency rejection passed;
context patch git --check/application passed; payload/ZIP CRC+SHA checks passed.

NOT executed: 16 real CPU NN integration cases, real saved-run preflight/checkpoints, full
repository suite, Ruff, CUDA or other native Python versions. Torch, PyArrow and SQLAlchemy are
unavailable in the authoring environment. These are explicit local acceptance gates, not skips
that should be counted as success. Stop on a native failure before using this for later packages.
