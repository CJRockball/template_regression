# NN regression lifecycle — Package 02

Base: feature/update_nn at bfdf2370f440a5bc134127ae8f730d8708da2113, including tested
Package 01, nullable output declaration and lightweight-config dispatch compatibility fixes.
This is a CANDIDATE pending local native acceptance. No GitHub writes or real-data fitting
were performed while authoring.

## Scope
Extends the EXISTING shared training.regression_calibration workflow. No second NN calibration
engine or duplicated conformal maths is introduced. New request field model_type: nn explicitly
chooses Package 01 predictor preparation; default xgb retains the old serialized request shape.

Supports final/cv_mean, correction none/bias/affine, and split-conformal coverages. Correction
uses only recalibration rows; after correction is frozen, conformal scores use only calibration
rows. Development/OOF/final holdout labels are excluded. No repartitioning, model training,
preprocessor fitting, checkpoint alteration, evaluation, release or deployment approval occurs.

Raw/schema-3 NN preparation is retained; calibration fits to predictions from the exact selected
predictor. CV mean is corrected/calibrated AFTER combining member predictions, not separately
per member. Existing NN-owned Gold stays frozen. Source role IDs are independently checked
before each reserved target read. Nulls remain subject to the original strict contracts.

The finite rank is ceil((n_calibration + 1) * coverage), with an observed order statistic (not an
interpolated percentile). Requests needing a rank beyond n_calibration are rejected. Intervals
are marginal under exchangeability, not conditional/subgroup guarantees or observed future
coverage. No lower-bound clipping or automatic nonnegative correction is added.

## Changed files
Only runtime modification:
  src/ml_template/training/regression_calibration.py
New:
  tests/calibration/test_nn_lifecycle_02.py
  tests/calibration/test_nn_lifecycle_02_native.py
  configs/calibration/concrete_nn_package_02.yaml

Package 01 source, training, checkpoint/preprocessor code, existing maths, dashboards and model
factory are untouched. Evaluation/release still reject NN until their later packages.

## Install
Extract outside the repo. BUNDLE is the extracted nn_lifecycle_package_02 folder. From repo root:

```bash
uv run python "$BUNDLE/install_nn_lifecycle_02.py" --repo . --check
uv run python "$BUNDLE/install_nn_lifecycle_02.py" --repo . --check \
  --emit-patch nn_lifecycle_02_exact.patch
uv run python "$BUNDLE/install_nn_lifecycle_02.py" --repo . --apply
```

Exact original/dependency Git byte hashes and payload SHA256 are checked before copying the
replacement module. No runtime ast.dump fingerprints. Whole modified Python is parsed with
3.11 grammar and compiled before writing. A source mismatch means STOP/reconcile, not reset
work or weaken the guard. --check writes nothing except an explicitly requested emitted patch.
No model/data/registry access occurs in installation. No lockfile or dependency changes.

Use installer OR patch, not both. The ZIP includes a standard full-source Git patch, built from
the inspected original calibration source (verified f763e... Git blob) and new files. Alternative
only after successful guarded --check/--emit-patch:

```bash
git apply --check nn_lifecycle_02_exact.patch
git apply nn_lifecycle_02_exact.patch
```

Git alone does not make installer rollback backups. Backups/exact applied.patch live in
.nn_lifecycle_02_backup. Do not commit that folder, delivery folder or generated exact patch.
Before formatting/later edits, installer rollback:

```bash
uv run python "$BUNDLE/install_nn_lifecycle_02.py" --repo . --rollback
```

Rollback verifies installed/original hashes and retains backup history. After later edits use
your reviewed Git diff. Partial write exceptions attempt restoration; conflicts are not forced.

## Acceptance first — synthetic data only
Your working Package 01 environment is sufficient. Native cases reuse its accepted real CPU
training fixture, including both raw/stateless features and explicit nullable ratio outputs.
No synthetic model is substituted in those NATIVE cases.

```bash
uv run pytest -v -rs -x --tb=long \
  tests/calibration/test_nn_lifecycle_02.py \
  tests/calibration/test_nn_lifecycle_02_native.py
```

Expect 49 pure cases + 28 native cases. Native cases must EXECUTE, not skip. They exercise:
- real NN final/cv_mean x raw/stateless x intervals/bias+intervals/affine+intervals/bias-only/affine-only;
- --check no source values/weights/DB changes/output files;
- exact logical source-row filters and role read order;
- forbidden model/Gold fitting, unchanged parent hashes;
- independent reconstruction of correction/quantiles;
- wrong predictor/unit and tampered artifact rejection;
- minimum support/finite rank/prior-exposure rejection;
- failed audit record and lock cleanup; continued NN evaluation blocking.

Then compatibility and full suite:

```bash
uv run pytest -v -rs -x --tb=long tests/scoring/test_regression_postprocessing_native.py
uv run pytest -q tests/models tests/calibration tests/evaluation tests/scoring
uv run pytest -q
```

Review and format BEFORE creating real calibration artifacts:

```bash
uv run ruff check --fix \
  src/ml_template/training/regression_calibration.py \
  tests/calibration/test_nn_lifecycle_02.py \
  tests/calibration/test_nn_lifecycle_02_native.py
uv run ruff format \
  src/ml_template/training/regression_calibration.py \
  tests/calibration/test_nn_lifecycle_02.py \
  tests/calibration/test_nn_lifecycle_02_native.py
uv run ruff check
uv run pytest -q
```

Formatting changes installer rollback hashes. No Python downgrade, unsafe Ruff fixes or new
uv resolution is needed. Existing Torch extra is pytorch if installation is needed, not nn.

## YAML and preflight
Installer places configs/calibration/concrete_nn_package_02.yaml. It deliberately starts with
correction:none (intervals only) and DOES NOT pre-authorise prior-exposure research reuse:

```yaml
model_type: nn
parent_run_id: run_20261111_135508_ccf0e8
predictor_kind: final
correction: none
coverages: [0.8, 0.9, 0.95]
target_unit: MPa
min_calibration_rows: 30
min_recalibration_rows: 10
intended_use: "Exploratory NN regression interval calibration; no deployment approval"
acknowledge_prior_exposure: false
prior_exposure_note: ""
```

Choose predictor/correction/levels BEFORE reserved labels are read. Do not run all variants as
a sweep and pick whichever looks best on calibration rows. For bias/affine change correction;
for correction-only use coverages: [] with correction bias/affine. Never use development OOF
residuals as the conformal calibration population. Do not relabel old dev/holdout rows.

```bash
uv run python -m ml_template.training.regression_calibration \
  --config configs/calibration/concrete_nn_package_02.yaml --check
```

Use this direct regression CLI, not the older classification calibration entry point.
NN CLI --check opens an existing SQLite DB in mode=ro. Optional --database /path/project.db
selects a different existing DB (not a field in the calibration YAML). Execution opens mode=rw;
no new database is silently created. API callers use their supplied engine.

Expected metadata only: model_type:nn, models_loaded:false, predictions_computed:false,
calibration_labels_loaded:false, recalibration_labels_loaded:false, holdout_labels_loaded:false,
files_written:false, evaluation_enabled:false, release_enabled:false. n_calibration and
n_recalibration count requested/used populations; an unused role is reported as zero.
Hashes read whole files as bytes; no target columns or checkpoint payloads are decoded by check.

Concrete has prior holdout exposure in the established history. If present, expect ready:false
and prior_exposures. That is the safeguard working. Stop or explicitly document a research-only
reuse by setting acknowledge_prior_exposure:true and a truthful prior_exposure_note. No default
note is supplied. Acknowledgement CANNOT restore independence or establish future coverage.

## Explicit fitting, only after acceptance/protocol approval
Implementation first on synthetic data. For real data, avoid consuming reserved labels just to
debug later packages; preferably wait until prediction/provenance code is stable. Once chosen:

```bash
uv run python -m ml_template.training.regression_calibration \
  --config configs/calibration/concrete_nn_package_02.yaml --execute
```

This is a side-effecting operation: new registry record + lock/audit timestamps + new regcal_...
artifact directory. It does NOT fit NN/Gold or read development/holdout target rows. Correction
runs on recalibration first, then interval quantiles on calibration; only required roles are read.
Uses filtered logical Parquet row selection; no claim is made about physical row-group decoding.
Failed attempts keep failure/access audit and release the lock. No silent retries or caching.

The returned calibration_id identifies NEW immutable postprocessing evidence. Outputs include
request.json and calibration.json with config, predictor/data/code/runtime/unit bindings,
partition key hashes, correction, finite-rank quantiles, payload hash and sealed artifact manifest.
This does NOT append a descriptor or calibration to the original training run.

Save the returned ID for later Package 03/04 requests. Do not run NN evaluate/release yet.
Changes to bound critical code/runtime/inputs may invalidate reuse. Never repair/reseal old
calibration/release evidence to pass a new runtime check. Preserve original source/environment
for frozen releases and do not assume old XGB calibration IDs load in a modified checkout.
A new calibration protocol may be necessary if critical code truly changes later.

## Validation here / remaining gates
49 pure cases PASSED on the actual modified module plus the unchanged shared maths module.
The authoring harness is an isolated package with those modules only; it is not full repo testing.
Original calibration and maths modules were reconstructed EXACTLY and verified against their
inspected Git blobs before testing/modifying. This checks old XGB serialized request shape too.

Six additional isolated SQLite workflow scenarios passed: five combinations and injected
failure/lock cleanup. They execute the actual modified workflow/SQL and shared maths with
STUBBED metadata, prediction, partition and artifact boundaries. They are NOT native NN or
production-registry acceptance and are reported separately from the 49 pytest cases.

Installer private-dependency fixture check/apply/idempotence/rollback/source-edit rejection and
full-context Git patch application passed. All Python parses with 3.11 grammar and compiles on
3.12.8. Archive CRC/member/payload hashes checked before delivery.

NOT RUN: 28 native CPU cases, old XGB native regression/calibration/release suite, full repo
suite, Ruff, real saved NN fitting, CUDA, native Python 3.11. Torch/PyArrow/SQLAlchemy are
unavailable here. Native acceptance locally is REQUIRED before treating Package 02 as complete.
