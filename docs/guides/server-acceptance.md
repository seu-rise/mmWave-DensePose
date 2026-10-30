# Develop-branch server acceptance

Use this runbook to test a fresh user checkout on a **glibc Linux x86_64 server
with an NVIDIA GPU** before merging `develop` into `main`. Commands use Bash,
Python 3.10.19 and the locked CUDA 11.8 profile. They explicitly specify model
and data arguments rather than loading experiment configuration files.

Run each stage separately and stop if it fails. Keep the same repository and
Bash session for the variables below; use another terminal for the WebGUI.
A `tmux` session is useful for the long download, cache preparation and training
stages. Record observations against the exact tested Git commit.

The minimum complete route is: **fresh clone → environment → full dataset →
short training/validation/test → full training/validation/test → official
asset downloads/evaluation → browser acceptance → merge review**.
The extended checks cover single-frame models and PCC/GlobalLocalNetwork.

## 1. Clone the development branch and record the environment

Start in a new working directory on the server. Do not copy an existing virtual
environment, numeric caches, weights or user logs into this checkout.

```bash
git clone --branch develop https://github.com/seu-rise/mmWave-DensePose.git mmWave-DensePose-acceptance
cd mmWave-DensePose-acceptance
git status --short
git branch --show-current
git log -1 --oneline

TESTED_COMMIT=$(git rev-parse HEAD)
ACCEPT_DIR="$(pwd)/.local/acceptance/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$ACCEPT_DIR"
set -o pipefail
printf '%s\n' "$TESTED_COMMIT" > "$ACCEPT_DIR/tested-commit.txt"
git status --porcelain=v1 > "$ACCEPT_DIR/git-status-before.txt"
uname -srm | tee "$ACCEPT_DIR/platform.txt"
ldd --version > "$ACCEPT_DIR/glibc.txt"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv | tee "$ACCEPT_DIR/gpu.txt"
df -h . | tee "$ACCEPT_DIR/disk-before.txt"
```

Expected: branch `develop`, no modified/untracked public files, `x86_64`, and a
working NVIDIA driver. The dataset archives occupy about 161 GiB and installed
sequence files occupy another 161 GiB. Allow additional space for staging,
numeric caches, checkpoints and prediction exports; 500 GiB free is a practical
starting budget for this route, not a limit for many experiments.

The clone fetches normal branch references, so `origin/main` remains available
for the final merge review. Do not start from a maintainer-only checkout.

## 2. Install uv and the locked compute environment

Install uv using its [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version | tee "$ACCEPT_DIR/uv-version.txt"
```

The project requires uv 0.12.12 or newer. Install the pinned interpreter and
dependencies:

```bash
uv python install 3.10.19
uv sync --locked --extra cu118
uv run --locked --extra cu118 python -m example.environment.verify --backend cu118 --require-cuda \
  2>&1 | tee "$ACCEPT_DIR/02-environment.log"
uv run --locked --extra cu118 python -m example.models.forward --all --device cuda --backward \
  2>&1 | tee "$ACCEPT_DIR/02-models.log"
uv run --locked --extra cu118 python scripts/train.py --list-models
```

Expected: isolated Python 3.10.19, the pinned PyTorch/PyG packages, CUDA/native
operator checks passing, and finite outputs/gradients for all 11 registered
models. This synthetic check does not train on the dataset or establish paper
accuracy.

**Keep `--extra cu118` on every `uv run` command in this GPU checkout**, including
download and verification commands. A base-only invocation may synchronize away
the compute packages. Do not use `--all-extras` or combine `cpu` and `cu118`.
For a CPU-only acceptance route, consistently use `cpu`, `--device cpu`, omit
GPU IDs and omit `--require-cuda`; do not change profiles midway through a run.

## 3. Inspect release availability and install the dataset

Inspect the manifests actually cloned from `develop`:

```bash
uv run --locked --extra cu118 python - <<'PY' | tee "$ACCEPT_DIR/03-release-state.json"
import json
from pathlib import Path

dataset = json.loads(Path("dataset/release-manifest.json").read_text())
assets = json.loads(Path("paper/download-manifest.json").read_text())
archives, packages = dataset["archives"], assets["packages"]
print(json.dumps({
    "dataset_status": dataset.get("status"),
    "dataset_packages": len(archives),
    "dataset_urls": sum(bool(p.get("url")) for p in archives),
    "dataset_archive_gib": sum(p["size_bytes"] for p in archives) / 2**30,
    "official_packages": len(packages),
    "official_urls": sum(bool(p.get("url")) for p in packages),
}, indent=2))
assert archives and all(p.get("url") for p in archives), "Dataset URLs are incomplete"
PY

uv run --locked --extra cu118 python scripts/download_dataset.py --dry-run \
  2>&1 | tee "$ACCEPT_DIR/03-dataset-plan.log"
uv run --locked --extra cu118 python scripts/download_dataset.py --progress-json \
  2>&1 | tee "$ACCEPT_DIR/03-dataset-install.log"
```

The data manifest declares 49 ScienceDB packages with production URLs and status
`published`. The official asset manifest includes production download URLs for
37 weight packages and five prediction packages. Use the printed state of your
tested commit as evidence, then verify actual online access in stage 8. A
configured URL alone does not establish a successful download. Missing official
asset URLs block stage 8, but do not prevent own-model training in stages 5–7.

Expected installation layout:

```text
data/mmwave_densepose/
  manifest.json
  raw/manifest.json
  raw/*.pkl
  keypoint/...
  keypoint_muti/...
  dense/...
```

The installer checks ZIP and member sizes/hashes before accepting the files.
The source PKLs retain the published synchronized radar, skeleton and anonymized
image modalities. NPY/NPZ training caches are derived automatically later.
Do not copy a developer cache or run ignored preparation scripts.

Check the installed identity and sequence count without loading PKLs:

```bash
uv run --locked --extra cu118 python - <<'PY' | tee "$ACCEPT_DIR/03-installed-identity.json"
import hashlib, json
from pathlib import Path

release = json.loads(Path("dataset/release-manifest.json").read_text())
path = Path("data/mmwave_densepose/manifest.json")
raw = path.read_bytes()
manifest = json.loads(raw)
assert hashlib.sha256(raw).hexdigest() == release["dataset_manifest_sha256"]
assert len(manifest["sequences"]) == release["expected_sequence_count"]
print(json.dumps({"manifest_sha256": release["dataset_manifest_sha256"],
                  "sequences": len(manifest["sequences"])}, indent=2))
PY
```

Use **Dataset → Verify dataset** for a separate non-mutating full-file checksum
scan. The dataset CLI has no `--verify-only`. If verification finds a damaged
part, `scripts/download_dataset.py --repair` scans and repairs affected packages;
with an unchanged complete installation it returns `status: verified` and zero
selected archives. Repair can replace corrupt files, so retain its report.
Retrying the installer reuses verified archives in `data/downloads/`. An
interrupted network transfer itself restarts; completed ZIPs are reusable.

## 4. Start the WebGUI for monitoring

In another server terminal, enter the same clone and run:

```bash
uv run --locked --extra cu118 python app.py \
  --data-root data/mmwave_densepose --host 127.0.0.1 --port 8765 --trust-pickle
```

Use the verified release as the trusted pickle source. On your workstation,
replace `USER@SERVER` and establish the SSH tunnel:

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@SERVER
```

Open the server's printed `http://127.0.0.1:8765/#token=...` URL in your local
browser, including the complete token. Keep the local forwarded port and server
port identical. If 8765 is occupied, change **both** to 8766. Restarting the
service creates a new token. Do not put tokens in screenshots or public reports.

Leave the service running while executing CLI training from the first terminal.
It discovers tasks in the same checkout on the same server. Confirm Dataset
can select a sequence, show synchronized images/points/skeleton and play at
6.5 FPS. Run the explicit dataset verification here and save its result.

## 5. Run a short train → validate → test route

Return to the first Bash session. Define the explicit shared arguments once:

```bash
PREPROCESS_ARGS=(
  --data-root data/mmwave_densepose
  --split-strategy sequence_chunk --seed 42 --partition 0.8 0.1 0.1
  --split-chunk-size 200 --max-timestamp-gap-ms 300 --min-valid-points 50
)
POSE_DATA_ARGS=(
  "${PREPROCESS_ARGS[@]}"
  --num-points 300 --input-dim 5 --seq-len 5 --num-joints 19
)
RUNTIME_ARGS=(--device cuda --gpu-ids 0 --batch-size 128 --num-workers 0)
OPTIMIZER_ARGS=(
  --learning-rate 0.0005 --weight-decay 0.00001 --optimizer adam
  --scheduler-name cosine --scheduler-eta-min 0.000001
)

uv run --locked --extra cu118 python scripts/train.py \
  --model PASTNet "${POSE_DATA_ARGS[@]}" "${RUNTIME_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" --ablation full --epochs 1 --limit-batches 2 \
  2>&1 | tee "$ACCEPT_DIR/05-smoke-train.log"
```

No `--checkpoint` is passed: training starts from scratch. First use may process
the entire source dataset to build caches before the two batches run. Monitor
memory/disk and the preparation phase; short training does not mean short cache
creation. With output piped through `tee`, progress is printed as bounded text
snapshots; in a TTY it uses a progress bar. Keep `pipefail` enabled to retain
the training process's failure status.

Expected: a registered CLI task in **Training & Testing → Execution**, finite
training/validation losses, a best checkpoint selected on validation data and
a completed execution. The batch limit applies to training **and** its internal
validation. This is a connectivity check, not a full experiment.

Define this helper to extract the exact completed checkpoint from the captured
training output and validate its metadata/hash. It avoids guessing run names or
selecting an unrelated earlier experiment:

```bash
checkpoint_from_log() {
  uv run --locked --extra cu118 python - "$1" "$2" <<'PY'
import hashlib, json, sys
from pathlib import Path

lines = Path(sys.argv[1]).read_text().splitlines()
matches = [line.removeprefix("Best checkpoint: ").strip()
           for line in lines if line.startswith("Best checkpoint: ")]
assert len(matches) == 1, "Expected one checkpoint announcement; check the training log"
root = Path.cwd()
checkpoint = Path(matches[0]).resolve()
assert checkpoint.is_relative_to(root / "mmwave_densepose/checkpoints")
model, run_id = checkpoint.parent.parent.name, checkpoint.parent.name
directory = root / "mmwave_densepose/logs/train" / model / run_id
record = json.loads((directory / "run.json").read_text())
assert record["origin"] == "user" and record["action"] == "train"
assert record["config"]["epochs"] == int(sys.argv[2])
assert json.loads((directory / "execution.json").read_text())["status"] == "Complete"
assert record.get("checkpoint_sha256"), "Training did not finish checkpoint publication"
digest = hashlib.sha256()
with checkpoint.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
assert digest.hexdigest() == record["checkpoint_sha256"]
print(checkpoint.relative_to(root).as_posix())
PY
}

SMOKE_CKPT=$(checkpoint_from_log "$ACCEPT_DIR/05-smoke-train.log" 1)
printf '%s\n' "$SMOKE_CKPT" | tee "$ACCEPT_DIR/smoke-checkpoint.txt"

uv run --locked --extra cu118 python scripts/validate.py \
  --model PASTNet "${POSE_DATA_ARGS[@]}" "${RUNTIME_ARGS[@]}" \
  --ablation full --checkpoint "$SMOKE_CKPT" --epochs 1 --limit-batches 2 \
  2>&1 | tee "$ACCEPT_DIR/05-smoke-validate.log"

uv run --locked --extra cu118 python scripts/evaluate.py \
  --model PASTNet "${POSE_DATA_ARGS[@]}" "${RUNTIME_ARGS[@]}" \
  --ablation full --checkpoint "$SMOKE_CKPT" --epochs 1 --limit-batches 2 --visual \
  2>&1 | tee "$ACCEPT_DIR/05-smoke-test.log"
```

`validate.py` uses the validation split; `evaluate.py` uses the test split and
is the GUI's **Test** task. `--visual` writes `test_results.pkl`. For the full
release these two-batch evaluations should have
`evaluation_complete: false` and fewer samples than `expected_samples`.
Do not use `--allow-partial`: it is for partial fixture datasets, not this
full-release acceptance route. Limited runs cannot establish paper results.
The `--epochs 1` values on these validation/test commands retain the short
training setup in their configuration records; they do not repeat evaluation
for multiple epochs.

## 6. Run full PAST-Net training and complete evaluation

Reuse the same explicit data/runtime arguments and remove all batch limits:

```bash
uv run --locked --extra cu118 python scripts/train.py \
  --model PASTNet "${POSE_DATA_ARGS[@]}" "${RUNTIME_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" --ablation full --epochs 200 \
  2>&1 | tee "$ACCEPT_DIR/06-full-train.log"

FULL_CKPT=$(checkpoint_from_log "$ACCEPT_DIR/06-full-train.log" 200)
printf '%s\n' "$FULL_CKPT" | tee "$ACCEPT_DIR/full-checkpoint.txt"

uv run --locked --extra cu118 python scripts/validate.py \
  --model PASTNet "${POSE_DATA_ARGS[@]}" "${RUNTIME_ARGS[@]}" \
  --ablation full --checkpoint "$FULL_CKPT" \
  2>&1 | tee "$ACCEPT_DIR/06-full-validate.log"

uv run --locked --extra cu118 python scripts/evaluate.py \
  --model PASTNet "${POSE_DATA_ARGS[@]}" "${RUNTIME_ARGS[@]}" \
  --ablation full --checkpoint "$FULL_CKPT" --visual \
  2>&1 | tee "$ACCEPT_DIR/06-full-test.log"
```

Expected: 200 completed epochs, validation-based selection, no test data used
to select the epoch, and complete validation/test populations. Record the
measured metrics. Paper references remain frozen; do not edit references or
change evaluation inputs to make a measurement match them.

The full run uses the cache created in stage 5 because its model/data/seed
combination is unchanged. Changing seed, split, point count, features or window
length selects a different cache. Epoch count is not a data-cache setting.
Keep `num-workers=0` for the baseline, then tune it separately if desired.

Training uses automatic parameter-based run directories. Repeated runs receive
new suffixes instead of overwriting earlier files. `--checkpoint` initializes
weights; optimizer/trainer-state resume is not implemented. An interrupted
200-epoch run must not be reported as a completed full-training acceptance.

Verify and summarize the completed full run:

```bash
uv run --locked --extra cu118 python - "$FULL_CKPT" <<'PY' | tee "$ACCEPT_DIR/06-full-artifacts.json"
import hashlib, json, math, sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
run_id = checkpoint.parent.name
train = Path("mmwave_densepose/logs/train/PASTNet") / run_id
record = json.loads((train / "run.json").read_text())
summary = json.loads((train / "training_summary.json").read_text())
assert summary["epochs"] == 200 and summary["selection_split"] == "val"
assert summary["test_used_for_selection"] is False
config = Path(record["config_file"])
assert hashlib.sha256(config.read_bytes()).hexdigest() == record["config_sha256"]
report = {"run_id": run_id, "checkpoint": str(checkpoint),
          "config_file": str(config), "config_id": record["config_id"], "evaluations": []}
for action, directory, pattern in [
    ("validate", train / "validation", "val_*/run.json"),
    ("evaluate", Path("mmwave_densepose/logs/test/PASTNet") / run_id, "test_*/run.json"),
]:
    candidates = []
    for path in directory.glob(pattern):
        item = json.loads(path.read_text())
        if item.get("action") == action and item.get("checkpoint_sha256") == record["checkpoint_sha256"]:
            candidates.append(path)
    assert candidates, f"Missing {action} record"
    path = max(candidates, key=lambda p: p.stat().st_mtime_ns)
    item = json.loads(path.read_text())
    metrics = json.loads((path.parent / "metrics.json").read_text())
    assert json.loads((path.parent / "execution.json").read_text())["status"] == "Complete"
    assert item["origin"] == "user" and item["limit_batches"] is None
    assert item["dataset_manifest_sha256"] == record["dataset_manifest_sha256"]
    assert item["partition_cache_id"] == record["partition_cache_id"]
    assert metrics["evaluation_complete"] is True
    assert metrics["samples"] == metrics["expected_samples"] and metrics["samples"] > 0
    assert metrics["joints"] == 19 and math.isfinite(metrics["mle_cm"])
    if action == "evaluate":
        assert (path.parent / "test_results.pkl").is_file()
    report["evaluations"].append({"action": action, "log_dir": str(path.parent), "metrics": metrics})
print(json.dumps(report, indent=2))
PY
```

## 7. Extended data/model routes

The all-model synthetic check in stage 2 does not test every model's data loader.
Before accepting these features, run the following additional short routes.
Their metrics are diagnostic, not paper comparisons.

### Single-frame data

```bash
uv run --locked --extra cu118 python scripts/train.py \
  --model PointMLP "${PREPROCESS_ARGS[@]}" "${RUNTIME_ARGS[@]}" \
  --num-points 300 --input-dim 5 --seq-len 1 --num-joints 19 \
  "${OPTIMIZER_ARGS[@]}" --epochs 1 --limit-batches 2 \
  2>&1 | tee "$ACCEPT_DIR/07-single-train.log"
SINGLE_CKPT=$(checkpoint_from_log "$ACCEPT_DIR/07-single-train.log" 1)
uv run --locked --extra cu118 python scripts/evaluate.py \
  --model PointMLP "${PREPROCESS_ARGS[@]}" "${RUNTIME_ARGS[@]}" \
  --num-points 300 --input-dim 5 --seq-len 1 --num-joints 19 \
  --checkpoint "$SINGLE_CKPT" --epochs 1 --limit-batches 2 --visual \
  2>&1 | tee "$ACCEPT_DIR/07-single-test.log"
```

### PCC completion and its explicit GlobalLocalNetwork dependency

```bash
PCC_DATA_ARGS=(
  "${PREPROCESS_ARGS[@]}"
  --num-points 80 --input-dim 3 --seq-len 1
  --num-out-points 196 --pcc-label-radius 0.05
)
uv run --locked --extra cu118 python scripts/train.py \
  --model PCCCompletion "${PCC_DATA_ARGS[@]}" "${RUNTIME_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" --epochs 1 --limit-batches 2 \
  2>&1 | tee "$ACCEPT_DIR/07-pcc-train.log"
PCC_CKPT=$(checkpoint_from_log "$ACCEPT_DIR/07-pcc-train.log" 1)
uv run --locked --extra cu118 python scripts/evaluate.py \
  --model PCCCompletion "${PCC_DATA_ARGS[@]}" "${RUNTIME_ARGS[@]}" \
  --checkpoint "$PCC_CKPT" --epochs 1 --limit-batches 2 \
  2>&1 | tee "$ACCEPT_DIR/07-pcc-test.log"

uv run --locked --extra cu118 python scripts/train.py \
  --model GlobalLocalNetwork "${POSE_DATA_ARGS[@]}" "${RUNTIME_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" --num-out-points 196 \
  --completion-checkpoint "$PCC_CKPT" --epochs 1 --limit-batches 2 \
  2>&1 | tee "$ACCEPT_DIR/07-global-local-train.log"
GLOBAL_CKPT=$(checkpoint_from_log "$ACCEPT_DIR/07-global-local-train.log" 1)
uv run --locked --extra cu118 python scripts/evaluate.py \
  --model GlobalLocalNetwork "${POSE_DATA_ARGS[@]}" "${RUNTIME_ARGS[@]}" \
  --num-out-points 196 --checkpoint "$GLOBAL_CKPT" --epochs 1 --limit-batches 2 --visual \
  2>&1 | tee "$ACCEPT_DIR/07-global-local-test.log"
```

Expected: PCC reports finite Chamfer distance; GlobalLocalNetwork training
records the explicitly selected PCC dependency. Its complete pose checkpoint
embeds that component, so the evaluation command supplies only the pose weight.
This short PCC weight tests the dependency path, not official PCC accuracy.

Also exercise **Subject-independent** (`--split-strategy subject`, omit
`--split-chunk-size`) and multiple GPUs if those are release acceptance targets.
For multi-GPU training select `--gpu-ids 0 1` with a suitable total batch size;
validation/test/prediction select only one GPU. Record these as separate cases.
A single-GPU success does not establish physical multi-GPU acceptance.

## 8. Official online assets and reference-weight evaluation

This stage requires the published package URLs in the tested branch. Check the
specific package before interpreting a missing download as a server failure:

```bash
uv run --locked --extra cu118 python scripts/download_paper_assets.py --asset main-pastnet --list \
  | tee "$ACCEPT_DIR/08-weight-status.json"
uv run --locked --extra cu118 python scripts/download_paper_assets.py \
  --kind predictions --asset main-pastnet --test-id test_001 --list \
  | tee "$ACCEPT_DIR/08-prediction-status.json"
```

If the packages are unpublished, mark online delivery **BLOCKED** and continue
recording the already tested own-model route. To complete this gate, upload the
prepared packages to ScienceDB, publish the real URLs in `develop`, then repeat
the affected acceptance stages against the updated commit. A local ZIP copied
from the developer workstation does not establish public online delivery.

When URLs are available, first test the GUI missing-file route on this fresh
checkout: **Test → PASTNet → official main weight → Download official weights**.
Observe progress/cancellation and Installed. Then **Results → Official
evaluation → PASTNet → main weight → test_001 → Download predictions**. Metrics
must remain visible; open visualization explicitly after installation.

Run the public CLI route as well (an already installed package is verified and
reused). Use another missing packaged asset to test a real CLI network transfer
if the GUI has already installed the selected one:

```bash
uv run --locked --extra cu118 python scripts/download_paper_assets.py --asset main-pastnet --progress-json \
  2>&1 | tee "$ACCEPT_DIR/08-weight-install.log"
uv run --locked --extra cu118 python scripts/download_paper_assets.py --asset main-pastnet --verify-only \
  2>&1 | tee "$ACCEPT_DIR/08-weight-verify.log"
uv run --locked --extra cu118 python scripts/download_paper_assets.py \
  --kind predictions --asset main-pastnet --test-id test_001 --progress-json \
  2>&1 | tee "$ACCEPT_DIR/08-prediction-install.log"

uv run --locked --extra cu118 python scripts/evaluate.py \
  --model PASTNet "${POSE_DATA_ARGS[@]}" "${RUNTIME_ARGS[@]}" --ablation full \
  --checkpoint paper/experiments/main-pastnet/checkpoints/best.ckpt --visual \
  2>&1 | tee "$ACCEPT_DIR/08-official-weight-test.log"
```

Expected: the installed checkpoint/member SHA256 values match the release and
the new evaluation is complete. Testing an official weight produces a new
**My evaluation** record; it does not replace the imported official test.
Compare its dataset identity, split, preprocessing, sample population and
measured metrics with the matching paper row. Keep the measured report intact.

For all assets, use `scripts/download_paper_assets.py --all` and
`--kind predictions --all`. Only five selected official test records currently
have prediction packages (six PKLs); the other 31 remain metrics-only.
Offline `--archive-dir PATH` is supported but is a separate installation case.

Preview the paper experiment selection before any full table sweep:

```bash
uv run --locked --extra cu118 python paper/run.py --all --action evaluate --device cuda --dry-run \
  2>&1 | tee "$ACCEPT_DIR/08-paper-plan.log"
```

Four historical assets (`main-dgcnn`, `main-point-transformer`, `main-pointmlp`,
`ablation-pointnet`) have unresolved original `single_intra_s42` partition
provenance. Their current JSON files can describe a supported split, but running
them with `sequence_chunk` does not recover the original partition. Treat their
exact official-result reproduction as unresolved until that identity is
established; do not report an unqualified all-table reproduction pass.

## 9. Browser and saved-artifact acceptance

| Check | Expected result |
| --- | --- |
| Dataset selection/playback | Anonymized images, radar points and skeleton align; playback remains usable. |
| Dataset verification | Complete installed release passes the frozen checksum inventory. |
| CLI task discovery | The CLI training run appears in Execution with progress, parameters and console output. |
| Test configuration | Own/official weights restore the right model/data settings; missing official weights show status/download. |
| Training logs | Own training scalar curves and Native TensorBoard can be opened; official logs remain separately identified. |
| New evaluations | Smoke and full tests appear under My evaluation with the correct training run and test ID. |
| Visual export | Own `--visual` test supports point/skeleton/prediction playback; recorded PKL download leaves metrics visible. |
| Completeness | Smoke tests are limited; full validation and tests cover the complete expected population. |
| Official identity | Table tags remain attached to official records; own tests are not relabeled official. |
| Run isolation | Repeated commands create new run/test IDs and do not overwrite official or earlier user artifacts. |

Where to inspect the evidence:

```text
mmwave_densepose/checkpoints/<Model>/<run_id>/best.ckpt
mmwave_densepose/logs/train/<Model>/<run_id>/
  run.json, execution.json, config-reference.json, hparams.yaml
  metrics.csv, events.out.tfevents.*, training_summary.json
  validation/val_001/...
mmwave_densepose/logs/test/<Model>/<run_id>/test_001/
  run.json, execution.json, metrics.json, per_joint_metrics.csv
  test_results.pkl                 # pose evaluation with --visual
configs/runs/<Model>/<action>-<config-id>.yaml
data/mmwave_densepose/cache/index.json
.local/acceptance/<acceptance-session>/
```

Paths printed by the commands are authoritative; later tests increment their
IDs. Shared numeric caches are indexed in the dataset cache tree. Read them
through **Dataset cache** in Training & Testing rather than inferring the split
from a cache filename. Automatically saved run YAML files contain the full
configuration, and identical configurations reuse the same file.

Distance fields ending `_cm` are centimeters. Legacy/TensorBoard distance
scalars and `per_joint_mle.csv` use meters; PCK/AUC scalar fractions become
percentages in the GUI. Record units when comparing exported values.

## 10. Acceptance report and merge gate

Create a report in `$ACCEPT_DIR` with these fields:

| Field | Record |
| --- | --- |
| Tested branch/commit | `develop`, exact `$TESTED_COMMIT` |
| Server | OS/glibc/architecture, GPU(s), driver and available memory/disk |
| Environment | uv version, Python/package versions, native-op/model-check logs |
| Release | Dataset version/status/manifest SHA256, asset manifest, URL availability |
| Data | Download, installed-file verification, cache preparation/reuse outcome |
| Own training | Smoke/full run IDs, configuration ID, checkpoint path/SHA256 |
| Validation/test | Log IDs, sample counts/completeness, metrics, visual PKL path |
| Extended cases | Single-frame, PCC, GlobalLocalNetwork, subject split, multi-GPU scope |
| Official delivery | CLI and GUI downloads, verification, selected reference-weight test |
| Browser | Screenshots of data, task discovery, logs, results and download states |
| Outcome | PASS / FAIL / BLOCKED / NOT RUN per case, with a cause and evidence path |

Before proposing the merge:

```bash
git diff --check
git status --short
git diff --exit-code
git diff --cached --exit-code
git ls-files --others --exclude-standard
git fetch origin
printf 'Tested commit: %s\n' "$TESTED_COMMIT"
git rev-parse origin/develop
```

Expected: no public source/config/reference changes caused by using the repo;
data, new weights, personal logs, caches and acceptance evidence remain ignored.
`origin/develop` must equal the commit actually tested. If it advanced, test
the additional changes before calling that new commit accepted.

Merge through a `develop` → `main` pull request only after all required cases
are PASS. Attach the report and tested commit; keep tokens and large binary
files out of the PR. Record optional untested hardware cases explicitly.
Missing official download URLs leave the complete public-delivery gate BLOCKED.
An all-paper-results release also requires resolving the historical partition
cases and verifying its claimed table coverage. A two-batch smoke success alone
is not a merge acceptance result.

This runbook does not merge or push either branch. After an approved merge,
perform one fresh clone of `main` and repeat installation plus the short route
against the released package manifests.

## Troubleshooting by stage

| Symptom | First check |
| --- | --- |
| Lock/package installation fails | Match `pyproject.toml`/`uv.lock`, supported Linux x86_64 and the complete `cu118` extra; retain the error log. |
| CUDA/native operators fail | Stage 2 GPU/driver output and exact wheel versions; do not silently continue on CPU. |
| Download returns HTML or a checksum error | Actual ScienceDB file URL/access and release manifest; do not bypass checksums. |
| Official package says unpublished | Stage 3/8 URL counts; publish the real URLs and retest the updated commit. |
| First training spends a long time preparing | Source-PKL cache generation, RAM and free space; inspect Execution rather than invoking a maintainer cache script. |
| CUDA out of memory | Record the failure; reduce the batch size consistently for a diagnostic rerun, recording it as a changed training setup. |
| Checkpoint helper fails | Original CLI exit status, `training_summary.json`, `execution.json` and checkpoint hash; do not use an interrupted run as a completed one. |
| CLI run is absent in the GUI | Same clone, same server/data root, current service version and `execution.json`; another machine's process is not local discovery. |
| New test missing from Results | My evaluation, matching model/training weight/test ID, then Refresh. |
| Metrics visible but no frame playback | `--visual`, the PKL's local presence, trusted loading and dataset availability. |
| SSH page reports token/host errors | Full newly printed URL and matching local/server tunnel ports. |
