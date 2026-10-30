# Training and evaluation

Run all commands from the repository root with a [compute profile](environment.md).
The examples below use CUDA; replace `--extra cu118` with `--extra cpu` and
`--device cuda` with `--device cpu` for CPU operation.

## Data and weights

Install the trusted dataset under `data/mmwave_densepose/`; see the
[dataset guide](dataset.md). The source PKLs, manifest and subject mapping are
required. Model tasks generate their numeric caches automatically on first use;
there is no required separate cache-preparation command. Allow time and disk
space for this first run.

Official experiment configurations are in [`paper/configs/`](../../paper/configs/).
Prepared weights and their hashes are registered in
[`paper/manifest.json`](../../paper/manifest.json). The code and dataset are
released; a fresh clone does not contain the large dataset and checkpoint
binaries. Download the dataset and obtain the matching weights separately.
See [official results](official-results.md) for asset locations and experiment associations.

No input configuration file is required: the commands below specify parameters
directly. `--model` selects the data family, while the split strategy, seed and
data settings select or generate the matching cache. The
[complete CLI parameter reference](cli-reference.md) lists every option, default
and model/action restriction. Prepared paper configurations remain in
`paper/configs/`; keep your custom inputs separate from those official materials.

## Train PAST-Net

```bash
uv run --locked --extra cu118 python scripts/train.py --model PASTNet --data-root data/mmwave_densepose --split-strategy sequence_chunk --seed 42 --partition 0.8 0.1 0.1 --split-chunk-size 200 --max-timestamp-gap-ms 300 --min-valid-points 50 --num-points 300 --input-dim 5 --seq-len 5 --num-joints 19 --epochs 200 --batch-size 128 --learning-rate 0.0005 --weight-decay 0.00001 --optimizer adam --ablation full --scheduler-name cosine --scheduler-eta-min 0.000001 --num-workers 0 --cache-size 141 --device cuda
```

This command trains from scratch. To initialize from existing weights, add
`--checkpoint PATH`. This loads model weights only; optimizer/trainer-state
resume is not implemented. A saved action-specific YAML can also preserve its
recorded initial checkpoint when replayed with the same script.

The terminal prints the allocated log directory and best checkpoint path.
Validation MLE selects the best pose checkpoint; validation Chamfer distance
selects the best `PCCCompletion` checkpoint. Test data does not select an epoch.

## Evaluate a trained checkpoint

Use the same model, input dimensions, window length, preprocessing and split
configuration used during training. Supplying a checkpoint to the CLI does not
automatically restore all of those data parameters.

```bash
uv run --locked --extra cu118 python scripts/evaluate.py --model PASTNet --checkpoint paper/experiments/main-pastnet/checkpoints/best.ckpt --data-root data/mmwave_densepose --split-strategy sequence_chunk --seed 42 --partition 0.8 0.1 0.1 --split-chunk-size 200 --max-timestamp-gap-ms 300 --min-valid-points 50 --num-points 300 --input-dim 5 --seq-len 5 --num-joints 19 --ablation full --batch-size 128 --num-workers 0 --cache-size 141 --device cuda --visual
```

The example explicitly selects the installed official PAST-Net weight. To test
your own training, replace `--checkpoint` with the path printed by that run and
repeat its model, data and split arguments. Optimization arguments such as
`--epochs` and `--learning-rate` are not needed for evaluation. A new evaluation
of an official checkpoint appears under **My evaluation**, separate from the
recorded official result.

`--visual` creates `test_results.pkl` for the Results panel's point-cloud and
Avatar views. It is available only for pose evaluation. `--save-predictions`
separately exports sharded numeric NPZ predictions. Metrics alone do not provide
frame visualization.

The other CLI workflows are:

```bash
uv run --locked --extra cu118 python scripts/validate.py --model PASTNet --checkpoint paper/experiments/main-pastnet/checkpoints/best.ckpt --data-root data/mmwave_densepose --split-strategy sequence_chunk --seed 42 --partition 0.8 0.1 0.1 --split-chunk-size 200 --max-timestamp-gap-ms 300 --min-valid-points 50 --num-points 300 --input-dim 5 --seq-len 5 --num-joints 19 --batch-size 128 --num-workers 0 --device cuda
uv run --locked --extra cu118 python scripts/infer.py --model PASTNet --checkpoint paper/experiments/main-pastnet/checkpoints/best.ckpt --data-root data/mmwave_densepose --split-strategy sequence_chunk --seed 42 --partition 0.8 0.1 0.1 --split-chunk-size 200 --max-timestamp-gap-ms 300 --min-valid-points 50 --num-points 300 --input-dim 5 --seq-len 5 --num-joints 19 --batch-size 128 --num-workers 0 --device cuda
```

`validate.py` uses the validation split. `infer.py` exports predictions for the
**labeled test split**; it is not an arbitrary unlabeled-file interface.
The GUI offers Train, Test and Predict; its Test task runs `evaluate.py`.

For short diagnostics, `--limit-batches N` limits computation.
`--allow-partial` permits intentional numeric fixtures marked partial; it does
not bypass the full published-release installation check. Limited/partial runs
are marked incomplete and are not equivalent to a complete benchmark test.

## Models and data parameters

Model identifiers are case-sensitive. List them with
`uv run --locked --extra cu118 python scripts/train.py --list-models`.

| Model identifiers | Data family |
| --- | --- |
| `DGCNN`, `PointTransformer`, `PointMLP`, `PoseGraphNet`, `PointNet2Baseline` | Single-frame pose, T=1 |
| `PASTNet`, `MMPoseFK`, `DualPathCNNBiLSTM`, `PntPoseAT`, `GlobalLocalNetwork` | Temporal pose, default T=5 |
| `PCCCompletion` | Single-frame XYZ point completion, T=1 |

| Option | Meaning |
| --- | --- |
| `--split-strategy sequence_chunk` | Allocate chunks using the selected seed and ratios; default chunk size is 200 frames. |
| `--split-strategy subject` | Separate subjects between splits. Omit chunk size for whole-sequence windows and subject-count allocation; an explicit positive chunk size selects the weighted/chunked variant. |
| `--partition TRAIN VAL TEST` | Three nonnegative ratios with a positive sum; default `0.8 0.1 0.1`. |
| `--seed N` | Split, ordering, initialization and sampling seed; default 42. |
| `--num-points N` | Input points per frame; pose default 300, PCC default 80. |
| `--input-dim C` | 3=XYZ, 4=XYZ+Doppler, 5=XYZ+Doppler+SNR; PCC requires 3. |
| `--seq-len T` | Positive temporal window length; single-frame models and PCC require 1. |
| `--min-valid-points N` | Each selected frame must contain strictly more than N finite XYZ points; default 50. |
| `--max-timestamp-gap-ms MS` | Adjacent temporal-frame gaps must be strictly below this threshold; default 300 ms. |
| `--ablation NAME` | PASTNet only: `full`, `spatial_only`, or `baseline`. Must match the checkpoint. |

The model determines its data family. A cache filename is not a split strategy.
Changing seed or data parameters selects a different cache/partition combination.
MMPoseFK reads 19-joint source labels but computes pose metrics over its supported
17-joint order.

See [all CLI parameters](cli-reference.md) for optimization, scheduler, device,
cache and export options, including exact defaults. Each script also supports
`--help`. Official baseline configurations can differ from general CLI defaults.

For **Subject-independent** partitioning without a config file, select
`--split-strategy subject` and omit `--split-chunk-size`. Keep all other
arguments appropriate for the selected model. With a config file, a recorded
chunk size remains in effect unless you remove it from that file. The CLI value
is `subject`; `Subject-independent` is the paper/GUI display name.

## Devices and PCC dependencies

Use `--device cpu`, `--device cuda`, or a specific device such as
`--device cuda:0`. CUDA evaluation, validation and prediction can select **one**
GPU using `--gpu-ids 0`. Training can select several, for example
`--device cuda --gpu-ids 0 1`; the first is primary. Batch size is the total
across the selected GPUs and must provide at least two samples per GPU during
multi-GPU training.

`PCCCompletion` uses 80 XYZ input points and predicts 196 XYZ points in the
official configurations. Its loss and metric are non-squared bidirectional
Chamfer distance, rather than pose MLE.

Train the PCC completion model directly:

```bash
uv run --locked --extra cu118 python scripts/train.py --model PCCCompletion --data-root data/mmwave_densepose --split-strategy sequence_chunk --seed 42 --partition 0.8 0.1 0.1 --split-chunk-size 200 --max-timestamp-gap-ms 300 --min-valid-points 50 --num-points 80 --input-dim 3 --seq-len 1 --num-out-points 196 --pcc-label-radius 0.05 --dense-label-mode pcc --epochs 100 --batch-size 128 --learning-rate 0.0005 --weight-decay 0.00001 --optimizer adam --scheduler-name cosine --scheduler-eta-min 0.000001 --num-workers 0 --device cuda
```

`GlobalLocalNetwork` training requires an explicitly selected PCC checkpoint.
This example uses the installed official dependency; alternatively supply the
checkpoint printed by your PCC training:

```bash
uv run --locked --extra cu118 python scripts/train.py --model GlobalLocalNetwork --completion-checkpoint paper/experiments/main-pcc-dense/checkpoints/best.ckpt --data-root data/mmwave_densepose --split-strategy sequence_chunk --seed 42 --partition 0.8 0.1 0.1 --split-chunk-size 200 --max-timestamp-gap-ms 300 --min-valid-points 50 --num-points 300 --input-dim 5 --seq-len 5 --num-joints 19 --epochs 200 --batch-size 128 --learning-rate 0.0005 --weight-decay 0.00001 --optimizer adam --scheduler-name cosine --scheduler-eta-min 0.000001 --num-workers 0 --device cuda
```

Choose a PCC weight trained with the intended data protocol. The completion
branch is frozen during pose training. A complete GlobalLocalNetwork pose
checkpoint embeds that branch, so ordinary evaluation needs only the pose
checkpoint. Do not add an unrelated PCC weight during evaluation.

## Optional configuration files

`--config` is a convenience for repeated runs. It accepts JSON or YAML and is
not required by any example above. The effective setting order is **explicit
CLI argument → configuration file → model-aware default**. For example:

```bash
uv run --locked --extra cu118 python scripts/train.py --config paper/configs/main-pastnet.json --batch-size 64 --device cuda
```

Here `--batch-size 64` overrides the file. Copy a paper configuration to your
own directory before changing it. Training ignores an evaluation checkpoint in
a shared paper config unless `--checkpoint` is explicitly supplied; replaying
a saved `action: train` snapshot can restore its recorded initial checkpoint.
Validation, evaluation and prediction can use the checkpoint in the input file,
but explicit `--checkpoint PATH` takes precedence.
A training snapshot records initialization weights; it does not automatically
point to the best checkpoint saved later by that run.

Every actual run records a complete resolved configuration automatically, even
when no input file is supplied. Identical settings reuse the same snapshot in
`configs/runs/`; logs record its path and identity. This generated snapshot is
a record of the command, not a required input file.

## Saved files

```text
mmwave_densepose/checkpoints/<Model>/<run_id>/best.ckpt
mmwave_densepose/logs/train/<Model>/<run_id>/
mmwave_densepose/logs/train/<Model>/<run_id>/validation/val_001/
mmwave_densepose/logs/test/<Model>/<run_id>/test_001/
mmwave_densepose/logs/test/<Model>/<run_id>/predict_001/
configs/runs/<Model>/<action>-<config-id>.yaml
```

Readable run IDs encode the major parameters. Exact identity comes from the
saved configuration and checkpoint hash. Repeated training gains `_r02`, `_r03`,
and later suffixes; repeated test/prediction/validation directories increment
their numbers. Existing runs are never overwritten. Leave `--output-dir`
unset for automatic allocation; overrides must stay in the corresponding
model's log tree and do not relocate checkpoints.

New runs write `hparams.yaml`, `metrics.csv` and native TensorBoard event files,
alongside `run.json` and configuration provenance. Pose evaluation also writes
`metrics.json`, `per_joint_metrics.csv` and compatible `per_joint_mle.csv`.
PCC writes completion/per-cloud metrics instead of pose-joint metrics.

TensorBoard/legacy CSV distance scalars, including `mle` and `per_joint_mle.csv`,
are in **meters**; PCK/AUC scalars are fractions. Summary fields ending `_cm`
are centimeters. CSV epochs are zero-based. The GUI converts display units to
centimeters and percentages. Keep these units in mind when exporting values.

Official and personal logs both live under `mmwave_densepose/logs/`; verified
origin metadata separates them. Prepared official weights stay under
`paper/experiments/<asset-id>/checkpoints/`. Personal logs, weights and generated
prediction files are excluded from Git.

## Monitor runs and compare results

Open [Training & Testing → Execution](web-gui.md) to monitor a local CLI run's
progress and console output, or launch a validated GUI task. CLI runs register
automatically in the same repository on the same machine. The GUI's Logs tab
lists training logs; completed tests open in Results.

Use **Logs → Native TensorBoard** to open all indexed training event logs in a
new browser tab, with a distinct name for each run. This includes both personal
and official logs regardless of the Logs panel's current filter. Alternatively,
start TensorBoard directly:

```bash
uv run --locked --extra cu118 python -m tensorboard.main --logdir mmwave_densepose/logs/train --host 127.0.0.1
```

Compare a measurement with an official reference only when the input population,
split, preprocessing, model variant and checkpoint agree. See
[official results](official-results.md) for reproduction workflows.
