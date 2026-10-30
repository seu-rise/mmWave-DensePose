# Training, validation, testing and prediction CLI

Run commands from the repository root. No input config file is required: choose
the script, specify `--model`, and pass the settings you need. See
[training and evaluation](training.md) for complete copyable commands.

## Entry points and precedence

| Script | Action and data split | Checkpoint |
| --- | --- | --- |
| `scripts/train.py` | Train on `train`, validate on `val` after each epoch | Optional weight initialization |
| `scripts/validate.py` | Measure the `val` split | Required |
| `scripts/evaluate.py` | Test on the `test` split | Required |
| `scripts/infer.py` | Export predictions for the labeled `test` split | Required |

The GUI's **Test** task calls `evaluate.py`; there is no separate `test.py`.
`infer.py` is not an arbitrary unlabeled-file prediction interface.

Explicit CLI arguments override an optional JSON/YAML `--config`. Unspecified
settings use the file's values or model-aware defaults. Defaults below describe
the public **no-config CLI**, not every official experiment or GUI selection.
The model selects single-frame pose, temporal pose or PCC completion data;
`--split-strategy` selects only the partition protocol.

```bash
uv run --locked --extra cu118 python scripts/train.py --help
uv run --locked --extra cu118 python scripts/evaluate.py --help
uv run --locked --extra cu118 python scripts/train.py --list-models
```

For CPU operation, use `--extra cpu` and `--device cpu`. Omit compute extras only
when browsing the WebGUI without executing model tasks.

## Model and weight selection

| Parameter | Default / choices | Meaning and applicability |
| --- | --- | --- |
| `--model NAME` | Required without `--config`; identifiers below | Case-sensitive implementation name. |
| `--checkpoint PATH` | None | Training: initialize model weights only. Other actions: required unless supplied by the config. Does not restore the data/split arguments automatically. |
| `--completion-checkpoint PATH` | None | Explicit PCC weights required for `GlobalLocalNetwork` training; invalid for other models. A complete pose checkpoint embeds its PCC branch, so ordinary GlobalLocalNetwork evaluation needs only `--checkpoint`. |
| `--ablation NAME` | `full`; `full`, `spatial_only`, `baseline` | PASTNet architecture variant; must match its checkpoint. Other model implementations do not use this selector. |
| `--num-joints N` | `19`; positive integer | Source/output pose joint count. Keep the checkpoint-compatible value. PCC does not produce pose joints. MMPoseFK still uses 19-joint source labels and its supported 17-joint metric subset. |
| `--list-models` | Off | Print identifiers and exit without training or loading data; no `--model` required. |

| Model identifiers | Data family / default window |
| --- | --- |
| `DGCNN`, `PointTransformer`, `PointMLP`, `PoseGraphNet`, `PointNet2Baseline` | Single-frame pose, `--seq-len 1` |
| `PASTNet`, `MMPoseFK`, `DualPathCNNBiLSTM`, `PntPoseAT`, `GlobalLocalNetwork` | Temporal pose, `--seq-len 5` |
| `PCCCompletion` | Point completion, `--seq-len 1`, XYZ only |

Training without `--checkpoint` starts from scratch. There is no `--resume`
option: supplying weights starts a new training run with a new optimizer and
scheduler. A shared paper config's evaluation checkpoint is not used as a
training initializer. A saved same-action `action: train` snapshot can preserve
its recorded initial checkpoint when replayed.

## Dataset, partitions and inputs

These parameters apply to training, validation, evaluation and prediction. Use
the same values that produced the selected checkpoint when measuring it.

| Parameter | No-config default / choices | Meaning |
| --- | --- | --- |
| `--data-root PATH` | `data/mmwave_densepose` | Installed dataset directory containing its manifest and source sequences. |
| `--split-strategy NAME` | `sequence_chunk`; `sequence_chunk`, `subject` | Paper labels: **Sequence-chunk** and **Subject-independent**, respectively. These are the only CLI split choices. |
| `--seed N` | `42`; nonnegative integer | Partition allocation, point sampling, loader order and model initialization seed. |
| `--partition TRAIN VAL TEST` | `0.8 0.1 0.1` | Three finite nonnegative ratios with positive sum; normalized to sum to one. |
| `--split-chunk-size N` | `200` for `sequence_chunk`; unset for `subject` | Positive chunk length in frames. Sequence-chunk allocates chunks before forming windows. Subject mode without this option uses whole-sequence windows and subject-count allocation; an explicit positive value requests its weighted/chunked variant. |
| `--max-timestamp-gap-ms MS` | `300` | Positive temporal continuity threshold: adjacent frame gaps must be strictly below it. |
| `--min-valid-points N` | `50`; nonnegative integer | A frame must contain strictly more than N points with finite XYZ coordinates. This is a filtering threshold, not the sampled input size. |
| `--num-points N` | Pose: `300`; PCC: `80` | Positive input point count per frame, after filtering and point sampling/padding. |
| `--input-dim C` | Pose: `5`; PCC: `3`; choices `3`, `4`, `5` | `3`: XYZ; `4`: XYZ + Doppler; `5`: XYZ + Doppler + SNR. PCC requires 3. |
| `--seq-len T` | Temporal pose: `5`; single-frame pose/PCC: `1` | Positive temporal window length. Single-frame models and PCC reject values other than 1. |

For Subject-independent runs without an input config, use
`--split-strategy subject` and omit `--split-chunk-size`. Overriding a file's
strategy does not remove a chunk size already recorded in that file.

The original PKLs remain the source dataset. Numeric sequence caches,
partition indexes and training-only normalization statistics are generated
automatically. Cache identity includes the model/data family, strategy, seed,
input/preprocessing settings and source fingerprint; the registry is
`data/mmwave_densepose/cache/index.json`. A changed seed does not reuse the
previous partition just because the strategy name is the same.

## Training and scheduler settings

These options control `train.py`; they do not change the measured weights in
validation, evaluation or prediction.

| Parameter | No-config default / choices | Meaning |
| --- | --- | --- |
| `--epochs N` | Pose: `200`; PCC: `100` | Positive number of training epochs. |
| `--learning-rate LR` | `0.0005` | Positive finite Adam learning rate. |
| `--weight-decay WD` | `0.00001` | Nonnegative finite weight decay. |
| `--optimizer NAME` | `adam`; only `adam` | Optimizer. |
| `--scheduler-name NAME` | `cosine`; `cosine`, `step`, `none` | Epoch-based learning-rate schedule. Cosine uses the configured epoch count as its period. |
| `--scheduler-eta-min LR` | `0.000001` | Nonnegative minimum learning rate; used by `cosine`. |
| `--scheduler-step-size N` | `30` | Positive number of epochs between decays; used by `step`. |
| `--scheduler-gamma FACTOR` | `0.1` | Positive finite multiplicative decay factor; used by `step`. |

The best pose checkpoint is selected by validation MLE; PCC uses validation
Chamfer distance. The test split does not select the best epoch.

## Runtime and memory

| Parameter | No-config default / choices | Meaning and applicability |
| --- | --- | --- |
| `--device DEVICE` | `auto`; use `auto`, `cpu`, `cuda`, `cuda:N` | Auto selects CUDA if available, otherwise CPU. Explicit CUDA requests fail if unavailable. |
| `--gpu-ids ID [ID ...]` | No explicit list | Visible CUDA device indices. Training can use multiple GPUs; validation/evaluation/prediction accept one. The first selected GPU is primary; keep `--device` compatible with it. |
| `--batch-size N` | `128` | Total batch size across selected GPUs. Evaluation requires a positive value; training requires at least two samples, and multi-GPU training at least two per selected GPU. |
| `--num-workers N` | `0` | DataLoader subprocess count; 0 loads in the main process. |
| `--torch-threads N` | `4` | PyTorch CPU thread count; applied when running on CPU. |
| `--cache-size N` | `141` | Positive maximum number of decoded numeric sequences retained by each dataset/loader process. This is an in-memory LRU size, not a disk-cache count or split size. Reduce it when RAM is limited; workers can each retain their own cache. |
| `--limit-batches N` | Unset | Positive batch limit for diagnostic runs. Training limits both train and validation loaders. Limited tests are recorded as incomplete. |

For example, add `--device cuda --gpu-ids 0 1 --batch-size 128` to a training
command to distribute the total batch across two visible GPUs. Evaluation uses
one, for example `--device cuda --gpu-ids 0`.

## PCC completion settings

Use these with `--model PCCCompletion`; this model trains point completion,
not human-pose regression. Its loss/metric is non-squared bidirectional Chamfer
distance. It requires single-frame XYZ input.

| Parameter | No-config default / choices | Meaning |
| --- | --- | --- |
| `--num-out-points N` | `196` | Positive number of completed XYZ points produced by PCC. GlobalLocalNetwork also accepts this architecture setting for its embedded completion branch; match the PCC checkpoint. |
| `--pcc-label-radius M` | `0.05` | Positive sphere-supervision radius in meters used to construct PCC dense labels. |
| `--dense-label-mode NAME` | `pcc`; only `pcc` | PCC sphere supervision; no alternative dense-label modes are exposed. |

GlobalLocalNetwork training uses an explicitly selected PCC checkpoint and
freezes that branch. Omit `--completion-checkpoint` when testing its complete
pose checkpoint, unless you intentionally supply the exact matching dependency;
conflicting PCC weights are rejected.

## Exports, diagnostics and files

| Parameter | Default | Meaning and applicability |
| --- | --- | --- |
| `--visual` / `--no-visual` | Off | Pose `evaluate.py` only. Export `test_results.pkl` containing inputs, predictions, ground truth and sample IDs for Results/Avatar playback. Rejected for train/validate/infer and PCC. |
| `--save-predictions` / `--no-save-predictions` | Off; always on for `infer.py` | Export sharded numeric `predictions/batch_*.npz` plus its manifest during validation/evaluation/prediction. Training does not export prediction shards. Infer always saves, even with the negative flag. |
| `--allow-partial` / `--no-allow-partial` | Off | Permit intentional numeric fixtures marked partial. Does not bypass incomplete published-release installation checks. Partial runs remain labeled incomplete. |
| `--output-dir PATH` | Automatic | Override the log directory only, within the action/model's prescribed tree. Must name a new direct child; validation/test/prediction names use `val_001` / `test_001` / `predict_001` numbering. Does not relocate model checkpoints. |
| `--config PATH` | None | Optional JSON/YAML input. Use underscore keys such as `learning_rate` inside the file; use hyphens such as `--learning-rate` on the CLI. |
| `--help` / `-h` | Off | Show accepted options and exit. |

Negative boolean flags can override `true` values in an input config. If omitted,
the corresponding file value is retained; otherwise the defaults above apply.

Every real CLI run writes or reuses a complete resolved YAML snapshot in
`configs/runs/<Model>/<action>-<config-id>.yaml`. Non-default parameters come
first, separated by comments from defaults. Logs retain its filename and hash.
This happens automatically even when the command does not use `--config`.

| Generated artifact | Default location |
| --- | --- |
| User training checkpoint | `mmwave_densepose/checkpoints/<Model>/<run_id>/best.ckpt` |
| User training log | `mmwave_densepose/logs/train/<Model>/<run_id>/` |
| User validation of a personal checkpoint | `mmwave_densepose/logs/train/<Model>/<run_id>/validation/val_001/` |
| User evaluation | `mmwave_densepose/logs/test/<Model>/<training-run-id>/test_001/` |
| User prediction export | `mmwave_densepose/logs/test/<Model>/<training-run-id>/predict_001/` |

Validation of a registered official checkpoint is nested under its registered
training-log directory. It still records a new user run, as does testing an
official weight. Existing run directories are preserved and numbered/versioned
automatically. For concrete file contents, metric units, monitoring and matching
official records, continue with [training](training.md#saved-files) and
[official results](official-results.md).
