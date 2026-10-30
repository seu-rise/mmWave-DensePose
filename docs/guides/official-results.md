# Official results and experiment tables

Use [Web GUI](web-gui.md) to browse the released figures, tables and recorded
experiments. Use the commands below to run a new evaluation with your installed
dataset and weights. All commands run from the repository root.

## Available materials

The [asset manifest](../../paper/manifest.json) associates **37 checkpoints**
with **37 official training-log groups** and **36 selected test records**.
There are 35 pose assets and two PCC completion dependencies. The subject-split
PCC dependency has a training log but no standalone test record.

| Material | Location |
| --- | --- |
| Checkpoint metadata, sizes and SHA256 | `paper/manifest.json` |
| Download packages, URLs and archive/member checksums | `paper/download-manifest.json` |
| Experiment configurations | `paper/configs/` |
| Official checkpoint files | `paper/experiments/<asset-id>/checkpoints/best.ckpt` |
| Official and user training logs | `mmwave_densepose/logs/train/<Model>/<run-id>/` |
| Official and user test logs | `mmwave_densepose/logs/test/<Model>/<training-run-id>/<test-id>/` |
| Manuscript table values | `paper/reference/table_01.csv` through `table_10.csv` |
| Frozen figure assets | `paper/figures/` |

The code and dataset are released. Dataset and checkpoint binaries are
distributed separately from Git. Dataset packages are listed in the release
manifest; checkpoint metadata and required file locations are listed in the
asset manifest. Use the [official asset installer](paper-assets.md) to download
weights or optional prediction packages. Download the dataset and obtain the matching weights before
running model evaluations.
Follow [dataset installation](dataset.md) and [environment setup](environment.md)
before running an evaluation. Check each downloaded weight against its manifest
size and SHA256 using the installer, which places it at the listed path.

## Browse recorded results

Official weight and prediction downloads can be continued after an interruption
or service restart. **Continue download** verifies and reuses completed cached
packages before fetching missing ones. **Verify installed files** checks existing
files that have not yet been verified by the installer. Files with conflicting
content require explicit **Repair**. An interrupted file transfer resumes at the
package level; an unfinished ZIP may need to be downloaded again.

- **Training & Testing → Logs** shows training runs, including their recorded
  parameters, scalar curves and native TensorBoard. It excludes test logs.
- **Results → Official evaluation** shows the selected official test records.
  **My evaluation** shows tests run from this checkout, including new evaluations
  of official weights. Both sources use `mmwave_densepose/logs/`.
- Select model, training weight and test ID to locate one test. Table tags show
  the checkpoint's association with the manuscript. A PCC tag identifies a
  completion dependency, rather than a pose-method row.
- A test needs its own usable prediction PKL for frame visualization. Metrics
  remain available without it. Prediction files are separate large assets.

Imported records preserve their available parameters and measured values.
An imported record marked `recorded` has not been established as a complete new
evaluation against the current dataset. Unknown values remain unknown.

The GUI restores and locks the selected weight's model, input and split
settings. Four historical assets use an unsupported `intra_sequence` protocol:
`main-dgcnn`, `main-point-transformer`, `main-pointmlp` and `ablation-pointnet`.
Their GUI evaluation entries remain disabled until the original partition can
be recovered. Selecting a current `sequence_chunk` configuration does not
establish an equivalent evaluation of those historical weights.

## Official reference tables and My results

In **Official results**, **Official** displays frozen manuscript values.
Tables 1–3 are informational and have **My results** disabled. In Tables 4–10,
**My results** uses compatible tests produced in this checkout. Missing eligible
measurements appear as a dash.

A candidate must match the model, seed, input/preprocessing configuration,
dataset manifest and generated partition identity, and evaluate the complete
test population. Partial datasets, batch-limited runs, wrong thresholds and
incompatible joint counts are excluded. The pose threshold is 0.15 m;
MMPoseFK uses its supported 17-joint subset and other pose models use 19 joints.

Cells independently select the best eligible value: minimum MLE/PA-MPJPE or
maximum PCK/AUC. Consequently, different cells can come from different tests.
Official imported records and manuscript values are not substituted for missing
user measurements.

The manuscript reports **570,124 valid synchronized frames** in Table 2;
Table 3 displays the rounded **570,000**. These differ in scope from the raw
dataset inventory used by installation verification.

## Run table experiments

For commands that spell out model, checkpoint, split and training parameters
without `--config`, see [training and evaluation](training.md). The
[CLI reference](cli-reference.md) explains every parameter and its default.
The table runner below is the convenience entry point for prepared paper selections.

The runner maps table selections to shared experiment configurations:

| Table | Experiment selection |
| --- | --- |
| 1, 3 | Descriptive references; no model execution |
| 2 | Installed dataset audit, separate from the manuscript reference |
| 4, 5 | Nine main pose models; Table 5 adds per-joint values |
| 6 | PASTNet and PntPoseAT, five seeds each |
| 7 | Nine main and nine subject-split pose configurations |
| 8 | baseline, spatial-only and full PASTNet |
| 9 | XYZ and full 5D PASTNet inputs |
| 10 | PASTNet at 25, 50, 75, 100, 125, 150 and 300 points |

Preview a selection without executing it:

```bash
uv run --locked --extra cu118 python paper/run.py --table 4 --dry-run
```

After installing the required assets, run a single compatible evaluation or a
table selection:

```bash
uv run --locked --extra cu118 python scripts/evaluate.py --config paper/configs/main-pastnet.json --device cuda
uv run --locked --extra cu118 python paper/run.py --table 10 --action evaluate --device cuda
```

Replace `--extra cu118` with `--extra cpu` and `--device cuda` with `--device cpu`
for CPU execution. `--data-root PATH` overrides the dataset directory.
`--all` selects every table. Review its dry run and historical protocol limits
before launching a full sweep; the complete sweep has not been verified.
`--limit-batches N` is a limited diagnostic run and cannot reproduce a full
table result.

For training, use `--action train`. Selections containing GlobalLocalNetwork
require an explicitly chosen PCC checkpoint; subject-split training needs its
corresponding subject PCC dependency. See [training](training.md). Evaluation
uses the complete pose checkpoint, including its PCC component.

New experiment logs stay in the managed `mmwave_densepose/logs/` directories;
new weights go to `mmwave_densepose/checkpoints/<Model>/<run-id>/`.
Aggregate reports are printed under `.local/runtime/paper-reproduction/` by
default. `--output-dir PATH` changes that report directory, not individual run
locations. Reports preserve execution status and available measured CSV values.
Table 6 exports mean and sample standard deviation only when all five values
for the model are available. Full from-scratch benchmark training remains
unverified.

## Figures

Figures use frozen static data; opening a figure does not scan the dataset or
run inference. Figures 1, 2, 3, 7 and 8 use raster images, Figure 5 uses local
ECharts, and Figures 4, 6, 9, 10 and 11 use Three.js snapshots.

Hover over an expandable subplot to highlight its border, then click it to open
an interactive dialog. The tab preview itself does not rotate or zoom. Close
the dialog to return to the unchanged reference view. Figure and table captions
appear below their display canvas.
