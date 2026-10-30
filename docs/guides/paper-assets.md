# Official weights and prediction files

Browse all released files on
[ScienceDB ↗](https://www.scidb.cn/detail?dataSetId=96bb16b514eb4f69b79293d230666c10).

Official checkpoints and prediction PKLs are distributed as independent ZIPs.
Small parameters, scalar metrics and TensorBoard records stay in the repository.
Install the [environment](environment.md) first and run these commands from the
repository root. No PyTorch/CUDA extra is required to download or verify files.

The experiment-to-weight/log/table mapping is in
[paper/manifest.json](../../paper/manifest.json). Download filenames, URLs, package
checksums, member checksums and installation paths are pinned by
[paper/download-manifest.json](../../paper/download-manifest.json).
Each weight has its own package. Each selected test with available prediction
files has its own prediction package; a package can contain more than one PKL.

## Download and install

Inspect availability without downloading or loading model/PKL content:

```bash
uv run --locked python scripts/download_paper_assets.py --list
uv run --locked python scripts/download_paper_assets.py --kind predictions --list
```

Download one weight, or all official weights:

```bash
uv run --locked python scripts/download_paper_assets.py --asset main-pastnet
uv run --locked python scripts/download_paper_assets.py --all
```

Download the selected PAST-Net test's predictions for Results playback:

```bash
uv run --locked python scripts/download_paper_assets.py --kind predictions --asset main-pastnet --test-id test_001
```

Use `--kind predictions --all` to install every packaged selected prediction
record. Tests without a prediction package remain metrics-only; another test's
predictions are never substituted. `--kind all --asset main-pastnet` selects both
that experiment's weight and its packaged predictions.

A package whose URL has not been published is shown as unavailable for online
download. The CLI reports this explicitly and the WebGUI disables its download
button. Add only actual ScienceDB file download URLs to the package manifest;
a publication landing-page URL does not return an installable ZIP.

## Local archives, verification and repair

You can download the ZIPs manually and install them without a network transfer:

```bash
uv run --locked python scripts/download_paper_assets.py --asset main-pastnet --archive-dir /path/to/downloaded-zips
uv run --locked python scripts/download_paper_assets.py --kind predictions --asset main-pastnet --test-id test_001 --archive-dir /path/to/downloaded-zips
```

Verify installed content against its SHA256, inspect a plan, or explicitly repair
a modified/corrupt official file:

```bash
uv run --locked python scripts/download_paper_assets.py --asset main-pastnet --verify-only
uv run --locked python scripts/download_paper_assets.py --all --dry-run
uv run --locked python scripts/download_paper_assets.py --asset main-pastnet --repair
```

Installation validates the ZIP size/SHA256, exact member inventory and each
member's size/SHA256 before placing files at their registered destinations.
Conflicting local files require `--repair`; the installer does not overwrite
different content by default. Checkpoints and PKLs are not deserialized during
download, extraction or verification. Verified installed files and cached ZIPs
are reused when retrying. A concurrent installer is rejected rather than writing
the same destinations twice. An interrupted network transfer is downloaded again;
completed verified packages are reused.

| Content | Destination |
| --- | --- |
| Official weight | `paper/experiments/<asset-id>/checkpoints/best.ckpt` |
| Official test predictions | The selected test directory in `paper/manifest.json`, under `mmwave_densepose/logs/test/` |
| Downloaded ZIP cache | `data/downloads/paper-assets/` |
| WebGUI transfer status | `.local/runtime/paper-installation/` |

Change the ZIP cache with `--cache-dir PATH`. Keep the exact package filenames
when using `--archive-dir`; do not unzip or move model files manually first.
Training a GlobalLocalNetwork requires an explicit PCC checkpoint: obtain
`main-pcc-dense` or `subject-pcc-dense` for the matching protocol. Prepared
GlobalLocalNetwork pose checkpoints already contain their PCC component.

## WebGUI

- **Training & Testing → Test → Weights / Training run** shows the selected
  official weight's installation state and a download/repair button when its
  package is published. Missing official entries can be selected for downloading;
  evaluation remains disabled until the weight passes the existing compatibility
  checks. Installing a file does not change its recorded protocol.
- **Results → Official evaluation → Model → Weights → Test ID** shows prediction
  package availability beside the frame visualization. Downloading predictions
  leaves the saved metrics visible. After installation, the selected record is
  refreshed so playback can discover the files.
- Both entries show background progress, failure/cancellation state and Cancel.
  Reloading or switching tabs does not start a second transfer. Closing the
  browser does not cancel the server job. The server must use `--trust-pickle`
  before Results may load trusted prediction PKLs.

Installing weights or predictions does not download the raw dataset. Radar/image
playback and a new evaluation also need the [dataset](dataset.md).
