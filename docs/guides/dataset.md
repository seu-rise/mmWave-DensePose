# Dataset installation and partitions

Run all commands from the repository root. Install the base environment first;
see [Environment](environment.md).

## Release availability

Browse the dataset, model weights and prediction packages on
[ScienceDB ↗](https://www.scidb.cn/detail?dataSetId=96bb16b514eb4f69b79293d230666c10).

The dataset is **released**. The installer reads the download URLs, package
sizes, checksums and file mappings from
[release-manifest.json](../../dataset/release-manifest.json).
Use the command below to download and install the release.

```bash
uv run --locked python scripts/download_dataset.py
```

The same workflow is available through **Dataset → Download dataset** in the
[web workbench](web-gui.md). Independently downloadable ZIP packages contain
metadata and groups of complete anonymized sequences. The installer verifies
archive and member checksums before accepting an installation.

The standard layout is:

```text
data/mmwave_densepose/
├── manifest.json
└── raw/
    ├── manifest.json
    └── *.pkl
```

Keep all release members in place, including additional indexes declared by
the published manifest. The original sequence files total about 161 GiB;
allow extra space for downloaded archives, installation staging and derived
caches. Dataset and generated files are excluded from Git.

## Contents and statistics

| Item | Value |
| --- | --- |
| Subjects | 101 |
| Continuous sequences | 141 |
| Motions | 19 designed motions and free motion |
| Valid synchronized frames | 570,124 |
| Radar attributes | XYZ, Doppler and SNR |
| Skeleton | 19 joints, coordinates in meters |
| Other modalities | Anonymized color, depth, infrared, body index and timestamps |

The paper-valid frame count describes its filtering protocol. It is different
from the raw inventory total and from the number of temporal samples produced
by an experiment. The overview displays frozen release statistics.

The installed manifest defines sequence identity and subject membership.
[subject-mapping.json](../../dataset/subject-mapping.json) and
[schema.json](../../dataset/schema.json) describe the reference mapping and file
format. Do not infer subject identities from sequence filenames.

## Verify and repair

An interrupted or partial installation shows **Continue download**, including
after restarting the service. Continuing first verifies completed cached ZIPs
and installs valid cached packages before downloading missing ones. Installed
sequences become available in Dataset immediately; the page shows their count
and the remaining installation status. Closing the progress dialog does not
stop the operation.

A partial release is **browsing only**. Training, validation, testing and
prediction remain disabled until the complete dataset is installed and verified.
The CLI enforces the same rule; `--allow-partial` does not bypass it for a
partially downloaded release. A cached package is not considered verified until
its checksum and members have been checked.

Use **Dataset → Verify dataset** to explicitly scan the local release against
its frozen baseline. The report lists missing, changed and unexpected files.
When a required file is corrupt or missing, **Repair packages** downloads the
affected packages rather than downloading every archive again. Verified cached
archives can be reused. Unrelated extra files are not automatically deleted.

```bash
uv run --locked python scripts/download_dataset.py --repair
```

For archives you have already downloaded, pass `--archive-dir PATH`. Use
`--dry-run` to inspect the installation plan without installing files. Closing
the GUI progress dialog leaves its background operation running; use Cancel
to request a stop.

PKL loading and cache preparation deserialize data. Use trusted releases and
start the workbench with `--trust-pickle` only for data whose source you trust.
Checksum verification alone does not require unpickling.

## Partition settings

Choose one of two strategies in [training](training.md):

- **Sequence-chunk** (`sequence_chunk`): split each sequence into nonoverlapping
  chunks, then assign chunks to train/validation/test using the seed. Windows
  stay inside their assigned chunk. The default chunk size is 200 frames.
- **Subject-independent** (`subject`): keep every sequence from a subject in
  one partition. By default, windows use whole sequences. Leave chunk size
  empty in the GUI, omit `--split-chunk-size`, or use `null` in a configuration.
  An explicit positive chunk size enables chunk-limited windows and
  candidate-window weighting during allocation.

The default proportions are 0.8/0.1/0.1. Model, seed, input dimensions, window
length and filtering settings determine the actual sample population. The
default valid frame has more than 50 finite-XYZ radar points; temporal windows
also require adjacent timestamp gaps below 300 ms. Feature normalization is
fitted from training frames only.

The first data load creates or reuses the matching derived cache automatically.
No separate cache preparation command or manual index selection is required.
Changing a relevant parameter selects a different cache. Caches are disposable;
the source release is not. **Dataset cache** in Training & Testing lets you
inspect and remove eligible caches.

Frozen [reference split counts](../../dataset/release-overview.json) belong to
their recorded experiments. Compare new measurements with
[official results](official-results.md) only when dataset identity, partition,
filtering and input settings match. Partial or batch-limited runs are not
complete benchmark evaluations.
