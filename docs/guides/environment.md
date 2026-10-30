# Environment setup

Run commands from the repository root. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
and use uv **0.12.12 or newer**. The repository pins Python **3.10.19** and
resolves dependencies in `uv.lock`; no Conda activation or separate pip setup is
needed.

## Choose one compute profile

| Use | Install |
| --- | --- |
| Web workbench, dataset download and existing-log browsing | `uv sync --locked` |
| CPU training, evaluation and first-run cache generation | `uv sync --locked --extra cpu` |
| NVIDIA CUDA 11.8 training and evaluation | `uv sync --locked --extra cu118` |
| Optional Qt desktop viewer | `uv sync --locked --extra gui` |

Choose **either `cpu` or `cu118`**, never both. Do not use `--all-extras`.
The web interface needs neither Qt nor PyTorch, but launching a model task from
that interface needs a compute profile. TensorBoard is included in the base
profile. The browser needs WebGL2; no Node.js, npm build or CDN is required.

Use the same compute extra when running model commands or starting a web
service that will launch them:

```bash
uv sync --locked --extra cu118
uv run --locked --extra cu118 python scripts/train.py --list-models
uv run --locked --extra cu118 python scripts/train.py --help
```

For CPU, replace `cu118` with `cpu` and set `--device cpu` on model commands.
For a combined compute and Qt environment, include both extras when syncing,
for example `uv sync --locked --extra cu118 --extra gui`.

Keep the full selected profile on `uv sync` and `uv run` commands. A base-only
invocation can remove the training backend from an environment previously
synchronized with compute extras. Keep `pyproject.toml` and `uv.lock`
together instead of installing unrelated package versions over `.venv`.

## Supported platforms

The locked native-wheel profiles target **Windows AMD64** and **glibc Linux
x86_64**, with CPython 3.10. macOS, ARM, Alpine/musl and other Python versions are
outside these profiles. Native wheels being listed for an operating system does
not establish that every workflow has been tested there.

Both compute profiles use PyTorch 2.6.0 and PyTorch Geometric 2.7.0, with matching
`torch-cluster` and `torch-scatter` wheels. The CUDA profile also requires a
compatible NVIDIA GPU and driver. Selecting that profile does not by itself
make CUDA available.

## Start the web workbench

```bash
# Dataset and existing-log browsing only:
uv run --locked python app.py --data-root data/mmwave_densepose --trust-pickle --open-browser

# Include the compute profile to launch model tasks from the GUI:
uv run --locked --extra cu118 python app.py --data-root data/mmwave_densepose --trust-pickle --open-browser
```

Use `--trust-pickle` only for trusted dataset and prediction files. Open the
complete local URL printed by the service, including its token fragment.
After restarting the service, use its newly printed URL. For headless or remote
access, see the [web workbench guide](web-gui.md).

## Troubleshooting

- **CUDA unavailable:** check the GPU/driver, or use the CPU profile with
  `--device cpu`.
- **PyG operator import failure:** sync one complete supported profile; do not
  mix CPU and CUDA extension wheels.
- **Lock mismatch:** use repository-matched `pyproject.toml` and `uv.lock`, then
  rerun `uv sync --locked`. Do not bypass a mismatch with arbitrary dependencies.
- **Offline installation:** first synchronize the chosen profile online. Add
  `--offline` only after Python and its packages are cached locally.
- **Wrong editor interpreter:** select `.venv/Scripts/python.exe` on Windows or
  `.venv/bin/python` on Linux.

Continue with [dataset installation](dataset.md), [training and evaluation](training.md),
or the [repository overview](../../README.md).
