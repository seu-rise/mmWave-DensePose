# Runnable examples

Run these diagnostics from the repository root with the
[uv environment](../docs/guides/environment.md). They use synthetic inputs and
do not require the dataset or pretrained weights.

## Check the environment

```bash
uv run --locked --extra cu118 python -m example.environment.verify --backend cu118 --require-cuda
```

For a CPU installation, use `--extra cpu` and `--backend cpu` without
`--require-cuda`.

## Check a model

```bash
uv run --locked --extra cu118 python -m example.models.forward --model PASTNet --backward
uv run --locked --extra cpu python -m example.models.forward --all --device cpu --backward
```

The model example checks output shapes, finite predictions and, with `--backward`,
finite gradients. It does not train on the benchmark or measure paper accuracy.

Use the actual workflow entrypoints for
[dataset installation](../docs/guides/dataset.md),
[weight and prediction downloads](../docs/guides/paper-assets.md),
[training and evaluation](../docs/guides/training.md),
[paper reproduction](../docs/guides/official-results.md) and the
[WebGUI](../docs/guides/web-gui.md).
