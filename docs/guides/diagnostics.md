# Environment and model diagnostics

These checks do not require the released dataset or pretrained checkpoints.
They check local setup and model interfaces, not benchmark accuracy. Use the
[environment guide](environment.md) to install one complete compute profile.

## Check the selected environment

```bash
uv run --locked --extra cu118 python -m example.environment.verify --backend cu118 --require-cuda
```

For CPU:

```bash
uv run --locked --extra cpu python -m example.environment.verify --backend cpu
```

The check verifies the project interpreter, pinned package versions and native
operators. An environment failure identifies the missing or mismatched component.

## Check a model interface

```bash
uv run --locked --extra cu118 python -m example.models.forward --model PASTNet --backward
```

The default example runs on CPU even inside the CUDA profile. Add
`--device cuda` to exercise the GPU. Use `--all` to check all registered models;
`--points`, `--frames` and `--input-dim` control synthetic input dimensions.

For a CPU-only environment, replace `--extra cu118` with `--extra cpu`.
Successful output means the model produces finite predictions of the expected
shape; it does not establish a paper result or verify a trained checkpoint.
