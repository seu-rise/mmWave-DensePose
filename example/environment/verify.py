"""Check the isolated uv environment and exercise the selected native backend.

    uv run --locked --extra cu118 --extra gui python -m example.environment.verify --backend cu118 --gui --require-cuda
    uv run --locked --extra cpu python -m example.environment.verify --backend cpu
    uv run --locked --extra gui python -m example.environment.verify --gui-only
"""

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cu118"))
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--gui-only", action="store_true", help="Also assert that Torch is not installed")
    parser.add_argument("--require-cuda", action="store_true", help="Exercise native operators on the GPU")
    args = parser.parse_args()
    if args.gui_only and args.backend:
        parser.error("--gui-only cannot be combined with --backend")
    if args.require_cuda and args.backend != "cu118":
        parser.error("--require-cuda needs --backend cu118")

    if sys.version_info[:3] != (3, 10, 19):
        raise RuntimeError(f"Expected Python 3.10.19, got {sys.version}")
    config = Path(sys.prefix) / "pyvenv.cfg"
    if sys.prefix == sys.base_prefix or not config.is_file():
        raise RuntimeError("Run this example inside the uv project environment")
    settings = dict(line.lower().split("=", 1) for line in config.read_text().splitlines() if "=" in line)
    settings = {key.strip(): value.strip() for key, value in settings.items()}
    if settings.get("include-system-site-packages") != "false":
        raise RuntimeError("The environment must not inherit Conda/system packages")

    expected = {"numpy": "2.2.6", "PyYAML": "6.0.3"}
    if args.gui or args.gui_only:
        expected.update({"PyQt5": "5.15.9", "PyQt5-Qt5": "5.15.2",
                         "PyQt5-sip": "12.17.1", "matplotlib": "3.10.7"})
    if args.backend:
        expected.update({"torch": f"2.6.0+{args.backend}", "torch-geometric": "2.7.0",
                         "torch-cluster": f"1.6.3+pt26{args.backend}",
                         "torch-scatter": f"2.1.2+pt26{args.backend}"})
    versions = {name: importlib.metadata.version(name) for name in expected}
    if versions != expected:
        raise RuntimeError(f"Pinned-version mismatch: expected {expected}, got {versions}")
    report = {"python": sys.version.split()[0], "executable": sys.executable,
              "isolated": True, "versions": versions}

    if args.gui or args.gui_only:
        from PyQt5.QtCore import QT_VERSION_STR
        from mmwave_densepose.gui.dataset_viewer import DatasetViewer

        if QT_VERSION_STR != "5.15.2":
            raise RuntimeError(f"Unexpected Qt runtime: {QT_VERSION_STR}")
        report["qt_runtime"] = QT_VERSION_STR
        report["viewer_imported"] = DatasetViewer.__name__
    if args.gui_only:
        if importlib.util.find_spec("torch") is not None:
            raise RuntimeError("Use a clean gui-only profile to prove Torch independence")
        report["torch_absent"] = True

    if args.backend:
        import torch
        from torch_cluster import fps
        from torch_scatter import scatter_add

        if args.backend == "cpu" and torch.version.cuda is not None:
            raise RuntimeError("CPU profile unexpectedly includes a CUDA build")
        if args.backend == "cu118" and torch.version.cuda != "11.8":
            raise RuntimeError("CUDA profile does not match CUDA 11.8")
        if args.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA execution requested but no compatible GPU is available")
        device = "cuda" if args.require_cuda else "cpu"
        points = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]], device=device)
        sampled = fps(points, ratio=0.5, random_start=False)
        if sampled.numel() != 2 or sampled.unique().numel() != 2:
            raise RuntimeError("torch-cluster FPS failed")
        values = torch.tensor([1., 2., 3.], device=device, requires_grad=True)
        reduced = scatter_add(values, torch.tensor([0, 1, 0], device=device))
        if not torch.equal(reduced, torch.tensor([4., 2.], device=device)):
            raise RuntimeError("torch-scatter reduction failed")
        reduced.sum().backward()
        if not torch.equal(values.grad, torch.ones_like(values)):
            raise RuntimeError("torch-scatter backward failed")
        report.update({"native_ops_device": device, "native_ops_passed": ["fps", "scatter_add", "backward"],
                       "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda})
        if args.require_cuda:
            report["gpu"] = torch.cuda.get_device_name()
    report["status"] = "passed"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
