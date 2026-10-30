"""Evaluate a mmWave-DensePose model from the repository root."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmwave_densepose.engine.runner import main

if __name__ == "__main__":
    raise SystemExit(main("evaluate"))
