#!/usr/bin/env python3
"""Compatibility entrypoint for scripts/data/build_augmented_datasets.py."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "data" / "build_augmented_datasets.py"
    runpy.run_path(str(target), run_name="__main__")
