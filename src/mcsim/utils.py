"""Shared utilities — RNG seeding, IO, config loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
