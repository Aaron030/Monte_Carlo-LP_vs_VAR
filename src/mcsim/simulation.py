"""Monte Carlo driver.

Runs many replications of (simulate from DGP) -> (fit each estimator)
-> (record IRFs and diagnostics), with optional parallelization via joblib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


@dataclass
class MCConfig:
    n_reps: int = 1000
    T: int = 240
    horizon: int = 20
    seed: int = 0
    n_jobs: int = 1
    estimators: dict[str, Callable] = field(default_factory=dict)


def run(dgp_spec, cfg: MCConfig) -> dict[str, np.ndarray]:
    """Run Monte Carlo. Returns dict of stacked IRFs per estimator. To be implemented."""
    raise NotImplementedError


def _single_rep(dgp_spec, cfg: MCConfig, seed: int):
    raise NotImplementedError
