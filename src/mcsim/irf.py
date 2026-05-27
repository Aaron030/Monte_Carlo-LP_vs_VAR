"""Impulse response utilities — identification, transforms, error metrics."""

from __future__ import annotations

import numpy as np


def cholesky_identification(sigma: np.ndarray) -> np.ndarray:
    """Lower-triangular Cholesky factor for short-run identification."""
    return np.linalg.cholesky(sigma)


def irf_bias(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return estimate - truth


def irf_rmse(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((estimate - truth) ** 2, axis=0))
