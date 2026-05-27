"""Data-generating processes for the Monte Carlo study.

Holds the *true* models used to simulate time series — ARMA / VAR /
extensions — and the closed-form impulse responses of those true
models, against which estimators are scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class ARMASpec:
    """Univariate ARMA(p, q): y_t = sum phi_i y_{t-i} + e_t + sum theta_j e_{t-j}."""

    ar: Sequence[float] = ()        # (phi_1, ..., phi_p)
    ma: Sequence[float] = ()        # (theta_1, ..., theta_q)
    sigma: float = 1.0


def simulate_arma(spec: ARMASpec, T: int, rng: np.random.Generator, burnin: int = 200) -> np.ndarray:
    """Simulate a length-T path from an ARMA spec, discarding burn-in."""
    ar = np.asarray(spec.ar, dtype=float)
    ma = np.asarray(spec.ma, dtype=float)
    p, q = ar.size, ma.size
    n = T + burnin
    e = rng.standard_normal(n) * spec.sigma
    y = np.zeros(n)
    for t in range(max(p, q), n):
        ar_part = ar @ y[t - p:t][::-1] if p else 0.0
        ma_part = ma @ e[t - q:t][::-1] if q else 0.0
        y[t] = ar_part + e[t] + ma_part
    return y[burnin:]


def arma_irf(spec: ARMASpec, horizon: int) -> np.ndarray:
    """Closed-form ARMA impulse response: psi_0..psi_H to a unit innovation."""
    ar = np.asarray(spec.ar, dtype=float)
    ma = np.asarray(spec.ma, dtype=float)
    p, q = ar.size, ma.size
    psi = np.zeros(horizon + 1)
    psi[0] = 1.0
    for h in range(1, horizon + 1):
        ar_part = sum(ar[j] * psi[h - j - 1] for j in range(p) if h - j - 1 >= 0)
        ma_part = ma[h - 1] if h - 1 < q else 0.0
        psi[h] = ar_part + ma_part
    return psi
