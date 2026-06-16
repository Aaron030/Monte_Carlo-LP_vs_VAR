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


# ---------------------------------------------------------------------------
# Vector autoregression (the baseline DGP of the study)
# ---------------------------------------------------------------------------


@dataclass
class VARSpec:
    """Stable K-variate VAR(p) with recursive (Cholesky) identification.

        y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + u_t,   u_t = B eps_t,
        eps_t ~ iid N(0, I_K).

    Parameters
    ----------
    A : array (p, K, K)
        Coefficient matrices A_1, ..., A_p stacked on the first axis.
    B : array (K, K)
        Structural impact matrix, lower-triangular (Cholesky factor of the
        reduced-form covariance Sigma_u = B B'). A one-unit shock to the m-th
        structural innovation is B[:, m]; recursive identification means the
        first variable does not respond on impact to later shocks.
    """

    A: np.ndarray
    B: np.ndarray

    def __post_init__(self):
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        if self.A.ndim != 3 or self.A.shape[1] != self.A.shape[2]:
            raise ValueError("A must have shape (p, K, K).")
        if self.B.shape != (self.A.shape[1], self.A.shape[1]):
            raise ValueError("B must have shape (K, K) matching A.")

    @property
    def p(self) -> int:
        return self.A.shape[0]

    @property
    def k(self) -> int:
        return self.A.shape[1]


def companion(A: np.ndarray) -> np.ndarray:
    """Companion matrix of a VAR(p): shape (pK, pK)."""
    A = np.asarray(A, dtype=float)
    p, k, _ = A.shape
    C = np.zeros((p * k, p * k))
    C[:k] = np.concatenate([A[i] for i in range(p)], axis=1)  # [A_1 ... A_p]
    if p > 1:
        C[k:, : (p - 1) * k] = np.eye((p - 1) * k)
    return C


def spectral_radius(A: np.ndarray) -> float:
    """Largest eigenvalue modulus of the companion matrix (persistence)."""
    return float(np.max(np.abs(np.linalg.eigvals(companion(A)))))


def scale_to_persistence(A: np.ndarray, target: float) -> np.ndarray:
    """Rescale a VAR's coefficients so the companion spectral radius equals
    ``target``, calibrating persistence as in the baseline design.

    Uses the exact geometric rescaling ``A_i <- c^i A_i`` with
    ``c = target / spectral_radius(A)``. Substituting ``z -> c z`` in the
    reduced-form lag polynomial shows this scales *every* companion eigenvalue
    by exactly ``c``, so the resulting spectral radius is exactly ``target``.
    This generalises the thesis's leading-term rescaling of ``A_1`` (the i=1
    case) to hit the target exactly when higher-lag matrices are nonzero, while
    preserving the qualitative IRF shape.
    """
    A = np.asarray(A, dtype=float)
    p = A.shape[0]
    c = target / spectral_radius(A)
    powers = c ** np.arange(1, p + 1)
    return A * powers[:, None, None]


def var_ma_matrices(A: np.ndarray, horizon: int) -> np.ndarray:
    """Reduced-form moving-average matrices Psi_0..Psi_H, shape (H+1, K, K).

    Psi_0 = I; Psi_h = sum_{i=1}^{min(h,p)} A_i Psi_{h-i}.
    """
    A = np.asarray(A, dtype=float)
    p, k, _ = A.shape
    psi = [np.eye(k)]
    for h in range(1, horizon + 1):
        s = np.zeros((k, k))
        for i in range(1, min(h, p) + 1):
            s += A[i - 1] @ psi[h - i]
        psi.append(s)
    return np.array(psi)


def var_irf(spec: VARSpec, horizon: int, shock: int = 0, response: int = 0) -> np.ndarray:
    """True structural IRF theta_h = (Psi_h B)[response, shock], h = 0..H.

    Defaults to the response of variable 1 to structural shock 1 -- the scalar
    object the LP regression targets -- returned as a length-(H+1) array.
    """
    psi = var_ma_matrices(spec.A, horizon)
    # np.errstate: macOS Accelerate BLAS spuriously raises FPE flags in matmul.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        theta = psi @ spec.B  # (H+1, K, K), each Psi_h @ B
    return theta[:, response, shock]


def simulate_var(
    spec: VARSpec,
    T: int,
    rng: np.random.Generator,
    burnin: int = 200,
    return_shocks: bool = False,
):
    """Simulate a length-T path from a VAR spec, discarding burn-in.

    Returns an array of shape (T, K). If ``return_shocks`` is True, returns the
    tuple ``(y, eps)`` where ``eps`` (shape (T, K)) are the structural shocks
    that generated ``y``, aligned row-for-row.
    """
    A, B = spec.A, spec.B
    p, k = spec.p, spec.k
    n = T + burnin
    eps = rng.standard_normal((n, k))
    y = np.zeros((n, k))
    # np.errstate: macOS Accelerate BLAS spuriously raises FPE flags in matmul.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        u = eps @ B.T  # u_t' = eps_t' B'
        for t in range(p, n):
            val = u[t].copy()
            for i in range(1, p + 1):
                val += A[i - 1] @ y[t - i]
            y[t] = val
    if return_shocks:
        return y[burnin:], eps[burnin:]
    return y[burnin:]
