"""Data-generating processes for the Monte Carlo study.

Holds the *true* models used to simulate time series — ARMA / VAR /
extensions — and the closed-form impulse responses of those true
models, against which estimators are scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


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


# ---------------------------------------------------------------------------
# VARMA(p, q): VAR baseline augmented with a moving-average component
# ---------------------------------------------------------------------------
#
# Used by Extension 1 (dynamic misspecification): the data are generated by a
# VARMA(p, q), but the LP and VAR estimators are applied unchanged, so the MA
# component is a *misspecification* of their (V)AR-only Wold approximation.


@dataclass
class VARMASpec:
    """Stable, invertible K-variate VARMA(p, q) with recursive identification.

        y_t = A_1 y_{t-1} + ... + A_p y_{t-p}
              + u_t + Theta_1 u_{t-1} + ... + Theta_q u_{t-q},
        u_t = B eps_t,   eps_t ~ iid N(0, I_K).

    Parameters
    ----------
    A : array (p, K, K)
        Autoregressive matrices A_1, ..., A_p. Stability is governed by the same
        companion matrix as a pure VAR (the MA part does not affect stability).
    Theta : array (q, K, K)
        Moving-average matrices Theta_1, ..., Theta_q acting on past reduced-form
        innovations u_t. Theta = 0 (q = 0) nests the baseline :class:`VARSpec`.
        Invertibility (existence of an infinite-order VAR representation) requires
        the roots of det(I + Theta_1 z + ... + Theta_q z^q) to lie outside the
        unit circle.
    B : array (K, K)
        Lower-triangular structural impact matrix, Sigma_u = B B' (as in VARSpec).
    """

    A: np.ndarray
    Theta: np.ndarray
    B: np.ndarray

    def __post_init__(self):
        self.A = np.asarray(self.A, dtype=float)
        self.Theta = np.asarray(self.Theta, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        if self.A.ndim != 3 or self.A.shape[1] != self.A.shape[2]:
            raise ValueError("A must have shape (p, K, K).")
        k = self.A.shape[1]
        if self.Theta.ndim != 3 or self.Theta.shape[1:] != (k, k):
            raise ValueError("Theta must have shape (q, K, K) matching A.")
        if self.B.shape != (k, k):
            raise ValueError("B must have shape (K, K) matching A.")

    @property
    def p(self) -> int:
        return self.A.shape[0]

    @property
    def q(self) -> int:
        return self.Theta.shape[0]

    @property
    def k(self) -> int:
        return self.A.shape[1]


def varma_ma_matrices(A: np.ndarray, Theta: np.ndarray, horizon: int) -> np.ndarray:
    """Reduced-form MA matrices Phi_0..Phi_H of a VARMA(p, q), shape (H+1, K, K).

    The Wold coefficients solve A(L) Phi(L) = Theta(L), i.e. matching the
    coefficient of L^h in (I - sum_i A_i L^i)(sum_h Phi_h L^h) = (I + sum_j Theta_j L^j):

        Phi_0 = I,
        Phi_h = sum_{i=1}^{min(h,p)} A_i Phi_{h-i}  +  Theta_h     (Theta_h = 0 for h > q).

    This is the pure-VAR recursion of :func:`var_ma_matrices` with the extra
    additive MA term Theta_h at horizons 1..q.
    """
    A = np.asarray(A, dtype=float)
    Theta = np.asarray(Theta, dtype=float)
    p, k, _ = A.shape
    q = Theta.shape[0]
    psi = [np.eye(k)]
    for h in range(1, horizon + 1):
        s = np.zeros((k, k))
        for i in range(1, min(h, p) + 1):
            s += A[i - 1] @ psi[h - i]
        if h <= q:
            s = s + Theta[h - 1]  # direct MA contribution at horizon h
        psi.append(s)
    return np.array(psi)


def varma_irf(
    spec: VARMASpec, horizon: int, shock: int = 0, response: int = 0
) -> np.ndarray:
    """True structural IRF theta_h = (Phi_h B)[response, shock], h = 0..H.

    Phi_h are the VARMA Wold matrices, so this is the estimand of Extension 1 --
    the response of variable 1 to structural shock 1 by default, returned as a
    length-(H+1) array.
    """
    psi = varma_ma_matrices(spec.A, spec.Theta, horizon)
    # np.errstate: macOS Accelerate BLAS spuriously raises FPE flags in matmul.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        theta = psi @ spec.B
    return theta[:, response, shock]


def simulate_varma(
    spec: VARMASpec,
    T: int,
    rng: np.random.Generator,
    burnin: int = 200,
    return_shocks: bool = False,
):
    """Simulate a length-T path from a VARMA spec, discarding burn-in.

    Returns an array of shape (T, K). If ``return_shocks`` is True, returns the
    tuple ``(y, eps)`` where ``eps`` (shape (T, K)) are the structural shocks
    that generated ``y``, aligned row-for-row.
    """
    A, Theta, B = spec.A, spec.Theta, spec.B
    p, q, k = spec.p, spec.q, spec.k
    n = T + burnin
    eps = rng.standard_normal((n, k))
    y = np.zeros((n, k))
    start = max(p, q)
    # np.errstate: macOS Accelerate BLAS spuriously raises FPE flags in matmul.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        u = eps @ B.T  # reduced-form innovations u_t' = eps_t' B'
        for t in range(start, n):
            val = u[t].copy()
            for i in range(1, p + 1):
                val += A[i - 1] @ y[t - i]
            for j in range(1, q + 1):
                val += Theta[j - 1] @ u[t - j]  # moving-average term
            y[t] = val
    if return_shocks:
        return y[burnin:], eps[burnin:]
    return y[burnin:]
