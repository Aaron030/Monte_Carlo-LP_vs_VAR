"""Estimators under comparison: AR(p) / VAR(p) and Local Projections.

Each estimator returns IRFs in a common shape so the simulation loop can
treat them interchangeably. For univariate cases the return is a 1-D array
of length horizon+1; for multivariate (VAR) it becomes H x k x k.
"""

from __future__ import annotations

import numpy as np


def fit_ar_ols(y: np.ndarray, p: int) -> tuple[np.ndarray, float]:
    """OLS fit of AR(p) without intercept. Returns (phi, sigma2)."""
    y = np.asarray(y, dtype=float)
    T = y.size
    X = np.column_stack([y[p - 1 - k : T - 1 - k] for k in range(p)])
    y_dep = y[p:]
    # np.errstate: macOS Accelerate BLAS spuriously raises FPE flags in matmul.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        phi = np.linalg.solve(X.T @ X, X.T @ y_dep)
        resid = y_dep - X @ phi
        sigma2 = float(resid @ resid) / (T - p - p)
    return phi, sigma2


def ar_irf_from_coefs(phi: np.ndarray, horizon: int) -> np.ndarray:
    """IRF of an AR(p) with coefficients phi to a unit innovation."""
    p = phi.size
    psi = np.zeros(horizon + 1)
    psi[0] = 1.0
    for h in range(1, horizon + 1):
        psi[h] = sum(phi[j] * psi[h - j - 1] for j in range(p) if h - j - 1 >= 0)
    return psi


def estimate_ar_irf(y: np.ndarray, p: int, horizon: int) -> np.ndarray:
    """Fit AR(p) by OLS and return its implied IRF."""
    phi, _ = fit_ar_ols(y, p)
    return ar_irf_from_coefs(phi, horizon)


def estimate_var_irf(y: np.ndarray, p: int, horizon: int) -> np.ndarray:
    """Fit VAR(p) and return impulse responses up to `horizon`. To be implemented."""
    raise NotImplementedError


def estimate_lp_irf(y: np.ndarray, p: int, horizon: int) -> np.ndarray:
    """Estimate IRFs by Jordà-style local projections. To be implemented."""
    raise NotImplementedError
