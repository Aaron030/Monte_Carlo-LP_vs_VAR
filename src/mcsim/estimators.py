"""Estimators under comparison, the VAR(p) and the Local Projection.

Each estimator returns the impulse response as a one-dimensional array of length
horizon+1, the response of one chosen variable to one chosen structural shock,
so the simulation loop can treat the two estimators interchangeably.
"""

from __future__ import annotations

import numpy as np

from .dgp import var_ma_matrices


# ---------------------------------------------------------------------------
# VAR(p): reduced-form OLS, recursive identification, structural IRF
# ---------------------------------------------------------------------------


def fit_var_ols(y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS fit of a VAR(p) with intercept, equation by equation.

    Parameters
    ----------
    y : array (T, K)
    p : int

    Returns
    -------
    (A_hat, sigma_u, resid)
        ``A_hat`` has shape (p, K, K) with A_hat[i] = \\hat A_{i+1}.
        ``sigma_u`` is the (K, K) residual covariance, and ``resid`` is the
        (T - p, K) residual matrix aligned with t = p, ..., T-1.
    """
    y = np.asarray(y, dtype=float)
    T, k = y.shape
    rows = T - p
    # Regressors [1, y_{t-1}, ..., y_{t-p}] for each t = p..T-1.
    X = np.ones((rows, 1 + p * k))
    for i in range(1, p + 1):
        X[:, 1 + (i - 1) * k : 1 + i * k] = y[p - i : T - i]
    Y = y[p:]
    # np.errstate guards against macOS Accelerate BLAS spuriously raising FPE flags in matmul.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        coef = np.linalg.solve(X.T @ X, X.T @ Y)  # (1 + pK, K)
        resid = Y - X @ coef
        dof = max(rows - (1 + p * k), 1)
        sigma_u = (resid.T @ resid) / dof
    # coef rows 1.. hold the lag blocks. A_i maps y_{t-i} -> y_t, so A_i = block.T.
    A_hat = np.empty((p, k, k))
    for i in range(p):
        A_hat[i] = coef[1 + i * k : 1 + (i + 1) * k, :].T
    return A_hat, sigma_u, resid


def estimate_var_irf(
    y: np.ndarray, p: int, horizon: int, shock: int = 0, response: int = 0
) -> np.ndarray:
    """Fit VAR(p), identify recursively, and return the structural IRF.

    The reduced-form coefficients are estimated by OLS. The impact matrix is the
    Cholesky factor \\hat B = chol(\\hat Sigma_u), and the structural IRF is
    \\hat theta_h = (\\hat Psi_h \\hat B)[response, shock]. Defaults to the
    response of variable 1 to structural shock 1, as a length-(H+1) array.
    """
    A_hat, sigma_u, _ = fit_var_ols(y, p)
    B_hat = np.linalg.cholesky(sigma_u)
    psi = var_ma_matrices(A_hat, horizon)  # (H+1, K, K)
    theta = psi @ B_hat
    return theta[:, response, shock]


# ---------------------------------------------------------------------------
# Local projections with a first-stage VAR(p) structural shock
# ---------------------------------------------------------------------------


def estimate_lp_irf(
    y: np.ndarray, p: int, horizon: int, shock: int = 0, response: int = 0
) -> np.ndarray:
    """Jorda-style local projections, identified via a first-stage VAR(p).

    The structural shock is recovered from the VAR(p) residuals,
    ``x_t = e_shock' \\hat B^{-1} \\hat u_t`` (a unit-variance innovation), and
    at each horizon h the response is the slope on x_t in

        y_{response, t+h} = mu_h + beta_h x_t + sum_{l=1}^p delta_{h,l}' y_{t-l} + xi.

    Returns \\hat theta_h^{LP} = beta_h for h = 0..H as a length-(H+1) array.
    """
    y = np.asarray(y, dtype=float)
    T, k = y.shape
    A_hat, sigma_u, resid = fit_var_ols(y, p)
    B_hat = np.linalg.cholesky(sigma_u)
    # Structural innovations eps_t = B^{-1} u_t. Take the shock-th component.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        eps = np.linalg.solve(B_hat, resid.T).T  # (rows, K), rows = T - p
    x = eps[:, shock]  # shock series, index r <-> t = p + r

    rows = T - p
    # Lag controls [y_{t-1}, ..., y_{t-p}] aligned with base times t = p..T-1.
    lags = np.empty((rows, p * k))
    for i in range(1, p + 1):
        lags[:, (i - 1) * k : i * k] = y[p - i : T - i]

    irf = np.full(horizon + 1, np.nan)
    for h in range(horizon + 1):
        m = rows - h  # usable base times r = 0..rows-1-h
        if m <= 0:
            break
        dep = y[p + h : p + h + m, response]
        design = np.column_stack([np.ones(m), x[:m], lags[:m]])
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            beta = np.linalg.lstsq(design, dep, rcond=None)[0]
        irf[h] = beta[1]  # slope on x_t
    return irf
