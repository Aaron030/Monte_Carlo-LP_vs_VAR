"""Coverage analysis functions for the Monte Carlo study.

Holds the *true* models used to simulate time series — ARMA / VAR /
extensions — and the closed-form impulse responses of those true
models, against which estimators are scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mcsim.dgp import var_ma_matrices
from mcsim.estimators import fit_var_ols


def var_theta_se(y, q, horizon, shock=0, response=0):
    """Structural IRF point estimate AND delta-method standard error for a VAR(q).

    Returns (theta_hat, se) each length horizon+1, for the response of variable
    `response` to structural shock `shock`.  The SE propagates the sampling
    uncertainty of the reduced-form OLS coefficients A through the nonlinear map
        theta_h = e_response' . Psi_h(A) . B . e_shock ,
    holding the impact matrix B = chol(Sigma_u) FIXED (Ch. 3's choice: only the
    reduced-form coefficients are propagated).
    """
    # --- 1. Reduced-form OLS fit -------------------------------------------------
    # A_hat: (q, k, k) lag matrices;  Sig: (k, k) residual covariance.
    A, Sig, _resid = fit_var_ols(y, q)
    k = A.shape[1]

    # --- 2. Structural objects (B held fixed for the delta method) ---------------
    Bm = np.linalg.cholesky(Sig)            # recursive impact matrix  B = chol(Sigma_u)
    c = Bm[:, shock]                        # column = impact vector of the chosen shock, B e_shock
    er = np.zeros(k); er[response] = 1.0    # selector e_response (picks the response variable)

    # Reduced-form MA matrices Psi_0..Psi_H and the point IRF theta_h = er' Psi_h c.
    Psi = var_ma_matrices(A, horizon)       # (H+1, k, k)
    theta = np.array([er @ Psi[h] @ c for h in range(horizon + 1)])

    # --- 3. OLS coefficient covariance  Cov(vec C) = Sigma_u (x) (X'X)^{-1} -------
    # Rebuild the regressor matrix X_t = [1, y_{t-1}, ..., y_{t-q}] (same as fit_var_ols).
    T = y.shape[0]; rows = T - q
    X = np.ones((rows, 1 + q * k))
    for i in range(1, q + 1):
        X[:, 1 + (i - 1) * k: 1 + i * k] = y[q - i: T - i]
    XtXi = np.linalg.inv(X.T @ X)           # (1+qk, 1+qk)

    # Parameters we propagate: every entry A_i[a, b].  Index convention matches
    # fit_var_ols, where A_i[a, b] = coef[row, a] with row = 1 + i*k + b
    # (i is 0-based lag, a = equation/response row, b = which variable's lag).
    params = [(i, a, b) for i in range(q) for a in range(k) for b in range(k)]

    # Sigma_alpha[p, p'] = Cov(A_i[a,b], A_{i'}[a',b']) = Sigma_u[a,a'] * (X'X)^{-1}[row, row'].
    # (From Cov(vec C) = Sigma_u (x) (X'X)^{-1}: equations couple via Sigma_u,
    #  regressors via (X'X)^{-1}.)
    Sig_a = np.array([[Sig[a, a2] * XtXi[1 + i * k + b, 1 + i2 * k + b2]
                       for (i2, a2, b2) in params] for (i, a, b) in params])

    # --- 4. Gradient d_h = d theta_h / d alpha via the differentiated MA recursion
    # Psi_h = sum_{j=1}^{min(h,q)} A_j Psi_{h-j}.  Differentiating w.r.t. the single
    # entry A_{i+1}[a,b] gives, with E_{ab} the unit matrix (1 at (a,b)):
    #   dPsi_h = sum_j A_j dPsi_{h-j}  +  E_{ab} Psi_{h-i-1}   (the +E term only when j=i+1)
    Em = {}
    for a in range(k):
        for b in range(k):
            mat = np.zeros((k, k)); mat[a, b] = 1.0; Em[(a, b)] = mat

    dPsi = [{p: np.zeros((k, k)) for p in params}]     # dPsi at h=0 is all zeros
    for h in range(1, horizon + 1):
        cur = {}
        for p in params:
            i, a, b = p
            acc = np.zeros((k, k))
            for j in range(1, min(h, q) + 1):
                acc = acc + A[j - 1] @ dPsi[h - j][p]       # chain-rule term
                if j - 1 == i:                              # direct term: this lag IS the parameter
                    acc = acc + Em[(a, b)] @ Psi[h - j]
            cur[p] = acc
        dPsi.append(cur)

    # --- 5. Var(theta_h) = d_h' Sigma_alpha d_h, with d_h[p] = er' dPsi_h[p] c ----
    se = np.zeros(horizon + 1)
    for h in range(horizon + 1):
        d = np.array([er @ dPsi[h][p] @ c for p in params])
        se[h] = np.sqrt(max(d @ Sig_a @ d, 0.0))           # clip tiny negatives from roundoff
    return theta, se


def lp_theta_se(y, p, horizon, shock=0, response=0):
    """LP(p) point IRF AND heteroskedasticity-robust (HC1) standard error per horizon.

    The shock is recovered from a first-stage VAR(p) (same as estimate_lp_irf);
    at each horizon h the IRF is the OLS slope on the shock x_t in
        y_{response, t+h} = mu + beta_h x_t + sum_l delta_l' y_{t-l} + xi,
    and its SE is the White/HC1 robust SE of that slope.  HC (not HAC) is used
    because the structural shock is a martingale difference (Plagborg-Moller & Wolf);
    note LP residuals are serially correlated for h>0, so this can mildly under-state
    uncertainty at long horizons.
    """
    y = np.asarray(y, dtype=float)
    T, k = y.shape

    # First-stage VAR(p): residuals -> structural shock  x_t = e_shock' B^{-1} u_t.
    A, Sig, resid = fit_var_ols(y, p)
    Bm = np.linalg.cholesky(Sig)
    eps = np.linalg.solve(Bm, resid.T).T          # structural innovations (rows aligned t=p..T-1)
    x = eps[:, shock]

    rows = T - p
    lags = np.empty((rows, p * k))                # controls [y_{t-1}, ..., y_{t-p}]
    for i in range(1, p + 1):
        lags[:, (i - 1) * k: i * k] = y[p - i: T - i]

    theta = np.full(horizon + 1, np.nan)
    se = np.full(horizon + 1, np.nan)
    for h in range(horizon + 1):
        m = rows - h                              # usable base times at this horizon
        if m <= 0:
            break
        dep = y[p + h: p + h + m, response]        # y_{response, t+h}
        Xr = np.column_stack([np.ones(m), x[:m], lags[:m]])   # [const, shock, lags]
        beta, *_ = np.linalg.lstsq(Xr, dep, rcond=None)
        e = dep - Xr @ beta                       # residuals
        XtXi = np.linalg.inv(Xr.T @ Xr)
        # HC1 robust covariance:  (X'X)^{-1} (X' diag(e^2) X) (X'X)^{-1} * m/(m-kreg)
        meat = Xr.T @ (Xr * (e ** 2)[:, None])
        V = XtXi @ meat @ XtXi * (m / (m - Xr.shape[1]))
        theta[h] = beta[1]                        # slope on the shock (col index 1)
        se[h] = np.sqrt(V[1, 1])
    return theta, se


