"""Standard errors and confidence-interval inputs for the impulse-response study.

This module turns one simulated data set into a point estimate of the impulse
response together with a standard error, for two methods. For a VAR it uses the
delta method, and for a Local Projection it uses a heteroskedasticity-robust
standard error. The notebooks call these functions many thousands of times to
measure how often each method's 95 percent confidence interval actually
contains the true response.
"""

import numpy as np

from mcsim.dgp import var_ma_matrices
from mcsim.estimators import fit_var_ols


def var_theta_se(y, q, horizon, shock=0, response=0):
    """Impulse response and its delta-method standard error for a VAR of order q.

    From one simulated sample this returns two arrays of length horizon+1. The
    first is the estimated impulse response of variable `response` to structural
    shock `shock`. The second is its standard error.

    The standard error comes from the delta method. The impulse response is a
    smooth nonlinear function of the estimated VAR coefficients A,
        theta_h = e_response' Psi_h(A) B e_shock,
    so we take the known sampling variability of A and carry it through that
    function. Following Chapter 3, the impact matrix B = chol(Sigma_u) is held
    fixed and only the coefficient uncertainty is propagated.
    """
    # Step 1. Fit the reduced-form VAR by OLS.
    # A holds the q lag matrices, each k by k. Sig is the k by k residual covariance.
    A, Sig, _resid = fit_var_ols(y, q)
    k = A.shape[1]

    # Step 2. Build the structural objects, with B held fixed for the delta method.
    Bm = np.linalg.cholesky(Sig)            # recursive impact matrix B = chol(Sigma_u)
    c = Bm[:, shock]                        # impact vector of the chosen shock, B e_shock
    er = np.zeros(k); er[response] = 1.0    # selector that picks out the response variable

    # Reduced-form moving-average matrices Psi_0..Psi_H, then the point impulse
    # response theta_h = er' Psi_h c.
    Psi = var_ma_matrices(A, horizon)       # shape (H+1, k, k)
    theta = np.array([er @ Psi[h] @ c for h in range(horizon + 1)])

    # Step 3. Covariance of the OLS coefficients, Cov(vec C) = Sigma_u kron (X'X)^{-1}.
    # Rebuild the regressor matrix X_t = [1, y_{t-1}, ..., y_{t-q}] (same as fit_var_ols).
    T = y.shape[0]; rows = T - q
    X = np.ones((rows, 1 + q * k))
    for i in range(1, q + 1):
        X[:, 1 + (i - 1) * k: 1 + i * k] = y[q - i: T - i]
    XtXi = np.linalg.inv(X.T @ X)           # shape (1+qk, 1+qk)

    # The parameters we propagate are all entries A_i[a, b]. The index convention
    # matches fit_var_ols, where A_i[a, b] = coef[row, a] with row = 1 + i*k + b.
    # Here i is the 0-based lag, a is the equation row, and b is the lagged variable.
    params = [(i, a, b) for i in range(q) for a in range(k) for b in range(k)]

    # Sig_a[p, p'] = Cov(A_i[a,b], A_{i'}[a',b']) = Sigma_u[a,a'] * (X'X)^{-1}[row, row'].
    # This follows from Cov(vec C) = Sigma_u kron (X'X)^{-1}, where equations couple
    # through Sigma_u and regressors through (X'X)^{-1}.
    Sig_a = np.array([[Sig[a, a2] * XtXi[1 + i * k + b, 1 + i2 * k + b2]
                       for (i2, a2, b2) in params] for (i, a, b) in params])

    # Step 4. Gradient d_h = d theta_h / d alpha from the differentiated MA recursion.
    # Since Psi_h = sum_{j=1}^{min(h,q)} A_j Psi_{h-j}, differentiating with respect to
    # the single entry A_{i+1}[a,b], with E_{ab} the unit matrix (1 at (a,b)), gives
    #   dPsi_h = sum_j A_j dPsi_{h-j} + E_{ab} Psi_{h-i-1}, the E term only when j=i+1.
    Em = {}
    for a in range(k):
        for b in range(k):
            mat = np.zeros((k, k)); mat[a, b] = 1.0; Em[(a, b)] = mat

    dPsi = [{p: np.zeros((k, k)) for p in params}]     # the derivative at h=0 is all zeros
    for h in range(1, horizon + 1):
        cur = {}
        for p in params:
            i, a, b = p
            acc = np.zeros((k, k))
            for j in range(1, min(h, q) + 1):
                acc = acc + A[j - 1] @ dPsi[h - j][p]       # chain-rule term
                if j - 1 == i:                              # direct term, this lag is the parameter
                    acc = acc + Em[(a, b)] @ Psi[h - j]
            cur[p] = acc
        dPsi.append(cur)

    # Step 5. Variance Var(theta_h) = d_h' Sig_a d_h, with d_h[p] = er' dPsi_h[p] c.
    se = np.zeros(horizon + 1)
    for h in range(horizon + 1):
        d = np.array([er @ dPsi[h][p] @ c for p in params])
        se[h] = np.sqrt(max(d @ Sig_a @ d, 0.0))           # clip tiny negatives from rounding
    return theta, se


def lp_theta_se(y, p, horizon, shock=0, response=0):
    """Local Projection impulse response and its robust standard error per horizon.

    From one simulated sample this returns two arrays of length horizon+1, the
    estimated impulse response and its standard error. The shock is recovered
    from a first-stage VAR of order p, exactly as in estimate_lp_irf. At each
    horizon h the impulse response is the OLS slope on the shock x_t in the
    regression
        y_{response, t+h} = mu + beta_h x_t + sum_l delta_l' y_{t-l} + xi,
    and the standard error is the heteroskedasticity-robust (White, HC1) standard
    error of that slope. A robust standard error suffices here, rather than a HAC
    one, because the structural shock is a martingale difference (Plagborg-Moller
    and Wolf). Note that the projection residuals are serially correlated for
    h>0, so this can mildly understate the uncertainty at long horizons.
    """
    y = np.asarray(y, dtype=float)
    T, k = y.shape

    # First-stage VAR of order p. Turn its residuals into the structural shock
    # x_t = e_shock' B^{-1} u_t.
    _A, Sig, resid = fit_var_ols(y, p)
    Bm = np.linalg.cholesky(Sig)
    eps = np.linalg.solve(Bm, resid.T).T          # structural innovations, rows aligned t=p..T-1
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
        # HC1 robust covariance, (X'X)^{-1} (X' diag(e^2) X) (X'X)^{-1} * m/(m-kreg)
        meat = Xr.T @ (Xr * (e ** 2)[:, None])
        V = XtXi @ meat @ XtXi * (m / (m - Xr.shape[1]))
        theta[h] = beta[1]                        # slope on the shock (column index 1)
        se[h] = np.sqrt(V[1, 1])
    return theta, se
