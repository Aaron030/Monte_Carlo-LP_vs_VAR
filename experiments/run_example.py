"""Worked example of a single Monte Carlo experiment.

This is a *runnable* template that uses only the currently-implemented pieces
(the univariate ARMA DGP, the AR(p) estimator, and the IRF metrics) to show how
the three contracts fit together:

* DGP        -- a callable ``(rng, T) -> data`` (here, simulate an ARMA path).
* Estimators -- callables ``(data) -> irf_array``, built with ``partial`` so a
                whole *sweep* of lag orders presents the same interface.
* Scoring    -- done *after* ``run`` against the closed-form true IRF, so the
                DGP (and thus the truth) can change without re-running anything.

To build the actual baseline experiment, copy this file and swap the ARMA DGP
for the bivariate VAR(4) DGP and the AR estimator for ``estimate_var_irf`` /
``estimate_lp_irf`` once those are implemented -- the ``run`` call stays the same.

Run with::

    python experiments/run_example.py
"""

from __future__ import annotations

from functools import partial

import numpy as np

from mcsim.dgp import ARMASpec, arma_irf, simulate_arma
from mcsim.estimators import estimate_ar_irf
from mcsim.simulation import MCConfig, run

# True data-generating process: a stationary AR(2). Its IRF is hump-free but
# rich enough that an *under*-specified AR(1) estimator is visibly biased while
# AR(2)/AR(3) recover it -- a univariate echo of the lag-order theme in the
# baseline study.
DGP_SPEC = ARMASpec(ar=(0.5, 0.3), sigma=1.0)


def arma_dgp(rng: np.random.Generator, T: int, spec: ARMASpec) -> np.ndarray:
    """DGP adapter matching the ``(rng, T) -> data`` contract.

    Defined at module level (not a lambda) so it pickles for ``n_jobs != 1``.
    """
    return simulate_arma(spec, T, rng)


def main() -> None:
    horizon = 20

    cfg = MCConfig(
        n_reps=2000,
        T=240,
        horizon=horizon,
        seed=20260527,
        n_jobs=-1,
        progress=True,
        # A small "order sweep": the same estimator at different lag orders,
        # each bound to the (data) -> irf contract via partial.
        estimators={
            f"AR({p})": partial(estimate_ar_irf, p=p, horizon=horizon)
            for p in (1, 2, 3)
        },
    )

    dgp = partial(arma_dgp, spec=DGP_SPEC)
    results = run(dgp, cfg)

    # Closed-form true estimand for this DGP, shape (horizon + 1,).
    truth = arma_irf(DGP_SPEC, horizon)

    # Score afterwards: bias and RMSE per horizon, NaN-safe against failed reps.
    report_horizons = [0, 1, 2, 5, 10, 20]
    header = "estimator | " + " | ".join(f"h={h:<2d}" for h in report_horizons)
    print(f"\nDGP: AR{tuple(DGP_SPEC.ar)},  n_reps={cfg.n_reps},  T={cfg.T}\n")
    print("RMSE of IRF estimate by horizon")
    print(header)
    print("-" * len(header))
    for name, stack in results["irfs"].items():
        rmse = np.sqrt(np.nanmean((stack - truth) ** 2, axis=0))
        row = " | ".join(f"{rmse[h]:.3f}" for h in report_horizons)
        print(f"{name:>9s} | {row}")

    failed = {k: v for k, v in results["n_failures"].items() if v}
    if failed:
        print(f"\nfailed replications: {failed}")


if __name__ == "__main__":
    main()
