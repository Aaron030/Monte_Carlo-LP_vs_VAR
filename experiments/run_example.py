"""Entry point for a single Monte Carlo experiment.

Copy this file per experiment (e.g. baseline, lag-misspec, omitted-var, ...)
and fill in the DGP spec and estimator wiring.
"""

from __future__ import annotations

from mcsim.estimators import estimate_lp_irf, estimate_var_irf
from mcsim.simulation import MCConfig, run


def main() -> None:
    cfg = MCConfig(
        n_reps=1000,
        T=240,
        horizon=20,
        seed=20260527,
        n_jobs=-1,
        estimators={
            "VAR": estimate_var_irf,
            "LP": estimate_lp_irf,
        },
    )
    dgp_spec = None  # TODO: define
    results = run(dgp_spec, cfg)
    print(results)


if __name__ == "__main__":
    main()
