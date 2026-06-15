"""Monte Carlo driver.

Runs many replications of (simulate from DGP) -> (fit each estimator)
-> (record IRFs and diagnostics), with optional parallelization via joblib.

The driver is deliberately agnostic about the model under study. It commits
to only two minimal contracts, so the DGP, the data dimensionality, and the
set of estimators can all change without editing this file:

* DGP contract       -- ``dgp_spec`` is a callable ``(rng, T) -> data`` (a DGP
                        that ignores the sample size may also be a plain
                        ``(rng) -> data``), or any object exposing a
                        ``.simulate(rng, T)`` method. ``data`` is whatever the
                        estimators understand (1-D series, T x k panel, ...).
* Estimator contract -- each value in ``cfg.estimators`` is a callable
                        ``(data) -> np.ndarray`` returning that estimator's IRF
                        for one sample. Bind extra arguments (lag order,
                        horizon, ...) with ``functools.partial`` so every
                        estimator presents the same one-argument interface,
                        e.g. ``partial(estimate_var_irf, p=4, horizon=20)``.

``run`` returns the *raw* stacked IRF estimates rather than summary statistics,
so bias / variance / MSE / RMSE / coverage can be computed afterwards against
whatever true estimand the chosen DGP implies (see :mod:`mcsim.irf`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


@dataclass
class MCConfig:
    n_reps: int = 1000
    T: int = 240
    horizon: int = 20
    seed: int = 0
    n_jobs: int = 1
    estimators: dict[str, Callable] = field(default_factory=dict)
    progress: bool = True  # show a tqdm progress bar over replications


def _simulate(dgp_spec, cfg: MCConfig, rng: np.random.Generator):
    """Draw one dataset from the DGP.

    Accepts either a callable or an object with a ``.simulate`` method, and
    tolerates DGPs that take ``(rng, T)`` or only ``(rng)``.
    """
    if dgp_spec is None:
        raise ValueError(
            "dgp_spec is None: pass a callable (rng, T) -> data, a callable "
            "(rng) -> data, or an object with a .simulate(rng, T) method."
        )
    sim = getattr(dgp_spec, "simulate", dgp_spec)
    try:
        return sim(rng, cfg.T)
    except TypeError:
        # DGP that ignores the sample size and takes only the generator.
        return sim(rng)


def _single_rep(dgp_spec, cfg: MCConfig, seed) -> dict:
    """One replication: simulate -> fit each estimator -> collect IRFs.

    Returns ``{name: irf_array or None}``. An estimator that raises on a given
    sample (e.g. a singular moment matrix) yields ``None`` for that replication
    instead of aborting the whole experiment.
    """
    rng = np.random.default_rng(seed)
    data = _simulate(dgp_spec, cfg, rng)
    out: dict = {}
    for name, est in cfg.estimators.items():
        try:
            out[name] = np.asarray(est(data), dtype=float)
        except Exception:
            out[name] = None
    return out


def run(dgp_spec, cfg: MCConfig) -> dict:
    """Run the Monte Carlo experiment.

    Parameters
    ----------
    dgp_spec
        The data-generating process (see the DGP contract in the module
        docstring). It is the only object that knows the true model, so
        swapping it swaps the experiment's DGP without touching this driver.
    cfg
        Experiment configuration. ``cfg.estimators`` maps a display label to a
        one-argument callable ``(data) -> irf_array`` (see the estimator
        contract in the module docstring).

    Returns
    -------
    dict
        ``{"irfs": {name: ndarray of shape (n_reps, *irf_shape)},
           "n_failures": {name: int}, "n_reps": int, "config": cfg}``.
        Replications in which an estimator failed (raised, or returned a
        differently shaped array) are filled with ``np.nan`` so every
        estimator's stack is rectangular; the count is reported in
        ``n_failures``.

    Notes
    -----
    Each replication gets an independent, reproducible RNG stream derived from
    ``cfg.seed`` via :class:`numpy.random.SeedSequence`, so results do not
    depend on ``n_jobs`` and are bit-for-bit reproducible.

    For ``n_jobs != 1`` joblib uses process-based parallelism, so ``dgp_spec``
    and every estimator must be picklable -- use module-level functions and
    ``functools.partial`` rather than lambdas or locally defined closures.
    """
    if not cfg.estimators:
        raise ValueError(
            "cfg.estimators is empty: register at least one "
            "name -> callable(data) -> irf estimator."
        )

    # Independent, reproducible substream per replication (parallel-safe).
    child_seeds = np.random.SeedSequence(cfg.seed).spawn(cfg.n_reps)

    jobs = (delayed(_single_rep)(dgp_spec, cfg, s) for s in child_seeds)
    bar = tqdm(
        jobs,
        total=cfg.n_reps,
        desc="MC reps",
        disable=not cfg.progress,
    )
    rep_results = Parallel(n_jobs=cfg.n_jobs)(bar)

    names = list(cfg.estimators.keys())

    # Reference IRF shape per estimator, taken from its first successful rep.
    ref_shape: dict = {}
    for name in names:
        for rep in rep_results:
            if rep[name] is not None:
                ref_shape[name] = rep[name].shape
                break

    irfs: dict = {}
    n_failures: dict = {}
    for name in names:
        shape = ref_shape.get(name)
        if shape is None:
            # Every replication failed for this estimator.
            irfs[name] = np.full((cfg.n_reps,), np.nan)
            n_failures[name] = cfg.n_reps
            continue
        stack = np.full((cfg.n_reps, *shape), np.nan)
        fails = 0
        for i, rep in enumerate(rep_results):
            val = rep[name]
            if val is None or val.shape != shape:
                fails += 1
                continue
            stack[i] = val
        irfs[name] = stack
        n_failures[name] = fails

    return {
        "irfs": irfs,
        "n_failures": n_failures,
        "n_reps": cfg.n_reps,
        "config": cfg,
    }
