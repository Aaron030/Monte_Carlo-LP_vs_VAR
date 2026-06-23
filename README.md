# Complexity-Adjusted Comparisons of LPs and VARs

Monte Carlo evidence on Vector Autoregressions (VARs) versus Local Projections (LPs) for impulse
response estimation, following the complexity-adjusted benchmark of Ludwig (2026). Instead of
comparing the two methods at a single common lag order, we locate a fixed LP(4) on the full
VAR-order frontier and track how the comparison behaves under misspecification.

The repository backs two seminar papers that share a common baseline and differ only in their
extension:

- **Paper A — Dynamic misspecification.** Augments the baseline VAR with a moving-average
  component (VARMA(4,1)).
- **Paper B — Functional misspecification.** Augments the baseline VAR with a mean-zero
  lag-quadratic term (a nonlinear DGP).

Authors: Aaron Liebig and Patrick Tenner. Supervisor: Prof. Dr. Derya Uysal, LMU Munich.

## Repository layout

```
src/mcsim/        Python package: DGPs, estimators, IRF tools, coverage, MC driver
experiments/      Jupyter notebooks that run the simulations and produce the figures
figures/          Generated figures, grouped by Base_Case, Dynamic_MisSpec, NonLinear_DGP
Documents/        LaTeX sources for both papers (main_ext_1.tex, main_ext_2.tex)
requirements.txt  Python dependencies
pyproject.toml    Package metadata (installable as mcsim)
```

The `mcsim` package separates concerns across modules: `dgp.py` (true models and their closed-form
IRFs), `estimators.py` (VAR and LP), `irf.py` (identification and error metrics), `coverage.py`
(delta-method and robust standard errors), and `simulation.py` (the replication driver).

## Setup

Requires Python 3.9+.

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # makes the mcsim package importable
```

## Reproducing the results

The notebooks in `experiments/` are the entry points. Each runs the Monte Carlo study and writes
its figures into `figures/`. They are split by persistence regime (LOW, MID, HIGH).

- `experiments/Base_Case/` — baseline under correct specification
- `experiments/Extension_Dynamic_Misspec/` — Paper A (VARMA)
- `experiments/Extension_Nonlinear/` — Paper B (nonlinear lag-quadratic)

Run them with JupyterLab, or headless via `jupyter nbconvert --execute`.

## Building the papers

From `Documents/`, with a LaTeX distribution and `latexmk`:

```sh
latexmk main_ext_1    # Paper A
latexmk main_ext_2    # Paper B
```

Each build also refreshes a `<root>.counts.txt` chapter word count via `wordcount.sh` (requires
`detex`).
