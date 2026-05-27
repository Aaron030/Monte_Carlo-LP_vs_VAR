"""Plotting helpers — IRF fan charts, bias/RMSE comparisons across estimators."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(context="paper", style="whitegrid")


def plot_irf_band(irfs: np.ndarray, ax=None, label: str | None = None):
    """Plot median IRF with 16/84 bands across replications. To be implemented."""
    raise NotImplementedError
