"""Thesis ensemble (ResNet-50 + DenseNet-121 + Inception-v3), built without the test set.

The 2023 thesis tuned the ensemble weights and the model choice on the official test
set. Here nothing about the ensemble sees test labels:

1. each member's logits are divided by its own temperature, fitted on val, so a more
   overconfident member does not dominate the average;
2. members are combined by soft voting (weighted mean probability), with either
   equal  1/k each, nothing fitted (primary), or
   val    weights minimising val log-loss on a simplex grid (does weighting help at all?).

The result has the same long format as the members' predictions (image_id, fold, y, logit,
split, arch, seed), with arch = "ens_<weighting>". It goes through metrics.evaluate_all
like any single model, so its threshold and temperature are also fitted on val only.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import metrics as M
from .calibrate import fit_temperature

THESIS_ARCHS = ("resnet50", "densenet121", "inception_v3")


def member_logits(preds: pd.DataFrame, split: str, seed: int, fold: str,
                  archs=THESIS_ARCHS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(image_ids, y, Z) with Z[:, j] = logits of archs[j], rows aligned by image_id."""
    d = preds[(preds["split"] == split) & (preds["seed"] == seed) & (preds["fold"] == fold)
              & preds["arch"].isin(archs)]
    wide = d.pivot_table(index="image_id", columns="arch", values="logit", aggfunc="first")
    missing = set(archs) - set(wide.columns)
    if missing or wide.isna().any().any():
        raise ValueError(f"incomplete members for {split}/seed{seed}/{fold}: {missing or 'NaN rows'}")
    y = d.groupby("image_id")["y"].first().loc[wide.index].to_numpy()
    return wide.index.to_numpy(), y, wide[list(archs)].to_numpy(float)


def combine(Z, w) -> np.ndarray:
    """Logit of the weighted mean probability (soft voting), computed stably.

    logit(sum_j w_j sigmoid(z_j)) = logsumexp(log w + log sigmoid(z)) - logsumexp(log w + log sigmoid(-z))
    """
    Z = np.asarray(Z, float)
    with np.errstate(divide="ignore"):
        lw = np.log(np.asarray(w, float))
    a = lw - np.logaddexp(0.0, -Z)   # log w + log sigmoid(z)
    b = lw - np.logaddexp(0.0, Z)    # log w + log sigmoid(-z)
    return _lse(a) - _lse(b)


def _lse(x):
    m = np.max(x, axis=1, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=1, keepdims=True)))[:, 0]


def weight_grid(k: int, step: float = 0.05) -> np.ndarray:
    """All weight vectors on the simplex with the given step (k=3, 0.05 -> 231 vectors)."""
    n = int(round(1 / step))
    rows = [c + (n - sum(c),) for c in itertools.product(range(n + 1), repeat=k - 1) if sum(c) <= n]
    return np.array(rows, float) / n


def fit_weights(y, Z, step: float = 0.05) -> np.ndarray:
    """Weights minimising log-loss of combine(Z, w) on (y, Z); ties go to the most equal weights."""
    grid = weight_grid(Z.shape[1], step)
    losses = np.array([M.nll(y, combine(Z, w)) for w in grid])
    best = np.flatnonzero(losses <= losses.min() + 1e-12)
    return grid[best[np.argmin(np.abs(grid[best] - 1 / Z.shape[1]).sum(axis=1))]]


def build(preds: pd.DataFrame, split: str, archs=THESIS_ARCHS,
          weighting: str = "equal", step: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensemble predictions for every seed of `split`, fitted on val only.

    Returns (ensemble preds in long format, per-seed fitted temperatures and weights).
    """
    if weighting not in ("equal", "val"):
        raise ValueError("weighting must be 'equal' or 'val'")
    rows, fitted = [], []
    for seed in sorted(preds.loc[preds["split"] == split, "seed"].unique()):
        _, yv, Zv = member_logits(preds, split, seed, "val", archs)
        T = np.array([fit_temperature(yv, Zv[:, j]) for j in range(len(archs))])
        w = np.full(len(archs), 1 / len(archs)) if weighting == "equal" else fit_weights(yv, Zv / T, step)
        fitted.append({"split": split, "seed": seed, "weighting": weighting,
                       **{f"T_{a}": t for a, t in zip(archs, T)},
                       **{f"w_{a}": x for a, x in zip(archs, w)}})
        for fold in ("val", "test"):
            ids, y, Z = member_logits(preds, split, seed, fold, archs)
            rows.append(pd.DataFrame({"image_id": ids, "fold": fold, "y": y,
                                      "logit": combine(Z / T, w), "split": split,
                                      "arch": f"ens_{weighting}", "seed": seed}))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(fitted)
