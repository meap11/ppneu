"""Thesis ensemble re-done honestly, to measure selection bias.

The 2023 thesis averaged ResNet-50, DenseNet-121 and Inception-v3 with weights (and the
model choice) tuned on the official test set. Here the same weighted soft-voting
ensemble is built per seed, with weights fitted three ways:

equal   1/3 each, nothing fitted
val     grid search on the val fold (honest)
test    grid search on the test fold itself (the thesis protocol, optimistic)

All three are scored on test. test minus val is the selection bias. The same is
done for picking the single best model.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import metrics as M

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


def accuracy(y, logit) -> float:
    return float(np.mean((np.asarray(logit) >= 0) == (np.asarray(y) == 1)))


def fit_weights(y, Z, objective=accuracy, step: float = 0.05, higher_is_better: bool = True) -> np.ndarray:
    """Grid-search weights maximising objective(y, combine(Z, w)).

    Ties go to the weights closest to equal, so a flat objective does not pick a corner.
    """
    grid = weight_grid(Z.shape[1], step)
    scores = np.array([objective(y, combine(Z, w)) for w in grid])
    if not higher_is_better:
        scores = -scores
    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))
    dist = np.abs(grid[best] - 1 / Z.shape[1]).sum(axis=1)
    return grid[best[np.argmin(dist)]]


def selection_bias(preds: pd.DataFrame, split: str = "official", archs=THESIS_ARCHS,
                   objective=accuracy, step: float = 0.05) -> pd.DataFrame:
    """One row per (seed, method): ensemble/single-model weights and their test metrics.

    method: equal | val | test (weights fitted where the name says), and
            single_val | single_test (best single member chosen on val / on test).
    """
    rows = []
    for seed in sorted(preds.loc[preds["split"] == split, "seed"].unique()):
        _, yv, Zv = member_logits(preds, split, seed, "val", archs)
        _, yt, Zt = member_logits(preds, split, seed, "test", archs)
        k = len(archs)
        choices = {
            "equal": np.full(k, 1 / k),
            "val": fit_weights(yv, Zv, objective, step),
            "test": fit_weights(yt, Zt, objective, step),
            "single_val": np.eye(k)[np.argmax([objective(yv, Zv[:, j]) for j in range(k)])],
            "single_test": np.eye(k)[np.argmax([objective(yt, Zt[:, j]) for j in range(k)])],
        }
        for method, w in choices.items():
            z = combine(Zt, w) if w.max() < 1 else Zt[:, int(np.argmax(w))]
            rows.append({"split": split, "seed": seed, "method": method,
                         **{f"w_{a}": float(x) for a, x in zip(archs, w)},
                         "test_acc": accuracy(yt, z), "test_auroc": M.auroc(yt, z),
                         "test_sens_05": M.sens_spec(yt, z, 0.0)[0],
                         "test_spec_05": M.sens_spec(yt, z, 0.0)[1]})
    return pd.DataFrame(rows)
