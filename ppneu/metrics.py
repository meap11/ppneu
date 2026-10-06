"""Step 4: metrics computed offline from per-run preds.csv files.

Metric families:
* threshold-free   AUROC, AUPRC
* operating points sensitivity / specificity at (a) the val-fitted 95%-sensitivity
                   threshold (primary) and (b) logit 0 = probability 0.5 (secondary)
* calibration      log-loss and Brier (primary), ECE with 15 equal-width bins
                   (secondary: most probabilities sit near 0 or 1, so binned ECE
                   barely reacts to overconfidence)
* uncertainty      bootstrap 95% CIs over test images, stratified by class; paired
                   bootstrap for differences on the same images

Everything works on logits; probabilities are derived with a stable sigmoid.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


# --------------------------------------------------------------------------- #
# Point metrics
# --------------------------------------------------------------------------- #

def sigmoid(z) -> np.ndarray:
    z = np.asarray(z, float)
    return np.exp(-np.logaddexp(0.0, -z))


def auroc(y, score) -> float:
    return float(roc_auc_score(y, score))


def auprc(y, score) -> float:
    return float(average_precision_score(y, score))


def nll(y, logit) -> float:
    """Mean binary log-loss from logits (no clipping needed)."""
    y, z = np.asarray(y, float), np.asarray(logit, float)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def brier(y, logit) -> float:
    return float(np.mean((sigmoid(logit) - np.asarray(y, float)) ** 2))


def ece(y, logit, n_bins: int = 15) -> float:
    """Expected calibration error, equal-width probability bins."""
    y, p = np.asarray(y, float), sigmoid(logit)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def sens_spec(y, score, threshold: float) -> tuple[float, float]:
    """Sensitivity and specificity of the rule score >= threshold."""
    y, pred = np.asarray(y), np.asarray(score, float) >= threshold
    return float(pred[y == 1].mean()), float((~pred[y == 0]).mean())


def point_metrics(y, logit, threshold: float, T: float = 1.0) -> dict:
    """All metrics for one set of predictions. threshold is on the raw logit scale."""
    se, sp = sens_spec(y, logit, threshold)
    se5, sp5 = sens_spec(y, logit, 0.0)
    zt = np.asarray(logit, float) / T
    return {
        "auroc": auroc(y, logit), "auprc": auprc(y, logit),
        "sens": se, "spec": sp, "sens_05": se5, "spec_05": sp5,
        "nll": nll(y, logit), "brier": brier(y, logit), "ece": ece(y, logit),
        "nll_ts": nll(y, zt), "brier_ts": brier(y, zt), "ece_ts": ece(y, zt),
    }


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #

def _strat_indices(y, n_boot: int, seed: int):
    """Yield resampled index arrays, resampling positives and negatives separately."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def bootstrap_ci(fn, y, *arrays, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """(estimate, lo, hi) for fn(y, *arrays), class-stratified percentile bootstrap."""
    y = np.asarray(y)
    arrays = [np.asarray(a) for a in arrays]
    est = fn(y, *arrays)
    bs = np.array([fn(y[i], *[a[i] for a in arrays]) for i in _strat_indices(y, n_boot, seed)])
    lo, hi = np.nanpercentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(est), float(lo), float(hi)


def paired_bootstrap_diff(fn, y, a, b, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """fn(y, a) - fn(y, b) on the same images: (diff, lo, hi, two-sided p)."""
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    diff = fn(y, a) - fn(y, b)
    bs = np.array([fn(y[i], a[i]) - fn(y[i], b[i]) for i in _strat_indices(y, n_boot, seed)])
    lo, hi = np.nanpercentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return float(diff), float(lo), float(hi), float(min(p, 1.0))


def bootstrap_diff(fn, ya, a, yb, b, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """fn(ya, a) - fn(yb, b) on two independent image sets: (diff, lo, hi, two-sided p)."""
    ya, a, yb, b = map(np.asarray, (ya, a, yb, b))
    diff = fn(ya, a) - fn(yb, b)
    ia, ib = _strat_indices(ya, n_boot, seed), _strat_indices(yb, n_boot, seed + 1)
    bs = np.array([fn(ya[i], a[i]) - fn(yb[j], b[j]) for i, j in zip(ia, ib)])
    lo, hi = np.nanpercentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return float(diff), float(lo), float(hi), float(min(p, 1.0))


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #

def load_preds(roots) -> pd.DataFrame:
    """Concatenate every runs/<split>/<arch>/seed<k>/preds.csv found under roots.

    roots: one path or a list (e.g. the downloaded 02_train_<arch>_output folders).
    Adds split, arch and seed (int) columns.
    """
    roots = [roots] if isinstance(roots, (str, Path)) else roots
    frames = []
    for root in roots:
        for f in sorted(Path(root).glob("**/runs/*/*/seed*/preds.csv")):
            d = pd.read_csv(f)
            d["split"], d["arch"] = f.parts[-4], f.parts[-3]
            d["seed"] = int(f.parts[-2][4:])
            frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no preds.csv under {roots}")
    out = pd.concat(frames, ignore_index=True)
    dup = out.duplicated(["split", "arch", "seed", "fold", "image_id"])
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicate prediction rows (same run found twice?)")
    return out


def evaluate_run(run: pd.DataFrame, target_sens: float = 0.95) -> dict:
    """Fit T and the threshold on val, then score test. run = preds of one run."""
    from .calibrate import fit_temperature, threshold_at_sensitivity

    v, t = run[run["fold"] == "val"], run[run["fold"] == "test"]
    T = fit_temperature(v["y"], v["logit"])
    thr = threshold_at_sensitivity(v["y"], v["logit"], target_sens)
    out = {"T": T, "threshold_logit": thr, "threshold_prob": float(sigmoid(thr)),
           "n_test": len(t), "prev_test": float(t["y"].mean())}
    out.update(point_metrics(t["y"], t["logit"], thr, T))
    return out


def evaluate_all(preds: pd.DataFrame, target_sens: float = 0.95) -> pd.DataFrame:
    """One row per (split, arch, seed) with val-fitted T/threshold and test metrics."""
    rows = []
    for (s, a, k), run in preds.groupby(["split", "arch", "seed"], sort=True):
        rows.append({"split": s, "arch": a, "seed": k, **evaluate_run(run, target_sens)})
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame, cols=None, by=("split", "arch")) -> pd.DataFrame:
    """Mean and SD over seeds."""
    cols = cols or ["auroc", "auprc", "sens", "spec", "sens_05", "spec_05",
                    "nll", "brier", "ece", "nll_ts", "T"]
    return table.groupby(list(by))[cols].agg(["mean", "std"])


def seen_in_train(manifest: pd.DataFrame) -> pd.Series:
    """Per image_id: does its patient cluster (split_group) also appear in the train fold?"""
    train_groups = set(manifest.loc[manifest["fold"] == "train", "split_group"])
    return manifest.set_index("image_id")["split_group"].isin(train_groups).rename("seen")


def run_meta(roots) -> pd.DataFrame:
    """meta.json of every run (best epoch, timing, versions) as a table."""
    roots = [roots] if isinstance(roots, (str, Path)) else roots
    rows = []
    for root in roots:
        for m in sorted(Path(root).glob("**/runs/*/*/seed*/meta.json")):
            d = json.loads(m.read_text())
            c = d["config"]
            rows.append({"split": c["split"], "arch": c["arch"], "seed": c["seed"],
                         "best_epoch": d["best_epoch"], "epochs_run": d["epochs_run"],
                         "minutes": d["minutes"], "gpu": d["gpu"], **d["versions"]})
    return pd.DataFrame(rows)
