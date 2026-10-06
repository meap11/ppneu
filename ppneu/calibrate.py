"""Step 5: post-hoc calibration and operating points, all fitted on validation only.

Two things are fitted per run, both on the val fold, then frozen and applied to test
(and later to external data and to compressed models):

temperature   a single T > 0 with p = sigmoid(logit / T), fitted by minimising val
              log-loss. It changes probabilities only, never decisions: logit / T has
              the same sign as logit, so a 0.5 cut-off makes identical calls before and
              after scaling.
threshold     the operating point for a target sensitivity (default 95%, screening use).
              Chosen on the logit scale, so saturated probabilities (many 1.0s in
              float32) cannot create ties.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .metrics import nll


def fit_temperature(y, logit, bounds=(0.05, 20.0)) -> float:
    """T minimising log-loss of sigmoid(logit / T) on (y, logit). T > 1 = overconfident."""
    y, logit = np.asarray(y, float), np.asarray(logit, float)
    res = minimize_scalar(lambda t: nll(y, logit / np.exp(t)),
                          bounds=tuple(np.log(bounds)), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(logit, T: float) -> np.ndarray:
    return np.asarray(logit, float) / T


def threshold_at_sensitivity(y, score, target: float = 0.95) -> float:
    """Largest threshold t with sensitivity(score >= t) >= target on (y, score).

    With the positives' scores sorted ascending, at most floor((1 - target) * n_pos)
    of them may fall below t, so t is the score of the next one.
    """
    y, score = np.asarray(y), np.asarray(score, float)
    pos = np.sort(score[y == 1])
    if len(pos) == 0:
        raise ValueError("no positives to set a sensitivity threshold")
    k = int(np.floor((1 - target) * len(pos) + 1e-9))
    return float(pos[k])
