"""Step 2: frozen split manifests.

Three conditions, all built from the audit index (index.csv from notebook 00):

official      train/val = official train+val folders re-split by patient group
              (clean validation); test = the official 624-image test folder.
              The honest version of the protocol most Kermany papers use.
image_random  all 5,856 images pooled and split 70/15/15 at the IMAGE level,
              stratified by label. Reproduces the common "merge folders and
              re-split" practice: images of the same child, and the 32 exact
              duplicate copies, can land on both sides.
grouped       all images pooled, redundant exact-duplicate copies dropped, split
              70/15/15 at the GROUP level (no group crosses splits).

Groups are deliberately conservative (over-merged): union of
  * group_coarse (person ID merged across bacteria/virus; IM and NORMAL2-IM merged)
  * exact-duplicate pixel sets
Over-merging can only make the grouped split stricter, never leakier.

Each manifest row: image_id, relpath, label, y (1 = pneumonia), subtype,
split_group (cluster id), fold (train/val/test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .audit import parse_name

FRACTIONS = (0.70, 0.15, 0.15)
FOLDS = ("train", "val", "test")


def reparse(index: pd.DataFrame) -> pd.DataFrame:
    """Re-run the (fixed) filename parser so old index.csv files stay usable."""
    parsed = index["filename"].str.rsplit(".", n=1).str[0].map(parse_name).apply(pd.Series)
    out = index.drop(columns=[c for c in parsed.columns if c in index.columns]).join(parsed)
    out["y"] = (out["label"] == "PNEUMONIA").astype(int)
    return out


def clusters(index: pd.DataFrame) -> pd.Series:
    """Union-find over group_coarse and exact-duplicate pixel sets -> cluster id."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for g, h in zip(index["group_coarse"], index["md5_pixels"]):
        union("g:" + g, "h:" + h)
    roots = index["group_coarse"].map(lambda g: find("g:" + g))
    codes, _ = pd.factorize(roots, sort=True)
    return pd.Series(codes, index=index.index, name="split_group")


def _assign_groups(groups: pd.DataFrame, fractions, rng) -> pd.Series:
    """Greedy stratified assignment of whole groups to folds.

    groups: one row per cluster with columns n (images) and n_pos (pneumonia images).
    Groups are shuffled, then each goes to the fold furthest below its target
    in its stratum (majority label).
    """
    groups = groups.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    groups = groups.sort_values("n", ascending=False, kind="stable")  # big groups first
    fold_of = {}
    for stratum, g in groups.groupby(groups["n_pos"] * 2 >= groups["n"], sort=False):
        target = np.array(fractions) * g["n"].sum()
        filled = np.zeros(len(fractions))
        for cid, n in zip(g.index, g["n"]):
            k = int(np.argmax(target - filled))
            fold_of[cid] = FOLDS[k]
            filled[k] += n
    return pd.Series(fold_of)


def _base(index: pd.DataFrame) -> pd.DataFrame:
    cols = ["image_id", "relpath", "label", "y", "subtype", "split_group", "orig_split"]
    return index.rename(columns={"split": "orig_split"})[cols].copy()


def make_official(index: pd.DataFrame, seed: int = 0, val_frac: float = 0.15) -> pd.DataFrame:
    df = _base(index)
    is_test = df["orig_split"] == "test"
    trval = df[~is_test]
    stats = trval.groupby("split_group").agg(n=("y", "size"), n_pos=("y", "sum"))
    folds = _assign_groups(stats, (1 - val_frac, val_frac, 0.0), np.random.default_rng(seed))
    df.loc[~is_test, "fold"] = trval["split_group"].map(folds).values
    df.loc[is_test, "fold"] = "test"
    return df


def make_image_random(index: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    df = _base(index)
    rng = np.random.default_rng(seed)
    df["fold"] = ""
    for _, g in df.groupby("y"):
        idx = rng.permutation(g.index.to_numpy())
        n_tr = int(round(FRACTIONS[0] * len(idx)))
        n_va = int(round(FRACTIONS[1] * len(idx)))
        df.loc[idx[:n_tr], "fold"] = "train"
        df.loc[idx[n_tr:n_tr + n_va], "fold"] = "val"
        df.loc[idx[n_tr + n_va:], "fold"] = "test"
    return df


def make_grouped(index: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    keep = ~index.duplicated("md5_pixels", keep="first")  # drop redundant exact copies
    df = _base(index[keep])
    stats = df.groupby("split_group").agg(n=("y", "size"), n_pos=("y", "sum"))
    folds = _assign_groups(stats, FRACTIONS, np.random.default_rng(seed))
    df["fold"] = df["split_group"].map(folds)
    return df


def build_all(index: pd.DataFrame, seed: int = 0) -> dict[str, pd.DataFrame]:
    index = reparse(index)
    index["split_group"] = clusters(index)
    return {
        "official": make_official(index, seed),
        "image_random": make_image_random(index, seed),
        "grouped": make_grouped(index, seed),
    }


EXTRA_SEEDS = (1, 2)


def build_extra(index: pd.DataFrame, seeds=EXTRA_SEEDS) -> dict[str, pd.DataFrame]:
    """Extra realisations of the two pooled conditions, named <condition>_s<seed>.

    The seed-0 manifests stay frozen; these only show that the image_random vs grouped
    gap does not depend on one particular split. official has a fixed test set, so it
    gets no extra realisations.
    """
    out = {}
    for seed in seeds:
        ms = build_all(index, seed)
        out[f"image_random_s{seed}"] = ms["image_random"]
        out[f"grouped_s{seed}"] = ms["grouped"]
    return out


def leakage_report(m: pd.DataFrame) -> dict:
    """Cross-fold sharing of clusters and of exact pixel duplicates (via image_id sets)."""
    g = {f: set(m.loc[m["fold"] == f, "split_group"]) for f in FOLDS}
    test_mask = m["fold"] == "test"
    shared_tr_te = g["train"] & g["test"]
    return {
        "n": {f: int((m["fold"] == f).sum()) for f in FOLDS},
        "pos_rate": {f: round(float(m.loc[m["fold"] == f, "y"].mean()), 3) for f in FOLDS},
        "clusters_shared_train_test": len(shared_tr_te),
        "clusters_shared_train_val": len(g["train"] & g["val"]),
        "clusters_shared_val_test": len(g["val"] & g["test"]),
        "test_images_in_clusters_seen_in_train": int(
            (test_mask & m["split_group"].isin(shared_tr_te)).sum()),
    }
