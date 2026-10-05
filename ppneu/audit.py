"""Step 1: audit of the Kermany pediatric chest X-ray dataset.

Works on the Kaggle copy ("paultimothymooney/chest-xray-pneumonia"), which ships
train/val/test folders with NORMAL/PNEUMONIA subfolders (plus a nested duplicate
copy and a __MACOSX folder, both of which are ignored).

What this module produces
-------------------------
build_index()        one row per image: split, label, subtype, patient-group keys,
                     size, mode, exact hashes and a 64-bit perceptual hash.
exact_duplicates()   sets of images with identical pixels (across or within splits).
near_duplicate_pairs() image pairs whose pHash Hamming distance <= max_dist.
group_overlap()      how many filename-derived patient groups appear in >1 split.
id_consistency()     do images sharing a group ID across train/test look as similar
                     as images sharing an ID within a split? (tests whether the
                     filename IDs really mean "same patient" across folders)

Patient-group keys (filename IDs are proxies, NOT verified patient identifiers)
------------------
group_fine   keeps namespaces separate:   person12_bacteria, person12_virus,
             IM-0115, NORMAL2-IM-0115
group_coarse merges namespaces:           person12, IM-0115 (IM and NORMAL2-IM merged)
             -> over-groups on purpose; use it for the leakage-safe split.
"""
from __future__ import annotations

import hashlib
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct

SPLITS = ("train", "val", "test")
LABELS = ("NORMAL", "PNEUMONIA")
IMG_EXT = {".jpeg", ".jpg", ".png"}

# --------------------------------------------------------------------------- #
# Locating the dataset
# --------------------------------------------------------------------------- #

def find_root(search_dir: str | Path = "/kaggle/input") -> tuple[Path, list[Path]]:
    """Return (chosen_root, all_candidates).

    A candidate is a folder containing train/val/test, each with NORMAL/PNEUMONIA.
    The shallowest one is chosen; the Kaggle copy also contains a nested duplicate.
    """
    candidates = []
    for p in Path(search_dir).rglob("train"):
        if "__MACOSX" in p.parts or not p.is_dir():
            continue
        root = p.parent
        if all((root / s / lab).is_dir() for s in SPLITS for lab in LABELS):
            candidates.append(root)
    if not candidates:
        raise FileNotFoundError(f"No train/val/test NORMAL/PNEUMONIA tree under {search_dir}")
    candidates = sorted(set(candidates), key=lambda r: (len(r.parts), str(r)))
    return candidates[0], candidates


# --------------------------------------------------------------------------- #
# Filename -> patient-group parsing
# --------------------------------------------------------------------------- #

_PERSON = re.compile(r"^person(\d+)_(bacteria|virus)_(\d+)(?:_\d+)*$", re.IGNORECASE)
_IM = re.compile(r"^(NORMAL2-)?IM-(\d+)-(\d+)(?:-\d+)*$", re.IGNORECASE)


def parse_name(stem: str) -> dict:
    """Parse a Kermany filename stem into subtype and group keys."""
    m = _PERSON.match(stem)
    if m:
        pid, sub, _ = m.groups()
        sub = sub.lower()
        return {
            "subtype": sub,
            "group_fine": f"person{int(pid)}_{sub}",
            "group_coarse": f"person{int(pid)}",
        }
    m = _IM.match(stem)
    if m:
        prefix, pid, _ = m.groups()
        tag = "NORMAL2-IM" if prefix else "IM"
        return {
            "subtype": "normal",
            "group_fine": f"{tag}-{int(pid):04d}",
            "group_coarse": f"IM-{int(pid):04d}",
        }
    # Unparsed names become their own singleton group and are reported.
    return {"subtype": "unknown", "group_fine": f"unparsed:{stem}", "group_coarse": f"unparsed:{stem}"}


# --------------------------------------------------------------------------- #
# Per-image description (runs in worker processes)
# --------------------------------------------------------------------------- #

def phash64(gray: Image.Image) -> int:
    """64-bit DCT perceptual hash (same algorithm as imagehash.phash, no dependency)."""
    small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    coeffs = dct(dct(small, axis=0), axis=1)[:8, :8]
    bits = (coeffs > np.median(coeffs)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _describe(path_str: str) -> dict:
    p = Path(path_str)
    raw = p.read_bytes()
    with Image.open(p) as im:
        mode = im.mode
        width, height = im.size
        gray = im.convert("L")
        arr = np.asarray(gray)
        md5_pixels = hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()
        ph = phash64(gray)
    return {
        "md5_bytes": hashlib.md5(raw).hexdigest(),
        "md5_pixels": md5_pixels,
        "phash": f"{ph:016x}",
        "width": width,
        "height": height,
        "mode": mode,
    }


def build_index(root: str | Path, workers: int | None = None) -> pd.DataFrame:
    """One row per image under root/{split}/{label}/."""
    root = Path(root)
    rows = []
    for split in SPLITS:
        for label in LABELS:
            for f in sorted((root / split / label).iterdir()):
                if f.suffix.lower() not in IMG_EXT or f.name.startswith("."):
                    continue
                rows.append({
                    "relpath": str(f.relative_to(root)),
                    "split": split,
                    "label": label,
                    "filename": f.name,
                    **parse_name(f.stem),
                })
    df = pd.DataFrame(rows)

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **_):
            return x

    paths = [str(root / r) for r in df["relpath"]]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        desc = list(tqdm(ex.map(_describe, paths, chunksize=32), total=len(paths), desc="hashing"))
    df = pd.concat([df, pd.DataFrame(desc)], axis=1)
    df.insert(0, "image_id", np.arange(len(df)))
    return df


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #

def exact_duplicates(df: pd.DataFrame, key: str = "md5_pixels") -> pd.DataFrame:
    """Sets of images with identical pixels. One row per duplicate set."""
    dup = df[df.duplicated(key, keep=False)]
    out = (
        dup.groupby(key)
        .agg(
            n=("image_id", "size"),
            image_ids=("image_id", list),
            filenames=("filename", list),
            splits=("split", lambda s: sorted(set(s))),
            labels=("label", lambda s: sorted(set(s))),
            subtypes=("subtype", lambda s: sorted(set(s))),
        )
        .reset_index()
    )
    out["cross_split"] = out["splits"].map(len) > 1
    out["label_conflict"] = out["labels"].map(len) > 1
    out["subtype_conflict"] = out["subtypes"].map(len) > 1
    return out


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hashes(df: pd.DataFrame) -> np.ndarray:
    return np.array([int(h, 16) for h in df["phash"]], dtype=np.uint64)


def _hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise Hamming distance between two broadcastable uint64 arrays."""
    x = np.ascontiguousarray(np.bitwise_xor(a, b))
    return _POP8[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1, dtype=np.int16)


def near_duplicate_pairs(df: pd.DataFrame, max_dist: int = 4, chunk: int = 512) -> pd.DataFrame:
    """All pairs (i < j) with pHash Hamming distance <= max_dist."""
    h = _hashes(df)
    found = []
    for s in range(0, len(h), chunk):
        d = _hamming(h[s:s + chunk, None], h[None, :])
        i, j = np.nonzero(d <= max_dist)
        keep = (i + s) < j
        i, j = i[keep], j[keep]
        found.append(np.stack([i + s, j, d[i, j]], axis=1))
    pairs = np.concatenate(found) if found else np.empty((0, 3), dtype=int)
    p = pd.DataFrame(pairs, columns=["a", "b", "dist"])
    cols = ["image_id", "split", "label", "subtype", "group_fine", "filename", "md5_pixels"]
    for side in ("a", "b"):
        side_df = df[cols].iloc[p[side]].reset_index(drop=True).add_suffix(f"_{side}")
        p = pd.concat([p, side_df], axis=1)
    p["exact"] = p["md5_pixels_a"] == p["md5_pixels_b"]
    p["cross_split"] = p["split_a"] != p["split_b"]
    p["same_group"] = p["group_fine_a"] == p["group_fine_b"]
    p["label_conflict"] = p["label_a"] != p["label_b"]
    return p.drop(columns=["a", "b"])


# --------------------------------------------------------------------------- #
# Patient-group overlap between splits
# --------------------------------------------------------------------------- #

def group_overlap(df: pd.DataFrame, col: str) -> dict:
    sets = {s: set(df.loc[df["split"] == s, col]) for s in SPLITS}
    res = {f"n_groups_{s}": len(sets[s]) for s in SPLITS}
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        shared = sets[a] & sets[b]
        res[f"shared_groups_{a}_{b}"] = len(shared)
        res[f"{b}_images_in_shared_groups_with_{a}"] = int(
            ((df["split"] == b) & df[col].isin(shared)).sum()
        )
    return res


def id_consistency(df: pd.DataFrame, col: str = "group_fine", n_random: int = 20000,
                   seed: int = 0) -> pd.DataFrame:
    """pHash distances for three kinds of image pairs:

    within_split_same_id : same group ID, same split  (should be "same patient")
    cross_split_same_id  : same group ID, train vs test (same patient, or reused number?)
    random_diff_id       : different group IDs, same label (baseline)

    If cross-split same-ID pairs look like within-split pairs, the shared IDs likely
    are the same patients. If they look like random pairs, the numbering was probably
    reset per folder and the "overlap" is a naming coincidence.
    """
    df = df.reset_index(drop=True)
    h = _hashes(df)
    rng = np.random.default_rng(seed)
    records = []

    for _, g in df.groupby(col):
        idx = g.index.to_numpy()
        spl = g["split"].to_numpy()
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if spl[a] == spl[b]:
                    kind = "within_split_same_id"
                elif {spl[a], spl[b]} == {"train", "test"}:
                    kind = "cross_split_same_id"
                else:
                    continue
                records.append((kind, idx[a], idx[b]))

    for label, g in df.groupby("label"):
        idx = g.index.to_numpy()
        grp = g[col].to_numpy()
        a = rng.choice(len(idx), n_random)
        b = rng.choice(len(idx), n_random)
        ok = grp[a] != grp[b]
        records += [("random_diff_id", idx[i], idx[j]) for i, j in zip(a[ok], b[ok])]

    kinds, ia, ib = zip(*records) if records else ((), (), ())
    ia, ib = np.array(ia, dtype=int), np.array(ib, dtype=int)
    return pd.DataFrame({"kind": kinds, "dist": _hamming(h[ia], h[ib]) if len(ia) else []})
