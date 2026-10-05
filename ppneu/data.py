"""Image cache, dataset and augmentation.

All 5,856 images are decoded once, converted to grayscale and resized to
CACHE_SIZE x CACHE_SIZE (aspect ratio not preserved, as in the thesis), then kept
in one uint8 array indexed by image_id. Every split condition reads from the same
cache, so decoding the large JPEGs never happens inside the training loop.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CACHE_SIZE = 256
MANIFEST_URL = "https://raw.githubusercontent.com/meap11/ppneu/main/manifests/{}.csv"


def load_manifest(name: str, local_dir: str | Path | None = None) -> pd.DataFrame:
    """Read a frozen manifest from a local repo checkout, else from GitHub."""
    if local_dir is not None and (Path(local_dir) / f"{name}.csv").exists():
        return pd.read_csv(Path(local_dir) / f"{name}.csv")
    return pd.read_csv(MANIFEST_URL.format(name))


def _load_one(args) -> np.ndarray:
    path, size = args
    with Image.open(path) as im:
        im = im.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def build_cache(manifest: pd.DataFrame, root: str | Path, out: str | Path,
                size: int = CACHE_SIZE, workers: int | None = None) -> np.ndarray:
    """Decode every image in the manifest into an (N, size, size) uint8 array.

    Row i holds image_id i. Saved to `out` (.npy) and reused if it already exists.
    """
    out = Path(out)
    if out.exists():
        arr = np.load(out, mmap_mode="r")
        if arr.shape[1:] == (size, size) and len(arr) > manifest["image_id"].max():
            return np.asarray(arr)
    root = Path(root)
    ids = manifest["image_id"].to_numpy()
    arr = np.zeros((ids.max() + 1, size, size), dtype=np.uint8)
    jobs = [(str(root / p), size) for p in manifest["relpath"]]
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **_):
            return x
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, img in zip(ids, tqdm(ex.map(_load_one, jobs, chunksize=32), total=len(jobs),
                                    desc="caching")):
            arr[i] = img
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)
    return arr


# --------------------------------------------------------------------------- #
# Torch parts (imported lazily so the audit/split code works without torch)
# --------------------------------------------------------------------------- #

def make_transforms(img_size: int, train: bool):
    """Augmentation on 1-channel uint8 tensors.

    No horizontal flip: it moves the heart to the right side, which never happens
    in real AP radiographs. Mild affine and intensity jitter only.
    """
    from torchvision.transforms import v2 as T
    import torch

    ops = []
    if train:
        ops += [
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    ops += [T.Resize((img_size, img_size), antialias=True), T.ToDtype(torch.float32, scale=True)]
    return T.Compose(ops)


class CXRDataset:
    """Returns (3xHxW float tensor normalised with the model's mean/std, label, image_id)."""

    def __init__(self, cache: np.ndarray, df: pd.DataFrame, img_size: int, train: bool,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        import torch
        self.cache = cache
        self.ids = df["image_id"].to_numpy()
        self.y = df["y"].to_numpy().astype(np.float32)
        self.tf = make_transforms(img_size, train)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import torch
        x = torch.from_numpy(np.ascontiguousarray(self.cache[self.ids[i]]))[None]  # 1xHxW uint8
        x = self.tf(x).expand(3, -1, -1)
        x = (x - self.mean) / self.std
        return x, torch.tensor(self.y[i]), int(self.ids[i])
