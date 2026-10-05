"""One training run = one (split condition, architecture, seed).

Rules that protect the paper:
* Model selection (best epoch) uses the validation fold only.
* The test fold is predicted once, after training, and never used for any decision.
* Loss is plain (unweighted) BCE so predicted probabilities stay interpretable;
  calibration is one of the paper's main metrics, and class weighting distorts it.

Outputs, in runs/<split>/<arch>/seed<k>/:
  preds.csv  image_id, fold, y, logit, prob   (val and test)
  log.csv    per-epoch train loss, val loss, val AUROC, lr, seconds
  meta.json  config, best epoch, val/test AUROC, versions, GPU, timing
  best.pt    best-epoch weights (state_dict)
"""
from __future__ import annotations

import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass
class RunConfig:
    split: str                    # official | image_random | grouped
    arch: str                     # key of models.ARCHS
    seed: int = 0
    img_size: int = 224
    epochs: int = 20
    patience: int = 5             # early stopping on val AUROC
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    workers: int = 4
    amp: bool = True
    limit: int | None = None      # smoke test: cap images per fold
    out_root: str = "/kaggle/working/runs"
    extra: dict = field(default_factory=dict)

    @property
    def out_dir(self) -> Path:
        return Path(self.out_root) / self.split / self.arch / f"seed{self.seed}"


def seed_everything(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def _predict(model, loader, device, amp):
    import torch
    model.eval()
    logits, ys, ids = [], [], []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        for x, y, i in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            logits.append(model(x).float().squeeze(1).cpu())
            ys.append(y)
            ids.append(i)
    return (torch.cat(logits).numpy(), torch.cat(ys).numpy(), torch.cat(ids).numpy())


def train_one(cfg: RunConfig, manifest: pd.DataFrame, cache: np.ndarray, force: bool = False) -> dict:
    """Train one run. Returns meta dict. Skips if preds.csv already exists."""
    import torch
    from torch.utils.data import DataLoader

    from . import models
    from .data import CXRDataset

    out = cfg.out_dir
    if (out / "preds.csv").exists() and not force:
        return json.loads((out / "meta.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg.amp and device.type == "cuda"

    folds = {f: manifest[manifest["fold"] == f] for f in ("train", "val", "test")}
    if cfg.limit:
        folds = {f: d.sample(frac=1.0, random_state=cfg.seed).groupby("y").head(cfg.limit // 2)
                 for f, d in folds.items()}

    model, mean, std = models.create(cfg.arch)
    model = model.to(device).to(memory_format=torch.channels_last)

    def loader(fold, train):
        ds = CXRDataset(cache, folds[fold], cfg.img_size, train, mean, std)
        g = torch.Generator().manual_seed(cfg.seed)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train, drop_last=train,
                          num_workers=cfg.workers, pin_memory=True, generator=g,
                          persistent_workers=cfg.workers > 0)

    dl_train, dl_val = loader("train", True), loader("val", False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * len(dl_train)
    warm = max(1, int(cfg.warmup_epochs * len(dl_train)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best = {"auc": -1.0, "loss": float("inf"), "epoch": -1}
    log, bad, t0 = [], 0, time.time()
    for epoch in range(cfg.epochs):
        te = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, y, _ in dl_train:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(x).squeeze(1), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)

        lv, yv, _ = _predict(model, dl_val, device, amp)
        val_loss = float(loss_fn(torch.from_numpy(lv), torch.from_numpy(yv)))
        val_auc = _safe_auc(yv, lv)
        log.append({"epoch": epoch, "train_loss": tot / max(n, 1), "val_loss": val_loss,
                    "val_auc": val_auc, "lr": sched.get_last_lr()[0], "sec": time.time() - te})
        print(f"[{cfg.split}/{cfg.arch}/s{cfg.seed}] ep{epoch:02d} "
              f"train {tot / max(n, 1):.4f} | val loss {val_loss:.4f} auc {val_auc:.4f} "
              f"| {time.time() - te:.0f}s", flush=True)

        improved = val_auc > best["auc"] + 1e-4 or (
            abs(val_auc - best["auc"]) <= 1e-4 and val_loss < best["loss"])
        if improved:
            best = {"auc": val_auc, "loss": val_loss, "epoch": epoch}
            torch.save(model.state_dict(), out / "best.pt")
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    pd.DataFrame(log).to_csv(out / "log.csv", index=False)

    # Final predictions with the best-epoch weights. Test is touched only here.
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    rows = []
    for fold in ("val", "test"):
        lg, yy, ii = _predict(model, loader(fold, False), device, amp)
        rows.append(pd.DataFrame({"image_id": ii, "fold": fold, "y": yy.astype(int),
                                  "logit": lg, "prob": 1 / (1 + np.exp(-lg))}))
    preds = pd.concat(rows, ignore_index=True)

    meta = {
        "config": {**asdict(cfg), "out_dir": str(out)},
        "best_epoch": best["epoch"],
        "epochs_run": len(log),
        "val_auc": _safe_auc(*preds.loc[preds.fold == "val", ["y", "prob"]].to_numpy().T),
        "test_auc": _safe_auc(*preds.loc[preds.fold == "test", ["y", "prob"]].to_numpy().T),
        "n": {f: int(len(d)) for f, d in folds.items()},
        "minutes": (time.time() - t0) / 60,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "versions": {"torch": torch.__version__, "python": platform.python_version()},
    }
    try:
        import timm
        meta["versions"]["timm"] = timm.__version__
    except ImportError:
        pass
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    preds.to_csv(out / "preds.csv", index=False)  # written last: marks the run as complete
    return meta


def collect(out_root: str | Path = "/kaggle/working/runs") -> pd.DataFrame:
    """Gather meta.json of all finished runs into one table."""
    rows = []
    for m in Path(out_root).glob("*/*/seed*/meta.json"):
        d = json.loads(m.read_text())
        c = d["config"]
        rows.append({"split": c["split"], "arch": c["arch"], "seed": c["seed"],
                     "best_epoch": d["best_epoch"], "epochs_run": d["epochs_run"],
                     "val_auc": d["val_auc"], "test_auc": d["test_auc"],
                     "minutes": round(d["minutes"], 1)})
    return pd.DataFrame(rows).sort_values(["split", "arch", "seed"]) if rows else pd.DataFrame()
