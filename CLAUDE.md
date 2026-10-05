# ppneu — project context for Claude

## Goal
Turn a 2023 undergrad thesis (pediatric pneumonia detection on the Kermany chest X-ray
dataset; ResNet-50 + DenseNet-121 + Inception-v3 weighted ensemble, 94.67% acc on the
official test set) into a short paper for **IEEE ISBI 2027 (4 pages, deadline 26 Oct 2026)**,
then post to arXiv (eess.IV). Purpose: credential for PhD applications (Dec 2026).

The thesis itself is not publishable as-is (abstract claims things the body doesn't do,
ensemble gains +0.09 pts, model choice used the test set). The new paper is a
**leakage-controlled, externally validated evaluation of compressed on-device classifiers**.

## Paper contributions (planned)
1. Dataset audit: duplicates and filename-ID overlap (Table 1). DONE.
2. Leakage effect: official vs image-level random re-split vs patient-grouped split (Table 2).
3. External validation on VinDr-PCXR (pediatric, Vietnam) + foundation-model linear probe (Table 3).
4. Headline: Core ML compression (FP32/FP16/INT8/palettized) effect on AUROC, calibration (ECE)
   and decision-threshold shift, plus iPhone latency/memory (Table 4).

## Hard rules
- **Test sets are never used for any decision** (epochs, thresholds, ensemble weights,
  temperature scaling all fitted on validation only).
- Every run writes per-image `preds.csv` (image_id, fold, y, logit, prob); all metrics are
  computed offline from these.
- **VinDr-PCXR must never be uploaded anywhere** (PhysioNet license: no sharing, keep secure,
  code must be public). It is used only for evaluation on the user's Mac. Strip DICOM metadata;
  no VinDr images in figures.
- Unweighted BCE (calibration is a headline metric); no horizontal flip augmentation.

## Environment
- Training on **Kaggle** (T4 GPU) via notebooks; shared code installed with
  `pip install --no-deps git+https://github.com/meap11/ppneu.git` (torch/timm preinstalled).
- Kaggle dataset: `paultimothymooney/chest-xray-pneumonia`
  (root: /kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray).
- External eval, Core ML conversion checks and iPhone benchmark on the user's **Mac**.
- The user is an iOS developer (2.5 yrs) with an MLOps background.

## Repo layout
- `ppneu/audit.py` — index, hashing, duplicates, filename-group parsing (`find_root` walks dirs only; rglob was very slow on Kaggle)
- `ppneu/splits.py` — the three split conditions, cluster-based grouping, leakage report
- `ppneu/data.py` — 256px grayscale uint8 cache of all images, Dataset, augmentation
- `ppneu/models.py` — timm backbones: resnet50, densenet121, inception_v3, mobilenetv3, efficientnet_b0
- `ppneu/train.py` — one run per (split, arch, seed); early stop on val AUROC; outputs to runs/<split>/<arch>/seed<k>/
- `manifests/` — FROZEN split CSVs + leakage_report.json (verified reproducible on Kaggle; do not change)
- `notebooks/00_audit`, `01_splits`, `02_train`
- `docs/audit_findings.md` — audit results in detail

## Key findings so far (Steps 1–2)
- 5,856 images; 30 exact-duplicate sets (32 redundant, 5,824 unique), all within one split, no label conflicts.
- Filename IDs with namespaces kept separate: **0** train/test overlap. The online "264 overlapping
  patients" appears only when merging namespaces (personN_bacteria vs personN_virus, IM vs NORMAL2-IM);
  every such collision is cross-namespace, so it can't be verified as the same child.
- pHash cannot identify patients on these CXRs (same-ID median 22 bits vs random 24); near-duplicate
  counts are NOT reported as findings.
- Real leakage is in the common "pool and re-split by image" practice: 605/879 (69%) test images share
  a conservative patient cluster with train.
- Split sizes: official 4448/784/624, image_random 4099/878/879, grouped 4076/874/874 (0 shared clusters).

## Status / next steps
- [x] Step 0: VinDr-PCXR access granted; GitHub repo created. arXiv endorser: email thesis supervisor AFTER results.
- [x] Step 1: audit (00_audit)
- [x] Step 2: frozen splits (01_splits; all three manifests verified identical on Kaggle)
- [ ] Step 3: training — `02_train.ipynb` written, GPU code NOT yet executed. Next: run with `SMOKE=True`
      on Kaggle, check epoch times/errors, then `SMOKE=False` via Save & Run All (45 runs).
- [ ] Step 4/5: `metrics.py`, `calibrate.py`, `ensemble.py`; notebook 03_eval: AUROC/AUPRC, sens/spec at
      val-chosen threshold, ECE, bootstrap 95% CI, temperature scaling; thesis ensemble with weights fitted
      on test vs on val (shows selection bias).
- [ ] Step 6 (Mac): VinDr DICOM→PNG conversion, external eval, BiomedCLIP/RAD-DINO linear probe.
- [ ] Step 7: `export.py` (coremltools FP32/FP16/INT8/palettized) + Mac eval + SwiftUI iPhone benchmark app.
- [ ] Step 8: figures, 4-page paper (abstract written last), README/model card. Step 9: submit 26 Oct.
- Fallback if behind by 16 Oct: drop external validation + probe; paper = leakage audit + on-device.
