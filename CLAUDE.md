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
2. Leakage effect (Table 2), reframed 2026-10-08: leakage is real but local. Within the image_random
   test set, children seen in train get ~half the pneumonia errors (3 split realisations, pooled);
   comparing AUROC between split conditions is NOT robust (sign flips across realisations; test
   composition differs by construction).
3. External validation on VinDr-PCXR (pediatric, Vietnam) + foundation-model linear probe (Table 3).
4. Headline: Core ML compression (FP32/FP16/INT8/palettized) effect on AUROC, calibration
   (log-loss/Brier first, ECE secondary) and decision shift at the frozen FP32 val threshold
   (flipped decisions, Δsens/Δspec), plus iPhone latency/memory (Table 4).

## Hard rules
- **Test sets are never used for any decision** (epochs, thresholds, ensemble weights,
  temperature scaling all fitted on validation only).
- Every run writes per-image `preds.csv` (image_id, fold, y, logit, prob); all metrics are
  computed offline from these.
- **VinDr-PCXR must never be uploaded anywhere** (PhysioNet license: no sharing, keep secure,
  code must be public). It is used only for evaluation on the user's Mac. Strip DICOM metadata;
  no VinDr images in figures.
- Unweighted BCE (calibration is a headline metric); no horizontal flip augmentation.
- Metrics from the `logit` column only (float32 probs saturate at 1.0 → ties). Final predictions
  are FP32 (training-time val uses fp16 autocast; that is fine for epoch selection only).
- Operating point: threshold for 95% sensitivity fitted on val (primary), 0.5 (secondary).
- The user commits; Claude gives commit messages (no double quotes inside them).

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
- `ppneu/train.py` — one run per (split, arch, seed); early stop on val AUROC; outputs to runs/<split>/<arch>/seed<k>/;
  `repredict_fp32` re-predicts finished runs from best.pt
- `ppneu/metrics.py` — point metrics, image/cluster/paired bootstrap, run loading + per-run evaluation
- `ppneu/calibrate.py` — temperature scaling, sensitivity-target threshold, val-only threshold-rule cross-fit
- `ppneu/ensemble.py` — thesis trio, members temperature-scaled on val, equal weights (primary); never sees test
- `manifests/` — FROZEN seed-0 split CSVs + leakage_report.json (do not change); extra realisations
  `image_random_s1/s2`, `grouped_s1/s2` + leakage_report_extra.json
- `notebooks/00_audit`, `01_splits`, `02_train` (RUN=main per arch, or extra_splits), `02b_repredict`, `03_eval` (local, CPU)
- Local, gitignored: `02_train_<arch>_output/` (weights), `02b_repredict_output/` (FP32 preds of the 45 main runs),
  `02_train_extra_splits_output/`; `results/` holds the CSV tables from 03_eval
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

## Key findings (Steps 3–5, 45 main runs + 20 extra-split runs, FP32 predictions)
- Training: 2–6 min/run on a T4; all 45 runs ≈ 2.5 GPU-h. Best epochs 1–17; all models overconfident (T ≈ 1.5–1.7).
- Leakage, between conditions: image_random − grouped AUROC = +0.0035 (s0), −0.0009 (s1), −0.0022 (s2) → not robust.
  grouped test sets hold 5.5% original-test-folder images vs ~11% and more virus (32% vs 25%), by construction.
- Leakage, within image_random test (pneumonia, val 95%-sens threshold, cluster bootstrap): unseen − seen error
  +3.6 / +6.0 / +3.7 pts; pooled +4.4 (1.9 to 7.2), p = 0.001. Normals: no effect. Confound check: in grouped,
  single- vs multi-image patients pooled −0.1 (−2.1 to 2.3); image_random multi-image only +4.4 (1.0 to 8.1).
- Official test (shift): AUROC 0.968–0.982, specificity 64–76% at 0.5 and 79–83% at the val threshold; log-loss
  0.45–0.74 vs ~0.1 in-distribution. MobileNetV3/EfficientNet transfer best.
- Thesis ensemble without test: equal-weight ensemble official AUROC 0.979, accuracy 88.6% (thesis: 94.67% with
  test-tuned weights/model choice). Val-fitted weights are erratic and do not help.
- fp16 vs FP32 logits: resnet/densenet median diff 0.005; mobilenetv3/effnet ~0.09 (max 1.85), yet ≤4 flipped
  decisions per test set → expect Core ML FP16 ≈ null; INT8/palettized levels needed for Table 4.

## Status / next steps
- [x] Step 0: VinDr-PCXR access granted; GitHub repo created. arXiv endorser: email thesis supervisor AFTER results.
- [x] Step 1: audit (00_audit)
- [x] Step 2: frozen splits (01_splits; all three manifests verified identical on Kaggle)
- [x] Step 3: training — 45 main runs (one Kaggle copy per arch), FP32 re-prediction (02b), 20 extra-split runs.
- [x] Step 4/5: metrics.py, calibrate.py, ensemble.py, 03_eval → results/*.csv (2026-10-08).
- [ ] Step 6 (Mac): VinDr DICOM→PNG conversion, external eval, BiomedCLIP/RAD-DINO linear probe.
- [ ] Step 7: `export.py` (coremltools FP32/FP16/INT8/palettized) + Mac eval + SwiftUI iPhone benchmark app.
- [ ] Step 8: figures, 4-page paper (abstract written last), README/model card. Step 9: submit 26 Oct.
- Fallback if behind by 16 Oct: drop external validation + probe; paper = leakage audit + on-device.
