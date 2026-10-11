# ppneu — project context for Claude

## Goal
Turn a 2023 undergrad thesis (pediatric pneumonia detection on the Kermany chest X-ray
dataset; ResNet-50 + DenseNet-121 + Inception-v3 weighted ensemble, 94.67% acc on the
official test set) into a short paper for **IEEE ISBI 2027 (4 pages, deadline 26 Oct 2026)**,
then post to arXiv (eess.IV). Purpose: credential for PhD applications (Dec 2026).

The thesis itself is not publishable as-is (abstract claims things the body doesn't do,
ensemble gains +0.09 pts, model choice and ensemble weights used the test set). The new paper is a
**leakage-controlled, externally validated evaluation of compressed on-device classifiers**.

## Paper contributions
1. Dataset audit: duplicates and filename-ID overlap (Table 1). DONE.
2. Leakage (Table 2). DONE. Leakage is real but local: within one test set, children whose patient
   cluster is also in train get about half the pneumonia errors (3 split realisations, pooled, confound
   checked). Comparing AUROC between a leaky and a clean split is NOT robust (the sign flips across
   realisations, and the grouping changes test-set composition). Methods message: measure leakage per
   patient inside one test set, not by comparing split-level AUROC.
3. External validation on VinDr-PCXR (pediatric, Vietnam) + foundation-model linear probe (Table 3). TODO, Mac.
4. Headline: Core ML compression (FP32/FP16/INT8/palettized) effect on AUROC, calibration
   (log-loss/Brier first, ECE secondary) and decisions at the frozen FP32 val threshold
   (flipped decisions, Δsens/Δspec), plus iPhone latency/memory (Table 4). TODO, Mac.
Also in the paper: thesis ensemble redone without the test set (88.6% vs the thesis's test-tuned 94.67%),
and the official-test distribution shift (calibration fitted on clean val does not transfer).

## Hard rules
- **Test sets are never used for any decision** (epochs, thresholds, temperatures, ensemble weights,
  model choice, metric/rule choices, quantization calibration data). Everything is fitted on val (or train).
- Every run writes per-image `preds.csv` (image_id, fold, y, logit, prob); all metrics are computed offline
  from these. **Use the `logit` column only**: float32 `prob` saturates at 1.0 and the ties lower AUROC.
- Final predictions are FP32. (Training-time val ran under fp16 autocast; fine for epoch selection only.)
  The original training `preds.csv` files are fp16; the FP32 ones are in `02b_repredict_output/`.
- Operating point: threshold for 95% sensitivity fitted on val (primary; chosen by a val-only cross-fit),
  0.5 (secondary). Temperature scaling changes probabilities only, never decisions.
- Unweighted BCE (calibration is a headline metric); no horizontal flip augmentation.
- **VinDr-PCXR must never be uploaded anywhere** (PhysioNet license: no sharing, keep secure,
  code must be public). Used only on the user's Mac. Strip DICOM metadata; no VinDr images in figures.
- `manifests/` seed-0 files are FROZEN. Never edit them.
- **The user commits.** Claude never runs git commit/push; it gives a commit message with no double
  quotes inside (they broke `git commit -m "..."` once).

## Environment
- Training on **Kaggle** (T4) via notebooks; shared code installed there with
  `pip install --no-deps git+https://github.com/meap11/ppneu.git` (torch/timm preinstalled).
  Kaggle limits: 2 concurrent GPU sessions, ~30 GPU-h/week. Attach earlier notebook outputs via Add Input
  (no upload needed).
- Kaggle dataset: `paultimothymooney/chest-xray-pneumonia`
  (root: /kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray).
- Training env (from meta.json): torch 2.11.0+cu128, timm 1.0.29, Python 3.13, Tesla T4.
- Local eval so far ran on Windows (Python 3.8, no torch): `03_eval` needs only numpy/pandas/scipy/sklearn.
- From 2026-10-12 work continues on the user's **Mac** (Steps 6–7): VinDr, Core ML conversion, iPhone benchmark.
- The user is an iOS developer (2.5 yrs) with an MLOps background.

## Repo layout
- `ppneu/audit.py` — index, hashing, duplicates, filename-group parsing (`find_root` walks dirs only; rglob was very slow on Kaggle)
- `ppneu/splits.py` — split conditions, cluster grouping (union of coarse filename groups and exact duplicates),
  leakage report; `build_extra` for extra realisations
- `ppneu/data.py` — 256×256 grayscale uint8 cache of all 5,856 images (bilinear, aspect ratio NOT kept),
  `CXRDataset`, augmentation (mild affine + brightness/contrast; no flip)
- `ppneu/models.py` — timm backbones with one logit: resnet50/densenet121/inception_v3 (`.tv_in1k`),
  mobilenetv3 (`mobilenetv3_large_100.ra_in1k`), efficientnet_b0 (`.ra_in1k`); mean/std from timm config
- `ppneu/train.py` — one run per (split, arch, seed): AdamW lr 2e-4, wd 1e-4, batch 32, 224 px, 1-epoch warmup + cosine,
  ≤20 epochs, early stop on val AUROC (patience 5), AMP; all layers fine-tuned. `repredict_fp32` re-predicts from best.pt
- `ppneu/metrics.py` — point metrics (from logits), image/cluster/paired bootstrap, `cluster_weights` (bootstrap shared by
  overlapping test sets), `load_preds`, `evaluate_run`/`evaluate_all` (fit T and threshold on val, score test)
- `ppneu/calibrate.py` — temperature scaling, sensitivity-target threshold, val-only threshold-rule cross-fit
- `ppneu/ensemble.py` — thesis trio; members temperature-scaled on val; equal weights (primary) or val log-loss weights;
  output has the preds format and goes through `evaluate_all`; never sees test labels (verified by scrambling them)
- `manifests/` — FROZEN seed-0 `official`, `image_random`, `grouped` + `leakage_report.json`; extra realisations
  `image_random_s1/s2`, `grouped_s1/s2` + `leakage_report_extra.json` (built locally with `splits.build_extra`
  from the audit `index.csv`; the same code rebuilt the seed-0 manifests byte-identically)
- `notebooks/` — `00_audit`, `01_splits` (Kaggle, CPU); `02_train` (Kaggle GPU; `RUN="main"` one copy per arch, or
  `RUN="extra_splits"`); `02b_repredict` (Kaggle GPU, FP32 re-prediction); `03_eval` (local CPU, ~2 min, writes `results/`)
- `results/` — committed CSV tables from 03_eval (reproduce byte-for-byte; bootstrap seeds fixed):
  `run_metrics` (per run), `summary_by_split_arch`, `threshold_rule_val_crossfit`, `test_composition`,
  `table2a_split_realisations`, `table2b_leakage_seen_unseen`, `table2b_confound_check`, `calibration_by_split`,
  `official_shift`, `auroc_ci_seed0`, `thesis_ensemble`, `ensemble_fitted_on_val`
- `docs/audit_findings.md` — audit results in detail
- Local only, gitignored (repo root): `02_train_<arch>_output/` (weights + fp16 preds), `02b_repredict_output/`
  (FP32 preds of the 45 main runs; what 03_eval uses), `02_train_extra_splits_output/` (20 runs, FP32).
  Outside the repo: `../00_audit_output/audit/index.csv` (audit index; needed only to rebuild manifests).

## Key findings (Steps 1–2)
- 5,856 images; 30 exact-duplicate sets (32 redundant, 5,824 unique), all within one split, no label conflicts.
- Filename IDs with namespaces kept separate: **0** train/test overlap. The online "264 overlapping
  patients" appears only when merging namespaces (personN_bacteria vs personN_virus, IM vs NORMAL2-IM);
  every such collision is cross-namespace, so it can't be verified as the same child.
- pHash cannot identify patients on these CXRs (same-ID median 22 bits vs random 24); near-duplicate
  counts are NOT reported as findings.
- Clusters are deliberately conservative (over-merged). Consequences: 274 official-test images share a
  cluster with official train (unverifiable cross-namespace merges); "seen" in Table 2b includes some
  different children, which can only dilute the leakage effect.
- Pooling and re-splitting by image (image_random): 605/879 (69%) test images share a cluster with train.
- Split sizes: official 4448/784/624, image_random 4099/878/879, grouped 4076/874/874 (0 shared clusters).

## Key findings (Steps 3–5; 45 main runs + 20 extra-split runs, FP32 predictions)
- Training: 2–6 min/run on a T4; 45 runs ≈ 2.5 GPU-h. Best epochs 1–17. All models overconfident
  (mean T 1.51–1.73 per split).
- Threshold rule (val-only cross-fit, SD across runs): 95%-sens rule sens SD 0.001–0.003 vs 0.009–0.022 at 0.5
  and 0.005–0.008 for Youden.
- Leakage, between conditions (Table 2a): image_random − grouped AUROC = +0.0035 (s0), −0.0009 (s1), −0.0022 (s2);
  image_random higher in 6/15 (arch, realisation) pairs → not robust. Test composition differs by construction:
  grouped tests hold 5.4–5.5% original-test-folder images vs 10.5–11.5%, and 31–33% virus vs 25–27%.
- Leakage, within the image_random test (Table 2b; pneumonia, val 95%-sens threshold, seed-0 models averaged over
  5 archs, cluster bootstrap): unseen − seen error +3.6 / +6.0 / +3.7 pts (s0/s1/s2); pooled +4.4 (95% CI 1.9 to 7.2),
  p = 0.001; at 0.5 pooled +2.5 (0.6 to 4.6). Normals: pooled +0.5 (−0.8 to 1.8), no effect.
  Confound check: grouped single- vs multi-image patients pooled −0.1 (−2.1 to 2.3); image_random restricted to
  multi-image patients +4.4 (1.0 to 8.1), p = 0.008.
- Official test (shift), per-arch means: AUROC 0.968–0.982; specificity 64–76% at 0.5 and 79–83% at the val
  threshold (sensitivity ~98–99%); log-loss 0.45–0.74 vs 0.07–0.14 in-distribution. MobileNetV3 (0.982) and
  EfficientNet-B0 (0.981) transfer best.
- Thesis ensemble without test (official): equal-weight AUROC 0.979, accuracy at 0.5 88.6%, beats every member's
  AUROC (0.968–0.976). Val-fitted weights are erratic across seeds and do not help.
- fp16 vs FP32 logits (same weights): resnet/densenet median |Δ| 0.005; mobilenetv3/effnet ~0.09 (max 1.85).
  Yet at the frozen FP32 val threshold ≤4 decisions flip per test set (<0.5 pts sens/spec). Expect Core ML FP16
  ≈ null; the dose-response for Table 4 must come from INT8 and 4/2-bit palettization.

## Mac handoff (Steps 6–7)
### Copy from the Windows machine (not in git; place in the repo root with the same names)
- `02b_repredict_output/` (3 MB) — required: FP32 reference preds for 03_eval and Table 4.
- `02_train_extra_splits_output/` — required for 03_eval; only `preds.csv`, `meta.json`, `log.csv` are needed
  (skip its `best.pt` files, ~1 GB).
- `02_train_mobilenetv3_output/` (147 MB) and `02_train_efficientnet_b0_output/` (141 MB) — required: weights for Table 4.
- Other `02_train_<arch>_output/` folders (resnet50 811 MB, inception_v3 752 MB, densenet121 245 MB) — only if
  Table 3 evaluates those archs on VinDr. 03_eval does not need them (it reads `02b_repredict_output/`).
- Optional: `../00_audit_output/audit/index.csv` (1.3 MB) to rebuild manifests.
- Do NOT copy `../git_token.txt` (sits next to the repo on Windows) into the repo or any shared bundle.
### Setup
- `git clone https://github.com/meap11/ppneu.git`, then a venv with torch + torchvision, `timm==1.0.29` (match training),
  coremltools (record its version), numpy, pandas, scipy, scikit-learn, tqdm, pydicom (VinDr); `pip install -e .`
- Kermany images on the Mac (Kaggle CLI: `kaggle datasets download paultimothymooney/chest-xray-pneumonia`), then
  `data.build_cache(load_manifest("official"), <chest_xray root>, <cache path>)` to rebuild the same 256 px cache.
### Step 7 plan (Core ML, Table 4)
- Models: mobilenetv3 and efficientnet_b0 (the deployable ones). Training split for the main table: `grouped`
  (clean in-distribution test); `official` as the shift condition. Seeds 0–2.
- First a parity check: PyTorch FP32 on the Mac (CPU) with the rebuilt cache must reproduce `02b_repredict_output`
  logits (expect max |Δ| ≲ 1e-3; if not, preprocessing differs). Then convert.
- Variants: FP32, FP16, INT8 weight-only (linear), palettized 6/4/2-bit (`coremltools.optimize.coreml`); optionally W8A8
  (activation calibration data from the **train** fold only).
- Inputs for the accuracy study: the exact normalized tensors from `CXRDataset(train=False)` (256 cache → Resize 224
  antialias → [0,1] → 3 channels → timm mean/std) as MLMultiArray, so only compression differs. On-device ImageType
  preprocessing parity is a separate check for the app.
- Compute units matter: "FP32" Core ML on the ANE runs in fp16. Evaluate with CPU_ONLY (true FP32 reference) and ALL
  (deployment), and record which.
- Metrics per variant vs FP32 on the same images: AUROC, flipped decisions and Δsens/Δspec at the frozen FP32 val
  threshold, log-loss/Brier/ECE, plus re-fitting the threshold on the variant's own val (does recalibration fix drift?).
  Paired bootstrap / flip counts; model size on disk; iPhone latency and memory (SwiftUI benchmark app or Xcode
  performance report).
- `inception_v3` was trained at 224 px (native 299): one sentence in the methods.
### Step 6 plan (VinDr-PCXR, Table 3)
- DICOM → PNG with metadata stripped, kept outside the repo and never uploaded. Fix the label mapping (which VinDr
  findings count as pneumonia) before looking at any model output. Preprocess like Kermany (grayscale, 256 → 224).
- Evaluate with thresholds/temperatures frozen from Kermany val; foundation-model linear probe (BiomedCLIP / RAD-DINO)
  trained on Kermany only.

## Status / next steps
- [x] Step 0: VinDr-PCXR access granted; GitHub repo created. arXiv endorser: email thesis supervisor AFTER results.
- [x] Step 1: audit (00_audit)
- [x] Step 2: frozen splits (01_splits; seed-0 manifests verified identical on Kaggle)
- [x] Step 3: training — 45 main runs, FP32 re-prediction (02b), 20 extra-split runs.
- [x] Step 4/5: metrics, calibration, ensemble, 03_eval → results/*.csv (verified reproducible 2026-10-12).
- [ ] Step 7 (Mac, do FIRST — it is the headline and part of the fallback): `ppneu/export.py`, Mac parity check, Core ML
      variants, eval notebook, iPhone benchmark.
- [ ] Step 6 (Mac): VinDr conversion, external eval, linear probe.
- [ ] Step 8: figures, 4-page paper (abstract written last), README/model card. Step 9: submit 26 Oct.
- Fallback if behind on 16 Oct: drop external validation + probe; paper = audit + leakage + on-device compression.
