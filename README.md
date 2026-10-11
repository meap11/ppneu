# ppneu

Leakage-controlled, externally validated evaluation of compressed on-device
pediatric pneumonia classifiers (Kermany CXR, VinDr-PCXR, Core ML).

Work in progress.

## Layout
- `ppneu/` shared code (audit, splits, data, models, training, metrics, calibration, ensemble)
- `notebooks/` one notebook per step: `00_audit`, `01_splits`, `02_train`, `02b_repredict` (Kaggle),
  `03_eval` (local, CPU)
- `manifests/` frozen split CSVs (seed 0) and two extra split realisations
- `results/` tables computed by `03_eval` from the per-run predictions
- `docs/` audit findings

## Install in a Kaggle notebook
```
!pip install -q --no-deps --force-reinstall --no-cache-dir git+https://github.com/meap11/ppneu.git
```

## Install locally
```
pip install -e .
```
`03_eval` needs only numpy, pandas, scipy and scikit-learn. Training and Core ML export also need
torch, torchvision and timm (and coremltools for export).
