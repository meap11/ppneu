# ppneu

Leakage-controlled, externally validated evaluation of compressed on-device
pediatric pneumonia classifiers (Kermany CXR, VinDr-PCXR, Core ML).

Work in progress.

## Layout
- `ppneu/` shared code (installed in Kaggle notebooks via pip)
- `notebooks/` one notebook per step (`00_audit`, `01_splits`, ...)
- `manifests/` frozen split CSVs

## Install in a Kaggle notebook
```
!pip install -q --no-deps --force-reinstall --no-cache-dir git+https://github.com/meap11/ppneu.git
```
