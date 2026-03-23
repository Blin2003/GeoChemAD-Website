# GeoChemAD

This repository now contains a runnable end-to-end implementation for the paper-driven geochemical anomaly detection workflow:

- benchmark dataset discovery from `../all_data`
- preprocessing for abnormal values and compositional transforms (`none`, `CLR`, `ILR`)
- unsupervised baselines (`zscore`, `mahalanobis`, `knn`, `iforest`, `ocsvm`, `ae`, `vae`, `transformer`)
- two-stage `GeoChemFormer`
- repeated AUC / AP evaluation against deposit sites
- spatial metrics and anomaly-score maps with `IDW` and `Kriging`
- feature strategies (`none`, `manual`, `causal`, `llm` proxy, `pca`)
- benchmark matrix export across multiple subsets and models
- FastAPI backend and browser dashboard

## Run

Install dependencies:

```bash
python3 -m pip install --user -r requirements.txt
```

Start the dashboard:

```bash
python3 scripts/serve.py
```

Train from CLI:

```bash
python3 scripts/train.py --dataset area1_sediment_au --model geochemformer
```

Run a benchmark matrix:

```bash
python3 scripts/benchmark.py --models geochemformer ae transformer zscore
```

Artifacts are written under `artifacts/runs/`.

## Project Layout

- `geochemad/`: data pipeline, models, training, API
- `scripts/`: CLI entrypoints
- `web/`: static dashboard assets
- `tests/`: smoke tests
