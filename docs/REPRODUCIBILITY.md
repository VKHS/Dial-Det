# Reproducibility levels

## Level 1: software verification

```bash
pytest -q
python scripts/release_check.py
```

This verifies contracts, geometry, hashing, objective arithmetic, exact/LP
bounds, DP, risk definitions, calibration arithmetic, strict manifests, and all
four synthetic end-to-end paths.

## Level 2: independent implementation study

A reader obtains a detector and dataset, exports candidate streams, trains the
supplied evidence encoder, freezes rules/splits, runs allocation and baselines,
and evaluates the generated predictions.

## Level 3: exact article replication

This additionally requires the authors' precise data identifiers, detector and
evidence checkpoints, every numerical training hyperparameter, environment,
hardware, and raw evaluator inputs. Those external/large assets are not
embedded in this source repository. Fields not numerically stated in the
article are listed explicitly in
`configs/paper_protocol/reported_settings.json`; they are never silently
invented.
