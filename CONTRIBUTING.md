# Contributing to DIAL-Det

Contributions should make DIAL-Det easier to inspect, execute and reproduce. Start with [the method guide](docs/METHOD.md), [implementation map](docs/IMPLEMENTATION_STATUS.md), and [reviewer guide](docs/REVIEWER_GUIDE.md). Preserve the mathematical contract, frozen candidate interface, data-role separation and traceability of reported results.

## Propose a focused change

Describe the concrete problem, the relevant paper section or observable failure, and the resulting behavior. Identify whether your change affects the frozen candidate interface, evidence training, report allocation, certificates, evaluation, or experiment provenance. A change to a scientific default needs a reason and an explicit configuration record.

For a bug report, include a minimal command, package version, environment, expected behavior and actual output. Prefer a small synthetic reproducer. When the problem depends on benchmark data, state its dataset, split, candidate exporter and checkpoint identity; provide the relevant manifests and log excerpts rather than bundling an entire dataset.

## Validate the behavior you changed

Install the dependencies for your task in a suitable environment, then run the relevant unit-test module. For example:

```bash
python -m unittest discover -s tests -p test_release.py -v
python run_dialdet_arrhenius.py smoke --output /absolute/path/dialdet-smoke --device cpu
```

The neural smoke adds `--neural` and performs real optimization on synthetic data. COCO evaluator tests require `pycocotools`; the external CrowdHuman evaluator tests additionally require `DIAL_CROWDHUMAN_EVALUATOR` to point to the upstream checkout. Skipped tests remain visible in the report and log. An installation or an import check alone does not validate a scientific result.

Tests should exercise observable numerical or workflow behavior: a violated invariant, an actual mismatch, a meaningful edge case, or a previously failing execution path. Preserve deterministic candidate identities, development/calibration/audit separation, and real failure statuses. Do not make a test pass by weakening an independently defined scientific check.

## Keep evidence traceable

If a result changes, record the code revision, configuration, candidate hashes, detector/evidence checkpoint identity, seed, population, evaluator and actual output files. Keep synthetic diagnostics and benchmark measurements clearly labeled. Do not transfer an earlier run's result to changed source by merely rewriting its manifest or package hashes.

The [COCO study workflow](docs/COCO_STUDY.md) plans tasks before execution, then records the work actually performed. Its JSON specification controls the scientific settings. Every ablation must be compared under the same candidate stream and evaluation population; see [the ablation guide](docs/ABLATIONS.md).

## Prepare a pull request or release

Explain what changed, why it matters, how it was checked, and any remaining limitations. Include actual commands and test summaries. Avoid claims that a local smoke run establishes benchmark accuracy, CUDA performance or server compatibility.

Before a release, refresh `PACKAGE_SHA256.json` for the reviewed source and execute the [release gate](docs/GITHUB_SETUP.md). The GitHub workflow checks these hashes; it never rewrites them to hide a source mismatch. Local runs are distinct from a GitHub Actions run on the uploaded commit.
