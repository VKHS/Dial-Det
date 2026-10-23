# Experiment workflows and artifact schemas

This guide describes input validation, complete-policy risk experiments, paired
statistical analysis, and evidence-backed result comparison. Use the
[COCO study guide](COCO_STUDY.md) for automated multi-seed cross-fitting and the
[dataset guide](DATASETS.md) for candidate export. The
[implementation notes](../IMPLEMENTATION_NOTES.md) define the computational
contracts behind these commands. Run all commands from the repository root.

## Validate inputs and preserve run evidence

```bash
python run_dialdet_arrhenius.py validate-manifest \
  --manifest /absolute/export/manifest.json --output /absolute/manifest_check
python run_dialdet_arrhenius.py verify-run \
  --run /absolute/completed_run --check-inputs --output /absolute/run_check
```

Manifest validation checks all candidate files and reports optional undeclared hashes honestly. Run verification checks manifest and recorded output hashes; `--check-inputs` additionally checks external files at their recorded paths. Use a separate verification output directory. Unlisted files are reported separately.

Successful CLI commands produce `RUN_MANIFEST.json` and `RUN_STATUS.json`; failures produce `FAILURE.json`. The manifest records source/runtime identity, arguments, explicit input files, artifacts and `synthetic: true`, `false` or unknown. Hashes establish consistency with a recorded execution. They do not authenticate training history or prove scientific reproduction.

## Complete-policy risk workflow

Before inspecting calibration losses, externally fix the detector, evidence checkpoint, full policy family, fallback, K and development scoring rule. Preserve the experiment's declaration and detector/data lineage records. The command chain cannot prove that this happened earlier.

```bash
python run_dialdet_arrhenius.py collect-risk \
  --manifest /absolute/focal_roles.json --checkpoint /absolute/calibrated.pt \
  --config /absolute/deployed_solver.json --budget 100 --device cuda \
  --output /absolute/risk_collection

python run_dialdet_arrhenius.py score-family \
  --manifest /absolute/focal_roles.json --checkpoint /absolute/calibrated.pt \
  --policy-records /absolute/risk_collection/policy_records.json \
  --crowdhuman-annotations /absolute/CrowdHuman/annotation_train.odgt \
  --crowdhuman-evaluator /absolute/CrowdDetection --device cuda \
  --output /absolute/development_family

python run_dialdet_arrhenius.py risk-report \
  --policy-records /absolute/risk_collection/policy_records.json \
  --development-scores /absolute/development_family/development_scores.json \
  --provenance /absolute/verified_population_provenance.json \
  --output /absolute/risk_report
```

Use `--device cpu` when appropriate. For COCO, replace both CrowdHuman evaluator options with `--coco-annotations FILE` and use a COCO manifest. Supply your actual paths; neither datasets nor model weights are bundled.

Collection includes every calibration image under all 24 policies, including uncertified and fallback outputs. Scoring uses exactly the checkpoint's reserved `dev` IDs and ignores calibration-loss values in the registry. It checks the same checkpoint/source and registered solver, temperature, quality, budget and fallback. These settings cannot be overridden in `score-family`. Separate out-of-fold development exports are explicitly identified.

`--allow-matching-proxy` permits diagnostic attainable-recall scoring without an official evaluator. It cannot enable focal risk-certificate eligibility. `DEVELOPMENT_FAMILY.json` records the metric and population; `development_scores.json` binds each measured score to its policy and development population.

The provenance JSON must contain these externally established facts for conditional eligibility:

| Field | Required meaning |
|---|---|
| `protocol` | `"independent_focal_detector"` |
| `complete_policy_including_fallback` | True; actual fallback and uncertified outputs included |
| `iid_deployment_population` | True only with justified sampling assumptions |
| `encoder_fixed_before_calibration`, `detector_unseen_calibration` | True only with supporting training/history records |
| `policy_family_fixed_before_calibration`, `budget_fixed_before_calibration` | True only with prior declaration evidence |
| `train_ids`, `dev_ids`, `audit_ids` | Actual image-ID arrays; checked against calibration disjointness and fitting bindings |
| `encoder_sha256`, `detector_sha256` | Actual 64-character hashes matching the checkpoint/export identities |
| `calibration_population_sha256` | Optional; if supplied, must match collection |

Do not fill these fields with assumed truths. Eligibility also requires verified policy/development bindings, an official development metric and nonsynthetic inputs. Public-checkpoint cross-fitting cannot establish the focal guarantee.

Statuses distinguish `no_feasible_policy`, `bounds_only_unverified_policy_binding`, `bounds_only_unverified_population` and `conditional_on_declared_provenance`. Even the last remains conditional on external assumptions.

## Paired-statistics specification

Save a JSON specification beside the referenced seed directories. Relative paths resolve against the specification file. Each encoder seed must contain the complete same held-out population; merge cross-fit predictions before evaluating each seed. Example matching diagnostic:

```json
{
  "schema_version": 1,
  "method_a": "dialdet",
  "method_b": "local",
  "budget": 100,
  "metric": "mean_image_shortfall",
  "subset": "all",
  "repeats": 2000,
  "seed": 1,
  "runs": [
    {"seed": 1, "audits": "seed1/evaluation/per_image_audits.json", "run_manifest": "seed1/RUN_MANIFEST.json"},
    {"seed": 2, "audits": "seed2/evaluation/per_image_audits.json", "run_manifest": "seed2/RUN_MANIFEST.json"},
    {"seed": 3, "audits": "seed3/evaluation/per_image_audits.json", "run_manifest": "seed3/RUN_MANIFEST.json"},
    {"seed": 4, "audits": "seed4/evaluation/per_image_audits.json", "run_manifest": "seed4/RUN_MANIFEST.json"},
    {"seed": 5, "audits": "seed5/evaluation/per_image_audits.json", "run_manifest": "seed5/RUN_MANIFEST.json"}
  ]
}
```

`run_manifest` is optional when adjacent evaluation/run evidence establishes synthetic classification. `audits` may also name the evaluation directory. The two distinct method names identify audit groups such as `dialdet_K100` and `local_K100`. At least two unique encoder seeds are required; five matches the paper's seed count.

| `metric` | Additional inputs |
|---|---|
| `mean_image_shortfall`, `mean_image_waste`, `pooled_shortfall` | Audit files above |
| `AP`, `AP50`, `AP75` | Top-level `coco_annotations` path and `predictions` in **every** run |

For COCO AP, set `"metric": "AP"`, add `"coco_annotations": "annotations/instances_val2017.json"`, and add, for example, `"predictions": "seed1/inference"` to the first run and corresponding paths for the others. `predictions` accepts an inference directory or its `predictions.jsonl` file. Existing frozen candidate and audit identities must match. pycocotools is required.

```bash
python run_dialdet_arrhenius.py paired-statistics \
  --specification /absolute/paired.json --output /absolute/paired_results
```

The bootstrap independently resamples images and encoder seeds while pairing both methods. COCO AP is recomputed per resampled dataset/seed, then averaged across sampled seeds. Repeated images receive separate evaluator IDs. This can be computationally expensive. Official CrowdHuman/CityPersons metric bootstrap adapters remain external.

`subset="high_pressure"` fixes the top `ceil(0.2*N)` images by H, with string-ID tie breaking, before resampling. It is retrospective and label-dependent. `PAIRED_STATISTICS.json` reports `estimate` as A minus B, `method_a_estimate`, `method_b_estimate`, `bootstrap_standard_error`, primary `normal_95_interval`, diagnostic `percentile_95_interval`, identities and synthetic classification. All values use fractions. Negative shortfall/waste contrasts and positive AP contrasts favor A.

## Reported versus generated result ledgers

Both input files use `schema="dialdet-results-v1"` and a nonempty `results` list. Their `kind` is respectively `reported` and `generated`.

| Required element | Contents |
|---|---|
| Each row | Unique nonempty `result_id`, finite numeric `value`, `context`, `provenance_status` |
| `context` | Nonempty `dataset`, `split`, `detector`, `metric`, `unit`, `track`, `protocol`; positive integer `budget` |
| Reported file's `source` | `title`, `version`, lowercase 64-hex `document_sha256` |
| Reported row | `provenance_status="reported_in_article"`, actual `paper_location` |
| Generated row | `provenance_status="generated_by_local_run"`, `evidence` object |
| `evidence` | `run_manifest`, `artifact`, `artifact_sha256`, `json_pointer` |

`run_manifest` resolves relative to the ledger (or can be absolute). `artifact` must stay **inside the run directory**, relative to `RUN_MANIFEST.json`; absolute paths and `..` are rejected. Its hash must match both the manifest and actual file. The JSON pointer must identify the exact numeric `value`.

For a pipeline, the artifact can be `evaluation/EVALUATION.json` and the pointer `/results/dialdet_K100/official_coco/metrics/AP`. A standalone evaluation's artifact is `EVALUATION.json`. Copy actual output values and hashes; do not invent a result to complete a ledger. The verified run must explicitly be nonsynthetic. This does not independently validate the context declarations or manuscript transcription.

```bash
python run_dialdet_arrhenius.py compare-results \
  --generated /absolute/generated.json --reference /absolute/reported.json \
  --output /absolute/comparison
```

Contexts must match exactly, including units and protocol. Missing generated entries fail unless `--allow-partial` is explicit; unexpected IDs and zero overlap fail. `RESULT_COMPARISON.json` records differences and completeness. Reported values must identify their source document and location.

## Runtime and artifact integrity

Unsealed risk records support bounds-only diagnostics. Predictions without policy bindings support numerical membership audits with unverified policy identity. Policy-bound workflows require matching source, settings, candidates and recorded artifacts. Keep the implementation fixed throughout collection and family scoring, and use separate directories for distinct experiments.

The optional `dpp_kernel` / `dpp_map` Python helpers are untuned and absent from default CLI comparison outputs. Development selection of scale/ridge and the intended benchmark protocol are still required.

Arrhenius uses account `naiss2025-5-643-gpu`, storage base `/nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls`, code under `code/dialdet_reconstruction` and environment under `code/venv_arrhenius_gh200`. The shell supports `collect-risk`, `score-family`, `tune-local`, `paired-statistics`; use `DIAL_POLICY_RECORDS`, `DIAL_DEV_ANNOTATIONS`, `DIAL_STATISTICS_SPEC` as applicable. `DIAL_BUDGET`/`DIAL_CONFIG` configure collection, not registered-family scoring. Direct CPU analysis commands need no GPU allocation. See [the Arrhenius guide](../README_ARRHENIUS.txt) for modules and exact launch examples.

```bash
python scripts/release_check.py --neural --output ../dialdet_release_check
```

This checks package hashes, imports/CLI, shell syntax, tests and actual synthetic execution without installing dependencies. Omit `--neural` for the non-neural checks. See [the validation report](../VALIDATION_REPORT.md) and `validation/` for measured validation. Synthetic checks and local shell tests establish neither real benchmark performance nor executed Arrhenius jobs.
