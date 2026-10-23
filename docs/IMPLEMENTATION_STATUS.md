# Implementation status and evidence map

**The core DIAL-Det computation is implemented:** learned relational evidence, identity-mass training, at-most-$K$ allocation, independent certificate replay and held-out evaluation. The repository also provides executable COCO study orchestration and reproducible run records. The tables below state the supported computation, evaluation interfaces and experiment requirements.

Use [VALIDATION_REPORT.md](../VALIDATION_REPORT.md) for recorded execution results, [METHOD.md](METHOD.md) for the mathematical contract, and [IMPLEMENTATION_NOTES.md](../IMPLEMENTATION_NOTES.md) for the detailed design specification. Test availability below identifies coverage; it does not by itself establish that a test ran in a particular environment. Optional neural and external-evaluator dependencies are identified in the validation report.

## Method-to-code-to-tests

Paper references identify the main paper or supplement; exact source files are identified by [SOURCE_PROVENANCE.json](../SOURCE_PROVENANCE.json).

| Contract | Main implementation | Focused tests |
|---|---|---|
| Frozen candidate identity and annotation roles — Appendix B.1/B.6 | [`data.py`](../dialdet/data.py), [`exporter.py`](../dialdet/exporter.py), [`detection_datasets.py`](../dialdet/detection_datasets.py) | [`test_data_metrics.py`](../tests/test_data_metrics.py), [`test_detection_datasets.py`](../tests/test_detection_datasets.py), [`test_evidence.py`](../tests/test_evidence.py) |
| Two support branches, gated encoder, nested quality, symmetric identity — Appendix B.2–B.4 | [`evidence.py`](../dialdet/evidence.py) | [`test_evidence.py`](../tests/test_evidence.py) |
| Mass/transitivity losses, checkpoint resume, development temperature — Appendix B.5 | [`evidence.py`](../dialdet/evidence.py), [`training.py`](../dialdet/training.py) | [`test_evidence.py`](../tests/test_evidence.py), [`test_detection_workflow.py`](../tests/test_detection_workflow.py), [`test_pipeline_integrity.py`](../tests/test_pipeline_integrity.py) |
| Unthresholded mass, objective, analytic admission/exchange rules — paper Propositions 1–2 | [`objective.py`](../dialdet/objective.py) | [`test_solver.py`](../tests/test_solver.py) |
| Cardinality search, split correction, DP, LP witnesses and checked gaps — Algorithm D1 | [`solver.py`](../dialdet/solver.py), [`certificate.py`](../dialdet/certificate.py) | [`test_solver.py`](../tests/test_solver.py) |
| Maximum matching, risk weighting and retrospective miss bound — Appendix C.1/C.8, E.1 | [`metrics.py`](../dialdet/metrics.py), [`audit_bound.py`](../dialdet/audit_bound.py) | [`test_data_metrics.py`](../tests/test_data_metrics.py), [`test_audit_bound.py`](../tests/test_audit_bound.py) |
| Official COCO evaluation, CrowdHuman adapter, fixed high-pressure subset — Appendix E | [`metrics.py`](../dialdet/metrics.py), [`crowdhuman.py`](../dialdet/crowdhuman.py), [`cli.py`](../dialdet/cli.py) | [`test_integration.py`](../tests/test_integration.py), [`test_crowdhuman.py`](../tests/test_crowdhuman.py), [`test_evaluation.py`](../tests/test_evaluation.py), [`test_detection_workflow.py`](../tests/test_detection_workflow.py) |
| Cross-fit roles and held-out union — Appendix B.6 | [`data.py`](../dialdet/data.py), [`crossfit.py`](../dialdet/crossfit.py) | [`test_data_metrics.py`](../tests/test_data_metrics.py), [`test_integration.py`](../tests/test_integration.py) |
| Complete 24-policy family and simultaneous bounds — Appendix C.9/E.10 | [`risk_runtime.py`](../dialdet/risk_runtime.py), [`development.py`](../dialdet/development.py), [`risk.py`](../dialdet/risk.py) | [`test_risk_provenance.py`](../tests/test_risk_provenance.py), [`test_integration.py`](../tests/test_integration.py), [`test_data_metrics.py`](../tests/test_data_metrics.py) |
| Paired image–seed bootstrap and full-dataset AP recomputation — Appendix E.9 | [`statistical_workflow.py`](../dialdet/statistical_workflow.py), [`statistics.py`](../dialdet/statistics.py) | [`test_statistics_workflow.py`](../tests/test_statistics_workflow.py) |
| Numerical, semantic, population and policy identities; verified results | [`provenance.py`](../dialdet/provenance.py), [`artifacts.py`](../dialdet/artifacts.py), [`results.py`](../dialdet/results.py) | [`test_provenance.py`](../tests/test_provenance.py), [`test_artifacts_results.py`](../tests/test_artifacts_results.py), [`test_pipeline_integrity.py`](../tests/test_pipeline_integrity.py) |
| COCO study planning/execution/resume and paired analysis | [`study.py`](../dialdet/study.py) | [`test_study.py`](../tests/test_study.py) |
| Package integrity, CLI execution and Arrhenius routing | [`release_check.py`](../scripts/release_check.py), root shell launchers | [`test_release.py`](../tests/test_release.py) |

The tests include exhaustive small-instance objective comparisons, LP-bound checks, split correction, negative cardinality values, empty reports, nonfinite-evidence fallback, certificate tampering, annotation semantics, disjoint data roles, and bootstrap image multiplicity. These targeted checks make the method inspectable beyond a successful import.

## Integration boundaries

| Area | Implemented and available | Boundary a reader should retain |
|---|---|---|
| Detector export | torchvision Faster R-CNN ResNet-50-FPN with frozen candidate IDs/features/scores | Native YOLO, DETR-family, CrowdDet/DDQ and MMDetection exporters are absent; other backbones or framework checkpoints need an explicit adapter |
| Dataset input | COCO instances JSON, CrowdHuman full-body ODGT, CityPersons MAT | Correct images, annotations and compatible checkpoints must be supplied |
| Benchmark evaluation | Official COCO AP; CrowdHuman AP50/MR through a local upstream evaluator checkout | CityPersons official MR and CrowdHuman JI are external; matching counts never substitute for these metrics |
| Comparison track | Membership selection with native scores restored for every implemented control | `coco_predictions(..., own_scores=q)` is an API hook; the complete own-score experiment track is not integrated into the CLI study |
| Controls | Matched local rule, greedy full objective, NMS, Gaussian Soft-NMS | The local rule has a 24-threshold development search; CLI NMS/Soft-NMS use documented defaults, not a completed equally tuned eleven-baseline comparison |
| DPP | PSD-projected kernel and at-most-$K$ greedy log-determinant Python helpers | Untuned defaults, excluded from default CLI controls; no reproduced DPP-MAP* study |
| Study automation | COCO two-fold/five-seed plan, execution and resume; union evaluation; paired statistics | Orchestration is implemented; real study output is not bundled as measured benchmark evidence |
| Ablations | Explicit fresh-training variants in the [ablation guide](ABLATIONS.md) | The complete Appendix E factorial, removals, oracle and corruption experiments still need implementation or execution |
| Risk | All policies, fallback-inclusive losses, development scoring and finite-family bounds | A population guarantee depends on external independence, detector history and prior fixing of the policy family; public-checkpoint cross-fitting remains ineligible |
| Statistical comparisons | Paired diagnostics and official COCO AP/AP50/AP75; normal primary and diagnostic percentile intervals | Native CrowdHuman/CityPersons metric bootstrap remains external; a Holm helper alone is not the paper's full predeclared comparison study |
| Runtime | Per-stage timing records in actual execution artifacts | No reproduced A100 latency, optimized native allocator or Arrhenius GH200 benchmark is supplied |

## Explicit reference choices

The following choices specify the reference implementation, including the network internals, training procedure, split construction and certificate arithmetic.

| Decision | Reference realization |
|---|---|
| Frozen appearance | Spatial mean of the detector's RoIAlign features |
| Rescue compatibility | Same-class filtering also applies to appearance-rescue neighbors |
| Network internals | Hidden width 160, specified symmetric geometry, LeakyReLU attention and GELU/LayerNorm residual transformations |
| Pair supervision | Every eligible support pair is used, so importance weights are one |
| Optimizer details | One image per step, gradient clipping at 5, no learning-rate schedule, development temperature in $[0.05,20]$ |
| Internal development split | The opposite cross-fit half is split 80/20 for fitting/development; split seed and IDs are recorded |
| Component splitting | Deterministic strongest-edge-first, size-limited Kruskal merging with a checked correction for removed interactions |
| Certificate arithmetic | Exact dyadic/rational witness checking and integer dynamic programs; a conservative realization distinct from the paper's floating directed-rounding implementation |
| Cut workload | At most 2,048 enumerated maximal cliques by default; omitted cuts loosen the bound |
| Local fallback | Threshold 0.5 until a development-tuned value is supplied; annotation-enabled pipelines tune it before audit inference |

Detector checkpoints, frozen candidate exports and benchmark experiment artifacts are external inputs. Each experiment records its own fitting populations, split seed, evidence checkpoint and evaluator settings.

## What makes a result ready to report

A completed real-data experiment should supply its source revision, frozen export identity, data roles, checkpoints, development-selected settings, all held-out predictions and evaluator identity. The five-seed study should additionally retain each seed's complete fold union, dataset metrics and paired-statistics specification/output. Allocation status shares must include uncertified and fallback reports; empty reports must remain in the evaluation population.

`RUN_MANIFEST.json` and the associated hashes connect artifacts to the recorded inputs and implementation. They detect inconsistencies and changes; they are not an external signature, a preregistration or a proof of detector-training independence. The [reviewer guide](REVIEWER_GUIDE.md) gives a practical inspection sequence and the [experiment guide](EXPERIMENTS.md) defines the evidence schemas.

The manuscript's sixteen settings, eleven external decoder comparisons, full ablations, statistical conclusions and latency results require their own completed runs. Repository validation supports the implementation claims recorded in [VALIDATION_REPORT.md](../VALIDATION_REPORT.md); it does not populate a benchmark results table.
