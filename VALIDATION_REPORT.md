# Validation

## Recorded outcome

**152 tests passed, with zero failures, errors or skips**, in the integrated local CPU run. All **11 executable release checks** also passed, including the actual synthetic neural smoke and offline wheel build/install/import checks. Test-suite duration is recorded in the execution logs and does not measure detector latency.

The authoritative local records are [validation/TEST_REPORT.json](validation/TEST_REPORT.json), [the full test log](validation/TEST_LOG.txt) and [the release report](validation/release_check/RELEASE_CHECK.json). [TEST_COVERAGE.json](validation/TEST_COVERAGE.json) records the counts below. These checks exercise computation on controlled fixtures; no real-dataset benchmark results are inferred from them.

| Area | Tests | Behavior checked |
|---|---:|---|
| Run records and result comparison | 8 | Changed artifacts/inputs, relocation, failed retries, synthetic classification and evidence-backed numeric values |
| Retrospective matching bound | 6 | Ideal blocks, price/mass/fragmentation terms, ambiguous matching and vacuous diagnostics |
| CrowdHuman interface and upstream metrics | 11 | Native output, ignore/duplicate matching, exact population and actual upstream AP/MR behavior |
| Frozen data, matching and risk definitions | 19 | Candidate immutability, disjoint roles, empty images, augmenting paths, risk weighting and all 48 hypotheses |
| Native dataset adapters | 10 | COCO-compatible export, CrowdHuman ODGT, CityPersons MATLAB/coordinates/subsets and actual torchvision export |
| Dataset workflows | 6 | Quality target, class table, evaluator selection, evaluated population and separate development/audit annotations |
| Evaluation and baseline interfaces | 8 | High-pressure population, COCO input guards, empty COCO reports and optional PSD DPP helper |
| Learned evidence and ablation behavior | 9 | Nested quality, symmetric identity, mass gradients, open wedges, resume, development temperature, frozen detector boundary and configured removals |
| Runtime integration | 4 | CLI guards, cross-fit union, official COCO cases and trained mass exponent across the 24-policy family |
| Candidate and complete-policy identity | 8 | Semantic metadata, relocation, population binding, checkpoint/source/config identity and manifest preflight |
| Release and Arrhenius helpers | 19 | Shell syntax/routing, study flags and specification-owned settings, preflight dependencies, dry-run submission and package hash coverage |
| Bound risk/development workflow | 10 | Sealed calibration/utilities, policy changes, fitting populations, conditional eligibility and trained family scoring |
| Objective, allocator and checker | 12 | Exhaustive small optima, marginals/exchanges, LP bounds, split correction, corrupted certificates and fallback |
| Paired statistical workflow | 8 | Shared image/seed resampling, exact populations, dataset AP, repeated-image multiplicity and source classification |
| COCO study orchestration | 13 | Strict planning, semantic hashes, all five ablation configurations, sequencing, verified resume, tampering, failed-task recovery and an executed synthetic study |
| Neural policy-binding integration | 1 | Trained inference plus missing/changed artifacts, altered selection, wrong IoU and development-to-audit relabeling |
| **Total** | **152** | **No failures, errors or skipped tests in this recorded run** |

## Synthetic study integration

The integration test starts from six invented COCO-format images with frozen candidates explicitly marked synthetic. It uses two encoder seeds, both cross-fit folds, one epoch per fold and budget 100. Through the outer CLI it:

1. Validates and writes a plan without launching computational tasks.
2. Executes the split and four actual evidence-training pipelines, including development temperature and local-threshold fitting.
3. Merges both held-out folds for each seed and evaluates each six-image union using official COCOeval.
4. Computes paired AP and mean-image-shortfall comparisons with two bootstrap replicates for this small functional test.
5. Verifies the complete study's recorded artifacts and inputs, then resumes with an assertion that no completed task launches again.

The test also checks that the split run and both generated fold manifests retain their synthetic classification when the source manifest itself is labeled COCO. Separate tests verify planning of all five registered ablation configurations, declared semantic hashes, malformed inputs, unsafe output paths, changed source/data/results and partial-task recovery.

The example scientific specification uses five seeds, 30 epochs and 2,000 replicates. The integration fixture is deliberately smaller and supplies no five-seed benchmark estimate. Actual study state and comparisons are documented in [COCO_STUDY.md](docs/COCO_STUDY.md).

## Executable release gate

The recorded command was:

```bash
export DIAL_CROWDHUMAN_EVALUATOR=/absolute/path/CrowdDetection
python scripts/release_check.py --neural --wheel --output /absolute/path/release_check
```

The validation environment includes compatible neural dependencies and an external upstream CrowdHuman evaluator checkout. The gate made no network requests, installed no dependency packages, downloaded no datasets and submitted no jobs.

| Release check | Recorded result |
|---|---|
| Distributed package hashes and required source/configuration coverage | Valid |
| Import all eligible package modules | Passed; 25 modules including `study` |
| CLI help | Passed |
| Syntax of all three shell scripts | All passed |
| Full test suite | 152 passed, zero skipped |
| Two-epoch synthetic neural smoke | Passed |
| Offline wheel build | Passed |
| Offline wheel installation into a temporary target | Passed |
| Imports resolving only to that installed target | Passed |
| Installed console-entry-point help | Passed |

The two-epoch smoke fits development temperature, evaluates four held-out synthetic images at budgets 2 and 3, emits five method groups, checks allocation certificates and matching diagnostics, and replays 16 analytic certificates. Its record is [SMOKE_REPORT.json](validation/release_check/SMOKE_REPORT.json).

The wheel checks build a source copy using `--no-index --no-deps --no-build-isolation`, then install the built package into a temporary target without resolving dependencies. Import-location assertions prevent the source checkout from silently satisfying the check. `__main__.py` is intentionally excluded from module-import enumeration; the CLI is checked separately. The distribution contains 26 package Python files. [WHEEL_CHECK.json](validation/WHEEL_CHECK.json) and the wheel logs preserve the observed result and built-wheel hash. The source archive includes the wheel verification records.

The package-hash snapshot used by the gate is retained at [TESTED_PACKAGE_SHA256.json](validation/release_check/TESTED_PACKAGE_SHA256.json). The final `PACKAGE_SHA256.json` covers the source, documentation and recorded validation evidence. The tested computational source is identical to the packaged computational source. [SOURCE_FINGERPRINT.json](validation/SOURCE_FINGERPRINT.json) identifies that source. Hashes establish byte consistency, not external authenticity or scientific replication.

## GitHub and Arrhenius scope

Both GitHub YAML files parsed successfully and all checked relative documentation links resolved. See [DOCS_CHECK.json](validation/DOCS_CHECK.json). The included workflow targets core Python 3.11/3.12 jobs and a Python 3.12 CPU neural job. **No hosted GitHub Actions execution was performed before delivery.** Its reports will describe what actually runs after the source is uploaded. CI intentionally omits the external CrowdHuman checkout and reports its dependent tests as skipped; core jobs also omit the neural/COCO extras.

Arrhenius tests use temporary local paths, mocked environment/preflight boundaries and dry-run submission. They preserve account `naiss2025-5-643-gpu`, storage root `/nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls`, the configured modules and GH200 environment path. `study` planning remains the default; `--execute` is explicit, and the specification owns device/epochs/seeds/budgets. No real Slurm submission, allocation access, ARM64/GH200 execution, CUDA timing or large-study resource sizing was tested.

## Recorded environment

| Dependency | Local version |
|---|---|
| Python | 3.12.14 |
| PyTorch | 2.5.1+cpu |
| torchvision | 0.20.1+cpu |
| NumPy | 2.5.2 |
| SciPy | 1.17.0 |
| pycocotools | 2.0.11 |
| Host | Linux, x86_64 |

[ENVIRONMENT.json](validation/ENVIRONMENT.json) records these values. They describe this validation environment; the cluster must use its own compatible ARM64/CUDA runtime. The upstream CrowdHuman code remains an external dependency and is not copied into the repository.

## Scope of the evidence

These checks verify the computation, integration and experiment controls on controlled inputs. Real COCO/CrowdHuman conclusions require the intended frozen candidates, compatible checkpoints, correct data roles, full training and complete held-out evaluation. The full external-decoder matrix, native alternative-detector exporters, complete ablation factorial, official CityPersons MR, CrowdHuman JI and optimized target-device runtime are outside the recorded validation scope.

The paper's population-risk guarantee additionally requires justified detector history, independence and prior fixing of the complete policy family. Recorded hashes cannot establish those assumptions. The implementation/status map and reviewer guide explain which claim each artifact can support.
