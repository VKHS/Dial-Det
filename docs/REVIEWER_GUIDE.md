# A practical review of DIAL-Det

This path lets a reader inspect the scientific decision, run a functional check, and then execute a controlled COCO experiment. [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) states the implemented scope; [VALIDATION_REPORT.md](../VALIDATION_REPORT.md) records the executed validation. The paper and supplement are identified in [SOURCE_PROVENANCE.json](../SOURCE_PROVENANCE.json).

## 1. Run the functional check

Use the dependency instructions in the [README](../README.md#quick-local-verification). With compatible PyTorch/torchvision installed, run from the repository root:

```bash
python run_dialdet_arrhenius.py smoke --neural --device cpu --output review_smoke
```

This performs actual synthetic training, development temperature fitting, held-out prediction, allocation, certificate replay and matching audits. Inspect the recorded execution status and artifacts. A small successful smoke establishes that these operations connect correctly; it supplies no estimate of real-dataset accuracy or target-server throughput. The nonneural `smoke` remains available for the numerical path.

For the package-level checks, including tests and shell syntax, use a separate destination outside the source tree:

```bash
python scripts/release_check.py --neural --output ../review_release_check
```

## 2. Trace the scientific claim to the executed objective

Read [METHOD.md](METHOD.md), then follow the [method-to-code-to-tests map](IMPLEMENTATION_STATUS.md#method-to-code-to-tests). Four details deserve explicit inspection: mass is summed **before** thresholding; the threshold acts on $r^p$; mass is fixed after decomposition/splitting; and the feasible set includes every cardinality from zero to $K$.

Check the ideal-block example and small exhaustive tests in `test_solver.py`. They establish that duplicate count does not inflate an ideal identity's normalized demand, and that the search/checker agrees with exhaustive solutions on the tested instances. Also inspect the tests for open-wedge transitivity, negative objective values, split corrections and nonfinite-evidence fallback.

To independently replay a certificate from an actual inference output, substitute its path:

```bash
python run_dialdet_arrhenius.py check-certificate \
  --certificate /path/to/inference/certificates/image_K100.json \
  --output review_certificate
```

The checked gap bounds optimization error for the recorded objective, including its coefficient-rounding envelope. It cannot establish that learned identities are correct or that AP improves. `uncertified` retains the incumbent; `fallback` uses the declared local rule and has no objective-gap certificate. Both remain part of performance and risk evaluation.

## 3. Inspect the COCO experiment before running it

Prepare a real frozen export and complete the example study specification using [COCO_STUDY.md](COCO_STUDY.md). Inspect the checkpoint identity, annotation source, candidate limit, split seed, fitting/development roles, encoder seeds, budgets and comparison endpoints.

```bash
python run_dialdet_arrhenius.py study \
  --specification configs/coco_study.example.json --output /path/review_coco
```

The default command writes a plan. The default design has five encoder seeds and two cross-fit folds per seed. In each fold, fitting and tuning use the opposite half; the selected half is reserved for audit. The internal 80/20 fitting/development split is an explicit reference choice. A completed plan is a specification of work, not an experimental result.

## 4. Execute on the real inputs and retain all reports

Run in the appropriate compute allocation; Arrhenius settings and launch previews are in [README_ARRHENIUS.txt](../README_ARRHENIUS.txt):

```bash
python run_dialdet_arrhenius.py study \
  --specification configs/coco_study.example.json --output /path/review_coco --execute
```

Use the documented `--resume` mode for interrupted study execution. Study stages, output identities and resume behavior are described in the [study guide](COCO_STUDY.md). A single pipeline or `full` shell job is only one fold/seed run. The [ablation guide](ABLATIONS.md) supplies focused variants requiring fresh training; it does not substitute for the complete manuscript factorial and external-baseline study.

Check that every audit image has a prediction record for every method/budget, including a record with an empty selection when nothing is reported. A missing image is not an empty report. Retain fallback and uncertified outputs. Their omission changes the deployed policy and the evaluated population.

## 5. Read the evidence and statistical comparison correctly

For each seed, merge both held-out folds and compute official COCO AP on the union. Average those dataset metrics across seeds. Averaging fold AP, averaging per-image AP, or pooling predictions from different seeds changes the estimand.

The paired bootstrap uses the same resampled images and seeds for both methods. Repeated image occurrences receive fresh evaluator IDs, preserving their multiplicity. Each sampled seed gets its own dataset-level AP calculation. Inspect the requested endpoint, observed contrast, seed count, population, replicate count and interval convention; fractions and percentage-point presentations must be labeled consistently. Native CrowdHuman/CityPersons metric bootstraps remain outside the supported workflow.

Verify a completed stage while its external inputs remain available:

```bash
python run_dialdet_arrhenius.py verify-run \
  --run /path/to/completed_stage --check-inputs --output /path/stage_verification
```

Keep verification output separate from the run being checked. Follow the [experiment guide](EXPERIMENTS.md) to connect each reported benchmark value to a numeric field in a verified, nonsynthetic run artifact. Retain the metric definition, evaluated population and source identity with that value.

## 6. Assess population risk as a separate claim

The focal risk workflow fixes the encoder, budget, complete policy family and fallback before calibration. It evaluates all 24 policies and the two image-wise risks, then forms 48 simultaneous bounds. Review the independence of calibration and deployment data, detector training history and prior choice of the family alongside the actual records. File hashes establish consistency of declared artifacts, not these external statistical facts. Public-checkpoint cross-fitting does not support the focal population-risk certificate.

An empirical release should therefore include the frozen input identities and roles, fitted checkpoints and development choices, complete held-out prediction/evaluator artifacts, numerical status shares, and seed-level metrics with the paired analysis specification. These let readers assess a reproducible result against the paper without confusing implementation verification, optimization certification and empirical performance.
