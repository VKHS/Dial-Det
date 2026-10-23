# Registered COCO public-checkpoint study

`study` turns a frozen COCO candidate population and a small JSON specification
into a reproducible command plan. Its default action validates the inputs and
writes that plan. Add `--execute` to run the registered commands sequentially.
The driver runs the DIAL-Det evidence model on a fixed candidate population and
records each training, evaluation and statistical-analysis task. Benchmark
comparisons require matching data, detector provenance and evaluation protocol.

## Prepare and review

Start with the local frozen candidate manifest produced by `export-coco`. The
source manifest must identify `dataset: "coco"`, contain at least six distinct
images, and assign every source record `role: "pool"`. Supply the original COCO
annotation JSON covering that entire population. An export of only a subset is
allowed, but the resulting metrics describe that subset. Keep the detector
checkpoint, export settings and candidate population fixed for all encoder
seeds and comparisons.

Copy [the example specification](../configs/coco_study.example.json), change its
two data paths and select `device: "cpu"` or `"cuda"` for your environment. Paths
inside the specification are relative to the specification's directory. The
example uses the full 30-epoch schedule, five encoder seeds and five report
budgets. No dataset or weights are downloaded by the study runner.

```bash
python run_dialdet_arrhenius.py study \
  --specification configs/coco_study.example.json \
  --output outputs/coco_public_study
```

Inspect `outputs/coco_public_study/STUDY_PLAN.json`. It contains exact argument
arrays, dependencies, generated statistics specifications, input hashes,
population identity and the implementation fingerprint. Validation streams all
candidate records to check identities, frozen file hashes, feature dimensions
and provenance. This may take time on a large export even though no model is
trained. A synthetic record remains synthetic when the manifest says COCO.

Run the reviewed plan from the same environment and source snapshot:

```bash
python run_dialdet_arrhenius.py study \
  --specification configs/coco_study.example.json \
  --output outputs/coco_public_study \
  --execute
```

The runner uses the current Python interpreter and the absolute path to the
repository CLI. For an installed package without the source-tree runner, it
uses that interpreter's `-m dialdet` entry point. Commands are passed as argument arrays without a shell. It
executes on the current machine or within the job allocation that launched it;
it does not submit further jobs. Resource allocation belongs in the Arrhenius
shell launcher. The study specification controls training device, epochs,
budgets and optional model/solver configuration for every fold.

## What one study computes

The plan creates one deterministic two-fold split using `split_seed`. For each
encoder seed, it fits one evidence model per fold on the opposite half, using
that opposite half's development partition for checkpoint selection,
calibration and the local rule's 24-threshold development search at budget 100.
Every candidate image is evaluated only by the model whose fitting and
development population excluded that image.

Each seed therefore produces two fold-level pipeline runs and one merged
held-out union. Official COCO evaluation is run again over that union. The
union AP is the seed's result: **the study never averages the two fold AP
values**. Pipeline fold evaluation files are retained for inspection, but they
are not the inputs to study-level statistics.

| Stage | Count for five seeds | Output |
| --- | ---: | --- |
| Frozen two-fold split | 1 | `folds/fold_0.json`, `folds/fold_1.json` |
| Evidence training, development tuning, inference and fold evaluation | 10 | `seed_S/fold_F/` |
| Merge complementary held-out predictions | 5 | `seed_S/union/` |
| Evaluate each merged held-out union | 5 | `seed_S/evaluation/` |
| Each predeclared paired comparison | 1 each | `statistics/comparison_NN/` |

The example declares two comparisons, making 23 tasks in total. Statistics
receive five actual union prediction files, five union audit files and the five
corresponding evaluation run manifests. They use the repository's paired
crossed image/seed bootstrap. For `AP`, `AP50` and `AP75`, COCOeval is rerun on
each resampled image population; these quantities are not approximated by
averaging per-image AP. The matching diagnostics have their own explicitly
named metrics. All reported contrasts are `method_a - method_b`, in fraction
units; negative shortfall differences favor `method_a`, whereas positive AP
differences favor it. Bootstrap AP with 2,000 repeats and five seeds can be
expensive: plan resources accordingly and predeclare any smaller exploratory
run as a separate study.

## Specification schema, version 1

Unknown fields, duplicate JSON keys, repeated seeds, repeated budgets, repeated
comparisons and non-finite numbers are rejected. Paths must identify existing
files before a plan can be created.

| Field | Default | Contract |
| --- | --- | --- |
| `schema_version` | Required | Integer 1 |
| `source_manifest` | Required | Frozen COCO pool manifest |
| `coco_annotations` | Required | Original COCO annotation JSON covering every source image |
| `seeds` | `[1,2,3,4,5]` | At least two unique integers, each 0 through 2³²−1 |
| `split_seed` | `20261022` | One fixed seed, 0 through 2³²−1 |
| `dev_fraction` | `0.2` | Strictly between 0 and 0.5, applied inside each fitting half |
| `epochs` | `30` | Positive integer |
| `budgets` | `[10,25,50,100,200]` | Unique positive integers; must include 100 |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` |
| `config` | Omitted | Optional existing JSON accepted by the pipeline's solver/model/training interface |
| `comparisons` | `[]` | Predeclared comparison objects described below |

Each comparison requires `metric`. The remaining defaults are
`method_a: "dialdet"`, `method_b: "local"`, `budget: 100`, `subset: "all"`,
`repeats: 2000`, and bootstrap `seed: 1`. Methods must be different members of
`dialdet`, `local`, `greedy`, `nms`, and `soft_nms`. The budget must appear in the
study budgets. Metrics are `AP`, `AP50`, `AP75`, `mean_image_shortfall`,
`mean_image_waste`, and `pooled_shortfall`. Subsets are `all` and `high_pressure`.
`repeats` must be at least 2. An empty `high_pressure` population cannot produce
that comparison and is rejected by the statistics command; choose comparisons
appropriate to your registered study.

Changing a registered specification, even its bytes without changing the JSON
meaning, requires a separate output directory. The specification file is itself a
hashed input. Keep the same interpreter, implementation and external files for
execution after planning. To compare ablations, use separate specifications
and output directories with the same pool and split seed, and route the
predeclared model changes through `config`.

## Execution records and verified resumption

The root directory contains two atomic control records:

- `STUDY_PLAN.json`: immutable plan, command identities, source fingerprint,
  hashes of the specification, annotations, optional configuration and every
  candidate file, population identity, and dependency graph.
- `STUDY_STATUS.json`: task status, exact task-input hashes, predecessor run
  identities, child run manifest digests and log paths.

Each executed task retains its ordinary `RUN_MANIFEST.json` and
`RUN_STATUS.json`. Its complete stdout/stderr is written to `logs/TASK.log`.
Statistics specifications are materialized in `specifications/` only when
execution is requested. The normal outer CLI also records the complete study
invocation using its standard run manifest.

After interruption, request verified reuse:

```bash
python run_dialdet_arrhenius.py study \
  --specification configs/coco_study.example.json \
  --output outputs/coco_public_study \
  --execute --resume
```

Before launching anything, the runner checks all completed tasks: recorded
commands and arguments, source identity, external input hashes, required
artifacts, all recorded output hashes, absence of unrecorded task artifacts,
and dependency run identities. A changed source or input, edited result,
missing output or mismatched run record causes rejection. This is content
consistency checking, not an external signature or proof that a declared
detector training history is true. An advisory POSIX lock prevents concurrent
runner processes from writing the same study output directory.

**`study --resume` does not recover a partially written training task.** If a
failed task directory contains files, move that task directory outside the
study and rerun with `--execute --resume`. Previously completed tasks are
verified and reused; the relocated task starts from scratch. The individual
`pipeline --resume .../training/last.pt` interface can be used separately for
manual checkpoint recovery, but this runner does not adopt such an independently
recovered directory. Do not delete completed directories or edit status files
to bypass a failed integrity check. If completed outputs or registered inputs
changed, start a separate study and preserve the earlier evidence.

## Scope of the scientific claim

This protocol holds the public detector's frozen candidate population fixed
and varies the learned evidence encoder. The run explicitly sets
`protocol: "public_checkpoint_crossfit"` and `risk_eligible: false`. It does not
train the paper's focal detector or supply the independent detector-development,
calibration and audit conditions needed for a separate statistical risk claim.
The six-image contract fixture and mock orchestration tests are marked
synthetic and do not provide benchmark performance evidence.

A reader-facing results release should provide the specification, frozen
candidate provenance, completed run manifests, per-seed union evaluations,
paired-statistics reports, runtime environment and exact implementation
revision. Report failures and compute/resource limits along with the results.
Only compare paper reference values after these real experiments complete on
the intended population and protocol.
