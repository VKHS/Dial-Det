DIAL-Det: object-detection jobs on Arrhenius
==========================================

The launcher runs DIAL-Det detection experiments on COCO, CrowdHuman and CityPersons. Dataset
selection controls the exporter, quality targets and evaluation options.

SERVER SETTINGS

  Compute account: naiss2025-5-643-gpu
  Partition:       gpu
  Storage root:    /nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls
  Python venv:   <root>/code/venv_arrhenius_gh200
  GPU module:      GPU/buildenv-nvhpc/25.9-cu13.0
  Python module:   GPU/Python/3.13.5-bare-gcc-2025b-eb
  Hardware:        ARM64 Grace CPU / NVIDIA GH200 GPU

Each job uses one node, one task and one GPU. CPU count, memory and walltime are tunable;
the detection defaults are 4 CPUs, 32 GB and 4 hours (smoke: 20 minutes). These
are launch defaults, not measured requirements for a complete benchmark run.

INSTALL AND CONFIGURE

Copy the repository contents into the configured code directory:

  /nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls/code/dialdet_reconstruction

Keep run_dialdet_arrhenius.py, dialdet/, configs/ and all three shell scripts.
The standalone Python entry point is not the complete implementation.

Copy configs/arrhenius.env.example to your own configuration file, fill in the
actual data and checkpoint paths, and select it with DIAL_ENV_FILE. The file is
Bash code and is sourced only when explicitly selected. The example preserves
nonempty environment overrides; additional Python CLI options take precedence
over the corresponding defaults.

  DIAL_ENV_FILE=/absolute/path/coco.env \
    DIAL_DRY_RUN=1 bash submit_dialdet_arrhenius.sh export

The dry run prints the effective Python command and sbatch invocation without
submitting. You can inspect only the Python stage, without modules or CUDA:

  DIAL_ENV_FILE=/absolute/path/coco.env \
    DIAL_PLAN=1 bash run_dialdet_arrhenius.sh full

The launcher reuses the existing environment. It validates Python >=3.10,
ARM64, NumPy and SciPy, then checks the libraries required by the chosen stage.
Neural stages require PyTorch and check CUDA when DIAL_DEVICE=cuda; exporters
also import torchvision/Pillow; explicit COCO evaluation requires pycocotools.
evaluate and paired-statistics use CPU runtime validation. Normal jobs never
install dependencies.
study planning also uses CPU validation. With --execute, study reads device
from the JSON specification and requires PyTorch plus pycocotools; per-stage
DIAL_DEVICE/EPOCHS/BUDGETS settings do not override the specification.
Inside an Arrhenius GPU allocation, inspect the environment with:

  DIAL_ENV_FILE=/absolute/path/coco.env bash setup_dialdet_arrhenius.sh --check

The explicit --install option adds the small non-PyTorch requirements only.
PyTorch and torchvision must be a compatible ARM64 CUDA pair supplied or
supported by the site; the setup script never installs or upgrades them.

DETECTION DATASETS

  DIAL_DATASET   Native annotation input          Quality target   Evaluation
  coco          COCO instances JSON              mean of 10 IoUs   COCO AP
  crowdhuman    CrowdHuman ODGT, full-body fbox    probability@.5   AP50/MR via official evaluator
  citypersons   CityPersons MATLAB annotations    probability@.5   matching audits/exported predictions
  generic       Prepared candidate manifest       auto             matching audits

COCO is the default. An explicit --dataset option overrides this default and
is checked against the manifest by Python. The smoke test always uses synthetic
detection data and does not read the selected benchmark.

The exporter expects a compatible torchvision Faster R-CNN R50-FPN checkpoint.
It preserves candidates before final detector NMS, native scores, source IDs,
boxes and frozen appearance features. The default detector class-table sizes
are 91 for COCO and 2 (background/person) for CrowdHuman or CityPersons. Override
DIAL_NUM_CLASSES only to match the actual checkpoint. An arbitrary detector
checkpoint from another framework is not interchangeable with this exporter.

COCO STARTING WORKFLOW

1. Check the pipeline and the allocated GPU:

     bash submit_dialdet_arrhenius.sh smoke

2. Set DIAL_DATASET=coco, DIAL_IMAGES, DIAL_ANNOTATIONS and
   DIAL_DETECTOR_WEIGHTS in your environment file; leave unknown paths empty
   until you supply the actual local locations. Export frozen candidates:

     DIAL_ENV_FILE=/absolute/path/coco.env \
       DIAL_OUTPUT=/absolute/path/coco_frozen \
       bash submit_dialdet_arrhenius.sh export

   Equivalent explicit options are accepted:

     bash submit_dialdet_arrhenius.sh export-coco \
       --images /absolute/path/coco/val2017 \
       --annotations /absolute/path/coco/annotations/instances_val2017.json \
       --weights /absolute/path/fasterrcnn_resnet50_fpn.pt

   --download-weights explicitly requests the public COCO checkpoint. No
   implicit checkpoint or dataset download is performed. --max-images N is
   available for an export pilot; it changes the exported population.

3. For a public-checkpoint experiment, make two cross-fit manifests in a
   compatible Python environment (lightweight CPU work):

     python run_dialdet_arrhenius.py make-crossfit \
       --manifest /absolute/path/coco_frozen/manifest.json \
       --output /absolute/path/coco_folds --seed 20261022

4. Point DIAL_MANIFEST at one fold and DIAL_EVAL_ANNOTATIONS at the matching
   COCO instances JSON. Run the two-epoch pilot or thirty-epoch pipeline:

     DIAL_ENV_FILE=/absolute/path/coco.env \
       DIAL_MANIFEST=/absolute/path/coco_folds/fold_0.json \
       DIAL_OUTPUT=/absolute/path/coco_fold0_pilot \
       bash submit_dialdet_arrhenius.sh pilot

     DIAL_ENV_FILE=/absolute/path/coco.env \
       DIAL_MANIFEST=/absolute/path/coco_folds/fold_0.json \
       DIAL_OUTPUT=/absolute/path/coco_fold0_full \
       DIAL_TIME=24:00:00 DIAL_MEM=64G \
       bash submit_dialdet_arrhenius.sh full

   Use a distinct output directory for each fold and encoder seed. The full
   mode means one 30-epoch train/infer/evaluate pipeline. It does not mean the
   complete detector-and-dataset comparison study. pilot uses all images in
   its manifest, so prepare a smaller population if needed.

5. Repeat for fold_1, then merge both held-out prediction populations before
   calculating dataset AP. The main README documents merge-crossfit and
   evaluate. Do not average the two fold AP values.

COCO STUDY ORCHESTRATION

The study command automates the public-checkpoint COCO two-fold workflow for
the specified encoder seeds, including held-out merge, evaluation and paired
statistics. configs/coco_study.example.json supplies a five-seed starting
specification; fill in the existing candidate-manifest and COCO annotation
paths and set its device to cuda for an intended GPU run. Review the complete
protocol and output layout in docs/COCO_STUDY.md.

Create and inspect the plan in a compatible CPU Python environment:

  python run_dialdet_arrhenius.py study \
    --specification /absolute/path/coco_study.json \
    --output /absolute/path/coco_study_run

The JSON specification owns device, epochs, seeds, budgets and solver config.
The shell mode passes the specification and output only, plus --execute and
--resume when explicitly supplied. DIAL_STUDY_SPEC selects the file; an explicit
--specification option takes precedence. DIAL_MANIFEST, DIAL_DEVICE,
DIAL_EPOCHS, DIAL_BUDGETS, DIAL_CONFIG and the checkpoint-valued DIAL_RESUME
do not alter the study. Its CLI rejects conflicting per-stage options.

Inspect the exact server command without execution or submission:

  DIAL_ENV_FILE=/absolute/path/coco.env \
    DIAL_STUDY_SPEC=/absolute/path/coco_study.json \
    DIAL_OUTPUT=/absolute/path/coco_study_run \
    DIAL_DRY_RUN=1 bash submit_dialdet_arrhenius.sh study --execute

After inspecting the plan and choosing appropriate resource requests:

  DIAL_ENV_FILE=/absolute/path/coco.env \
    DIAL_STUDY_SPEC=/absolute/path/coco_study.json \
    DIAL_OUTPUT=/absolute/path/coco_study_run \
    bash submit_dialdet_arrhenius.sh study --execute

Add --resume to reuse verified completed tasks. This checks recorded task,
input and source identities; it is distinct from a training checkpoint path.
A full five-seed plan contains ten fold-training pipelines and subsequent
evaluation/statistics stages. The default 4-hour/32-GB server requests are not
measured requirements for this work. Set DIAL_TIME and DIAL_MEM according to
your pilot and site limits. The command runs tasks sequentially in one job.

Without --execute, study only writes its plan. Use direct Python for planning
when a GPU allocation is unnecessary; submitting any mode through this helper
uses the configured one-GPU partition convention. This COCO public-checkpoint
study does not establish the paper's separate focal-detector risk certificate.

CROWDHUMAN

Set DIAL_DATASET=crowdhuman and point DIAL_IMAGES to the images directory,
DIAL_ANNOTATIONS to annotation_train.odgt or annotation_val.odgt, and
DIAL_DETECTOR_WEIGHTS to the compatible person detector checkpoint. Export:

  DIAL_ENV_FILE=/absolute/path/crowdhuman.env \
    DIAL_OUTPUT=/absolute/path/crowdhuman_frozen \
    bash submit_dialdet_arrhenius.sh export

DIAL_DATASET_SPLIT can explicitly select train or val. The default is inferred
from the canonical annotation filename. The adapter uses full-body fbox and
retains ignore regions; it does not substitute head boxes for person boxes.

For pipeline/evaluate, DIAL_EVAL_ANNOTATIONS selects the original ODGT file.
DIAL_DEV_ANNOTATIONS separately selects the pipeline's development annotations.
For the focal experiment, development candidates come from out-of-fold train
images, so use annotation_train.odgt for development and annotation_val.odgt
for audit. Public cross-fitting can leave DIAL_DEV_ANNOTATIONS empty to reuse
the same evaluation annotation source for both disjoint image populations.
DIAL_CROWDHUMAN_EVALUATOR optionally selects the official CrowdDetection source
checkout (repository root, evaluate/, or APMRToolkits/ are accepted). With it,
the runner invokes the upstream AP50/MR evaluator. Without it, it writes native
CrowdHuman prediction files and matching audits. It does not relabel COCO AP
as CrowdHuman AP. JI is not supplied by this evaluator adapter.

  DIAL_ENV_FILE=/absolute/path/crowdhuman.env \
    DIAL_MANIFEST=/absolute/path/crowdhuman_roles.json \
    DIAL_DEV_ANNOTATIONS=/absolute/path/annotation_train.odgt \
    DIAL_EVAL_ANNOTATIONS=/absolute/path/annotation_val.odgt \
    DIAL_CROWDHUMAN_EVALUATOR=/absolute/path/CrowdDetection \
    DIAL_OUTPUT=/absolute/path/crowdhuman_run \
    bash submit_dialdet_arrhenius.sh full

The paper's focal CrowdHuman experiment needs out-of-fold detector exports for
encoder fitting, a separately deployed detector, and disjoint calibration and
audit populations. Annotation conversion alone does not establish those roles.
Use prepared manifests whose provenance matches the intended protocol.

CITYPERSONS

Set DIAL_DATASET=citypersons, DIAL_IMAGES to the Cityscapes image tree,
DIAL_ANNOTATIONS to the MATLAB annotations and DIAL_DETECTOR_WEIGHTS to the
compatible person detector checkpoint. DIAL_COORDINATE_ORIGIN=0 preserves
the published annotation coordinates. Use 1 only for an explicit conversion
of a different source whose values are one-based.

  DIAL_ENV_FILE=/absolute/path/citypersons.env \
    DIAL_COORDINATE_ORIGIN=0 \
    DIAL_OUTPUT=/absolute/path/citypersons_frozen \
    bash submit_dialdet_arrhenius.sh export

DIAL_CITYPERSONS_SUBSET=all is the default. The optional reasonable subset
marks short or heavily occluded persons as ignored during target preparation;
it does not run the benchmark evaluator. Native export, learned evidence,
allocation and matching audits are supported. An
exact CityPersons MR benchmark still requires its external evaluator and
subset settings; no COCO or CrowdHuman evaluator is selected for CityPersons.

SEPARATE STAGES AND RESUME

  DIAL_ENV_FILE=/absolute/path/coco.env \
    bash submit_dialdet_arrhenius.sh train --epochs 30 --seed 1

  DIAL_ENV_FILE=/absolute/path/coco.env \
    DIAL_CHECKPOINT=/absolute/path/calibrated.pt \
    bash submit_dialdet_arrhenius.sh infer --role audit --budgets 10 25 50 100 200

  DIAL_ENV_FILE=/absolute/path/coco.env \
    DIAL_RESUME=/absolute/path/last.pt DIAL_OUTPUT=/absolute/path/run \
    bash submit_dialdet_arrhenius.sh train --epochs 30

For evaluation, DIAL_PREDICTIONS points to the predictions directory or JSONL.
The Python evaluate command can be run directly in a compatible CPU environment;
it does not need a GPU. Slurm --requeue allows scheduler restarts but does not
select a checkpoint automatically. Resume explicitly from the desired saved
checkpoint, with the same data, model and training configuration.

DEVELOPMENT TUNING AND COMPLETE-POLICY RISK COLLECTION

Local-rule tuning has its own job mode. The dataset-specific annotation option
uses DIAL_DEV_ANNOTATIONS first, then DIAL_EVAL_ANNOTATIONS/DIAL_ANNOTATIONS.
It must describe the manifest's development population; the Python command
checks the model's fitting roles. For example:

  DIAL_ENV_FILE=/absolute/path/coco.env \
    DIAL_CHECKPOINT=/absolute/path/training/calibrated.pt \
    DIAL_DEV_ANNOTATIONS=/absolute/path/coco/instances_train2017.json \
    DIAL_BUDGET=100 DIAL_OUTPUT=/absolute/path/local_tuning \
    bash submit_dialdet_arrhenius.sh tune-local

CrowdHuman AP50 tuning additionally requires DIAL_CROWDHUMAN_EVALUATOR. A
missing official evaluator does not silently switch the tuning objective to
a proxy. --allow-matching-proxy explicitly requests diagnostic matching-based
tuning and records that choice.

For the independent focal-detector risk protocol, collect all 24 complete
policies on the calibration role, including uncertified and fallback outputs:

  DIAL_ENV_FILE=/absolute/path/crowdhuman.env \
    DIAL_MANIFEST=/absolute/path/focal_roles.json \
    DIAL_CHECKPOINT=/absolute/path/training/calibrated.pt \
    DIAL_CONFIG=/absolute/path/deployed_solver.json \
    DIAL_BUDGET=100 DIAL_OUTPUT=/absolute/path/risk_collection \
    bash submit_dialdet_arrhenius.sh collect-risk

The solver JSON must contain the intended deployed settings, including the
development-tuned fallback threshold. DIAL_CONFIG is passed to train, infer,
pipeline and collect-risk. Explicit CLI options override it as documented by
the Python runner. DIAL_BUDGET is one K for tune-local and collect-risk;
DIAL_BUDGETS remains the list used by infer/pipeline. Changing the checkpoint,
support, solver, fallback, budget or scoring convention changes policy identity.
Population independence still requires the actual detector training history.

Score the same registered family on development images with score-family:

  DIAL_ENV_FILE=/absolute/path/crowdhuman.env \
    DIAL_MANIFEST=/absolute/path/focal_roles.json \
    DIAL_CHECKPOINT=/absolute/path/training/calibrated.pt \
    DIAL_POLICY_RECORDS=/absolute/path/risk_collection/policy_records.json \
    DIAL_DEV_ANNOTATIONS=/absolute/path/annotation_train.odgt \
    DIAL_CROWDHUMAN_EVALUATOR=/absolute/path/CrowdDetection \
    DIAL_OUTPUT=/absolute/path/development_family_scores \
    bash submit_dialdet_arrhenius.sh score-family

For COCO, DIAL_DEV_ANNOTATIONS supplies the development instances JSON. This
mode requires the registered policy-records file and uses its fixed budget,
solver, quality and fallback definitions; DIAL_CONFIG, DIAL_BUDGET and
DIAL_BUDGETS do not replace them. It runs evidence inference on DIAL_DEVICE
and scores the manifest's development role with the selected official
evaluator. The registered policy definitions are reused; calibration loss
values are not development utility. --allow-matching-proxy is available only
for explicitly labeled diagnostics and cannot enable risk certification.

PAIRED STATISTICAL ANALYSIS

The paired-statistics command reads a JSON specification that identifies the
actual runs, images, encoder seeds, metric and paired methods. See the main
README for the specification schema and the distinction between matching
diagnostics and official dataset AP. Run it in a compatible CPU environment:

  python run_dialdet_arrhenius.py paired-statistics \
    --specification /absolute/path/paired_runs.json \
    --output /absolute/path/paired_statistics

For long evaluation jobs the same command can be dispatched through Slurm:

  DIAL_STATISTICS_SPEC=/absolute/path/paired_runs.json \
    DIAL_OUTPUT=/absolute/path/paired_statistics \
    bash submit_dialdet_arrhenius.sh paired-statistics

This stage performs CPU evaluation and is not passed neural --device or
--dataset options. The submission helper uses the configured GPU partition
and one-GPU allocation convention; the direct Python command requires no GPU
allocation. No unverified site CPU partition is invented in these scripts.

LOCAL RELEASE GATE

The distribution includes a portable release checker:

  python scripts/release_check.py --output /absolute/path/release_check
  python scripts/release_check.py --neural --output /absolute/path/release_check_neural

It verifies PACKAGE_SHA256.json, rejects missing or unrecorded executable
source, imports the package modules, runs CLI help, checks each shell file,
runs the project's unittest selection, and executes a real synthetic smoke
in a temporary directory. --neural additionally requires PyTorch and runs the
two-epoch CPU neural smoke. Logs, test counts/skips, smoke results and the
actual exit status of every command are saved in RELEASE_CHECK.json and
companion logs. A successful gate establishes local release checks only;
it does not establish benchmark accuracy, server access or GPU performance.
An intentional source modification invalidates the distributed byte hashes.

OUTPUTS AND MONITORING

Default outputs: <root>/outputs/DIALDET_<dataset>_<mode>_<job-id>/

The submission helper creates log directories before sbatch opens stdout and
stderr. Use it when changing DIAL_BASE; #SBATCH directives cannot expand shell
variables and direct sbatch uses the literal fallback log paths at the top of
the job script.

Each job records JOB_STATUS.json, its effective command, source hashes, an
immutable source snapshot, module/Python versions and hardware details.
000_MANIFEST/effective_launch.json records the actual argv, selected module
names, an explicit small settings allowlist and the SHA-256 of the selected
Bash configuration. The configuration contents and unrelated environment
variables are not copied. Requeues check the retained source snapshot for
both changed hashes and missing/unrecorded files. A
COMPLETED status means the process exited successfully. Scientific results
remain in the training, inference and evaluation outputs.

Monitor the newest job:

  tail -n 100 -F \
    /nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls/logs/DIALDET_LIVE.out \
    /nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls/logs/DIALDET_LIVE.err

Concurrent runs retain separate logs; LIVE points to the newest started job.
No credentials or full process environment are copied into the run manifest.

SETTING REFERENCE

  DIAL_ENV_FILE        trusted Bash configuration path
  DIAL_DATASET         coco, crowdhuman, citypersons, generic
  DIAL_BASE            preserved storage root by default
  DIAL_CODE_DIR        default <base>/code/dialdet_reconstruction
  DIAL_VENV_DIR        default <base>/code/venv_arrhenius_gh200
  DIAL_PYTHON          explicit existing interpreter, overrides the venv path
  DIAL_IMAGES          native dataset image directory for export
  DIAL_ANNOTATIONS     native annotations for export; fallback evaluation source
  DIAL_DETECTOR_WEIGHTS local compatible frozen detector checkpoint
  DIAL_NUM_CLASSES     detector label-table size, including background
  DIAL_DATASET_SPLIT   explicit native dataset split for pedestrian exporters
  DIAL_PERSON_CLASS_ID person label in the supplied detector (default 1)
  DIAL_COORDINATE_ORIGIN 0 preserves CityPersons annotations; 1 converts another source
  DIAL_CITYPERSONS_SUBSET all (default) or reasonable target preparation
  DIAL_MAX_IMAGES      optional export limit
  DIAL_SCORE_FLOOR     optional detector candidate score threshold
  DIAL_ROLE           initial exported candidate role (default pool)
  DIAL_MANIFEST       role-assigned candidate manifest
  DIAL_CHECKPOINT     learned evidence checkpoint for infer/tune-local/collect-risk/score-family
  DIAL_PREDICTIONS    predictions for evaluate
  DIAL_EVAL_ANNOTATIONS native evaluation annotations; overrides DIAL_ANNOTATIONS
  DIAL_DEV_ANNOTATIONS separate pipeline development source; defaults to evaluation source
  DIAL_CROWDHUMAN_EVALUATOR optional official evaluator source directory
  DIAL_OUTPUT         absolute stage output directory
  DIAL_SEED           encoder training seed (default 1)
  DIAL_EPOCHS         override pilot/full/train epoch default
  DIAL_QUALITY_MODE   optional explicit quality target override
  DIAL_BUDGETS        whitespace-separated report budgets
  DIAL_BUDGET         single report budget for tune-local/collect-risk (default 100)
  DIAL_CONFIG         solver JSON for train/infer/pipeline/collect-risk
  DIAL_POLICY_RECORDS registered complete-policy file for score-family
  DIAL_STATISTICS_SPEC actual run specification for paired-statistics
  DIAL_STUDY_SPEC     COCO study JSON; --execute starts work, --resume resumes
  DIAL_RESUME         training checkpoint for an explicit resume
  DIAL_ACCOUNT       Slurm compute account
  DIAL_PARTITION     default gpu
  DIAL_CPUS/MEM/TIME  resource requests
  DIAL_GPU_MODULE/DIAL_PYTHON_MODULE site module overrides
  DIAL_DEVICE        cuda by default
  DIAL_DRY_RUN       submit helper: print commands without submitting
  DIAL_PLAN          job wrapper: print Python command without executing

LOCAL VALIDATION LIMIT

Shell syntax, dataset routing, quoting and mocked submission were tested
locally. Active account acceptance, shared environment state and execution on
Arrhenius require a real server run and are not claimed as verified here.
