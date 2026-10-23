# Dataset and frozen-candidate workflows

Run the commands below from the repository root after installing the appropriate
runtime. The native detector adapter is torchvision Faster R-CNN R50-FPN. COCO,
CrowdHuman and CityPersons adapters read their native annotations and export the
same frozen candidate contract. These commands train the DIAL-Det evidence model;
they do not train a detector from scratch.

For the automated multi-seed COCO workflow, use [COCO study](COCO_STUDY.md) after
exporting the candidates. The individual commands below remain useful for
inspecting a stage or extending a dataset adapter. Use separate output directories
for different checkpoints, variants and source versions.

## Real COCO workflow

The dataset and detector weights are not bundled. Provide actual local paths. Official COCO weights can be fetched only with the explicit `--download-weights` option; otherwise use `--weights /absolute/checkpoint.pt`.

### 1. Export frozen candidates

```bash
DIAL_OUTPUT=/absolute/path/coco_frozen \
  DIAL_DATASET=coco bash submit_dialdet_arrhenius.sh export \
  --images /absolute/path/coco/val2017 \
  --annotations /absolute/path/coco/annotations/instances_val2017.json \
  --weights /absolute/path/fasterrcnn_resnet50_fpn.pt
```

This writes `manifest.json` and compressed numerical records. It keeps the detector's native score floor, at most 500 candidates by native score, and proposal/class source IDs. It bypasses final RoI detection NMS while preserving RPN proposal processing. The exact RoI feature-pooling choice is recorded in every export. Local person-only torchvision checkpoints can use `--num-classes 2`. Arbitrary MMDetection checkpoints are not interchangeable with torchvision state dictionaries.

### 2. Create two cross-fit manifests

Run the following lightweight command in a Python environment with NumPy/SciPy:

```bash
python run_dialdet_arrhenius.py make-crossfit \
  --manifest /absolute/path/coco_frozen/manifest.json \
  --output /absolute/path/coco_folds --seed 20261022
```

For each fold, the opposite half supplies training/development data and the current half supplies audit data. Each image is audited exactly once across the two folds. An internal 80/20 split of the fitting half reserves development data; this is an explicit implementation choice. Public-checkpoint experiments do not receive the paper's focal-detector population risk certificate.

### 3. Train and evaluate each fold

```bash
DIAL_OUTPUT=/absolute/path/coco_fold0_run \
  bash submit_dialdet_arrhenius.sh pipeline \
  --manifest /absolute/path/coco_folds/fold_0.json \
  --quality-mode coco --epochs 30 --seed 1 \
  --coco-annotations /absolute/path/coco/annotations/instances_val2017.json
```

Repeat using `fold_1.json` and a distinct output directory. When official COCO annotations are supplied, the pipeline tunes the 24 local-rule thresholds on development AP, then uses the chosen threshold for both the matched local baseline and numerical fallback. The DIAL-Det objective constants use the paper defaults. Without annotations, the pipeline reports matching diagnostics and identifies the local threshold as an untuned default. Official AP requires the annotation file.

`full` is an alias for one 30-epoch pipeline; `pilot` uses two epochs. Neither means the complete sixteen-setting manuscript study. Use a deliberately smaller input manifest for a smaller image population.

### 4. Merge the held-out predictions before computing AP

```bash
python run_dialdet_arrhenius.py merge-crossfit \
  --manifests /absolute/path/coco_folds/fold_0.json /absolute/path/coco_folds/fold_1.json \
  --prediction-dirs /absolute/path/coco_fold0_run/inference /absolute/path/coco_fold1_run/inference \
  --output /absolute/path/coco_crossfit_union

python run_dialdet_arrhenius.py evaluate \
  --manifest /absolute/path/coco_crossfit_union/manifest.json \
  --predictions /absolute/path/coco_crossfit_union \
  --coco-annotations /absolute/path/coco/annotations/instances_val2017.json \
  --output /absolute/path/coco_crossfit_metrics
```

Do not average the two fold APs. COCO AP is computed on their combined held-out predictions. The global image budget is imposed first; standard COCO per-category `maxDets=[1,10,100]` remains in force afterwards. Five-seed runs use separate output directories.

## CrowdHuman workflow

Use the original images directory and `annotation_train.odgt` or `annotation_val.odgt`. The native adapter reads **full-body `fbox`**, retains `mask` and `extra.ignore` regions, and keeps head-only ignore attributes separate. The ODGT schema is documented by the [dataset authors](https://www.crowdhuman.org/download.html).

```bash
DIAL_DATASET=crowdhuman DIAL_OUTPUT=/absolute/path/crowdhuman_frozen \
  bash submit_dialdet_arrhenius.sh export \
  --images /absolute/path/CrowdHuman/Images \
  --annotations /absolute/path/CrowdHuman/annotation_val.odgt \
  --weights /absolute/path/crowdhuman_fasterrcnn_r50_fpn.pt \
  --num-classes 2 --split val
```

The detector checkpoint must be a compatible torchvision state dictionary. The default has background plus person. An explicitly supplied multiclass checkpoint can use `--num-classes` and `--person-class-id`; its training history is recorded as unknown unless separately established. Only person candidates enter the pool, and filtering happens before the 500-candidate limit. Native detector scores and IDs remain unchanged. Internal image IDs include dataset and split, while the original ODGT IDs are retained for evaluation.

For the public-checkpoint protocol, use the same `make-crossfit`, per-fold pipeline and `merge-crossfit` sequence as for COCO, with `DIAL_DATASET=crowdhuman`. For the paper's focal experiment, supply its separate out-of-fold fitting and calibration/audit manifests. Set `--development-annotations` to `annotation_train.odgt` for development tuning, and `--crowdhuman-annotations` to `annotation_val.odgt` for audit evaluation; the shell equivalent is `DIAL_DEV_ANNOTATIONS` plus `DIAL_EVAL_ANNOTATIONS`. The conversion command does not manufacture detector training histories or independent data roles.

Official AP50/MR uses a local checkout of the authors' [CrowdDetection evaluator](https://github.com/megvii-model/CrowdDetection/tree/master/evaluate/APMRToolkits):

```bash
DIAL_DATASET=crowdhuman DIAL_OUTPUT=/absolute/path/crowdhuman_fold0_run \
  bash submit_dialdet_arrhenius.sh pipeline \
  --manifest /absolute/path/crowdhuman_folds/fold_0.json \
  --crowdhuman-annotations /absolute/path/CrowdHuman/annotation_val.odgt \
  --crowdhuman-evaluator /absolute/path/CrowdDetection \
  --epochs 30 --seed 1
```

The evaluator adapter calls the upstream full-body matching, AP integration and `CALTECH_-2` miss-rate routines. It stores the exact filtered population, predictions and evaluator source hashes. With these evaluation options, local-rule thresholds are tuned on development AP50. Without an evaluator checkout, the pipeline still writes native prediction ODGT and matching audits; benchmark AP50/MR is explicitly absent. JI is not computed. All reported metric values use fractions, not percentages.

## CityPersons workflow

The native importer supports the official MATLAB annotation structure described in the [CityPersons repository](https://github.com/cvgroup-njust/CityPersons/blob/master/annotations/README.txt). Point the image root at `leftImg8bit/<split>`, containing the city subdirectories:

```bash
DIAL_DATASET=citypersons DIAL_OUTPUT=/absolute/path/citypersons_frozen \
  bash submit_dialdet_arrhenius.sh export \
  --images /absolute/path/Cityscapes/leftImg8bit/val \
  --annotations /absolute/path/CityPersons/anno_val.mat \
  --weights /absolute/path/citypersons_fasterrcnn_r50_fpn.pt \
  --num-classes 2 --split val --coordinate-origin 0
```

The default coordinate setting preserves the published numeric coordinates. `--coordinate-origin 1` is an explicit conversion for a separate source that requires subtracting one from x/y; it is not applied automatically to MATLAB files. Source label 1 is pedestrian; the other source labels become ignored regions.

`--citypersons-subset all` retains every pedestrian target. `--citypersons-subset reasonable` additionally ignores targets shorter than 50 pixels or with visible/full area below 0.65. These are recorded target-preparation settings. Use the public-checkpoint cross-fit workflow with `DIAL_DATASET=citypersons` for training and allocation. Benchmark MR requires the exact official CityPersons evaluator and evaluation settings; this package does not substitute CrowdHuman MR or COCO AP.

## Generic detection manifests

A version-1 JSON manifest has `dataset`, `protocol`, and `records`; each record has `path`, `image_id`, `role`, and optionally `sha256`. Allowed roles are `train`, `dev`, `calibration`, `audit`, `pool`. An image cannot occupy multiple roles in one manifest. The candidate NPZ schema is documented in `IMPLEMENTATION_NOTES.md` and enforced by `CandidateRecord`.

Validate an export before training, and verify an existing run before using its results:

```bash
python run_dialdet_arrhenius.py validate-manifest \
  --manifest /absolute/path/coco_frozen/manifest.json \
  --output /absolute/path/coco_manifest_check

python run_dialdet_arrhenius.py verify-run \
  --run /absolute/path/coco_fold0_run \
  --check-inputs --output /absolute/path/coco_fold0_verification
```

These are direct CPU commands. Manifest validation streams every candidate file and reports role counts, dimensions, declared file hashes and the selected population identity. Missing optional source hashes are reported rather than called verified. Run verification checks recorded output hashes; `--check-inputs` additionally requires original external inputs to remain at their recorded paths. Its output directory must be separate from the run being verified.

The focal CrowdHuman risk experiment needs correctly generated out-of-fold detector exports, an independently trained deployed detector, and separate calibration/audit populations. Those detector training histories require their own experiment records. Additional frozen detectors can use the same candidate interface, provided their source IDs, features, native scores, class expansion and preprocessing are recorded faithfully.

